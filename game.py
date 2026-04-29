import os
import random
import sys
import time

import cv2
import joblib
import mediapipe as mp
import pygame

from unknown_detection import predict_with_unknown
from utils import landmarks_to_array, normalize_landmarks


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_model.pkl")
ICON_DIR = os.path.join(BASE_DIR, "..", "project_oldwithCNN", "game_icon")
BACKGROUND_PATH = os.path.join(ICON_DIR, "background.jpg")
SPRITE_SHEET_PATH = os.path.join(BASE_DIR, "assets", "pixel_characters.png")
CHARACTER_HEIGHT = 210
ANIMATION_FRAME_TIME = 0.12

# Sprite sheet crop boxes.
SPRITE_RECTS = {
    "player": {
        # Girl frames.
        "idle": [(35, 80, 210, 305)],
        "attack": [(35, 80, 210, 305), (690, 85, 255, 300), (1000, 85, 285, 300)],
        "hit": [(35, 80, 210, 305), (1280, 90, 235, 295)],
    },
    "enemy": {
        # Enemy frames.
        "idle": [(30, 520, 270, 285)],
        "attack": [(30, 520, 270, 285), (670, 510, 315, 300), (985, 510, 330, 300)],
        "hit": [(30, 520, 270, 285), (1270, 515, 255, 295)],
    },
}

WIDTH, HEIGHT = 1000, 650
HP_BAR_WIDTH = 240
HP_BAR_HEIGHT = 26
ROUND_TIME = 120
SEQ_LENGTH = 4
CONF_THRESHOLD = 0.70
GESTURE_HOLD_TIME = 0.45
GESTURE_COOLDOWN = 0.70
ENEMY_ATTACK_TIME = 6.0
ENEMY_ATTACK_DAMAGE = 0.10
ENEMY_ATTACK_DELAY_ON_GESTURE = 1.0
CAMERA_BOX = pygame.Rect(35, 430, 285, 170)

DIR_GESTURES = ["up", "down", "left", "right"]

GESTURE_SYMBOL = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}


def load_image(path, size):
    image = pygame.image.load(path).convert_alpha()
    return pygame.transform.smoothscale(image, size)


def remove_sprite_sheet_background(frame):
    # Remove only the outside dark background.
    frame = frame.convert_alpha()
    width, height = frame.get_size()

    def is_bg_like(x, y):
        r, g, b, a = frame.get_at((x, y))
        if a == 0:
            return True
        # Dark blue-gray from the sprite sheet.
        return r <= 34 and g <= 38 and b <= 52 and b >= r - 2 and abs(r - g) <= 18

    visited = set()
    stack = []

    for x in range(width):
        stack.append((x, 0))
        stack.append((x, height - 1))
    for y in range(height):
        stack.append((0, y))
        stack.append((width - 1, y))

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= width or y < 0 or y >= height or (x, y) in visited:
            continue
        visited.add((x, y))
        if not is_bg_like(x, y):
            continue

        r, g, b, a = frame.get_at((x, y))
        frame.set_at((x, y), (r, g, b, 0))

        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return frame


def trim_sprite(surface):
    # Trim empty transparent space.
    rect = surface.get_bounding_rect(min_alpha=1)
    if rect.width <= 0 or rect.height <= 0:
        return surface
    return surface.subsurface(rect).copy()


def scale_sprite_by_height(surface, target_height):
    # Keep characters the same height.
    surface = trim_sprite(surface)
    width, height = surface.get_size()
    if height <= 0:
        return surface
    target_width = max(1, int(width * target_height / height))
    return pygame.transform.smoothscale(surface, (target_width, target_height))


def load_sprite_frame(sheet, rect, target_height, mirror=False):
    # Prepare one animation frame.
    frame = sheet.subsurface(pygame.Rect(rect)).copy()
    frame = remove_sprite_sheet_background(frame)
    frame = trim_sprite(frame)
    if mirror:
        frame = pygame.transform.flip(frame, True, False)
    return scale_sprite_by_height(frame, target_height)


def load_character_frames(sheet, character_name, mirror=False):
    frames = {}
    for state, rects in SPRITE_RECTS[character_name].items():
        frames[state] = [
            load_sprite_frame(sheet, rect, CHARACTER_HEIGHT, mirror=mirror)
            for rect in rects
        ]
    return frames


class CharacterAnimator:
    def __init__(self, frames):
        self.frames = frames
        self.state = "idle"
        self.started_at = 0.0
        self.duration = None

    def play(self, state, now, duration=None):
        self.state = state
        self.started_at = now
        self.duration = duration

    def current_frame(self, now):
        frames = self.frames.get(self.state, self.frames["idle"])
        if self.state == "idle":
            frame_index = int(now / 0.45) % len(frames)
            return frames[frame_index]

        elapsed = now - self.started_at
        frame_index = min(int(elapsed / ANIMATION_FRAME_TIME), len(frames) - 1)

        if self.duration is not None and elapsed >= self.duration:
            self.state = "idle"
            self.started_at = now
            self.duration = None
            return self.frames["idle"][0]

        return frames[frame_index]


def make_camera_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (CAMERA_BOX.width, CAMERA_BOX.height))
    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_rgb = cv2.flip(frame_rgb, 0)
    return pygame.surfarray.make_surface(frame_rgb)


def new_sequence():
    return [random.choice(DIR_GESTURES) for _ in range(SEQ_LENGTH)]


def apply_damage(success, player_hp, enemy_hp):
    damage = 0.2
    if success:
        enemy_hp = max(0.0, enemy_hp - damage)
    else:
        player_hp = max(0.0, player_hp - damage)
    return player_hp, enemy_hp


def read_gesture(model, class_names, hands, drawing_utils, frame_bgr):
    frame_bgr = cv2.flip(frame_bgr, 1)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label = "no hand"
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        drawing_utils.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
        )
        landmarks = landmarks_to_array(hand_landmarks)
        vec = normalize_landmarks(landmarks)
        pred, confidence = predict_with_unknown(model, vec, threshold=CONF_THRESHOLD)

        if pred == "unknown":
            label = "unknown"
        else:
            label = class_names[pred]

    if label not in DIR_GESTURES:
        return frame_bgr, "none", confidence

    return frame_bgr, label, confidence


def draw_health_bars(surface, small_font, player_hp, enemy_hp, enemy_name):
    x_margin = 40
    y_bar = 28

    pygame.draw.rect(surface, (75, 82, 95), (x_margin, y_bar, HP_BAR_WIDTH, HP_BAR_HEIGHT), border_radius=8)
    pygame.draw.rect(
        surface,
        (75, 82, 95),
        (WIDTH - x_margin - HP_BAR_WIDTH, y_bar, HP_BAR_WIDTH, HP_BAR_HEIGHT),
        border_radius=8,
    )

    pygame.draw.rect(
        surface,
        (63, 209, 255),
        (x_margin, y_bar, int(HP_BAR_WIDTH * player_hp), HP_BAR_HEIGHT),
        border_radius=8,
    )
    pygame.draw.rect(
        surface,
        (255, 112, 112),
        (WIDTH - x_margin - HP_BAR_WIDTH, y_bar, int(HP_BAR_WIDTH * enemy_hp), HP_BAR_HEIGHT),
        border_radius=8,
    )

    surface.blit(small_font.render("Ipunors HP", True, (245, 245, 245)), (x_margin, y_bar - 24))
    surface.blit(
        small_font.render(f"{enemy_name} HP", True, (245, 245, 245)),
        (WIDTH - x_margin - HP_BAR_WIDTH, y_bar - 24),
    )


def draw_timer(surface, font, remaining_time):
    text = font.render(f"{int(remaining_time):02d}s", True, (248, 248, 248))
    surface.blit(text, (WIDTH // 2 - text.get_width() // 2, 48))


def draw_enemy_attack_bar(surface, enemy_pos, progress):
    bar_width = 150
    bar_height = 14
    bar_x = enemy_pos[0] - bar_width // 2
    # Put charge bar under the enemy.
    bar_y = enemy_pos[1] + 16
    progress = max(0.0, min(progress, 1.0))

    pygame.draw.rect(surface, (55, 64, 79), (bar_x, bar_y, bar_width, bar_height), border_radius=7)
    pygame.draw.rect(
        surface,
        (255, 84, 84),
        (bar_x, bar_y, int(bar_width * progress), bar_height),
        border_radius=7,
    )
    pygame.draw.rect(surface, (255, 220, 128), (bar_x, bar_y, bar_width, bar_height), width=2, border_radius=7)


def draw_characters(surface, player_anim, enemy_anim, player_pos, enemy_pos, enemy_attack_progress, now):
    player_img = player_anim.current_frame(now)
    enemy_img = enemy_anim.current_frame(now)

    # Keep both characters on the same floor line.
    surface.blit(player_img, player_img.get_rect(midbottom=player_pos))
    surface.blit(enemy_img, enemy_img.get_rect(midbottom=enemy_pos))
    draw_enemy_attack_bar(surface, enemy_pos, enemy_attack_progress)


def draw_sequence(surface, font, small_font, sequence, current_idx, current_gesture, message):
    # Bottom command panel.
    panel = pygame.Rect(345, 420, 620, 150)
    panel_bg = pygame.Surface((panel.width, panel.height), pygame.SRCALPHA)
    panel_bg.fill((12, 30, 28, 185))
    surface.blit(panel_bg, panel.topleft)
    pygame.draw.rect(surface, (125, 215, 200), panel, width=2, border_radius=10)

    prompt = "Hold the matching gesture in front of the camera."
    surface.blit(small_font.render(prompt, True, (225, 238, 235)), (panel.x + 22, panel.y + 22))

    current_symbol = GESTURE_SYMBOL.get(current_gesture, "-")
    target_text = f"Target {current_idx + 1}/{SEQ_LENGTH}: {GESTURE_SYMBOL[sequence[current_idx]]}"
    live_text = f"Live: {current_symbol}"
    surface.blit(small_font.render(target_text, True, (255, 225, 90)), (panel.x + 22, panel.y + 56))
    surface.blit(small_font.render(live_text, True, (235, 240, 245)), (panel.x + 245, panel.y + 56))

    if len(message) > 48:
        message = message[:45] + "..."
    surface.blit(small_font.render(message, True, (175, 245, 190)), (panel.x + 22, panel.y + 86))

    gap = 135
    start_x = panel.x + 45
    y = panel.y + 105

    for index, gesture in enumerate(sequence):
        color = (255, 225, 90) if index == current_idx else (235, 240, 245)
        label = font.render(GESTURE_SYMBOL[gesture], True, color)
        surface.blit(label, (start_x + index * gap, y))

    pygame.draw.line(surface, (125, 215, 200), (panel.x + 18, panel.bottom - 18), (panel.right - 18, panel.bottom - 18), 1)

def draw_camera_panel(surface, small_font, frame_surface, current_gesture, confidence, hold_progress):
    # Camera preview panel.
    outer = CAMERA_BOX.inflate(18, 46)
    panel_bg = pygame.Surface((outer.width, outer.height), pygame.SRCALPHA)
    panel_bg.fill((16, 23, 34, 185))
    surface.blit(panel_bg, outer.topleft)
    pygame.draw.rect(surface, (110, 140, 190), outer, width=2, border_radius=10)
    surface.blit(frame_surface, CAMERA_BOX)

    status_text = f"Live gesture: {current_gesture.upper()}   conf: {confidence:.2f}"
    surface.blit(small_font.render(status_text, True, (235, 240, 245)), (CAMERA_BOX.x, CAMERA_BOX.bottom + 8))

    hold_box = pygame.Rect(CAMERA_BOX.x, CAMERA_BOX.bottom + 32, CAMERA_BOX.width, 10)
    pygame.draw.rect(surface, (55, 64, 79), hold_box, border_radius=5)
    pygame.draw.rect(
        surface,
        (255, 210, 78),
        (hold_box.x, hold_box.y, int(hold_box.width * hold_progress), hold_box.height),
        border_radius=5,
    )

def draw_status_panel(surface, small_font, required_gesture, sequence_index, current_gesture, message):
    # Kept so old calls do not break.
    pass

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MediaPipe Gesture Battle")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 30, bold=True)
    small_font = pygame.font.SysFont("Arial", 22)
    arrow_font = pygame.font.SysFont("Segoe UI Symbol", 44, bold=True)

    background = None
    if os.path.exists(BACKGROUND_PATH):
        background = pygame.transform.smoothscale(pygame.image.load(BACKGROUND_PATH).convert(), (WIDTH, HEIGHT))

    if not os.path.exists(SPRITE_SHEET_PATH):
        print(f"[ERROR] Sprite sheet not found: {SPRITE_SHEET_PATH}")
        print("[INFO] Put pixel_characters.png inside an assets folder next to game.py.")
        pygame.quit()
        sys.exit(1)

    sprite_sheet = pygame.image.load(SPRITE_SHEET_PATH).convert_alpha()
    player_anim = CharacterAnimator(load_character_frames(sprite_sheet, "player", mirror=False))
    # Flip enemy so it faces the player.
    enemy_anim = CharacterAnimator(load_character_frames(sprite_sheet, "enemy", mirror=True))
    enemy_name = "Sickle"

    model, class_names = joblib.load(MODEL_PATH)

    mp_hands = mp.solutions.hands
    drawing_utils = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam could not be opened.")
        pygame.quit()
        sys.exit(1)

    player_hp = 1.0
    enemy_hp = 1.0
    start_time = time.time()
    current_sequence = new_sequence()
    sequence_index = 0
    game_over = False
    game_result = ""
    current_gesture = "none"
    current_confidence = 0.0
    active_gesture = None
    gesture_started_at = 0.0
    last_accept_time = 0.0
    enemy_attack_started_at = time.time()
    enemy_attack_progress = 0.0
    status_message = "Match the highlighted direction."
    frame_surface = pygame.Surface((CAMERA_BOX.width, CAMERA_BOX.height))
    frame_surface.fill((15, 15, 15))

    # Character floor positions.
    player_pos = (190, 350)
    enemy_pos = (810, 350)

    running = True
    while running:
        dt = clock.tick(30)
        now = time.time()
        remaining_time = max(0.0, ROUND_TIME - (now - start_time))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if ret:
            frame, current_gesture, current_confidence = read_gesture(
                model, class_names, hands, drawing_utils, frame
            )
            frame_surface = make_camera_surface(frame)
        else:
            current_gesture = "none"
            current_confidence = 0.0

        hold_progress = 0.0

        if not game_over:
            required_gesture = current_sequence[sequence_index]
            enemy_attack_progress = max(0.0, min((now - enemy_attack_started_at) / ENEMY_ATTACK_TIME, 1.0))

            if remaining_time <= 0:
                if player_hp > enemy_hp:
                    game_result = "TIME UP - YOU WIN!"
                elif enemy_hp > player_hp:
                    game_result = "TIME UP - YOU LOSE!"
                else:
                    game_result = "TIME UP - DRAW!"
                game_over = True
            elif now - last_accept_time < GESTURE_COOLDOWN:
                status_message = "Nice. Get ready for the next direction."
            else:
                if current_gesture == required_gesture:
                    if active_gesture != current_gesture:
                        active_gesture = current_gesture
                        gesture_started_at = now
                    hold_elapsed = now - gesture_started_at
                    hold_progress = min(hold_elapsed / GESTURE_HOLD_TIME, 1.0)
                    status_message = "Hold steady to confirm this move."

                    if hold_elapsed >= GESTURE_HOLD_TIME:
                        enemy_attack_started_at += ENEMY_ATTACK_DELAY_ON_GESTURE
                        sequence_index += 1
                        last_accept_time = now
                        active_gesture = None
                        gesture_started_at = 0.0

                        if sequence_index >= SEQ_LENGTH:
                            player_hp, enemy_hp = apply_damage(True, player_hp, enemy_hp)
                            player_anim.play("attack", now, duration=0.48)
                            enemy_anim.play("hit", now, duration=0.40)
                            current_sequence = new_sequence()
                            sequence_index = 0
                            status_message = "Combo complete. Enemy took damage."
                        else:
                            status_message = "Direction accepted."
                else:
                    active_gesture = None
                    gesture_started_at = 0.0

                    if current_gesture == "none":
                        status_message = "Show your hand and match the highlighted direction."
                    else:
                        status_message = "Wrong direction. Switch to the highlighted move."

            if not game_over and enemy_attack_progress >= 1.0:
                player_hp = max(0.0, player_hp - ENEMY_ATTACK_DAMAGE)
                enemy_anim.play("attack", now, duration=0.48)
                player_anim.play("hit", now, duration=0.40)
                enemy_attack_started_at = now
                enemy_attack_progress = 0.0
                status_message = "Enemy attacked. Player took damage."

            if enemy_hp <= 0 or player_hp <= 0:
                if player_hp <= 0 and enemy_hp <= 0:
                    game_result = "BOTH KO - DRAW!"
                elif enemy_hp <= 0:
                    game_result = "YOU WIN!"
                else:
                    game_result = "YOU LOSE!"
                game_over = True
        else:
            required_gesture = current_sequence[sequence_index]

        if background is not None:
            screen.blit(background, (0, 0))
            tint = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            tint.fill((5, 8, 14, 205))
            screen.blit(tint, (0, 0))
        else:
            screen.fill((10, 10, 20))

        draw_health_bars(screen, small_font, player_hp, enemy_hp, enemy_name)
        draw_timer(screen, font, remaining_time)
        draw_characters(screen, player_anim, enemy_anim, player_pos, enemy_pos, enemy_attack_progress, now)
        draw_camera_panel(screen, small_font, frame_surface, current_gesture, current_confidence, hold_progress)
        draw_sequence(
            screen,
            arrow_font,
            small_font,
            current_sequence,
            sequence_index,
            current_gesture,
            status_message,
        )

        vs_text = font.render("VS", True, (245, 245, 245))
        screen.blit(vs_text, (WIDTH // 2 - vs_text.get_width() // 2, 86))

        helper_text = small_font.render(
            "ESC quits. Keep your hand visible and hold each direction briefly.",
            True,
            (208, 214, 223),
        )
        screen.blit(helper_text, (345, HEIGHT - 38))

        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            result_text = font.render(game_result, True, (255, 255, 255))
            info_text = small_font.render("Press ESC to quit the match.", True, (220, 220, 220))
            screen.blit(result_text, (WIDTH // 2 - result_text.get_width() // 2, HEIGHT // 2 - 30))
            screen.blit(info_text, (WIDTH // 2 - info_text.get_width() // 2, HEIGHT // 2 + 12))

        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

    cap.release()
    hands.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

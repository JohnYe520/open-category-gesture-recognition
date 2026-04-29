def predict_with_unknown(model, X, threshold=0.6):
    # Get confidence for each class.
    probs = model.predict_proba([X])[0]
    max_prob = max(probs)

    # Low confidence means it is probably not one of our gestures.
    if max_prob < threshold:
        return "unknown", max_prob

    # Otherwise use the best class.
    pred_idx = probs.argmax()
    return pred_idx, max_prob

"""Portable, non-pickle representation of a standardized linear classifier."""

import numpy as np


def export_model(scaler, classifier, labels):
    return {
        "format_version": 1,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
        "labels": labels,
    }


def predict(model, features):
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coef = np.asarray(model["coefficients"], dtype=float)
    intercept = np.asarray(model["intercepts"], dtype=float)
    labels = model["labels"]
    if (
        x.shape != (16,)
        or mean.shape != (16,)
        or scale.shape != (16,)
        or coef.shape != (len(labels), 16)
        or intercept.shape != (len(labels),)
        or len(labels) < 2
        or not all(np.isfinite(a).all() for a in (x, mean, scale, coef, intercept))
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid input or model parameters")
    logits = coef @ ((x - mean) / scale) + intercept
    exp = np.exp(logits - np.max(logits))
    probabilities = exp / exp.sum()
    return {
        "label": labels[int(np.argmax(probabilities))],
        "labels": labels,
        "probabilities": probabilities.tolist(),
    }

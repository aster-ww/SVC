"""Evaluation metrics shared by every dataset's evaluate.py."""

import numpy as np


def pcc_rowwise(P, T):
    """Row-wise Pearson correlation. NaN where a row has zero variance."""
    Pm = P - P.mean(axis=1, keepdims=True)
    Tm = T - T.mean(axis=1, keepdims=True)
    num = (Pm * Tm).sum(axis=1)
    den = np.sqrt((Pm * Pm).sum(axis=1) * (Tm * Tm).sum(axis=1))
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def cosine_similarity(x1, x2, axis=1, eps=1e-8):
    dot = np.sum(x1 * x2, axis=axis)
    denom = np.linalg.norm(x1, axis=axis) * np.linalg.norm(x2, axis=axis)
    return np.where(denom == 0, 0.0, dot / (denom + eps))


def spatial_pcc(pred_fg, truth_fg):
    """Mean over cells of the within-cell correlation between the predicted and
    measured map, on foreground pixels.

    pred_fg, truth_fg: (n_cells, n_foreground_pixels) for ONE gene.
    """
    non_zero = truth_fg.sum(axis=1) != 0
    valid = non_zero & (np.ptp(pred_fg, axis=1) > 0) & (np.ptp(truth_fg, axis=1) > 0)
    if not valid.any():
        return np.nan
    return float(np.nanmean(pcc_rowwise(pred_fg[valid], truth_fg[valid])))

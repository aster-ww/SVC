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

_ii, _jj = np.indices((12, 12))
DIST_WEIGHT = np.sqrt((_ii + 0.5 - 6) ** 2 + (_jj + 0.5 - 6) ** 2) / 6


def distance_to_center(maps, chunk=4096):
    """Count-weighted mean distance from the nuclear center of each registered map.

    maps    : (..., 12, 12) counts, real or predicted
    returns : (...) in [0, 1], 0 at the nuclear center and 1 on the cell boundary;
              0 where a map holds no counts
    """
    maps = np.asarray(maps)
    total = maps.sum(axis=(-1, -2))
    weighted = np.empty(maps.shape[:-2], dtype=np.float64)
    for s in range(0, len(maps), chunk):
        weighted[s:s + chunk] = (maps[s:s + chunk] * DIST_WEIGHT).sum(axis=(-1, -2))
    ratio = np.where(total != 0, weighted / np.where(total == 0, 1, total), 0.0)
    return np.where(total != 0, np.minimum(ratio, 1.0), 0.0)

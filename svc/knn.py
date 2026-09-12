"""
Per-cell k-NN context builder for the neighbor-context SVC variant.

For each cell, returns the indices and Euclidean distances of its k spatially
nearest OTHER cells WITHIN the same batch / FOV (or within the full provided
set if batch_id is None — e.g. xenium where each ROI is a single section).


Usage:
    from svc.knn import build_knn
    nb_idx, nb_dist = build_knn(location, k=10, batch_id=None)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn(location, k=10, batch_id=None):
    """Compute per-cell k-NN within batch (or full set if batch_id is None).

    Args:
        location: (N, 2) float array of (x, y) coordinates.
        k: int, number of neighbors per cell (excluding self).
        batch_id: optional (N,) int/str array; kNN runs INDEPENDENTLY within
            each unique batch. None ≡ single batch over the whole input.

    Returns:
        nb_idx:  (N, k) int64 — neighbor indices into `location`.
        nb_dist: (N, k) float32 — Euclidean distances.

    """
    N = location.shape[0]
    assert location.shape == (N, 2), f"location must be (N, 2), got {location.shape}"
    if batch_id is None:
        batch_id = np.zeros(N, dtype=np.int64)
    else:
        batch_id = np.asarray(batch_id)
        assert batch_id.shape == (N,), f"batch_id must be (N,), got {batch_id.shape}"

    nb_idx  = np.full((N, k), -1, dtype=np.int64)
    nb_dist = np.full((N, k), 1e9, dtype=np.float32)

    unique_batches = np.unique(batch_id)
    for b in unique_batches:
        sel = np.where(batch_id == b)[0]                         # global indices in this batch
        n_b = sel.shape[0]
        if n_b < 2:
            print(f"  [warn] batch {b!r}: only {n_b} cell(s) — neighbors padded with self")
            for i in sel:
                nb_idx[i, :] = i                                  # pad with self
            continue
        k_eff = min(k, n_b - 1)                                  # can't return more than n_b - 1 real neighbors
        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='auto', n_jobs=-1)
        nn.fit(location[sel])
        d, ii = nn.kneighbors(location[sel])                     # (n_b, k_eff+1), local indices
        # drop self (col 0)
        d_local  = d[:, 1:k_eff + 1].astype(np.float32)          # (n_b, k_eff)
        ii_local = ii[:, 1:k_eff + 1]                            # (n_b, k_eff), local
        ii_global = sel[ii_local]                                # map back to global

        for col in range(k_eff):
            nb_idx[sel, col]  = ii_global[:, col]
            nb_dist[sel, col] = d_local[:, col]
        if k_eff < k:
            for i in sel:
                for col in range(k_eff, k):
                    nb_idx[i, col]  = i
                    nb_dist[i, col] = 1e9
            print(f"  [warn] batch {b!r}: {n_b} cells < k+1={k+1} — last {k-k_eff} slot(s) padded")

    assert (nb_idx >= 0).all(), "some nb_idx left at -1 — batch handling bug"

    print(f"  built kNN: N={N}, k={k}")
    return nb_idx, nb_dist

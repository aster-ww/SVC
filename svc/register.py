"""
Cell registration: transcript coordinates -> the nucleus-centered unit circle -> gene maps.

center:

    theta = atan2(y - centerY, x - centerX)          the angle, rounded to 0.5 degrees
    rho   = l1 / (l1 + l2)                           l1 = distance to the nuclear center,
                                                     l2 = distance from the transcript to
                                                     the cell boundary along theta
    (x_norm, y_norm) = rho * (cos theta, sin theta)

rho is 0 at the nuclear center and 1 on the cell boundary, so cells of different size and
shape share one coordinate system. The registered transcripts are then binned into a
48x48 image per (cell, gene) and 4x4 sum-pooled to the 12x12 grid.

    from svc.register import register_transcripts, gene_maps, morphology_masks

    df = register_transcripts(transcripts, cell_contour)
    maps = gene_maps(df, cell_names, gene_names)               # (n_cells, n_genes, 12, 12)
    cell_morph = morphology_masks(cell_mask, centers, cell_names)      # (n_cells, 48, 48)
"""

import numpy as np
import pandas as pd

ANGLE_STEP = 0.5       
HIGH_RES = 48
POOL = 4
LOW_RES = HIGH_RES // POOL
MASK_SCALE = 34.0        


def _boundary_distance(tx, contour):
    """l2 for every transcript of one cell: the distance to the contour point whose angle
    is closest to the transcript's own. Ties go to the first such contour point, which is
    what the original per-row argmin did."""
    c_ang = contour['direction_vec'].to_numpy()
    c_xy = contour[['x', 'y']].to_numpy(dtype=np.float64)

    t_ang = tx['direction_vec'].to_numpy()
    uniq, inverse = np.unique(t_ang, return_inverse=True)
    nearest = np.abs(c_ang[None, :] - uniq[:, None]).argmin(axis=1)   # first minimum
    picked = c_xy[nearest[inverse]]

    t_xy = tx[['x', 'y']].to_numpy(dtype=np.float64)
    return np.sqrt(((t_xy - picked) ** 2).sum(axis=1))


def register_transcripts(data_df, cell_contour_df):
    """Add distance_to_center, direction_vec, ratio, x_norm and y_norm to a transcript table.

    data_df          one row per transcript, with x, y, cell and the cell's nuclear center
                     in centerX / centerY.
    cell_contour_df  one row per cell-boundary point, with cell, x, y and direction_vec
                     (the same 0.5-degree grid used here).

    Returns a copy; the input is not modified.
    """
    df = data_df.copy()
    df['distance_to_center'] = np.sqrt((df['x'] - df['centerX']) ** 2 +
                                       (df['y'] - df['centerY']) ** 2)
    df['direction_vec'] = (np.degrees(np.arctan2(df['y'] - df['centerY'],
                                                 df['x'] - df['centerX']))
                           / ANGLE_STEP).round() * ANGLE_STEP

    ratio = np.empty(len(df), dtype=np.float64)
   
    contour_of = dict(tuple(cell_contour_df.groupby('cell', sort=False, observed=True)))
    for cell, idx in df.groupby('cell', sort=False, observed=True).indices.items():
        l2 = _boundary_distance(df.iloc[idx], contour_of[cell])
        l1 = df['distance_to_center'].to_numpy()[idx]
        ratio[idx] = l1 / (l1 + l2)
    df['ratio'] = ratio

    theta = np.radians(df['direction_vec'].to_numpy())
    df['x_norm'] = df['ratio'].to_numpy() * np.cos(theta)
    df['y_norm'] = df['ratio'].to_numpy() * np.sin(theta)
    return df


def gene_maps(df, cell_names, gene_names, high_res=HIGH_RES, pool=POOL):
    """Bin registered transcripts into one image per (cell, gene).

    df          output of register_transcripts, with x_norm, y_norm, cell and gene.
    cell_names  row order of the output; cells absent from df give empty maps.
    gene_names  gene order of the output; genes outside this list are ignored.

    Returns (n_cells, n_genes, high_res/pool, high_res/pool) uint64 counts.
    """
    low_res = high_res // pool
    edges = np.linspace(-1 - 1e-10, 1 + 1e-10, high_res + 1)
    x_bin = pd.cut(df['x_norm'], bins=edges, labels=False).to_numpy()
    y_bin = pd.cut(df['y_norm'], bins=edges, labels=False).to_numpy()

    cell_pos = {c: i for i, c in enumerate(cell_names)}
    gene_pos = {g: i for i, g in enumerate(gene_names)}
    cell_idx = df['cell'].map(cell_pos).to_numpy()
    gene_idx = df['gene'].map(gene_pos).to_numpy()

    keep = ~(pd.isna(cell_idx) | pd.isna(gene_idx) | pd.isna(x_bin) | pd.isna(y_bin))
    cell_idx = cell_idx[keep].astype(np.int64)
    gene_idx = gene_idx[keep].astype(np.int64)
    y_bin = y_bin[keep].astype(np.int64)
    x_bin = x_bin[keep].astype(np.int64)

    flat = ((cell_idx * len(gene_names) + gene_idx) * high_res + y_bin) * high_res + x_bin
    counts = np.bincount(flat, minlength=len(cell_names) * len(gene_names) * high_res * high_res)
    maps = counts.reshape(len(cell_names), len(gene_names), high_res, high_res)

    return (maps.reshape(len(cell_names), len(gene_names), low_res, pool, low_res, pool)
                .sum(axis=(3, 5))
                .astype(np.uint64))


def morphology_masks(mask_df, centers, cell_names, size=HIGH_RES, scale=MASK_SCALE):
    """Rasterize segmentation masks into one binary image per cell.

    A fixed window is centered on each cell's nuclear center and divided into
    size x size bins of `scale` pixels; a bin is 1 when any mask pixel falls in it.

    mask_df     one row per mask pixel, with cell, x and y.
    centers     {cell: (x, y)} nuclear center per cell.
    cell_names  row order of the output; 

    Returns (n_cells, size, size) uint64, values 0 or 1.
    """
    out = np.zeros((len(cell_names), size, size), dtype=np.uint64)
    half = size // 2
    groups = dict(tuple(mask_df.groupby('cell', sort=False, observed=True)))
    for i, cell in enumerate(cell_names):
        g = groups.get(cell)
        if g is None:
            continue
        cx, cy = centers[cell]
        xb = np.floor((g['x'].to_numpy() - cx) / scale + half).astype(np.int64)
        yb = np.floor((g['y'].to_numpy() - cy) / scale + half).astype(np.int64)
        keep = (xb >= 0) & (xb < size) & (yb >= 0) & (yb < size)
        out[i, yb[keep], xb[keep]] = 1
    return out

"""
Turn SVC's predicted 12x12 negative-binomial parameters back into transcript
coordinates.

Four steps:
  postprocess_sampling            (mu, r) -> sampled counts on a 48x48 grid
  postprocess_predictions         48x48 counts -> a table of (cell, gene, count,
                                  radial ratio, angle)
  postprocess_predictions_original that table -> (x, y) in the original section,
                                  by mapping each radial ratio onto the cell's
                                  own contour along the transcript's angle
  expand_transcripts              one row per pixel -> one row per transcript,
                                  optionally jittered so transcripts sharing a
                                  pixel do not render as a single dot

"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def postprocess_sampling(prediction_mu, prediction_r, seed):
    """Sample counts on a 48x48 grid from the predicted NB parameters."""
    np.random.seed(seed)
    count_data = np.zeros((prediction_mu.shape[0], prediction_mu.shape[1], 48, 48))
    for i in tqdm(range(prediction_mu.shape[0])):
        for j in range(prediction_mu.shape[1]):

            cell_gene_mu = prediction_mu[i, j]
            cell_gene_r = prediction_r[i, j]
            cell_gene_mu = np.repeat(cell_gene_mu / 4, 4, axis=0)
            cell_gene_mu = np.repeat(cell_gene_mu / 4, 4, axis=1)

            cell_gene_r = np.repeat(cell_gene_r / 4, 4, axis=0)
            cell_gene_r = np.repeat(cell_gene_r / 4, 4, axis=1)
            mask = cell_gene_r > 0
            count_data_0 = np.zeros((48, 48))
            n = cell_gene_r[mask]
            p = cell_gene_r[mask] / (cell_gene_r[mask] + cell_gene_mu[mask])
            count_data_0[mask] = np.random.negative_binomial(n, p)
            count_data[i, j] = count_data_0
    return count_data


def postprocess_predictions(count_data, selected_gene, test_cell_names):
    """48x48 sampled counts -> long table with radial ratio and angle per transcript."""
    predictions_pixel = pd.DataFrame()
    for i in selected_gene:
        count_data_i = count_data[:, selected_gene.index(i)]
        non_zero_indices = np.argwhere(count_data_i != 0)
        non_zero_values = count_data_i[non_zero_indices[:, 0], non_zero_indices[:, 1],
                                       non_zero_indices[:, 2]]

        center_x, center_y = 24, 24
        x, y = non_zero_indices[:, 2], non_zero_indices[:, 1]
        ratio = np.sqrt((x + 0.5 - center_x) ** 2 + (y + 0.5 - center_y) ** 2) / 24
        ratio = np.minimum(ratio, 1)
        angles = (np.degrees(np.arctan2(y + 0.5 - center_y, x + 0.5 - center_x)) * 2).round() / 2
        cell = np.array(test_cell_names)[non_zero_indices[:, 0]]
        gene = np.repeat(i, len(non_zero_values))
        count = non_zero_values.flatten()
        predictions_pixel_i = pd.DataFrame({
            'cell': cell,
            'gene': gene,
            'count': count,
            'x': x,
            'y': y,
            'ratio': ratio,
            'direction_vec': angles,
        })

        predictions_pixel = pd.concat([predictions_pixel, predictions_pixel_i], axis=0)

    return predictions_pixel.reset_index(drop=True)


def find_closest_point_postprocess(df1, df2, angle_col, type_col):
    """For each row of df1, the contour distance of the nearest-angle point of
    df2, scaled by that row's radial ratio."""
    closest_points = []
    for _, row in df1.iterrows():
        angle_diffs = np.abs(df2[angle_col] - row[angle_col])
        idx = angle_diffs.argmin()
        df2_idx = df2.iloc[idx]
        if df2_idx.empty:
            print("empty")
            continue
        closest_points.append(df2_idx['distance_to_center'] * row['ratio'])

    return closest_points


def postprocess_predictions_original(predictions_pixel, df_cell_contour):
    """Map the radial-ratio table back to (x, y) in the original section, using
    each cell's own contour."""
    predictions_pixel[['centerX', 'centerY']] = 0
    predictions_pixel['distance_to_center'] = 0.0

    for cell in tqdm(predictions_pixel.cell.unique()):
        cell_center_x = df_cell_contour.centerX[df_cell_contour.cell == cell].values[0]
        cell_center_y = df_cell_contour.centerY[df_cell_contour.cell == cell].values[0]

        mask = predictions_pixel["cell"] == cell

        dist = find_closest_point_postprocess(
            predictions_pixel.loc[mask],
            df_cell_contour.loc[df_cell_contour["cell"] == cell],
            "direction_vec",
            "cell",
        )

        predictions_pixel.loc[mask, ["centerX", "centerY"]] = [cell_center_x, cell_center_y]
        predictions_pixel.loc[mask, "distance_to_center"] = dist

    predictions_pixel['angle_radians'] = predictions_pixel['direction_vec'].apply(
        lambda x: np.radians(x))
    predictions_pixel['x_original'] = predictions_pixel.apply(
        lambda row: row['distance_to_center'] * np.cos(row['angle_radians']) + row['centerX'], axis=1)
    predictions_pixel['y_original'] = predictions_pixel.apply(
        lambda row: row['distance_to_center'] * np.sin(row['angle_radians']) + row['centerY'], axis=1)

    return predictions_pixel


def expand_transcripts(predictions_pixel, jitter=False, jitter_sd=10.0, seed=None):
    """One row per predicted transcript rather than per occupied pixel.

    jitter displaces every transcript by N(0, jitter_sd) in the section frame. It has to be
    applied here rather than earlier: the transcripts of one pixel all map to the same
    coordinate, so without it a pixel holding several of them still renders as a single dot.
    Off by default -- only the MERFISH prediction panel turns it on; everywhere else the
    transcripts stay exactly on their pixel's mapped position.
    """
    df = predictions_pixel.reset_index(drop=True)
    df = df.loc[df.index.repeat(df['count'].round().astype(int))].reset_index(drop=True)
    if jitter and len(df):
        rng = np.random.default_rng(seed)
        df[['x_original', 'y_original']] += rng.normal(0, jitter_sd, (len(df), 2))
    return df

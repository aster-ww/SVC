"""
Turn SVC's cross-validated predictions into transcript coordinates (seqFISH+).

Inputs:
    --pred-dir   directory holding prediction_mu.npz / prediction_r.npz, written
                 by `evaluate.py --save-predictions`
    <data-root>/seqfish/cell_mask_contour_preprocessed.pkl
                 per-cell contour table with columns cell, x, y, centerX,
                 centerY, direction_vec, distance_to_center

Output: a CSV of predicted transcripts with cell, gene, count and (x_original,
y_original) in the coordinate frame of the original section.

Usage:
    python postprocess.py --data-root /path/to/data \
        --pred-dir ./output/seqfish --out predicted_transcripts.csv \
        [--genes GeneA GeneB ...] [--seed 2026]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from svc.postprocess import (postprocess_predictions,
                             postprocess_predictions_original,
                             postprocess_sampling)

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data')
ap.add_argument('--pred-dir', required=True,
                help="directory with prediction_mu.npz / prediction_r.npz")
ap.add_argument('--out', required=True, help="output CSV")
ap.add_argument('--genes', nargs='*', default=None,
                help="genes to reconstruct; default = all (slow and large)")
ap.add_argument('--seed', type=int, default=2026, help="seed for the NB sampling")
args = ap.parse_args()

dataset_dir = os.path.join(args.data_root, 'seqfish')

gene_names = np.loadtxt(f'{dataset_dir}/gene_names.txt', dtype=str).tolist()
test_data = np.load(f"{dataset_dir}/test_seqfish.npz", allow_pickle=True)
test_cell_names = test_data['cell_names']

mu = np.load(f"{args.pred_dir}/prediction_mu.npz")['prediction']
r  = np.load(f"{args.pred_dir}/prediction_r.npz")['prediction']
print(f"predictions {mu.shape}")

selected_gene = args.genes if args.genes else gene_names
gene_idx = [gene_names.index(g) for g in selected_gene]
print(f"reconstructing {len(selected_gene)} gene(s)")

count_data = postprocess_sampling(mu[:, gene_idx], r[:, gene_idx], args.seed)
pred_pixel = postprocess_predictions(count_data, list(selected_gene), test_cell_names)
print(f"sampled {len(pred_pixel):,} transcripts")

df_cell_contour = pd.read_pickle(f"{dataset_dir}/cell_mask_contour_preprocessed.pkl")
pred_pixel = postprocess_predictions_original(pred_pixel, df_cell_contour)

pred_pixel.to_csv(args.out, index=False)
print(f"saved {args.out}")

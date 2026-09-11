"""
Build the seqFISH+ model inputs from the measured transcripts (NIH/3T3).

This is the registration step: transcripts are placed in polar coordinates around
their own cell's nuclear center and binned into the 12x12 grid the model reads, and
the segmentation masks are rasterized into the matching 48x48 morphology images.
prepare_data.py then splits the result into the train / test .npz.

Reads from <data-root>/seqfish/:
    seqfish_data_dict.pkl               data_df        one row per transcript: x, y,
                                                       gene, cell, nuclear center
                                        cell_mask_df   one row per cell-mask pixel
                                        nuclear_mask_df  the same for nuclei
    cell_mask_contour_preprocessed.pkl  cell boundary points with their angle, from the
                                        segmentation; an input, not produced here
    gene_names.txt                      the analysed panel (the 1,000 most expressed
                                        genes within the Gene2vec ortholog set)

Writes into the same directory:
    cell_gene_map_low_res.npz  "image" (n_cells, n_genes, 12, 12)
    cell_morphology.npy        (n_cells, 48, 48)
    nuclear_morphology.npy     (n_cells, 48, 48)
    cell_location.npy          (n_cells, 2) nuclear centers
    cell_names.txt
    cell_batch.txt             the field of view each cell sits in, same order

Usage:
    python preprocess.py --data-root /path/to/data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import natsort
import numpy as np
import pandas as pd

from svc.register import gene_maps, morphology_masks, register_transcripts

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data',
                help="directory holding data/seqfish/ inputs")
args = ap.parse_args()

dataset_dir = os.path.join(args.data_root, 'seqfish')

data = pd.read_pickle(f'{dataset_dir}/seqfish_data_dict.pkl')
transcripts = data['data_df']
contour = pd.read_pickle(f'{dataset_dir}/cell_mask_contour_preprocessed.pkl')
gene_names = np.loadtxt(f'{dataset_dir}/gene_names.txt', dtype=str).tolist()

cell_names = natsort.natsorted(transcripts['cell'].unique().tolist())
centers = {cell: (g['centerX'].iloc[0], g['centerY'].iloc[0])
           for cell, g in contour.groupby('cell', observed=True)}
print(f"[seqfish] {len(transcripts):,} transcripts, {len(cell_names)} cells, "
      f"{len(gene_names)} genes")

print("Registering transcripts onto the unit circle ...")
registered = register_transcripts(transcripts, contour)

print("Binning into 12x12 gene maps ...")
maps = gene_maps(registered, cell_names, gene_names)
np.savez_compressed(f'{dataset_dir}/cell_gene_map_low_res.npz', image=maps)
print(f"  cell_gene_map_low_res.npz {maps.shape}, {int(maps.sum()):,} transcripts kept")

print("Rasterizing the segmentation masks ...")
cell_morphology = morphology_masks(data['cell_mask_df'], centers, cell_names)
nuclear_morphology = morphology_masks(data['nuclear_mask_df'], centers, cell_names)
np.save(f'{dataset_dir}/cell_morphology.npy', cell_morphology)
np.save(f'{dataset_dir}/nuclear_morphology.npy', nuclear_morphology)
print(f"  cell_morphology.npy {cell_morphology.shape}, "
      f"nuclear_morphology.npy {nuclear_morphology.shape}")

location = np.array([centers[c] for c in cell_names], dtype=np.float64)
np.save(f'{dataset_dir}/cell_location.npy', location)
np.savetxt(f'{dataset_dir}/cell_names.txt', cell_names, fmt='%s')

batch = transcripts.groupby('cell', observed=True)['batch'].first().reindex(cell_names)
assert batch.notna().all(), "some cells have no field-of-view id"
np.savetxt(f'{dataset_dir}/cell_batch.txt', batch.to_numpy().astype(np.int64), fmt='%d')
print(f"  cell_location.npy {location.shape}, cell_names.txt / cell_batch.txt "
      f"{len(cell_names)} cells across {batch.nunique()} fields of view")

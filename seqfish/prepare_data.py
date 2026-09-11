"""
Build the seqFISH+ train / test .npz files consumed by train.py and evaluate.py.

Reads from <data-root>/seqfish/:
    cell_gene_map_low_res.npz   "image" (n_cells, n_genes, 12, 12)
    cell_location.npy           (n_cells, 2)
    cell_morphology.npy         (n_cells, 48, 48)
    nuclear_morphology.npy      (n_cells, 48, 48)
    gene_names.txt, cell_names.txt, cell_cycle.txt
    cell_batch.txt              the field of view each cell sits in, in cell_names order

Test set = field of view 3 (cells whose name contains "-3"); everything else is
training. The neighbor graph is not stored here: seqFISH+ builds it at run time
within each field of view, so the field-of-view id travels with the split as
"batch": train.py holds out VAL_BATCHES with it, and evaluate.py and
extract_latent.py pass it to build_knn so neighbors never cross a field.

Usage:
    python prepare_data.py --data-root /path/to/data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

TEST_BATCH = '3'

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data',
                help="directory holding data/seqfish/ inputs")
args = ap.parse_args()

dataset_dir = os.path.join(args.data_root, 'seqfish')

gene_names = np.loadtxt(f'{dataset_dir}/gene_names.txt', dtype=str)
cell_names = np.loadtxt(f'{dataset_dir}/cell_names.txt', dtype=str)

data = np.load(f'{dataset_dir}/cell_gene_map_low_res.npz')["image"]
location = np.load(f"{dataset_dir}/cell_location.npy")
cell_morphology = np.load(f"{dataset_dir}/cell_morphology.npy")
nuclear_morphology = np.load(f"{dataset_dir}/nuclear_morphology.npy")
print(f"gene maps {data.shape}, location {location.shape}, "
      f"cell morphology {cell_morphology.shape}, nuclear morphology {nuclear_morphology.shape}")

# field of view per cell, in cell_names order
batch = np.loadtxt(f'{dataset_dir}/cell_batch.txt', dtype=np.int64)
assert len(batch) == len(cell_names), \
    f"cell_batch.txt has {len(batch)} rows but cell_names.txt has {len(cell_names)}"
print(f"fields of view: {len(np.unique(batch))} -> {np.unique(batch)}")

# cell identity = cell-cycle phase, one-hot
cell_cycle = np.loadtxt(f'{dataset_dir}/cell_cycle.txt', dtype=str)
cell_cycle_uni, cell_cycle_indices = np.unique(cell_cycle, return_inverse=True)
# float32: what every consumer casts to anyway, and what the released .npz holds
cell_cycle_label = np.eye(len(cell_cycle_uni), dtype=np.float32)[cell_cycle_indices]
print(f"cell-cycle phases: {list(cell_cycle_uni)} -> one-hot {cell_cycle_label.shape}")

train_indices, test_indices = [], []
for i in range(len(cell_names)):
    if f'-{TEST_BATCH}' in cell_names[i]:
        test_indices.append(i)
    else:
        train_indices.append(i)

for split, idx in [('train', train_indices), ('test', test_indices)]:
    idx = np.asarray(idx)
    out = f"{dataset_dir}/{split}_seqfish.npz"
    np.savez_compressed(
        out,
        # counts fit int32, masks are binary; every consumer casts to these anyway
        data_ori=data[idx].astype(np.int32),
        cell_morphology=np.asarray(cell_morphology)[idx].astype(np.uint8),
        nuclear_morphology=np.asarray(nuclear_morphology)[idx].astype(np.uint8),
        location=location[idx],
        identity_label=np.asarray(cell_cycle_label)[idx],
        identity=np.asarray(cell_cycle)[idx],
        cell_names=cell_names[idx],
        batch=batch[idx],
    )
    print(f"  wrote {out}  ({len(idx)} cells)")

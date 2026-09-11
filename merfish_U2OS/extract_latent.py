"""
Extract per-(cell, gene) latent embeddings for MERFISH (U2OS).

Uses the mean of the first three Performer blocks (see svc/latent.py). Nothing
is masked: the model sees every gene. The neighbor graph is rebuilt within each
field of view, exactly as at training and evaluation time.

Output: <out> .npz with emb (n_cells, n_genes, 384) float16, cell_name, split, genes.

Usage:
    python extract_latent.py --data-root /path/to/data \
        --ckpt ./output/merfish_U2OS/checkpoint.pth \
        --out latent_merfish_U2OS.npz [--l2-normalize]
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from svc.knn import build_knn
from svc.latent import LayerAverage
from svc.losses import BACKGROUND_PIXELS
from svc.model import SVC

K_SCALES = (4,)
K_NB = max(K_SCALES)
N_LAYERS = 3
BATCH = 32
USE_CELL_IDENTITY = False
SPLITS = ['train', 'test']

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--l2-normalize', action='store_true',
                help="L2-normalize each gene token (use for cosine/co-localization)")
args = ap.parse_args()

dataset_dir = os.path.join(args.data_root, 'merfish_U2OS')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_fg = np.ones((12, 12), dtype=bool)
for (r, c) in BACKGROUND_PIXELS:
    _fg[r, c] = False
FG = _fg.reshape(1, 1, 12, 12).astype(np.float32)

gene_names = np.loadtxt(f'{dataset_dir}/gene_names.txt', dtype=str)
G = len(gene_names)

train_z = np.load(f'{dataset_dir}/train_merfish_U2OS.npz', allow_pickle=True)
train_image = train_z['data_ori']
cell_median_train = float(np.median(
    train_image.reshape(train_image.shape[0], -1).sum(axis=1)))

gene2vec_weight = torch.from_numpy(
    np.load(f'{dataset_dir}/gene2vec_weight_merfish_U2OS.npy')).float()
ckpt = torch.load(args.ckpt, map_location=device)
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
state.pop('cell_median_train', None)
state.pop('tau_per_scale', None)


def build(tau):
    m = SVC(gene2vec_weight=gene2vec_weight, n_genes_for_sn=G, k_scales=K_SCALES,
            tau_init_per_scale=tau, cell_morphology=True, nuclear_morphology=True,
            use_cell_identity=USE_CELL_IDENTITY,
            emb_dropout=0.0).to(device)
    m.load_state_dict(state, strict=True)
    m.eval()
    m.set_tau_per_scale(tau)
    m.set_cell_median(cell_median_train)
    return m


EMB, CELLNAME, SPLIT = [], [], []
for split in SPLITS:
    z = np.load(f'{dataset_dir}/{split}_merfish_U2OS.npz', allow_pickle=True)
    img = np.ascontiguousarray(z['data_ori'], np.float32)
    loc = np.ascontiguousarray(z['location'], np.float32)
    batch_id = z['batch']

    nb_idx, nb_dist = build_knn(loc, k=K_NB, batch_id=batch_id)
    tau = [float(np.median(nb_dist[:, k - 1])) for k in K_SCALES]
    full_expr = (img * FG).sum((-1, -2)).astype(np.float32)

    model = build(tau)
    n = img.shape[0]
    print(f'[{split}] {n} cells', flush=True)

    out = np.zeros((n, G, 384), np.float16)
    with LayerAverage(model, n_layers=N_LAYERS, l2_normalize=args.l2_normalize) as tap:
        with torch.no_grad():
            for s in range(0, n, BATCH):
                idx = np.arange(s, min(s + BATCH, n))
                b = len(idx)
                io = torch.from_numpy(img[idx]).to(device)
                sf = io.reshape(b, -1).sum(1) / cell_median_train
                inp = io / sf.view(b, 1, 1, 1)
                mask = torch.zeros(b, G, dtype=torch.bool, device=device)
                sn_e = torch.from_numpy(full_expr[nb_idx[idx]]).to(device)
                sn_d = torch.from_numpy(nb_dist[idx]).to(device)
                cm = torch.from_numpy(z['cell_morphology'][idx].astype(np.float32)).to(device)
                nm = torch.from_numpy(z['nuclear_morphology'][idx].astype(np.float32)).to(device)
                model(inp, mask, sn_e, sn_d, cm, nm, None)
                out[s:s + b] = tap.embedding().half().cpu().numpy()

    EMB.append(out)
    CELLNAME.append(z['cell_names'].astype(str))
    SPLIT.append(np.array([split] * n))
    del model
    torch.cuda.empty_cache()

emb = np.concatenate(EMB)
np.savez(args.out, emb=emb, cell_name=np.concatenate(CELLNAME),
         split=np.concatenate(SPLIT), genes=np.array(gene_names))
print(f'saved {args.out}  {emb.shape}')

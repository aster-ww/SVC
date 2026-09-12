"""
Impute the genes of the training panel that a target dataset does not measure.

The model is applied unchanged: the measured genes are placed at their column in
the training vocabulary, every remaining column is left at zero and masked, and
the model predicts those columns from the observed genes, the morphology and the
spatial neighborhood. The size factor is the observed count over the median of
the training cells restricted to the same measured genes, so a target panel that
is a subset of the vocabulary is not read as a set of unusually small cells.

Output: <out> .npz with prediction (n_cells, n_imputed, 12, 12) float16, r,
size_factor, genes, cell_name.

Usage:
    python impute.py --data-root /path/to/data \
        --dataset xenium_mouse_brain_tau \
        --ckpt ../checkpoints/checkpoint_Xenium_mouse_brain.pth \
        --out pred_impute_tau.npz
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from svc.losses import BACKGROUND_PIXELS
from svc.model import SVC

K_SCALES = (4, 16, 64)
BATCH = 256
VOCABULARY = 'xenium_mouse_brain'

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data')
ap.add_argument('--dataset', required=True,
                help="directory under --data-root holding the target panel")
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

vocabulary_dir = os.path.join(args.data_root, VOCABULARY)
dataset_dir = os.path.join(args.data_root, args.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocabulary = np.loadtxt(f'{vocabulary_dir}/gene_names.txt', dtype=str).tolist()
measured = np.loadtxt(f'{dataset_dir}/gene_names.txt', dtype=str).tolist()
G = len(vocabulary)
column = {gene: i for i, gene in enumerate(vocabulary)}
missing = [gene for gene in measured if gene not in column]
if missing:
    raise ValueError(f"{len(missing)} measured genes are outside the training panel: "
                     f"{missing[:5]}")
measured_cols = [column[gene] for gene in measured]
imputed_cols = sorted(set(range(G)) - set(measured_cols))
imputed_genes = [vocabulary[i] for i in imputed_cols]
print(f'{len(measured_cols)} measured / {len(imputed_cols)} imputed of {G} genes')

z = np.load(f'{dataset_dir}/test_{args.dataset}.npz', allow_pickle=True)
n = z['data_ori'].shape[0]
image = np.zeros((n, G, 12, 12), np.float32)
image[:, measured_cols] = z['data_ori'].astype(np.float32)
identity = z['identity_label'].astype(np.float32)
nb_idx = z['nb_idx'].astype(np.int64)
nb_dist = z['nb_dist'].astype(np.float32)

train_image = np.load(f'{vocabulary_dir}/train_{VOCABULARY}.npz')['data_ori']
cell_median = float(np.median(
    train_image[:, measured_cols].reshape(len(train_image), -1).sum(1)))
del train_image
print(f'{n} cells | cell median over the measured genes, on the training cells '
      f'= {cell_median:.1f}')

_fg = np.ones((12, 12), dtype=bool)
for r, c in BACKGROUND_PIXELS:
    _fg[r, c] = False
FG = _fg.reshape(1, 1, 12, 12).astype(np.float32)
full_expr = (image * FG).sum((-1, -2)).astype(np.float32)
tau = [float(np.median(nb_dist[:, k - 1])) for k in K_SCALES]

gene2vec_weight = torch.from_numpy(
    np.load(f'{vocabulary_dir}/gene2vec_weight_{VOCABULARY}.npy')).float()
ckpt = torch.load(args.ckpt, map_location=device)
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}

model = SVC(gene2vec_weight=gene2vec_weight, n_genes_for_sn=G, k_scales=K_SCALES,
            tau_init_per_scale=tau, cell_morphology=True, nuclear_morphology=True,
            use_cell_identity=True, cell_identity_dim=identity.shape[1],
            emb_dropout=0.0).to(device)
model.load_state_dict(state, strict=True)
model.eval()
model.set_tau_per_scale(tau)
model.set_cell_median(cell_median)
print(f'loaded epoch {ckpt["best_epoch"]}')

background = torch.tensor(BACKGROUND_PIXELS, device=device, dtype=torch.long)
columns = torch.as_tensor(imputed_cols, device=device)
prediction = np.zeros((n, len(imputed_cols), 12, 12), np.float16)
dispersion = np.zeros_like(prediction)
size_factor = np.zeros(n, np.float32)

with torch.no_grad():
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        io = torch.from_numpy(image[s:e]).to(device)
        sf = io.reshape(e - s, -1).sum(1) / cell_median
        mask = torch.zeros(e - s, G, dtype=torch.bool, device=device)
        mask[:, columns] = True
        sn_e = torch.from_numpy(full_expr[nb_idx[s:e]]).to(device)
        sn_d = torch.from_numpy(nb_dist[s:e]).to(device)
        cm = torch.from_numpy(z['cell_morphology'][s:e].astype(np.float32)).to(device)
        nm = torch.from_numpy(z['nuclear_morphology'][s:e].astype(np.float32)).to(device)
        ci = torch.from_numpy(identity[s:e]).to(device)
        _, mu, r = model(io / sf.view(-1, 1, 1, 1), mask, sn_e, sn_d, cm, nm, ci)
        mu = mu * sf.view(-1, 1, 1, 1)
        mu[:, :, background[:, 0], background[:, 1]] = 0
        r[:, :, background[:, 0], background[:, 1]] = 0
        prediction[s:e] = mu[:, columns].half().cpu().numpy()
        dispersion[s:e] = r[:, columns].half().cpu().numpy()
        size_factor[s:e] = sf.cpu().numpy()

np.savez(args.out, prediction=prediction, r=dispersion, size_factor=size_factor,
         genes=np.array(imputed_genes), cell_name=z['cell_names'].astype(str))
print(f'saved {args.out}  {prediction.shape}')

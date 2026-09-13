"""
Deplete one gene in silico and read the response of the others.

The depleted gene's input is clamped to a low percentile of its own distribution
among the perturbed cells, and the same clamp is applied to that gene in the
neighbor-expression context of every neighboring cell. Each downstream gene is
then masked from the input in turn and predicted from the unperturbed and the
depleted input.

For each downstream-gene prediction, the size factor is computed from the
unperturbed input and held fixed for both predictions.

Output: <out> .npz with genes, cell_name, truth_count, size_factor and
count/distance before and after, plus each cell's predicted 12x12 map in both
arms when --save-maps is given.

Usage:
    python perturb.py --data-root /path/to/data \
        --ckpt ../checkpoints/checkpoint_Xenium_breast_cancer.pth \
        --deplete LUM --cell-types Stromal --genes CRISPLD2 --save-maps \
        --out lum_depletion_CRISPLD2.npz
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
from svc.metrics import DIST_WEIGHT
from svc.model import SVC

DATASET = 'xenium_breast_cancer'
SPLITS = ['train', 'val', 'test']
K_SCALES = (4, 16, 64)
BATCH = 512            # part of the numerics; see extract_latent.py
PERCENTILE = 5
MIN_COUNTS = 5
MIN_CELLS = 20
DETECTED_IN = 0.5

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--deplete', required=True, help="the gene to clamp")
ap.add_argument('--cell-types', required=True,
                help="comma-separated cell types the depletion is applied to")
ap.add_argument('--genes', default=None,
                help="comma-separated downstream genes; default = every broadly "
                     "expressed gene of the perturbed population")
ap.add_argument('--save-maps', action='store_true',
                help="also store every cell's predicted 12x12 map in both arms; "
                     "one gene's maps are ~18 MB, so pair it with --genes")
args = ap.parse_args()

dataset_dir = os.path.join(args.data_root, DATASET)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cell_types = args.cell_types.split(',')

gene_names = np.loadtxt(f'{dataset_dir}/gene_names.txt', dtype=str).tolist()
G = len(gene_names)
depleted = gene_names.index(args.deplete)

_fg = np.ones((12, 12), dtype=bool)
for r, c in BACKGROUND_PIXELS:
    _fg[r, c] = False
FG = torch.from_numpy(_fg.astype(np.float32)).to(device)
WEIGHT = torch.from_numpy(DIST_WEIGHT.astype(np.float32)).to(device)

split = {}
for name in SPLITS:
    z = np.load(f'{dataset_dir}/{name}_{DATASET}.npz', allow_pickle=True)
    counts = z['data_ori']
    full_expr = (counts * _fg.reshape(1, 1, 12, 12)).sum((-1, -2)).astype(np.float32)
    if name == 'train':
        cell_median = float(np.median(counts.reshape(len(counts), -1).sum(1)))
    keep = np.flatnonzero(np.isin(z['identity'].astype(str), cell_types))
    nb_idx = np.asarray(z['nb_idx'], np.int64)
    nb_dist = z['nb_dist'].astype(np.float32)
    split[name] = dict(
        image=counts[keep].astype(np.float32), truth=full_expr[keep],
        cell_morphology=z['cell_morphology'][keep].astype(np.float32),
        nuclear_morphology=z['nuclear_morphology'][keep].astype(np.float32),
        identity=z['identity_label'][keep].astype(np.float32),
        sn_expr=full_expr[nb_idx[keep]], sn_dist=nb_dist[keep],
        cell_name=z['cell_names'].astype(str)[keep],
        tau=[float(np.median(nb_dist[:, k - 1])) for k in K_SCALES])
    print(f'[{name}] {len(keep)} of {len(counts)} cells are {cell_types}', flush=True)

truth = np.concatenate([split[s]['truth'] for s in SPLITS])
clamp = float(np.percentile(truth[:, depleted], PERCENTILE))
if args.genes:
    downstream = [gene_names.index(g) for g in args.genes.split(',')]
else:
    detected = (truth >= 1).mean(0)
    enough = (truth >= MIN_COUNTS).sum(0)
    downstream = [j for j in range(G)
                  if detected[j] >= DETECTED_IN and enough[j] >= MIN_CELLS and j != depleted]
print(f'{args.deplete} p{PERCENTILE} = {clamp:.0f} counts, already at or below it in '
      f'{(truth[:, depleted] <= clamp).mean() * 100:.1f}% of cells')
print(f'{len(downstream)} downstream gene(s)')
del truth

gene2vec_weight = torch.from_numpy(
    np.load(f'{dataset_dir}/gene2vec_weight_{DATASET}.npy')).float()
ckpt = torch.load(args.ckpt, map_location=device)
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model = SVC(gene2vec_weight=gene2vec_weight, n_genes_for_sn=G, k_scales=K_SCALES,
            tau_init_per_scale=split['test']['tau'], cell_morphology=True,
            nuclear_morphology=True, use_cell_identity=True,
            cell_identity_dim=split['test']['identity'].shape[1],
            emb_dropout=0.0).to(device)
model.load_state_dict(state, strict=True)
model.eval()
model.set_cell_median(cell_median)
print(f'loaded epoch {ckpt["best_epoch"]}')

ARMS = ['before', 'after']
observed = (G - 1) / float(G)
out = {f'{k}_{a}': [] for k in ('count', 'distance') for a in ARMS}
maps = {a: [] for a in ARMS}
cell_name, truth_count, size_factor, n_cells = [], [], [], 0

for name in SPLITS:
    z = split[name]
    model.set_tau_per_scale(z['tau'])
    n = len(z['image'])
    n_cells += n
    piece = {k: np.empty((n, len(downstream)), np.float32) for k in out}
    factor = np.empty((n, len(downstream)), np.float32)
    if args.save_maps:
        piece_maps = {a: np.empty((n, len(downstream), 12, 12), np.float32) for a in ARMS}
    with torch.no_grad():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            b = e - s
            image = torch.from_numpy(z['image'][s:e]).to(device)
            sn = torch.from_numpy(z['sn_expr'][s:e]).to(device)
            rest = (torch.from_numpy(z['sn_dist'][s:e]).to(device),
                    torch.from_numpy(z['cell_morphology'][s:e]).to(device),
                    torch.from_numpy(z['nuclear_morphology'][s:e]).to(device),
                    torch.from_numpy(z['identity'][s:e]).to(device))

            total = image[:, depleted].sum((-1, -2))
            scale = torch.clamp(clamp / total.clamp(min=1e-6), max=1.0).view(b, 1, 1)
            depleted_image = image.clone()
            depleted_image[:, depleted] = image[:, depleted] * scale
            depleted_sn = sn.clone()
            depleted_sn[:, :, depleted] = sn[:, :, depleted].clamp(max=clamp)
            arm_input = {'before': (image, sn), 'after': (depleted_image, depleted_sn)}

            for column, j in enumerate(downstream):
                mask = torch.zeros(b, G, dtype=torch.bool, device=device)
                mask[:, j] = True
                reference = image.clone()
                reference[:, j] = 0
                sf = reference.reshape(b, -1).sum(1) / (cell_median * observed)
                factor[s:e, column] = sf.cpu().numpy()
                for arm in ARMS:
                    this_image, this_sn = arm_input[arm]
                    scaled = this_image / sf.view(b, 1, 1, 1)
                    scaled[:, j] = 0
                    context = this_sn.clone()
                    context[:, :, j] = 0
                    _, mu, _ = model(scaled, mask, context, *rest)
                    predicted = (mu[:, j] * sf.view(b, 1, 1)) * FG
                    count = predicted.sum((-1, -2))
                    piece[f'count_{arm}'][s:e, column] = count.cpu().numpy()
                    piece[f'distance_{arm}'][s:e, column] = torch.where(
                        count > 0, (predicted * WEIGHT).sum((-1, -2)) / count.clamp(min=1e-12),
                        count.new_zeros(())).clamp(max=1).cpu().numpy()
                    if args.save_maps:
                        piece_maps[arm][s:e, column] = predicted.cpu().numpy()
            if s % (BATCH * 8) == 0:
                print(f'  [{name}] {e}/{n}', flush=True)

    for k in out:
        out[k].append(piece[k])
    if args.save_maps:
        for a in ARMS:
            maps[a].append(piece_maps[a])
    size_factor.append(factor)
    cell_name.append(z['cell_name'])
    truth_count.append(z['truth'][:, downstream])

saved = {k: np.concatenate(v) for k, v in out.items()}
saved.update(genes=np.array([gene_names[j] for j in downstream]),
             cell_name=np.concatenate(cell_name),
             truth_count=np.concatenate(truth_count),
             size_factor=np.concatenate(size_factor),
             clamp=np.float32(clamp))
if args.save_maps:
    saved.update({f'map_{a}': np.concatenate(maps[a]) for a in ARMS})
np.savez(args.out, **saved)
print(f'saved {args.out}  {n_cells} cells x {len(downstream)} gene(s)')

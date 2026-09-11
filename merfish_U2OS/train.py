"""
Train SVC on MERFISH (U2OS). Single-scale k=4 neighborhood, kNN within each field of view, D4 augmentation on. No cell
identity (single clonal line). Validation = fields of view 1-5.

Usage:
    python train.py --data-root /path/to/data --out-dir ./output [--seed 2026]
"""

import argparse
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from svc.augment import _D4_ELEMENTS, apply_d4
from svc.knn import build_knn
from svc.losses import (BACKGROUND_PIXELS, compute_cell_sum_loss,
                        compute_loss_per_gene, compute_size_factor_obs,
                        foreground_mask, lambda_sum_at)
from svc.model import SVC
from svc.scheduler import CosineAnnealingWarmupRestarts

# ---- dataset-specific settings ----
VAL_BATCHES = [1, 2, 3, 4, 5]   # fields of view held out for validation
K_SCALES = (4,)                
K_NB = max(K_SCALES)
USE_CELL_IDENTITY = False     
USE_AUGMENT = True              # D4 augmentation 
BATCH_SIZE = 32
NUM_EPOCHS = 400
LEARNING_RATE = 1e-4
WEIGHT_DECAY, FILM_WEIGHT_DECAY, GENE_EMB_WEIGHT_DECAY = 1e-2, 1e-3, 1e-3
LAMBDA_WARMUP_START, LAMBDA_WARMUP_END = 50, 100
EARLY_STOP_PATIENCE = 50
MASK_PROB = 0.1
AMP_MIN_BATCH = 128

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data',
                help="directory holding data/merfish_U2OS/ inputs")
ap.add_argument('--out-dir', default='./output/merfish_U2OS',
                help="where the checkpoint and loss curves are written")
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--lambda-cell', type=float, default=10.0,
                help="final weight of the cell-level expression loss")
ap.add_argument('--no-augment', action='store_true', help="disable D4 augmentation")
ap.add_argument('--amp', choices=['auto', 'on', 'off'], default='auto')
args = ap.parse_args()

seed = args.seed
dataset_dir = os.path.join(args.data_root, 'merfish_U2OS')
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

LAMBDA_SUM_MAX = args.lambda_cell
use_augment = USE_AUGMENT and not args.no_augment


def compute_tau_per_scale(nb_dist, k_scales):
    """tau_s = median distance to the k_s-th neighbor."""
    return [float(np.median(nb_dist[:, k - 1])) for k in k_scales]


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


_fg = np.ones((12, 12), dtype=bool)
for (r, c) in BACKGROUND_PIXELS:
    _fg[r, c] = False
_FG_FILTER = _fg.reshape(1, 1, 12, 12).astype(np.float32)


def compute_full_expr(data_ori):
    return (data_ori * _FG_FILTER).sum(axis=(-1, -2)).astype(np.float32)


# ---- load, and split off the validation fields of view ----
train_data = np.load(f"{dataset_dir}/train_merfish_U2OS.npz", allow_pickle=True)
batch_of_train = train_data['batch']
val_mask_cells = np.isin(batch_of_train, VAL_BATCHES)
train_mask_cells = ~val_mask_cells

tr_image = _f32(train_data['data_ori'][train_mask_cells])
tr_loc   = _f32(train_data['location'][train_mask_cells])
tr_cm    = _f32(train_data['cell_morphology'][train_mask_cells])
tr_nm    = _f32(train_data['nuclear_morphology'][train_mask_cells])
tr_batch = batch_of_train[train_mask_cells]

va_image = _f32(train_data['data_ori'][val_mask_cells])
va_loc   = _f32(train_data['location'][val_mask_cells])
va_cm    = _f32(train_data['cell_morphology'][val_mask_cells])
va_nm    = _f32(train_data['nuclear_morphology'][val_mask_cells])
va_batch = batch_of_train[val_mask_cells]

n_train, n_val = tr_image.shape[0], va_image.shape[0]
n_genes = tr_image.shape[1]
print(f"[merfish_U2OS] train={n_train}  val(batches={VAL_BATCHES})={n_val}  "
      f"genes={n_genes}")

cell_median_train = np.median(tr_image.reshape(n_train, -1).sum(axis=1))

# ---- neighborhood graph, built within each field of view ----
print(f"Building per-cell k-NN (k_max={K_NB}, K_SCALES={K_SCALES}) within each batch ...")
tr_nb_idx, tr_sn_dist = build_knn(tr_loc, k=K_NB, batch_id=tr_batch)
va_nb_idx, va_sn_dist = build_knn(va_loc, k=K_NB, batch_id=va_batch)
tau_per_scale_train = compute_tau_per_scale(tr_sn_dist, K_SCALES)
tau_per_scale_val   = compute_tau_per_scale(va_sn_dist, K_SCALES)
print(f"  tau_per_scale_train={[f'{t:.2f}' for t in tau_per_scale_train]}, "
      f"tau_per_scale_val={[f'{t:.2f}' for t in tau_per_scale_val]}")

tr_full_expr = compute_full_expr(tr_image)                                # (n_train, G)
va_full_expr = compute_full_expr(va_image)
tr_sn_expr = tr_full_expr[tr_nb_idx]                                      # (n_train, k, G)
va_sn_expr = va_full_expr[va_nb_idx]


class SVC_Dataset_SN(Dataset):
    """D4 augmentation applies the SAME group element to the gene map and both
    morphology masks. The neighbor context is a per-gene total, so invariant."""

    def __init__(self, data_ori, cm, nm, sn_expr, sn_dist, augment=False):
        self.data_ori = data_ori
        self.cm = cm; self.nm = nm
        self.sn_expr = sn_expr; self.sn_dist = sn_dist
        self.augment = augment

    def __len__(self):
        return len(self.data_ori)

    def __getitem__(self, i):
        gm = self.data_ori[i]; cm = self.cm[i]; nm = self.nm[i]
        sn_expr = self.sn_expr[i]; sn_dist = self.sn_dist[i]
        if self.augment:
            elem = _D4_ELEMENTS[np.random.randint(8)]
            gm = apply_d4(gm, elem, (1, 2))
            cm = apply_d4(cm, elem, (0, 1))
            nm = apply_d4(nm, elem, (0, 1))
        return gm, cm, nm, sn_expr, sn_dist


train_dataset = SVC_Dataset_SN(tr_image, tr_cm, tr_nm,
                               tr_sn_expr, tr_sn_dist, augment=use_augment)
print(f"  D4 augmentation: {use_augment}")


def _seed_worker(worker_id):
    np.random.seed((seed + worker_id) % (2**31 - 1))


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                          worker_init_fn=_seed_worker, pin_memory=True, prefetch_factor=4)

use_amp = {'on': True, 'off': False}.get(args.amp, BATCH_SIZE >= AMP_MIN_BATCH)
print(f"  AMP(bf16 autocast): {use_amp}")

# ---- validation tensors, kept on the GPU ----
val_inputs_ori = torch.from_numpy(va_image).float().to(device)
val_cm = torch.from_numpy(va_cm).float().to(device)
val_nm = torch.from_numpy(va_nm).float().to(device)
val_sn_expr_t = torch.from_numpy(va_sn_expr).float().to(device)
val_sn_dist_t = torch.from_numpy(va_sn_dist).float().to(device)

_g = torch.Generator().manual_seed(seed)
_rand = torch.rand(n_val, n_genes, generator=_g)
_num_mask = int(MASK_PROB * n_genes)
_, _idx = torch.topk(-_rand, _num_mask, dim=1)
val_mask = torch.zeros(n_val, n_genes, dtype=torch.bool)
val_mask.scatter_(1, _idx, True); val_mask = val_mask.to(device)

# ---- model ----
gene2vec_weight = torch.from_numpy(
    np.load(f'{dataset_dir}/gene2vec_weight_merfish_U2OS.npy')).float()
model = SVC(
    gene2vec_weight=gene2vec_weight,
    n_genes_for_sn=n_genes,
    k_scales=K_SCALES,
    tau_init_per_scale=tau_per_scale_train,
    cell_morphology=True,
    nuclear_morphology=True,
    use_cell_identity=USE_CELL_IDENTITY,
    emb_dropout=0.0,
).to(device)
model.set_cell_median(float(cell_median_train))

foreground_expanded = foreground_mask(device)
film_param_names = set(model.film_gamma_param_names())
gene_emb_names   = set(model.gene_embedding_param_names())
film_params, gene_params, other_params = [], [], []
for name, p in model.named_parameters():
    if name in film_param_names: film_params.append(p)
    elif name in gene_emb_names: gene_params.append(p)
    else: other_params.append(p)
param_groups = [{"params": other_params, "weight_decay": WEIGHT_DECAY}]
if film_params: param_groups.append({"params": film_params, "weight_decay": FILM_WEIGHT_DECAY})
if gene_params: param_groups.append({"params": gene_params, "weight_decay": GENE_EMB_WEIGHT_DECAY})

optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=NUM_EPOCHS, cycle_mult=2.0,
                                          max_lr=LEARNING_RATE, min_lr=LEARNING_RATE / 100)


def compute_val_loss():
    model.eval()
    model.set_tau_per_scale(tau_per_scale_val)
    with torch.no_grad():
        sf_input, sf_mu = compute_size_factor_obs(val_inputs_ori, val_mask, cell_median_train)
        inputs = val_inputs_ori / sf_input
        obs_mask_4d = (1 - val_mask.float()).view(*val_inputs_ori.shape[:2], 1, 1)
        inputs = inputs * obs_mask_4d
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            _, mu, r = model(inputs, val_mask, val_sn_expr_t, val_sn_dist_t, val_cm, val_nm, None)
        mu = mu.float() * sf_mu
        loss_nb = (compute_loss_per_gene(val_inputs_ori, mu, r.float(), foreground_expanded)
                   * val_mask).sum() / val_mask.sum()
        loss_cell = compute_cell_sum_loss(mu, val_inputs_ori, val_mask, foreground_expanded)

    model.set_tau_per_scale(tau_per_scale_train)
    return loss_nb.item(), loss_cell.item()


train_losses, val_losses = [], []
best_val_loss, best_val_state, best_val_epoch = float("inf"), None, None
stop_patience_counter = 0

print(f"Starting merfish_U2OS training "
      f"(lambda 0->{LAMBDA_SUM_MAX} over ep {LAMBDA_WARMUP_START}->{LAMBDA_WARMUP_END}, "
      f"patience={EARLY_STOP_PATIENCE})...")
for epoch in range(NUM_EPOCHS):
    lambda_sum_t = lambda_sum_at(epoch, LAMBDA_SUM_MAX, LAMBDA_WARMUP_START, LAMBDA_WARMUP_END)
    model.train()
    running_loss = running_nb = running_cell = 0.0
    for i, batch in enumerate(train_loader):
        gm, cm, nm, sn_expr, sn_dist = batch
        gm = gm.to(device, non_blocking=True).float()
        cm = cm.to(device, non_blocking=True).float()
        nm = nm.to(device, non_blocking=True).float()
        sn_expr = sn_expr.to(device, non_blocking=True).float()
        sn_dist = sn_dist.to(device, non_blocking=True).float()

        rand = torch.rand(gm.shape[0], gm.shape[1], device=device)
        num_mask = int(MASK_PROB * gm.shape[1])
        _, idx = torch.topk(-rand, num_mask, dim=1)
        mask = torch.zeros(gm.shape[0], gm.shape[1], dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)

        sf_input, sf_mu = compute_size_factor_obs(gm, mask, cell_median_train)
        inputs = gm / sf_input
        obs_mask_4d = (1 - mask.float()).view(gm.shape[0], gm.shape[1], 1, 1)
        inputs = inputs * obs_mask_4d
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            _, mu, r = model(inputs, mask, sn_expr, sn_dist, cm, nm, None)
        mu = mu.float() * sf_mu
        loss_nb = (compute_loss_per_gene(gm, mu, r.float(), foreground_expanded)
                   * mask).sum() / mask.sum()
        loss_cell = compute_cell_sum_loss(mu, gm, mask, foreground_expanded)
        loss = loss_nb + lambda_sum_t * loss_cell
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        running_loss += loss.item(); running_nb += loss_nb.item(); running_cell += loss_cell.item()

    avg_train_loss = running_loss / (i + 1)
    avg_train_nb   = running_nb / (i + 1)
    avg_train_cell = running_cell / (i + 1)
    avg_val_nb, avg_val_cell = compute_val_loss()
    val_loss_for_sel = avg_val_nb + LAMBDA_SUM_MAX * avg_val_cell
    train_losses.append(avg_train_loss); val_losses.append(val_loss_for_sel)

    # Selection only once the cell-level expression loss is at full weight.
    in_stable = (epoch + 1) > LAMBDA_WARMUP_END
    if in_stable:
        if val_loss_for_sel < best_val_loss:
            best_val_loss = val_loss_for_sel
            best_val_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_epoch = epoch + 1
            stop_patience_counter = 0
        else:
            stop_patience_counter += 1
    scheduler.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  train={avg_train_loss:.4f}  "
              f"val={val_loss_for_sel:.4f}  "
              f"patience={stop_patience_counter}/{EARLY_STOP_PATIENCE}")

    if stop_patience_counter >= EARLY_STOP_PATIENCE:
        print(f"[EARLY STOP] validation did not improve for {EARLY_STOP_PATIENCE} "
              f"consecutive stable epochs. Exit at epoch {epoch+1}.")
        break

if best_val_state is not None:
    torch.save({"model_state_dict": best_val_state, "best_epoch": best_val_epoch},
               f"{out_dir}/checkpoint.pth")
    np.save(f"{out_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{out_dir}/val_losses.npy", np.array(val_losses))
    print(f"\nmerfish_U2OS done. Best epoch: {best_val_epoch}, "
          f"train: {train_losses[best_val_epoch-1]:.4f}, val: {best_val_loss:.4f}")
else:
    print("\n[WARN] no checkpoint saved (training ended before the stable phase)")

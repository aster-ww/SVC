"""
Train SVC on the Xenium mouse brain section. Multi-scale k=4/16/64
neighborhood, each scale with its own temperature; the kNN graph is
precomputed by prepare_data.py.

Usage:
    torchrun --nproc_per_node=4 train.py --data-root /path/to/data --out-dir ./output
    python train.py --data-root /path/to/data --out-dir ./output      # single GPU
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from svc.losses import (BACKGROUND_PIXELS, compute_cell_sum_loss,
                        compute_loss_per_gene, compute_size_factor_obs,
                        foreground_mask, lambda_sum_at)
from svc.model import SVC
from svc.scheduler import CosineAnnealingWarmupRestarts

if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    dist.init_process_group(backend='nccl')
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    DDP_MODE = WORLD_SIZE > 1
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f'cuda:{LOCAL_RANK}')
else:
    LOCAL_RANK = 0; WORLD_SIZE = 1; DDP_MODE = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IS_RANK0 = (LOCAL_RANK == 0)


def rank0_print(*a, **kw):
    if IS_RANK0:
        print(*a, **kw)


K_SCALES = (4, 16, 64)       
USE_CELL_IDENTITY = True
TOTAL_BATCH = 128
ACCUM_ITER = 2
NUM_EPOCHS = 400
LEARNING_RATE = 1e-4
WEIGHT_DECAY, FILM_WEIGHT_DECAY, GENE_EMB_WEIGHT_DECAY = 1e-2, 1e-3, 1e-3
LAMBDA_WARMUP_START, LAMBDA_WARMUP_END = 50, 100
EARLY_STOP_PATIENCE = 50
MASK_PROB = 0.1
EVAL_CHUNK = 256            

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data',
                help="directory holding data/xenium_mouse_brain/ inputs")
ap.add_argument('--out-dir', default='./output/xenium_mouse_brain',
                help="where the checkpoint and loss curves are written")
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--lambda-cell', type=float, default=10.0,
                help="final weight of the cell-level expression loss")
args = ap.parse_args()

seed = args.seed
dataset_dir = os.path.join(args.data_root, 'xenium_mouse_brain')
out_dir = args.out_dir
if IS_RANK0:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(seed + LOCAL_RANK)
random.seed(seed + LOCAL_RANK)
np.random.seed(seed + LOCAL_RANK)

LAMBDA_SUM_MAX = args.lambda_cell


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


_fg = np.ones((12, 12), dtype=bool)
for (r, c) in BACKGROUND_PIXELS:
    _fg[r, c] = False
_FG_FILTER = _fg.reshape(1, 1, 12, 12).astype(np.float32)


def compute_full_expr(data_ori):
    return (data_ori * _FG_FILTER).sum(axis=(-1, -2)).astype(np.float32)


def compute_tau_per_scale(nb_dist, k_scales):
    """tau_s = median distance to the k_s-th neighbor."""
    return [float(np.median(nb_dist[:, k - 1])) for k in k_scales]


class SVC_Dataset_SN(Dataset):
    """Neighbor context is gathered once at construction from `nb_idx`."""

    def __init__(self, data_ori, cm, nm, ci, nb_idx, nb_dist):
        self.data_ori = data_ori
        self.cm = cm; self.nm = nm; self.ci = ci
        full_expr = compute_full_expr(data_ori)                      # (N, G)
        self.sn_expr = full_expr[nb_idx]                             # (N, k_max, G)
        self.sn_dist = nb_dist

    def __len__(self):
        return len(self.data_ori)

    def __getitem__(self, i):
        return (self.data_ori[i], self.cm[i], self.nm[i], self.ci[i],
                self.sn_expr[i], self.sn_dist[i])


train_data = np.load(f"{dataset_dir}/train_xenium_mouse_brain.npz")
val_data   = np.load(f"{dataset_dir}/val_xenium_mouse_brain.npz")

train_image = np.ascontiguousarray(train_data['data_ori'], dtype=np.float32)
val_image   = np.ascontiguousarray(val_data['data_ori'], dtype=np.float32)

tr_nb_idx  = np.asarray(train_data['nb_idx'], dtype=np.int64)
tr_sn_dist = _f32(train_data['nb_dist'])
va_nb_idx  = np.asarray(val_data['nb_idx'], dtype=np.int64)
va_sn_dist = _f32(val_data['nb_dist'])

assert tr_nb_idx.shape[1] >= max(K_SCALES), \
    f"npz has k_max={tr_nb_idx.shape[1]}, need >= {max(K_SCALES)}; re-run prepare_data.py"

tau_per_scale_train = compute_tau_per_scale(tr_sn_dist, K_SCALES)
tau_per_scale_val   = compute_tau_per_scale(va_sn_dist, K_SCALES)
n_identity_types = train_data['identity_label'].shape[1]
rank0_print(f"  K_SCALES={K_SCALES}, n_identity_types={n_identity_types}")
rank0_print(f"  tau_per_scale_train = {[f'{t:.2f}' for t in tau_per_scale_train]}")
rank0_print(f"  tau_per_scale_val   = {[f'{t:.2f}' for t in tau_per_scale_val]}")

_train_base = SVC_Dataset_SN(
    data_ori=train_image,
    cm=_f32(train_data["cell_morphology"]),
    nm=_f32(train_data["nuclear_morphology"]),
    ci=_f32(train_data["identity_label"]),
    nb_idx=tr_nb_idx, nb_dist=tr_sn_dist,
)
cell_median_train = np.median(train_image.reshape(train_image.shape[0], -1).sum(axis=1))

batch_size = TOTAL_BATCH // WORLD_SIZE


def _seed_worker(worker_id):
    np.random.seed((seed + LOCAL_RANK * 1000 + worker_id) % (2**31 - 1))


train_sampler = DistributedSampler(_train_base, num_replicas=WORLD_SIZE,
                                   rank=LOCAL_RANK, shuffle=True, seed=seed,
                                   drop_last=False) if DDP_MODE else None
train_loader = DataLoader(
    _train_base, batch_size=batch_size, shuffle=(train_sampler is None),
    sampler=train_sampler, num_workers=2, worker_init_fn=_seed_worker,
    pin_memory=True, persistent_workers=True, prefetch_factor=4,
)

if IS_RANK0:
    val_inputs_ori = torch.from_numpy(val_image).float().to(device)
    val_cm_t = torch.from_numpy(_f32(val_data['cell_morphology'])).to(device)
    val_nm_t = torch.from_numpy(_f32(val_data['nuclear_morphology'])).to(device)
    val_id_t = torch.from_numpy(_f32(val_data['identity_label'])).to(device)
    val_full_expr = compute_full_expr(val_image)
    val_sn_expr_t = torch.from_numpy(val_full_expr[va_nb_idx]).to(device)
    val_sn_dist_t = torch.from_numpy(va_sn_dist).to(device)
    n_val, n_genes = val_image.shape[0], val_image.shape[1]

    _g = torch.Generator().manual_seed(seed)
    _rand = torch.rand(n_val, n_genes, generator=_g)
    _num_mask = int(MASK_PROB * n_genes)
    _, _idx = torch.topk(-_rand, _num_mask, dim=1)
    val_mask = torch.zeros(n_val, n_genes, dtype=torch.bool)
    val_mask.scatter_(1, _idx, True)
    val_mask = val_mask.to(device)
else:
    n_val = None
    n_genes = train_image.shape[1]

rank0_print(f"[xenium_mouse_brain] Train: {train_image.shape[0]}, Val: {n_val}")
rank0_print(f"  WORLD_SIZE={WORLD_SIZE}, DDP_MODE={DDP_MODE}, "
            f"total_batch={TOTAL_BATCH}, per_rank_batch={batch_size}")

gene2vec_weight = torch.from_numpy(
    np.load(f'{dataset_dir}/gene2vec_weight_xenium_mouse_brain.npy')).float()
model = SVC(
    gene2vec_weight=gene2vec_weight,
    n_genes_for_sn=n_genes,
    k_scales=K_SCALES,
    tau_init_per_scale=tau_per_scale_train,
    cell_morphology=True,
    nuclear_morphology=True,
    use_cell_identity=USE_CELL_IDENTITY,
    cell_identity_dim=n_identity_types,
    emb_dropout=0.0,
).to(device)
model.set_cell_median(float(cell_median_train))

base_model = model
if DDP_MODE:
    model = DDP(base_model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    rank0_print(f"  DDP across {WORLD_SIZE} ranks  (this rank: cuda:{LOCAL_RANK})")

foreground_expanded = foreground_mask(device)

film_param_names = set(base_model.film_gamma_param_names())
gene_emb_names   = set(base_model.gene_embedding_param_names())
film_params, gene_params, other_params = [], [], []
for name, p in base_model.named_parameters():
    if name in film_param_names:
        film_params.append(p)
    elif name in gene_emb_names:
        gene_params.append(p)
    else:
        other_params.append(p)
rank0_print(f"  param groups: FiLM={sum(p.numel() for p in film_params):,}  "
            f"gene={sum(p.numel() for p in gene_params):,}  "
            f"other={sum(p.numel() for p in other_params):,}")

param_groups = [{"params": other_params, "weight_decay": WEIGHT_DECAY}]
if film_params:
    param_groups.append({"params": film_params, "weight_decay": FILM_WEIGHT_DECAY})
if gene_params:
    param_groups.append({"params": gene_params, "weight_decay": GENE_EMB_WEIGHT_DECAY})

optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer, first_cycle_steps=NUM_EPOCHS, cycle_mult=2.0,
    max_lr=LEARNING_RATE, min_lr=LEARNING_RATE / 100,
)


def get_inputs(batch, device):
    inputs_ori, cm, nm, ci, sn_expr, sn_dist = batch
    inputs_ori = inputs_ori.to(device, non_blocking=True).float()
    cm_in = cm.to(device, non_blocking=True).float()
    nm_in = nm.to(device, non_blocking=True).float()
    ci_in = ci.to(device, non_blocking=True).float() if USE_CELL_IDENTITY else None
    sn_expr_in = sn_expr.to(device, non_blocking=True).float()
    sn_dist_in = sn_dist.to(device, non_blocking=True).float()
    return inputs_ori, cm_in, nm_in, ci_in, sn_expr_in, sn_dist_in


def compute_val_loss():
    if not IS_RANK0:
        return None
    base_model.eval()
    base_model.set_tau_per_scale(tau_per_scale_val)
    with torch.no_grad():
        ci_in = val_id_t if USE_CELL_IDENTITY else None
        sf_input, sf_mu = compute_size_factor_obs(val_inputs_ori, val_mask, cell_median_train)
        inputs = val_inputs_ori / sf_input
        val_obs_mask_4d = (1 - val_mask.float()).view(*val_inputs_ori.shape[:2], 1, 1)
        inputs = inputs * val_obs_mask_4d
        mus, rs = [], []
        for i in range(0, inputs.shape[0], EVAL_CHUNK):
            sl = slice(i, i + EVAL_CHUNK)
            _, mu_c, r_c = base_model(
                inputs[sl], val_mask[sl], val_sn_expr_t[sl], val_sn_dist_t[sl],
                val_cm_t[sl], val_nm_t[sl],
                ci_in[sl] if ci_in is not None else None,
            )
            mus.append(mu_c); rs.append(r_c)
        mu = torch.cat(mus, dim=0) * sf_mu
        r  = torch.cat(rs,  dim=0)
        loss_nb = (compute_loss_per_gene(val_inputs_ori, mu, r, foreground_expanded)
                   * val_mask).sum() / val_mask.sum()
        loss_cell = compute_cell_sum_loss(mu, val_inputs_ori, val_mask, foreground_expanded)
        loss = loss_nb + LAMBDA_SUM_MAX * loss_cell

    base_model.set_tau_per_scale(tau_per_scale_train)
    return {'total_loss': loss.item(), 'nb_loss': loss_nb.item(),
            'cell_loss': loss_cell.item()}




train_losses, val_losses = [], []
best_val_loss = float("inf")
best_state_dict = None; best_epoch = None
stop_patience_counter = 0

rank0_print(f"Starting xenium_mouse_brain training "
            f"(lambda 0->{LAMBDA_SUM_MAX} over ep {LAMBDA_WARMUP_START}->{LAMBDA_WARMUP_END}, "
            f"patience={EARLY_STOP_PATIENCE})...")
num_batches = len(train_loader)
for epoch in range(NUM_EPOCHS):
    lambda_sum_t = lambda_sum_at(epoch, LAMBDA_SUM_MAX, LAMBDA_WARMUP_START, LAMBDA_WARMUP_END)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    model.train()
    running_loss = running_nb = running_cell = 0.0
    n_batches_seen = 0
    for i, batch in enumerate(train_loader):
        inputs_ori, cm_in, nm_in, ci_in, sn_expr_in, sn_dist_in = get_inputs(batch, device)
        rand = torch.rand(inputs_ori.shape[0], inputs_ori.shape[1], device=device)
        num_mask = int(MASK_PROB * inputs_ori.shape[1])
        _, idx = torch.topk(-rand, num_mask, dim=1)
        mask = torch.zeros(inputs_ori.shape[0], inputs_ori.shape[1], dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        sf_input, sf_mu = compute_size_factor_obs(inputs_ori, mask, cell_median_train)
        inputs = inputs_ori / sf_input
        obs_mask_4d = (1 - mask.float()).view(inputs.shape[0], inputs.shape[1], 1, 1)
        inputs = inputs * obs_mask_4d
        _, mu, r = model(inputs, mask, sn_expr_in, sn_dist_in, cm_in, nm_in, ci_in)
        mu = mu * sf_mu
        loss_nb = (compute_loss_per_gene(inputs_ori, mu, r, foreground_expanded)
                   * mask).sum() / mask.sum()
        loss_cell = compute_cell_sum_loss(mu, inputs_ori, mask, foreground_expanded)
        loss = loss_nb + lambda_sum_t * loss_cell
        loss.backward()
        if ((i + 1) % ACCUM_ITER == 0) or (i + 1 == num_batches):
            optimizer.step(); optimizer.zero_grad()
        running_loss += loss.item(); running_nb += loss_nb.item(); running_cell += loss_cell.item()
        n_batches_seen += 1

    avg_loc = running_loss / max(n_batches_seen, 1)
    avg_nb_loc = running_nb / max(n_batches_seen, 1)
    avg_cell_loc = running_cell / max(n_batches_seen, 1)
    if DDP_MODE:
        t = torch.tensor([avg_loc, avg_nb_loc, avg_cell_loc], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM); t = t / WORLD_SIZE
        avg_train_loss, avg_train_nb, avg_train_cell = t[0].item(), t[1].item(), t[2].item()
    else:
        avg_train_loss, avg_train_nb, avg_train_cell = avg_loc, avg_nb_loc, avg_cell_loc
    train_losses.append(avg_train_loss)

    if IS_RANK0:
        vinfo = compute_val_loss()
        val_losses.append(vinfo['total_loss'])
        # Selection only once the cell-level expression loss is at full weight.
        in_stable = (epoch + 1) > LAMBDA_WARMUP_END
        if in_stable:
            if vinfo['total_loss'] < best_val_loss:
                best_val_loss = vinfo['total_loss']
                sd = base_model.state_dict()
                best_state_dict = {k: v.detach().cpu().clone() for k, v in sd.items()}
                best_epoch = epoch + 1
                stop_patience_counter = 0
            else:
                stop_patience_counter += 1
    scheduler.step()

    should_stop_local = 1 if (IS_RANK0 and stop_patience_counter >= EARLY_STOP_PATIENCE) else 0
    if DDP_MODE:
        st = torch.tensor([should_stop_local], device=device, dtype=torch.int64)
        dist.broadcast(st, src=0)
        should_stop = bool(st.item())
    else:
        should_stop = bool(should_stop_local)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        if IS_RANK0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}  train={avg_train_loss:.4f}  "
                  f"val={vinfo['total_loss']:.4f}  "
                  f"patience={stop_patience_counter}/{EARLY_STOP_PATIENCE}")

    if DDP_MODE:
        dist.barrier()
    if should_stop:
        rank0_print(f"[EARLY STOP] validation did not improve for {EARLY_STOP_PATIENCE} "
                    f"consecutive stable epochs. Exit at epoch {epoch+1}.")
        break

if IS_RANK0:
    if best_state_dict is not None:
        torch.save({"model_state_dict": best_state_dict, "best_epoch": best_epoch},
                   f"{out_dir}/checkpoint.pth")
        np.save(f"{out_dir}/train_losses.npy", np.array(train_losses))
        np.save(f"{out_dir}/val_losses.npy", np.array(val_losses))
        print(f"\nxenium_mouse_brain done. Best epoch: {best_epoch}, "
              f"train: {train_losses[best_epoch-1]:.4f}, val: {best_val_loss:.4f}")
    else:
        print("\n[WARN] no checkpoint saved")

if DDP_MODE:
    dist.barrier()
    dist.destroy_process_group()

"""
Evaluate SVC on the Xenium breast cancer test region by k-fold cross-validation
over GENES: each fold's genes are masked in every test cell and predicted from
the remaining genes, the auxiliary modalities and the neighbor context.

Metrics: per-gene spatial PCC / RMSE / cosine; per-gene cell-level PCC / RMSE /
cosine on per-(cell, gene) totals; per-cell cross-gene PCC / RMSE; and
dist_center, the correlation between predicted and measured mean
distance-to-center.

Usage:
    python evaluate.py --data-root /path/to/data --ckpt-dir ./output/xenium_breast_cancer
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from svc.losses import BACKGROUND_PIXELS
from svc.metrics import (cosine_similarity, distance_to_center, pcc_rowwise,
                         spatial_pcc)
from svc.model import SVC

K_SCALES = (4, 16, 64)
USE_CELL_IDENTITY = True
N_FOLDS = 10
KFOLD_RANDOM_STATE = 2026
BATCH_SIZE = 32

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default='./data',
                help="directory holding data/xenium_breast_cancer/ inputs")
ap.add_argument('--ckpt-dir', default='./output/xenium_breast_cancer',
                help="directory holding checkpoint.pth (or the released"
                     " checkpoint_Xenium_breast_cancer.pth)")
ap.add_argument('--save-predictions', action='store_true',
                help="also write prediction_mu.npz and prediction_r.npz into --ckpt-dir, "
                     "the cross-validated per-gene predictions for downstream use")
args = ap.parse_args()

dataset_dir = os.path.join(args.data_root, 'xenium_breast_cancer')
base_dir = args.ckpt_dir

RELEASED_CKPT = 'checkpoint_Xenium_breast_cancer.pth'

if os.path.exists(f'{base_dir}/checkpoint.pth'):
    ckpt_path = f'{base_dir}/checkpoint.pth'
elif os.path.exists(f'{base_dir}/{RELEASED_CKPT}'):
    ckpt_path = f'{base_dir}/{RELEASED_CKPT}'
else:
    raise SystemExit(
        f"no checkpoint found under {base_dir}\n"
        f"  expected {base_dir}/checkpoint.pth (written by train.py)\n"
        f"  or       {base_dir}/{RELEASED_CKPT} (from checkpoints.zip on Zenodo)")
print(f"checkpoint: {ckpt_path}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_tau_per_scale(nb_dist, k_scales):
    return [float(np.median(nb_dist[:, k - 1])) for k in k_scales]


test_data  = np.load(f"{dataset_dir}/test_xenium_breast_cancer.npz")
train_data = np.load(f"{dataset_dir}/train_xenium_breast_cancer.npz")

test_image  = test_data["data_ori"]
train_image = train_data["data_ori"]
test_image_eval = test_image

test_cell_morphology     = test_data["cell_morphology"].astype(np.float32)
test_nuclear_morphology  = test_data["nuclear_morphology"].astype(np.float32)
test_cell_identity_label = test_data["identity_label"].astype(np.float32)
test_nb_idx  = np.asarray(test_data["nb_idx"], dtype=np.int64)
test_sn_dist = test_data["nb_dist"].astype(np.float32)
tau_per_scale_test = compute_tau_per_scale(test_sn_dist, K_SCALES)
assert test_nb_idx.shape[1] >= max(K_SCALES), \
    f"test npz has k_max={test_nb_idx.shape[1]}, need >= {max(K_SCALES)}; re-run prepare_data.py"

_fg = np.ones((12, 12), dtype=bool)
for (r, c) in BACKGROUND_PIXELS:
    _fg[r, c] = False
_FG_FILTER = _fg.reshape(1, 1, 12, 12).astype(np.float32)
test_full_expr = (test_image * _FG_FILTER).sum(axis=(-1, -2)).astype(np.float32)   # (N, G)
test_sn_expr   = test_full_expr[test_nb_idx]                                      # (N, k, G)

train_count_sum = train_image.reshape(*train_image.shape[:2], -1).sum(axis=-1)
cell_median_train = float(np.median(train_count_sum.sum(axis=1)))

gene_names = np.loadtxt(f'{dataset_dir}/gene_names.txt', dtype=str).tolist()
n_identity_types = test_cell_identity_label.shape[1]
print(f"[xenium_breast_cancer] Test cells: {test_image.shape[0]}, "
      f"Genes: {len(gene_names)}, K_SCALES={K_SCALES}, "
      f"tau_per_scale_test={[f'{t:.2f}' for t in tau_per_scale_test]}")


class SVC_TestDataset_SN(Dataset):
    def __init__(self, data_ori, cm, nm, ci, sn_expr, sn_dist):
        self.data_ori = data_ori
        self.cm = cm; self.nm = nm; self.ci = ci
        self.sn_expr = sn_expr; self.sn_dist = sn_dist

    def __len__(self):
        return len(self.data_ori)

    def __getitem__(self, i):
        return (self.data_ori[i], self.cm[i], self.nm[i], self.ci[i],
                self.sn_expr[i], self.sn_dist[i])


test_dataset = SVC_TestDataset_SN(
    test_image, test_cell_morphology, test_nuclear_morphology,
    test_cell_identity_label, test_sn_expr, test_sn_dist,
)

background_pixel = torch.tensor(BACKGROUND_PIXELS, device=device, dtype=torch.long)
foreground = torch.ones((12, 12), dtype=torch.bool, device=device)
foreground[background_pixel[:, 0], background_pixel[:, 1]] = False
foreground_flat = foreground.flatten().cpu()
fg_mask = foreground_flat.numpy().astype(bool)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=KFOLD_RANDOM_STATE)
groups = [np.array(gene_names)[idx] for _, idx in kf.split(gene_names)]



def evaluate_model(gene_names, groups):

    gene2vec_weight = torch.from_numpy(
        np.load(f'{dataset_dir}/gene2vec_weight_xenium_breast_cancer.npy')).float()
    model = SVC(
        gene2vec_weight=gene2vec_weight,
        n_genes_for_sn=len(gene_names),
        k_scales=K_SCALES,
        tau_init_per_scale=tau_per_scale_test,
        cell_morphology=True,
        nuclear_morphology=True,
        use_cell_identity=USE_CELL_IDENTITY,
        cell_identity_dim=n_identity_types,
        emb_dropout=0.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    state.pop('cell_median_train', None)
    state.pop('tau_per_scale', None)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.set_tau_per_scale(tau_per_scale_test)
    model.set_cell_median(float(cell_median_train))
    print(f"  Loaded checkpoint from epoch {ckpt['best_epoch']}, "
          f"tau_per_scale = {model.tau_per_scale.tolist()}, "
          f"cell_median_train = {model.cell_median_train.item():.3f}")

    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    pred_mu_all = torch.zeros(test_image_eval.shape).to(device)
    pred_r_all = torch.zeros(test_image_eval.shape).to(device) if args.save_predictions else None

    for selected_gene in groups:
        gene_to_impute = [gene_names.index(g) for g in selected_gene if g in gene_names]
        pred_mu_list, pred_r_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs_ori, cm, nm, cell_id, sn_expr, sn_dist = batch
                inputs_ori = inputs_ori.to(device).float()
                inputs_mask = inputs_ori.clone()
                inputs_mask[:, gene_to_impute] = 0

                # Observed genes only, rescaled by the observed fraction.
                B = inputs_mask.shape[0]
                n_genes_local = inputs_mask.shape[1]
                n_obs_per_cell = float(n_genes_local - len(gene_to_impute))
                obs_frac = n_obs_per_cell / float(n_genes_local)
                sf_factor = inputs_mask.reshape(B, -1).sum(dim=1) / (cell_median_train * obs_frac)
                sf_input = sf_factor.view(B, *([1] * (inputs_mask.ndim - 1)))
                sf_mu    = sf_factor.view(B, 1, 1, 1)
                inputs = inputs_ori / sf_input

                mask = torch.zeros(inputs.shape[0], inputs.shape[1], device=device).bool()
                mask[:, gene_to_impute] = True
                inputs_masked = inputs.clone()
                inputs_masked[:, gene_to_impute] = 0

                cm_in = cm.to(device).float()
                nm_in = nm.to(device).float()
                ci_in = cell_id.to(device).float() if USE_CELL_IDENTITY else None
                sn_expr_in = sn_expr.to(device).float()
                sn_expr_in[:, :, gene_to_impute] = 0
                sn_dist_in = sn_dist.to(device).float()

                _, mu, r = model(inputs_masked, mask, sn_expr_in, sn_dist_in,
                                   cm_in, nm_in, ci_in)
                pred_mu_list.append(mu * sf_mu)
                if args.save_predictions:
                    pred_r_list.append(r)

        pred_mu_cat = torch.cat(pred_mu_list, dim=0)
        pred_mu_cat[:, :, background_pixel[:, 0], background_pixel[:, 1]] = 0
        pred_mu_all[:, gene_to_impute] = pred_mu_cat[:, gene_to_impute]
        if args.save_predictions:
            pred_r_cat = torch.cat(pred_r_list, dim=0)
            pred_r_cat[:, :, background_pixel[:, 0], background_pixel[:, 1]] = 0
            pred_r_all[:, gene_to_impute] = pred_r_cat[:, gene_to_impute]

    pred = pred_mu_all.cpu().numpy()

    if args.save_predictions:
        np.savez_compressed(f'{base_dir}/prediction_mu.npz', prediction=pred)
        np.savez_compressed(f'{base_dir}/prediction_r.npz',
                            prediction=pred_r_all.cpu().numpy())
        print(f"  wrote {base_dir}/prediction_mu.npz / prediction_r.npz")

    # spatial metrics
    pcc_per_gene, rmse_per_gene, cos_per_gene = [], [], []
    for i in range(len(gene_names)):
        pred_i, truth_i = pred[:, i], test_image_eval[:, i]
        non_zero = truth_i.sum((-1, -2)) != 0
        p_flat = pred_i.reshape(pred_i.shape[0], -1); t_flat = truth_i.reshape(truth_i.shape[0], -1)
        p, t = p_flat[:, fg_mask], t_flat[:, fg_mask]
        pcc_per_gene.append(spatial_pcc(p, t))
        rmse_per_gene.append(np.sqrt(((p - t) ** 2).mean(axis=1)).mean())
        cs = cosine_similarity(p, t, axis=1)
        cos_per_gene.append(cs[non_zero].mean() if non_zero.any() else np.nan)
    pcc_per_gene = np.array(pcc_per_gene); rmse_per_gene = np.array(rmse_per_gene)
    cos_per_gene = np.array(cos_per_gene)

    # cell-level metrics on per-(cell, gene) totals
    pred_cell  = pred.sum((-1, -2))
    truth_cell = test_image_eval.sum((-1, -2))
    cell_pcc, cell_rmse, cell_cos = [], [], []
    for i in range(len(gene_names)):
        pc, tc = pred_cell[:, i], truth_cell[:, i]
        non_zero = tc != 0
        cell_pcc.append(np.corrcoef(pc[non_zero], tc[non_zero])[0, 1] if non_zero.sum() > 1 else np.nan)
        cell_rmse.append(np.sqrt(((pc - tc) ** 2).mean()))
        denom = np.linalg.norm(pc) * np.linalg.norm(tc)
        cell_cos.append(np.dot(pc, tc) / (denom + 1e-8) if denom > 0 else 0.0)
    cell_pcc, cell_rmse, cell_cos = np.array(cell_pcc), np.array(cell_rmse), np.array(cell_cos)

    # cross-gene metrics, per cell
    per_cell_xgene_pcc = pcc_rowwise(pred_cell, truth_cell)
    valid = (truth_cell.sum(axis=1) > 0) & (pred_cell.std(axis=1) > 0) & (truth_cell.std(axis=1) > 0)
    per_cell_xgene_pcc = np.where(valid, per_cell_xgene_pcc, np.nan)
    per_cell_xgene_rmse = np.sqrt(((pred_cell - truth_cell) ** 2).mean(axis=1))
    cross_gene_pcc = float(np.nanmean(per_cell_xgene_pcc))
    cross_gene_rmse = float(per_cell_xgene_rmse.mean())
    pred_gene_mean  = pred_cell.mean(0); truth_gene_mean = truth_cell.mean(0)

    # mean distance-to-center of each gene's transcripts
    pred_ratio  = distance_to_center(pred)
    truth_ratio = distance_to_center(test_image_eval)
    pred_ratio_mean  = pred_ratio.sum(0) / np.maximum((pred_ratio != 0).sum(0), 1)
    truth_ratio_mean = truth_ratio.sum(0) / np.maximum((truth_ratio != 0).sum(0), 1)
    dist_pcc = np.corrcoef(pred_ratio_mean, truth_ratio_mean)[0, 1]

    return {
        'best_epoch': ckpt['best_epoch'],
        'per_gene_pcc': pcc_per_gene, 'mean_pcc': float(np.nanmean(pcc_per_gene)),
        'per_gene_rmse': rmse_per_gene, 'mean_rmse': float(rmse_per_gene.mean()),
        'per_gene_cosine': cos_per_gene, 'mean_cosine': float(np.nanmean(cos_per_gene)),
        'cell_per_gene_pcc': cell_pcc, 'cell_mean_pcc': float(np.nanmean(cell_pcc)),
        'cell_per_gene_rmse': cell_rmse, 'cell_mean_rmse': float(cell_rmse.mean()),
        'cell_per_gene_cosine': cell_cos, 'cell_mean_cosine': float(cell_cos.mean()),
        'per_cell_cross_gene_pcc': per_cell_xgene_pcc,
        'per_cell_cross_gene_rmse': per_cell_xgene_rmse,
        'cross_gene_pcc': cross_gene_pcc, 'cross_gene_rmse': cross_gene_rmse,
        'pred_gene_mean': pred_gene_mean, 'truth_gene_mean': truth_gene_mean,
        'dist_center_pcc': float(dist_pcc),
        'pred_ratio_mean': pred_ratio_mean, 'truth_ratio_mean': truth_ratio_mean,
    }


r = evaluate_model(gene_names, groups)
print(f"  PCC={r['mean_pcc']:.4f}, RMSE={r['mean_rmse']:.4f}, Cos={r['mean_cosine']:.4f}, "
      f"CellPCC={r['cell_mean_pcc']:.4f}, CellRMSE={r['cell_mean_rmse']:.4f}, "
      f"CellCos={r['cell_mean_cosine']:.4f}, CrossGenePCC={r['cross_gene_pcc']:.4f}, "
      f"CrossGeneRMSE={r['cross_gene_rmse']:.4f}, Dist={r['dist_center_pcc']:.4f}")

np.save(f'{base_dir}/eval_results.npy', np.array([r], dtype=object))
print(f"\nResults saved to {base_dir}/eval_results.npy")

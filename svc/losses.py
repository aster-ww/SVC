"""
Training objective:  loss = loss_nb + lambda * loss_cell,  both over masked
genes and foreground pixels only. 

"""

import torch
import torch.nn.functional as F

# Background pixels of the 12x12 image: grid locations outside the registered unit
# circle, i.e. outside the cell mask. Excluded from every loss and every metric.
BACKGROUND_PIXELS = [(0, 0), (0, 1), (0, 10), (0, 11),
                     (1, 0), (1, 11),
                     (10, 0), (10, 11),
                     (11, 0), (11, 1), (11, 10), (11, 11)]


def foreground_mask(device=None, height=12, width=12):
    """(1, 1, H, W) bool mask, False at the background corner pixels."""
    fg = torch.ones((height, width), dtype=torch.bool, device=device)
    for r, c in BACKGROUND_PIXELS:
        fg[r, c] = False
    return fg.reshape(1, 1, height, width)


def negative_binomial_loss(y_true, mu, r):
    """Per-pixel negative-binomial negative log-likelihood."""
    y = y_true.float()

    term1 = torch.special.gammaln(y + r)
    term2 = torch.special.gammaln(y + 1)
    term3 = torch.special.gammaln(r)
    term4 = r * (torch.log(r) - torch.log(r + mu))
    term5 = y * (torch.log(mu) - torch.log(r + mu))

    log_likelihood = term1 - term2 - term3 + term4 + term5
    return -log_likelihood


def compute_size_factor_obs(inputs_ori, mask, cell_median_train):
    """Returns (sf_input, sf_mu): one scalar per cell, shaped to divide the model
    input and to multiply the predicted mu."""
    B, G = inputs_ori.shape[:2]
    obs_mask_4d = (1 - mask.float()).view(B, G, 1, 1)
    per_cell_obs_total = (inputs_ori * obs_mask_4d).reshape(B, -1).sum(dim=1)
    obs_frac_per_cell = (1 - mask.float()).sum(dim=1) / float(G)
    sf = per_cell_obs_total / (cell_median_train * obs_frac_per_cell + 1e-8)
    sf_input = sf.view(B, *([1] * (inputs_ori.ndim - 1)))
    sf_mu    = sf.view(B, 1, 1, 1)
    return sf_input, sf_mu


def compute_loss_per_gene(target, mu, r, fg_e):
    """NB loss summed over foreground pixels -> (B, G)."""
    return (negative_binomial_loss(target, mu, r) * fg_e).sum(dim=(-1, -2))


def compute_cell_sum_loss(mu, inputs_ori, mask, fg_e):
    """Smooth-L1 on log1p of per-(cell, gene) foreground totals, masked genes only."""
    pred_cell  = (mu * fg_e).sum(dim=(-1, -2))
    truth_cell = (inputs_ori * fg_e).sum(dim=(-1, -2))
    L_sum_per  = F.smooth_l1_loss(
        torch.log1p(pred_cell.clamp(min=0)),
        torch.log1p(truth_cell.clamp(min=0)),
        reduction='none',
    )
    return (L_sum_per * mask).sum() / mask.sum()


def lambda_sum_at(epoch, lambda_max, warmup_start, warmup_end):
    """0 up to warmup_start, linear to lambda_max at warmup_end, constant after."""
    if epoch <= warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return lambda_max
    return lambda_max * (epoch - warmup_start) / (warmup_end - warmup_start)

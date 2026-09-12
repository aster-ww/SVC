"""SVC — predicting the subcellular spatial distribution of gene expression."""

from svc.model import SVC, BACKGROUND_PIXELS
from svc.knn import build_knn
from svc.losses import (
    negative_binomial_loss,
    foreground_mask,
    compute_size_factor_obs,
    compute_loss_per_gene,
    compute_cell_sum_loss,
    lambda_sum_at,
)
from svc.metrics import (cosine_similarity, distance_to_center, pcc_rowwise,
                         spatial_pcc)
from svc.scheduler import CosineAnnealingWarmupRestarts
from svc.augment import apply_d4, d4_augment, _D4_ELEMENTS

__all__ = [
    'SVC',
    'BACKGROUND_PIXELS',
    'build_knn',
    'negative_binomial_loss',
    'foreground_mask',
    'compute_size_factor_obs',
    'compute_loss_per_gene',
    'compute_cell_sum_loss',
    'lambda_sum_at',
    'cosine_similarity',
    'distance_to_center',
    'pcc_rowwise',
    'spatial_pcc',
    'CosineAnnealingWarmupRestarts',
    'apply_d4',
    'd4_augment',
]

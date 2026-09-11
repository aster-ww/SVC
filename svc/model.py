"""

forward(img, mask, sn_expr, sn_dist, cell_morph, nuclear_morph, cell_identity)
    img           (B, G, 12, 12)   per-gene subcellular expression maps
    mask          (B, G)           True = gene is masked and must be predicted
    sn_expr       (B, k_max, G)    k nearest cells' per-gene foreground totals,
                                   sorted by ascending distance
    sn_dist       (B, k_max)       distances to those neighbors
    cell_morph    (B, 48, 48)      cell mask
    nuclear_morph (B, 48, 48)      nucleus mask
    cell_identity (B, C)           one-hot cell type / state (optional)
  ->
    gene_encoding (B, G, dim)
    mu, r         (B, G, 12, 12)   negative-binomial parameters

Backbone: 12-layer Performer, dim 384, 12 heads. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from svc.losses import BACKGROUND_PIXELS
from svc.performer import Performer


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val


class TrainableGeneEmbedding(nn.Module):

    def __init__(self, gene2vec_weight, dim, n_aux):
        super().__init__()
        self.n_aux = n_aux
        self.register_buffer('gene2vec', gene2vec_weight.float())   # (n_genes, 200), frozen
        self.proj = nn.Linear(200, dim)
        self.aux_emb = nn.Parameter(torch.zeros(n_aux, dim))
        nn.init.normal_(self.aux_emb, std=0.02)

    def forward(self, x):
        gene_emb = self.proj(self.gene2vec)                          # (n_genes, dim)
        full = torch.cat([self.aux_emb, gene_emb], dim=0)            # (n_aux + n_genes, dim)
        t = torch.arange(x.shape[1], device=x.device)
        return full[t]


class MorphPatchEncoder(nn.Module):
    """Conv2d(1, dim, k=8, s=8) -> mean over patches -> LayerNorm."""

    def __init__(self, dim, in_h=48, in_w=48, patch_size=8):
        super().__init__()
        assert in_h % patch_size == 0 and in_w % patch_size == 0
        self.patch_emb = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)                                       # (B, 1, H, W)
        x = self.patch_emb(x)                                        # (B, dim, H/p, W/p)
        x = x.flatten(2).mean(-1)                                    # (B, dim)
        return self.norm(x)


class FiLMHead(nn.Module):
    """(gamma, beta) from one cell-level token. fc2 is zero-initialized: no-op at step 0."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 2 * dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(x)))                          # (B, 2*dim)
        gamma, beta = h.chunk(2, dim=-1)
        return gamma.unsqueeze(1), beta.unsqueeze(1)                 # (B, 1, dim) each


class SVC(nn.Module):
    """
    gene2vec_weight    (n_genes, 200) frozen Gene2Vec features; a trainable
                       Linear(200 -> dim) projects them.
    n_genes_for_sn     gene count of the neighbor-context input (= n_genes).
    k_scales           neighborhood sizes; (4, 16, 64) for tissue sections,
                       (4,) for cultured cells in small fields of view.
    tau_init_per_scale per-scale softmax temperature; set to the median distance
                       to the k_s-th neighbor of the set being processed, and
                       updated by set_tau_per_scale when the set changes.

    """

    def __init__(self, *,
                 gene2vec_weight,
                 n_genes_for_sn,
                 k_scales=(4, 16, 64),
                 tau_init_per_scale=None,
                 image_size=12,
                 dim=384,
                 depth=12,
                 heads=12,
                 cell_morphology=True,
                 nuclear_morphology=True,
                 use_cell_identity=True,
                 cell_morphology_dim=(48, 48),
                 nuclear_morphology_dim=(48, 48),
                 cell_identity_dim=32,
                 dim_head=64,
                 emb_dropout=0.0,
                 aux_dropout=0.1,
                 film_gate_init=-3.0,
                 local_attn_heads=0,
                 local_window_size=256,
                 causal=False,
                 ff_mult=4,
                 nb_features=None,
                 feature_redraw_interval=1000,
                 reversible=False,
                 ff_chunks=1,
                 ff_glu=False,
                 ff_dropout=0.,
                 attn_dropout=0.,
                 generalized_attention=False,
                 kernel_fn=nn.ReLU(),
                 use_scalenorm=False,
                 use_rezero=False,
                 cross_attend=False,
                 no_projection=False,
                 auto_check_redraw=True,
                 qkv_bias=False,
                 ):
        super().__init__()
        patch_height, patch_width = pair(image_size)
        patch_dim = patch_height * patch_width

        n_aux = (1 + int(cell_morphology) +
                 int(nuclear_morphology) + int(use_cell_identity))
        self.n_aux = n_aux
        self.aux_dropout = aux_dropout

        self.k_scales = tuple(int(k) for k in k_scales)
        self.k_max = max(self.k_scales)
        self.n_scales = len(self.k_scales)
        self.n_genes_for_sn = n_genes_for_sn

        self.pos_emb = TrainableGeneEmbedding(gene2vec_weight, dim, n_aux=n_aux)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        if cell_morphology:
            self.cell_morphology_token = MorphPatchEncoder(dim, *cell_morphology_dim)
        if nuclear_morphology:
            self.nuclear_morphology_token = MorphPatchEncoder(dim, *nuclear_morphology_dim)
        if use_cell_identity:
            self.cell_identity_token = nn.Sequential(
                nn.Linear(cell_identity_dim, dim),
                nn.LayerNorm(dim),
            )

        if cell_morphology:
            self.film_cm = FiLMHead(dim)
            self.film_gate_cm = nn.Parameter(torch.tensor(float(film_gate_init)))
        if nuclear_morphology:
            self.film_nm = FiLMHead(dim)
            self.film_gate_nm = nn.Parameter(torch.tensor(float(film_gate_init)))
        if use_cell_identity:
            self.film_ci = FiLMHead(dim)
            self.film_gate_ci = nn.Parameter(torch.tensor(float(film_gate_init)))

        self.embed_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.layer_pos_emb = Always(None)

        self.performer = Performer(
            dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal,
            ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks,
            generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu,
            ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias,
        )
        self.norm = nn.LayerNorm(dim)
        self.mu_head = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b c (h w) -> b c h w', h=patch_height, w=patch_width),
        )
        self.r_head = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b c (h w) -> b c h w', h=patch_height, w=patch_width),
        )

        fg = torch.ones(patch_height, patch_width, dtype=torch.float32)
        for r, c in BACKGROUND_PIXELS:
            fg[r, c] = 0.0
        self.register_buffer('_fg_mask', fg.view(1, 1, patch_height, patch_width),
                             persistent=False)

        self.mag_proj = nn.Linear(1, dim)

        self.sn_proj = nn.Sequential(
            nn.Linear(self.n_scales * n_genes_for_sn, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )
        self.film_sn = FiLMHead(dim)
        self.film_gate_sn = nn.Parameter(torch.tensor(float(film_gate_init)))

        if tau_init_per_scale is None:
            init_vals = [1.0] * self.n_scales
        else:
            init_vals = [float(v) for v in tau_init_per_scale]
            assert len(init_vals) == self.n_scales, \
                f"tau_init_per_scale length {len(init_vals)} != n_scales {self.n_scales}"
        self.register_buffer('tau_per_scale', torch.tensor(init_vals, dtype=torch.float32),
                             persistent=False)
        self._tau_set = tau_init_per_scale is not None

        self.register_buffer('cell_median_train', torch.tensor(1.0), persistent=False)
        self._cell_median_set = False

    def set_tau_per_scale(self, values):
        assert len(values) == self.n_scales, \
            f"expected {self.n_scales} values, got {len(values)}"
        with torch.no_grad():
            self.tau_per_scale.copy_(torch.tensor([float(v) for v in values],
                                                  dtype=torch.float32))
        self._tau_set = True

    def set_cell_median(self, value):
        with torch.no_grad():
            self.cell_median_train.fill_(float(value))
        self._cell_median_set = True

    def film_gamma_param_names(self):
        names = []
        for m in ['film_sn', 'film_cm', 'film_nm', 'film_ci']:
            if hasattr(self, m):
                names += [f'{m}.fc2.weight', f'{m}.fc2.bias']
        return names

    def gene_embedding_param_names(self):
        return [n for n, p in self.named_parameters()
                if n.startswith('pos_emb.') and p.requires_grad]

    def _build_sn_ctx(self, sn_expr, cell_mask, sn_dist):
        """Multi-scale, distance-weighted aggregation of neighbor expression.

        Args:
            sn_expr:   (B, k_max, G) neighbors' per-gene foreground totals,
                       sorted by ascending distance, so [:, :k_s] are the k_s
                       nearest.
            cell_mask: (B, G) the focal cell's mask.
            sn_dist:   (B, k_max) distances to those neighbors.
        Returns:
            (B, dim) the projected context token.
        """
        B, k_max, G = sn_expr.shape
        assert k_max >= self.k_max, f"sn_expr's k ({k_max}) < required k_max ({self.k_max})"

        mask_obs = (1.0 - cell_mask.float()).unsqueeze(1)               # (B, 1, G), 1 = observed
        obs_frac = mask_obs.sum(dim=2) / float(G)                       # (B, 1)

        contexts = []
        for s, k in enumerate(self.k_scales):
            nb_e_k = sn_expr[:, :k]                                     # (B, k, G)
            nb_d_k = sn_dist[:, :k]                                     # (B, k)
            tau_s = self.tau_per_scale[s]

   
            nb_masked = nb_e_k * mask_obs                               # (B, k, G)
            obs_sum_per_nb = nb_masked.sum(dim=2)                       # (B, k)
            sf_per_nb = obs_sum_per_nb / (self.cell_median_train * obs_frac + 1e-8)
            nb_obs_norm = nb_masked / (sf_per_nb.unsqueeze(-1) + 1e-8)  # (B, k, G)
            nb_obs_norm = torch.log1p(nb_obs_norm.clamp(min=0))

            w = torch.softmax(-nb_d_k / tau_s, dim=1).unsqueeze(-1)     # (B, k, 1)
            contexts.append((w * nb_obs_norm).sum(dim=1))               # (B, G)

        ctx_concat = torch.cat(contexts, dim=-1)                        # (B, S*G)
        return self.sn_proj(ctx_concat)                                 # (B, dim)

    def forward(self, img, mask, sn_expr, sn_dist,
                cell_morphology_vec=None, nuclear_morphology_vec=None,
                cell_identity_vec=None, output_attentions=False):
        if not self._cell_median_set:
            raise RuntimeError(
                "cell_median_train was never set: call set_cell_median(m) with the median "
                "per-cell total of the training data over the gene panel in use")
        if not self._tau_set:
            raise RuntimeError(
                "tau_per_scale was never set: pass tau_init_per_scale to SVC(...) or call "
                "set_tau_per_scale(v) with the median distance to the k-th neighbor of "
                "the cell set being processed")
        B, G = img.shape[:2]
        mask_4d = mask.float().view(B, G, 1, 1)

        img_obs = img * (1.0 - mask_4d)
        obs_fg = (1.0 - mask_4d) * self._fg_mask
        n_obs_fg = obs_fg.sum(dim=(1, 2, 3), keepdim=True)
        r_i = (img * obs_fg).sum(dim=(1, 2, 3), keepdim=True) / (n_obs_fg + 1e-5)
        img_normed = img_obs / (r_i + 1e-5)

        x = self.to_patch_embedding(img_normed)
        b, l, d = x.shape

        # Auxiliary tokens.
        sn_token = self._build_sn_ctx(sn_expr, mask, sn_dist)
        cm_proj = self.cell_morphology_token(cell_morphology_vec) \
            if (cell_morphology_vec is not None and hasattr(self, 'cell_morphology_token')) else None
        nm_proj = self.nuclear_morphology_token(nuclear_morphology_vec) \
            if (nuclear_morphology_vec is not None and hasattr(self, 'nuclear_morphology_token')) else None
        ci_proj = self.cell_identity_token(cell_identity_vec) \
            if (cell_identity_vec is not None and hasattr(self, 'cell_identity_token')) else None

        gate_gamma = x.new_zeros(b, 1, d)
        gate_beta = x.new_zeros(b, 1, d)
        for proj, head_name, gate_name in [
            (sn_token, 'film_sn', 'film_gate_sn'),
            (cm_proj,  'film_cm', 'film_gate_cm'),
            (nm_proj,  'film_nm', 'film_gate_nm'),
            (ci_proj,  'film_ci', 'film_gate_ci'),
        ]:
            if proj is not None and hasattr(self, head_name):
                g, bt = getattr(self, head_name)(proj)
                s = torch.sigmoid(getattr(self, gate_name))
                gate_gamma = gate_gamma + s * g
                gate_beta = gate_beta + s * bt

        mask_token = self.mask_token.expand(b, l, -1)
        w = mask.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        log_mag = torch.log1p((img * self._fg_mask).sum(dim=(-1, -2)))
        log_mag = log_mag * (1.0 - mask.float())
        mag_emb = self.mag_proj(log_mag.unsqueeze(-1))
        mag_emb = mag_emb * (1.0 - mask.float()).unsqueeze(-1)
        x = x + mag_emb

        # FiLM modulation.
        x = x * (1 + gate_gamma) + gate_beta

        aux_tokens = []
        for proj in [sn_token, cm_proj, nm_proj, ci_proj]:
            if proj is not None:
                aux_tokens.append(proj.unsqueeze(1))
        if self.aux_dropout > 0:
            aux_tokens = [F.dropout(t, p=self.aux_dropout, training=self.training)
                          for t in aux_tokens]
        aux_tokens.append(x)
        x = torch.cat(aux_tokens, dim=1)

        x = x + self.pos_emb(x)
        x = self.embed_norm(x)
        x = self.dropout(x)

        layer_pos_emb = self.layer_pos_emb(x)
        if output_attentions:
            x, attn = self.performer(x, pos_emb=layer_pos_emb, output_attentions=True)
        else:
            x = self.performer(x, pos_emb=layer_pos_emb)

        encoding = self.norm(x)
        n_aux = self.n_aux
        gene_encoding = encoding[:, n_aux:, :]
        mu = torch.exp(self.mu_head(encoding))[:, n_aux:, :, :]
        r = torch.exp(self.r_head(encoding))[:, n_aux:, :, :]

        if output_attentions:
            return gene_encoding, mu, r, attn
        return gene_encoding, mu, r

"""Cross-modal retrieval between SVC representations.

A linear probe fitted post hoc on the frozen representations: query and target
are reduced to `n_components` principal components on a fitting set, a ridge map
is learned from query to target space, and held-out queries are matched against
held-out targets by cosine similarity.

    from svc.retrieval import CrossModalRetriever, morphology_repr, gene_pattern_repr

    Q = morphology_repr(model, cell_mask, nuclear_mask)
    T = gene_pattern_repr(model, img, sn_expr, sn_dist, cell_mask, nuclear_mask,
                          identity, genes=top100, ablate=('morphology',))
    r = CrossModalRetriever().fit(Q, T, stratify=cell_type)
    r.retrieve(top_k=5)
    r.score()

"""

import numpy as np
import torch

from svc.latent import LayerAverage

__all__ = [
    'CrossModalRetriever',
    'morphology_repr',
    'gene_pattern_repr',
    'spatial_context_repr',
]

_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 1e4)


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _ridge(a, b, alpha):
    return np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ b)


def _pca_fit_transform(x, fit_idx, n_components, seed, side):
    from sklearn.decomposition import PCA
    if x.shape[1] < n_components:
        raise ValueError(
            f"{side} has {x.shape[1]} features, fewer than n_components="
            f"{n_components}; pass pca_{side}=False to use it as is")
    p = PCA(n_components, svd_solver='randomized', random_state=seed).fit(x[fit_idx])
    return p.transform(x), p


def _split_random(n, fit_frac, stratify, seed):
    rng = np.random.RandomState(seed)
    is_fit = np.zeros(n, bool)
    groups = [np.arange(n)] if stratify is None else \
        [np.where(np.asarray(stratify) == g)[0] for g in np.unique(stratify)]
    for idx in groups:
        rng.shuffle(idx)
        is_fit[idx[:int(fit_frac * len(idx))]] = True
    return np.where(is_fit)[0], np.where(~is_fit)[0]


def _split_spatial(coords, fit_frac, grid, seed):
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    nx, ny = grid
    xs = np.linspace(coords[:, 0].min(), coords[:, 0].max(), nx + 1)
    ys = np.linspace(coords[:, 1].min(), coords[:, 1].max(), ny + 1)
    bx = np.clip(np.digitize(coords[:, 0], xs[1:-1]), 0, nx - 1)
    by = np.clip(np.digitize(coords[:, 1], ys[1:-1]), 0, ny - 1)
    block = bx * ny + by
    blocks = np.unique(block)
    rng = np.random.RandomState(seed)
    rng.shuffle(blocks)
    is_fit = np.zeros(n, bool)
    total = 0
    for b in blocks:
        if total >= fit_frac * n:
            break
        m = block == b
        is_fit[m] = True
        total += m.sum()
    return np.where(is_fit)[0], np.where(~is_fit)[0]


class CrossModalRetriever:
    """Ridge probe from a query representation to a target representation.

    n_components  PCA dimension, fitted on the fitting set only.
    alphas        ridge penalties searched inside the fitting set.
    seed          controls the split and the PCA.
    """

    def __init__(self, n_components=50, alphas=_ALPHAS, seed=0):
        self.n_components = n_components
        self.alphas = tuple(alphas)
        self.seed = seed

    def fit(self, query, target, fit_idx=None, held_out_idx=None,
            fit_frac=0.7, stratify=None, split='random', coords=None,
            grid=(10, 8), pca_query=True, pca_target=True):
        """Fit the probe and compute the held-out similarity matrix.

        query, target      row-aligned (N, D); higher dims are flattened from axis 1.
        fit_idx            explicit split; otherwise `fit_frac` of the cells are
                           taken at random (`split='random'`, optionally
                           stratified by `stratify`) or by contiguous blocks of a
                           `grid` over `coords` (`split='spatial'`).
        pca_query/_target  which side is reduced to `n_components`; set False to
                           use a low-dimensional representation as is.
        """
        query = np.asarray(query, dtype=np.float64).reshape(len(query), -1)
        target = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
        if len(query) != len(target):
            raise ValueError(f"query and target must be row-aligned, got {len(query)} and {len(target)}")
        n = len(query)

        if fit_idx is not None:
            fit_idx = np.asarray(fit_idx)
            if held_out_idx is None:
                held_out_idx = np.setdiff1d(np.arange(n), fit_idx)
            else:
                held_out_idx = np.asarray(held_out_idx)
                overlap = np.intersect1d(fit_idx, held_out_idx)
                if overlap.size:
                    raise ValueError(f"fit_idx and held_out_idx overlap in {overlap.size} cells")
        elif split == 'random':
            fit_idx, held_out_idx = _split_random(n, fit_frac, stratify, self.seed)
        elif split == 'spatial':
            if coords is None:
                raise ValueError("split='spatial' requires coords")
            fit_idx, held_out_idx = _split_spatial(coords, fit_frac, grid, self.seed)
        else:
            raise ValueError(f"unknown split {split!r}")

        if (pca_query or pca_target) and len(fit_idx) <= self.n_components:
            raise ValueError(
                f"fitting set has {len(fit_idx)} cells, too few for "
                f"n_components={self.n_components}")

        if pca_query:
            qp, self.query_pca_ = _pca_fit_transform(query, fit_idx, self.n_components, self.seed, 'query')
        else:
            qp, self.query_pca_ = query, None
        if pca_target:
            tp, self.target_pca_ = _pca_fit_transform(target, fit_idx, self.n_components, self.seed, 'target')
        else:
            tp, self.target_pca_ = target, None

        o = np.random.RandomState(1).permutation(len(fit_idx))
        inner_fit = fit_idx[o[:int(0.8 * len(fit_idx))]]
        inner_val = fit_idx[o[int(0.8 * len(fit_idx)):]]

        def _instance_rank(s):
            m = s.shape[0]
            return np.mean([(s[i] < s[i, i]).sum() / (m - 1) for i in range(m)])

        self.alpha_ = max(
            self.alphas,
            key=lambda a: _instance_rank(
                _l2(qp[inner_val] @ _ridge(qp[inner_fit], tp[inner_fit], a))
                @ _l2(tp[inner_val]).T))

        self.coef_ = _ridge(qp[fit_idx], tp[fit_idx], self.alpha_)
        self.fit_idx_ = fit_idx
        self.held_out_idx_ = held_out_idx
        self.similarity_ = _l2(qp[held_out_idx] @ self.coef_) @ _l2(tp[held_out_idx]).T
        return self

    def retrieve(self, top_k=5):
        """(n_held_out, top_k) indices into `held_out_idx_`, best match first."""
        return np.argsort(-self.similarity_, axis=1)[:, :top_k]

    def roc(self, relevance='instance', ground_truth=None, top_k=5, groups=None,
            n_grid=301):
        """Per-query ROC over held-out cells, averaged on a common grid.

        Returns (false positive rate grid, mean true positive rate, mean AUROC).

        relevance='instance'    only the query's own target is positive.
        relevance='similarity'  the `top_k` targets most similar to the query's
                                under `ground_truth`, an (N, D) array scored by
                                cosine similarity on the held-out rows.
        groups                  restrict each query's candidate pool to cells
                                sharing its label, e.g. one cell type.
        """
        s = self.similarity_
        n = s.shape[0]
        if relevance == 'instance':
            rel = None
        elif relevance == 'similarity':
            if ground_truth is None:
                raise ValueError("relevance='similarity' requires ground_truth")
            g = np.asarray(ground_truth, dtype=np.float64)
            g = g.reshape(len(g), -1)[self.held_out_idx_]
            rel = _l2(g) @ _l2(g).T
        else:
            raise ValueError(f"unknown relevance {relevance!r}")

        if groups is None:
            pool = np.ones((n, n), bool)
        else:
            g = np.asarray(groups)[self.held_out_idx_]
            pool = g[:, None] == g[None, :]

        grid = np.linspace(0, 1, n_grid)
        tprs, aucs = [], []
        for i in range(n):
            m = pool[i]
            sc = s[i][m]
            if rel is None:
                lab = np.where(m)[0] == i
            else:
                r = rel[i][m]
                if len(r) < top_k:
                    continue
                lab = r >= np.sort(r)[-top_k]
            n_pos = int(lab.sum())
            n_neg = len(sc) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            ls = lab[np.argsort(-sc)]
            tpr = np.concatenate([[0], np.cumsum(ls) / n_pos])
            fpr = np.concatenate([[0], np.cumsum(~ls) / n_neg])
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(np.trapz(tpr, fpr))
        return grid, np.mean(tprs, 0), float(np.mean(aucs))

    def score(self, relevance='instance', ground_truth=None, top_k=5, groups=None):
        """Mean per-query AUROC over held-out cells; see `roc`."""
        return self.roc(relevance, ground_truth, top_k, groups)[2]


@torch.no_grad()
def morphology_repr(model, cell_morphology, nuclear_morphology,
                    batch_size=16, device=None):
    """(N, 2 * dim) cell- and nucleus-mask tokens, read before the Performer.

    """
    device = device or next(model.parameters()).device
    cm = torch.as_tensor(np.asarray(cell_morphology), dtype=torch.float32)
    nm = torch.as_tensor(np.asarray(nuclear_morphology), dtype=torch.float32)
    out = []
    for s in range(0, len(cm), batch_size):
        e = slice(s, min(s + batch_size, len(cm)))
        out.append(torch.cat([model.cell_morphology_token(cm[e].to(device)),
                              model.nuclear_morphology_token(nm[e].to(device))],
                             dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def gene_pattern_repr(model, img, sn_expr, sn_dist,
                      cell_morphology, nuclear_morphology, identity,
                      genes=None, ablate=(), n_layers=3, cell_median=None,
                      batch_size=16, device=None):
    """(N, n_genes, dim) per-gene tokens averaged over the first `n_layers`
    Performer blocks, with every gene observed.
    """
    device = device or next(model.parameters()).device
    unknown = set(ablate) - {'morphology', 'spatial_context', 'identity'}
    if unknown:
        raise ValueError(f"unknown ablation target(s): {sorted(unknown)}")

    img = torch.as_tensor(np.asarray(img), dtype=torch.float32)
    sn_e = torch.as_tensor(np.asarray(sn_expr), dtype=torch.float32)
    sn_d = torch.as_tensor(np.asarray(sn_dist), dtype=torch.float32)
    cm = torch.as_tensor(np.asarray(cell_morphology), dtype=torch.float32)
    nm = torch.as_tensor(np.asarray(nuclear_morphology), dtype=torch.float32)
    ci = torch.as_tensor(np.asarray(identity), dtype=torch.float32)

    n, n_genes = img.shape[:2]
    if cell_median is None:
        cell_median = float(np.median(img.reshape(n, -1).sum(1).numpy()))
    gene_idx = torch.arange(n_genes) if genes is None else torch.as_tensor(np.asarray(genes))
    id_const = ci.mean(0, keepdim=True).to(device)

    out = None
    with LayerAverage(model, n_layers=n_layers) as tap:
        for s in range(0, n, batch_size):
            e = slice(s, min(s + batch_size, n))
            b = e.stop - e.start
            im = img[e].to(device)
            sf = im.reshape(b, -1).sum(1) / cell_median
            mask = torch.zeros(b, n_genes, dtype=torch.bool, device=device)
            cm_b = torch.zeros_like(cm[e]).to(device) if 'morphology' in ablate else cm[e].to(device)
            nm_b = torch.zeros_like(nm[e]).to(device) if 'morphology' in ablate else nm[e].to(device)
            sn_b = torch.zeros_like(sn_e[e]).to(device) if 'spatial_context' in ablate else sn_e[e].to(device)
            ci_b = id_const.expand(b, -1) if 'identity' in ablate else ci[e].to(device)
            model(im / sf.view(b, 1, 1, 1), mask, sn_b, sn_d[e].to(device),
                  cm_b, nm_b, ci_b)
            emb = tap.embedding()[:, gene_idx.to(device)].cpu().numpy()
            if out is None:
                out = np.zeros((n,) + emb.shape[1:], np.float32)
            out[e] = emb
    return out


def spatial_context_repr(nb_idx, identity, k=20):
    """(N, n_types) cell-type composition of each cell's `k` nearest neighbors.
    """
    nb_idx = np.asarray(nb_idx)
    identity = np.asarray(identity, dtype=np.float32)
    if nb_idx.shape[1] < k:
        raise ValueError(f"nb_idx has only {nb_idx.shape[1]} neighbors, need k={k}")
    return identity[nb_idx[:, :k]].mean(1)

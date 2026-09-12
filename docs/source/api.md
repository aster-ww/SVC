# API reference

## `svc.SVC`

SVC is implemented with a Vision Transformer backbone and a Performer encoder, comprising
12 Performer layers with 12 attention heads per layer. Each gene within each cell is
represented by two gene-level embeddings, and each cell is represented by three types of
cell-level embeddings; these representations are combined and fed into the encoder. SVC is
trained using a self-supervised masked image modeling procedure, in which a random subset of
gene expression images in each cell is masked, and a decoder reconstructs their spatial
expression patterns.

### Inputs

```python
gene_encoding, mu, r = model(
    img,
    mask,
    sn_expr,
    sn_dist,
    cell_morphology_vec,
    nuclear_morphology_vec,
    cell_identity_vec,
)
```

| argument | shape | |
|---|---|---|
| `img` | (cells, genes, 12, 12) | registered gene images, divided by the cell's size factor |
| `mask` | (cells, genes) bool | the genes to predict; their images are replaced by a mask token |
| `sn_expr` | (cells, k, genes) | each neighbor's expression |
| `sn_dist` | (cells, k) | each neighbor's distance |
| `cell_morphology_vec` | (cells, 48, 48) | cell mask |
| `nuclear_morphology_vec` | (cells, 48, 48) | nuclear mask |
| `cell_identity_vec` | (cells, types) | cell type or state, one-hot; optional |

### Returns

`(gene_encoding, mu, r)` — the per-(cell, gene) representation, (cells, genes, 384), and the
mean and dispersion of each gene's predicted image, both (cells, genes, 12, 12).

### Construction

```python
from svc.model import SVC

model = SVC(
    gene2vec_weight=gene2vec_weight,
    n_genes_for_sn=n_genes,
    k_scales=(4, 16, 64),
    tau_init_per_scale=tau,
    cell_morphology=True,
    nuclear_morphology=True,
    use_cell_identity=True,
    cell_identity_dim=n_types,
    emb_dropout=0.0,
)
model.load_state_dict(state)
model.eval()
```

`k_scales` is `(4, 16, 64)` for tissue sections and `(4,)` for cultured cells in small fields
of view. Each dataset's `train.py`, `evaluate.py` and `extract_latent.py` build the model and
call it end to end; the tutorials run those scripts.

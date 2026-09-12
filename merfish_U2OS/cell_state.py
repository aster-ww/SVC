"""
Cell states from per-(cell, gene) SVC embeddings.
"""

import numpy as np
from sklearn.cluster import SpectralClustering

SEED = 0


def cell_states(emb, n_clusters, seed=SEED):
    """Cluster cells on their gene-gene cosine similarity matrices.

    emb        : (n_cells, n_genes, dim) latent embeddings
    returns    : cosine (n_cells, n_genes, n_genes), labels (n_cells,)
    """
    norm = np.linalg.norm(emb, axis=2, keepdims=True)
    cosine = np.einsum('cgd,chd->cgh', emb, emb) / (norm * norm.transpose(0, 2, 1))

    n_cells = len(cosine)
    distance = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        d = np.sqrt(((cosine[i] - cosine[i + 1:]) ** 2).sum(axis=(1, 2)))
        distance[i, i + 1:] = distance[i + 1:, i] = d

    sigma = np.median(distance)
    affinity = np.exp(-distance ** 2 / (2. * sigma ** 2))
    labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                random_state=seed).fit_predict(affinity)
    return cosine, labels

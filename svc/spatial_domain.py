"""
Spatial domains from an SVC co-localization graph plus a spatial graph.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

SEED = 6


def spatial_domains(svc_graph, spatial_graph, k, seed=SEED):
    """Cluster cells into `k` spatial domains.

    svc_graph, spatial_graph : (n_cells, n_cells) sparse, same node order
    returns                  : (n_cells,) int labels, -1 where the node is isolated
    """
    rng_state = np.random.RandomState(seed)
    total = svc_graph + spatial_graph
    degree = np.asarray(total.sum(axis=1)).ravel()
    n = total.shape[0]

    laplacian = csr_matrix((degree, (np.arange(n), np.arange(n)))) - total
  
    with np.errstate(divide='ignore'):
        inv_sqrt = csr_matrix((np.sqrt(1.0 / degree), (np.arange(n), np.arange(n))))
    normalized = inv_sqrt @ laplacian @ inv_sqrt

    _, vectors = eigsh(normalized, k=k, which='SM', v0=rng_state.rand(n))

    connected = degree != 0
    embedding = vectors[connected]
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    labels = np.full(n, -1)
    labels[connected] = KMeans(n_clusters=k, random_state=seed).fit_predict(embedding)
    return labels

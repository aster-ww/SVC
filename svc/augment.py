"""
D4 augmentation: the four 90-degree rotations, with and without a reflection.

Used for cultured cells only (seqFISH+ 3T3, MERFISH U2OS), which have no fixed
orientation. The 12x12 gene-map frame and the 48x48 morphology frame share
orientation, so the same element must be applied to every spatial tensor of a
cell. 
"""

import numpy as np

_D4_ELEMENTS = [(0, False), (1, False), (2, False), (3, False),
                (0, True),  (1, True),  (2, True),  (3, True)]   # (k_rot90, do_flip)


def apply_d4(arr, elem, axes):
    """Apply one D4 element to `arr` over the two spatial dims given in `axes`.
    elem is (k_rot90, do_flip) from _D4_ELEMENTS; returns a contiguous array."""
    k, do_flip = elem
    out = np.rot90(arr, k, axes=axes)
    if do_flip:
        out = np.flip(out, axis=axes[0])
    return np.ascontiguousarray(out)


def d4_augment(gene_map, cell_morph, nuclear_morph, elem=None):
    """Apply ONE D4 element (random if elem is None) to all three spatial tensors.
    gene_map (n_genes, 12, 12) -> spatial axes (1, 2); cell_morph / nuclear_morph
    (48, 48) -> spatial axes (0, 1). location / identity are not passed (a cell's
    absolute coordinate and class label are orientation-invariant)."""
    if elem is None:
        elem = _D4_ELEMENTS[np.random.randint(8)]
    return (apply_d4(gene_map, elem, (1, 2)),
            apply_d4(cell_morph, elem, (0, 1)),
            apply_d4(nuclear_morph, elem, (0, 1)))

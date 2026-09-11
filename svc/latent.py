"""
Per-(cell, gene) latent representations from a trained SVC.
        
"""

import types

import torch

from svc.reversible import route_args


class LayerAverage:
    """Context manager that captures each Performer block's gene tokens."""

    def __init__(self, model, n_layers=3, l2_normalize=False):
        self.model = model
        self.n_layers = n_layers
        self.l2_normalize = l2_normalize
        self.buf = []
        self._orig = None

    def __enter__(self):
        net = self.model.performer.net
        n_aux = self.model.n_aux
        buf = self.buf
        self._orig = net.forward

        def _patched(self, x, output_attentions=False, **kw):
            a = route_args(self.args_route, kw, len(self.layers))
            outs = []
            for (f, g), (fa, ga) in zip(self.layers, a):
                x = x + f(x, **fa)
                x = x + g(x, **ga)
                outs.append(x[:, n_aux:, :])
            buf.clear()
            buf.extend(outs)
            return x

        net.forward = types.MethodType(_patched, net)
        return self

    def __exit__(self, *exc):
        self.model.performer.net.forward = self._orig
        return False

    def embedding(self):
        """(B, G, dim) mean of the first `n_layers` blocks' gene tokens."""
        if not self.buf:
            raise RuntimeError("no forward pass was run inside the LayerAverage block")
        emb = torch.stack(self.buf[:self.n_layers], 0).mean(0)
        if self.l2_normalize:
            emb = torch.nn.functional.normalize(emb, dim=2)
        return emb

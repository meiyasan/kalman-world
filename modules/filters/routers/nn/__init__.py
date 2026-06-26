"""Neural architectures for NeuralRouter.

All follow the same three-method protocol recognised by NeuralRouter:

* ``build(d_in, d_out, seed=None)`` — lazy parameter initialisation
* ``fit(X, y, **kwargs)``           — training on (N, d_in) features
* ``__call__(x)``                   — forward: (d_in,) → (d_out,) probs

MLP is a working pure-numpy implementation.
TransformerNN, LSTM, CNN are stubs — see their module docstrings for
complete PyTorch implementation examples.
"""
from .mlp         import MLP
from .transformer import TransformerNN
from .lstm        import LSTM
from .cnn         import CNN

__all__ = ["MLP", "TransformerNN", "LSTM", "CNN"]

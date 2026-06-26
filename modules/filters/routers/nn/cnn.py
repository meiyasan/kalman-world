"""CNN — stub for 1-D temporal convolutional architectures.

A 1-D CNN treats the NIS history (window × M) as a multichannel sequence.
Convolutional filters learn local temporal patterns across models
(e.g., sudden NIS spikes indicating a regime switch).

Plug into NeuralRouter::

    router = NeuralRouter(model=MyCNN(), window=10)
    ikf.fit(training_zk_list, labels)

Implement three methods:

* ``build(d_in, d_out, seed=None)`` — initialise
* ``fit(X, y, **kwargs)``           — train
* ``__call__(x)``                   — forward: (d_in,) → (d_out,) probs

PyTorch example
---------------
.. code-block:: python

    import torch, torch.nn as nn

    class MyCNN(CNN):

        def build(self, d_in, d_out, seed=None):
            M = d_out
            W = d_in // M
            self._net = nn.Sequential(
                # (batch, M, W) — M channels, W time steps
                nn.Conv1d(M, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),      # global average pooling
                nn.Flatten(),
                nn.Linear(32, d_out),
                nn.Softmax(dim=-1),
            )
            self._opt = torch.optim.Adam(self._net.parameters(), lr=2e-3)
            self._M   = M
            self._W   = W

        def fit(self, X, y, epochs=200, **_):
            N = len(X)
            # reshape flat (N, M*W) → (N, M, W)
            Xt = torch.tensor(
                X.reshape(N, self._M, self._W), dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(epochs):
                self._opt.zero_grad()
                loss_fn(self._net(Xt), yt).backward()
                self._opt.step()

        def __call__(self, x):
            with torch.no_grad():
                t = torch.tensor(
                    x.reshape(1, self._M, self._W), dtype=torch.float32)
                return self._net(t).squeeze(0).numpy()
"""
from __future__ import annotations
import numpy as np


class CNN:
    """Stub: override ``build``, ``fit``, ``__call__`` with your framework."""

    def build(self, d_in: int, d_out: int, seed=None) -> None:
        raise NotImplementedError(
            "CNN.build: implement with your framework.\n"
            "See the module docstring for a PyTorch example."
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        raise NotImplementedError("CNN.fit")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CNN.__call__")

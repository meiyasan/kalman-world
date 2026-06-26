"""LSTM — stub for recurrent architectures.

An LSTM sees the full NIS sequence step-by-step, making it well-suited
for online routing where the window need not be fixed in advance.

Plug into NeuralRouter::

    router = NeuralRouter(model=MyLSTM(), window=10)
    ikf.fit(training_zk_list, labels)

The NeuralRouter presents the last ``window`` NIS steps as a flat feature
vector.  To use the full sequence, override ``NeuralRouter._window_feat``
or provide raw NIS arrays directly to a custom fit method.

Implement three methods:

* ``build(d_in, d_out, seed=None)`` — initialise
* ``fit(X, y, **kwargs)``           — train
* ``__call__(x)``                   — forward: (d_in,) → (d_out,) probs

PyTorch example
---------------
.. code-block:: python

    import torch, torch.nn as nn

    class MyLSTM(LSTM):

        def build(self, d_in, d_out, seed=None):
            M = d_out
            self._lstm = nn.LSTM(input_size=M, hidden_size=32, batch_first=True)
            self._head = nn.Linear(32, d_out)
            self._opt  = torch.optim.Adam(
                list(self._lstm.parameters()) + list(self._head.parameters()))
            self._M    = M
            self._window = d_in // M

        def fit(self, X, y, epochs=200, **_):
            # reshape flat (N, M*W) → (N, W, M) sequences
            N, W, M = len(X), self._window, self._M
            Xt = torch.tensor(X.reshape(N, W, M), dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(epochs):
                self._opt.zero_grad()
                out, _ = self._lstm(Xt)
                logits = self._head(out[:, -1])
                loss_fn(logits, yt).backward()
                self._opt.step()

        def __call__(self, x):
            with torch.no_grad():
                W, M = self._window, self._M
                t   = torch.tensor(x.reshape(1, W, M), dtype=torch.float32)
                out, _ = self._lstm(t)
                logits = self._head(out[0, -1])
                return torch.softmax(logits, dim=-1).numpy()
"""
from __future__ import annotations
import numpy as np


class LSTM:
    """Stub: override ``build``, ``fit``, ``__call__`` with your framework."""

    def build(self, d_in: int, d_out: int, seed=None) -> None:
        raise NotImplementedError(
            "LSTM.build: implement with your framework.\n"
            "See the module docstring for a PyTorch example."
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        raise NotImplementedError("LSTM.fit")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LSTM.__call__")

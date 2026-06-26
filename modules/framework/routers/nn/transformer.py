"""TransformerNN — stub for self-attention architectures.

The NIS history (T × M) is a natural sequence for a Transformer: each step
is a token, each model's NIS is a feature.  Self-attention can capture
long-range regime correlations that a fixed-window MLP misses.

Plug into NeuralRouter::

    router = NeuralRouter(model=MyTransformerNN(), window=16)
    ikf.fit(training_zk_list, labels)

Implement three methods:

* ``build(d_in, d_out, seed=None)`` — initialise your architecture
* ``fit(X, y, **kwargs)``           — train on (N, d_in) features, (N,) labels
* ``__call__(x)``                   — forward pass: (d_in,) → (d_out,) probs

PyTorch example
---------------
.. code-block:: python

    import torch, torch.nn as nn

    class MyTransformerNN(TransformerNN):

        def build(self, d_in, d_out, seed=None):
            if seed is not None:
                torch.manual_seed(seed)
            self._net = nn.Sequential(
                nn.Linear(d_in, 64),
                nn.ReLU(),
                nn.Linear(64, d_out),
                nn.Softmax(dim=-1),
            )
            self._opt = torch.optim.Adam(self._net.parameters(), lr=2e-3)

        def fit(self, X, y, epochs=200, **_):
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(epochs):
                self._opt.zero_grad()
                loss_fn(self._net(Xt), yt).backward()
                self._opt.step()

        def __call__(self, x):
            with torch.no_grad():
                t = torch.tensor(x, dtype=torch.float32)
                return self._net(t).numpy()

sklearn example
---------------
.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier

    class ForestNN(TransformerNN):

        def build(self, d_in, d_out, seed=None):
            self._clf = RandomForestClassifier(n_estimators=100,
                                               random_state=seed)

        def fit(self, X, y, **_):
            self._clf.fit(X, y)

        def __call__(self, x):
            return self._clf.predict_proba(x.reshape(1, -1))[0]
"""
from __future__ import annotations
import numpy as np


class TransformerNN:
    """Stub: override ``build``, ``fit``, ``__call__`` with your framework."""

    def build(self, d_in: int, d_out: int, seed=None) -> None:
        raise NotImplementedError(
            "TransformerNN.build: implement with your framework.\n"
            "See the module docstring for PyTorch and sklearn examples."
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        raise NotImplementedError("TransformerNN.fit")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TransformerNN.__call__")

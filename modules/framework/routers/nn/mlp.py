"""MLP — 2-hidden-layer tanh MLP over the flattened NIS window.

Plug into NeuralRouter::

    from kalman.framework.routers import NeuralRouter
    from kalman.framework.routers.nn import MLP

    router = NeuralRouter(model=MLP(hidden=32), window=10)
    ikf.fit(training_zk_list, labels_list)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

_DEVICE = ("mps"  if torch.backends.mps.is_available()  else
           "cuda" if torch.cuda.is_available()           else "cpu")


class MLP:
    """Two-hidden-layer tanh MLP + softmax head, trained with Adam.

    Parameters
    ----------
    hidden : int
        Width of each hidden layer.
    """

    def __init__(self, hidden: int = 32):
        self.hidden = int(hidden)
        self._net: nn.Module | None = None
        self._opt: torch.optim.Optimizer | None = None

    def build(self, d_in: int, d_out: int, seed=None) -> "MLP":
        if seed is not None:
            torch.manual_seed(seed)
        h = self.hidden
        self._net = nn.Sequential(
            nn.Linear(d_in, h), nn.Tanh(),
            nn.Linear(h, h),    nn.Tanh(),
            nn.Linear(h, d_out),
        ).to(_DEVICE)
        self._opt = torch.optim.Adam(self._net.parameters())
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 300, lr: float = 3e-3,
            batch_size: int = 64, **_) -> "MLP":
        self._opt.param_groups[0]["lr"] = lr
        Xt = torch.tensor(X, dtype=torch.float32, device=_DEVICE)
        yt = torch.tensor(y, dtype=torch.long,    device=_DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        N = len(Xt)
        for _ in range(epochs):
            idx = torch.randperm(N, device=_DEVICE)
            for s in range(0, N, batch_size):
                b = idx[s:s + batch_size]
                self._opt.zero_grad()
                loss_fn(self._net(Xt[b]), yt[b]).backward()
                self._opt.step()
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=_DEVICE)
            logits = self._net(xt).cpu().numpy()
        e = np.exp(logits - logits.max())
        return e / e.sum()

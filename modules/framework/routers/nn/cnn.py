"""CNN — 1-D temporal convolution over the NIS window.

Architecture: Conv1d (M channels × window positions) → tanh → global average
pool → linear softmax head.  Input layout: (batch, M, window) — M models are
treated as channels, window steps as the temporal axis.

Plug into NeuralRouter::

    from kalman.framework.routers import NeuralRouter
    from kalman.framework.routers.nn import CNN

    router = NeuralRouter(model=CNN(out_channels=8, kernel_size=3), window=10)
    ikf.fit(training_zk_list, labels_list)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

_DEVICE = ("mps"  if torch.backends.mps.is_available()  else
           "cuda" if torch.cuda.is_available()           else "cpu")


class CNN:
    """1-D convolution + global average pool + dense softmax head.

    Parameters
    ----------
    out_channels : int
        Number of convolutional filters.
    kernel_size : int
        Temporal kernel width (clipped to window length at build time).
    """

    def __init__(self, out_channels: int = 8, kernel_size: int = 3):
        self.out_channels = int(out_channels)
        self.kernel_size  = int(kernel_size)
        self._M: int | None = None
        self._W: int | None = None
        self._net: nn.Module | None = None
        self._opt: torch.optim.Optimizer | None = None
        self.loss_history: list[float] = []

    def build(self, d_in: int, d_out: int, seed=None) -> "CNN":
        if seed is not None:
            torch.manual_seed(seed)
        M = d_out
        W = d_in // M
        k = min(self.kernel_size, W)
        self._M, self._W = M, W
        h = self.out_channels
        self._net = nn.Sequential(
            nn.Conv1d(M, h, kernel_size=k, padding=k // 2),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1),   # → (batch, h, 1)
            nn.Flatten(),              # → (batch, h)
            nn.Linear(h, d_out),
        ).to(_DEVICE)
        self._opt = torch.optim.Adam(self._net.parameters())
        self.loss_history = []
        return self

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """(N, M*W) → (N, M, W)."""
        return X.reshape(-1, self._M, self._W)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 300, lr: float = 3e-3,
            batch_size: int = 64, **_) -> "CNN":
        self._opt.param_groups[0]["lr"] = lr
        Xt = torch.tensor(self._reshape(X), dtype=torch.float32, device=_DEVICE)
        yt = torch.tensor(y, dtype=torch.long, device=_DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        N = len(Xt)
        for _ in range(epochs):
            idx = torch.randperm(N, device=_DEVICE)
            epoch_loss, n_batches = 0.0, 0
            for s in range(0, N, batch_size):
                b = idx[s:s + batch_size]
                self._opt.zero_grad()
                loss = loss_fn(self._net(Xt[b]), yt[b])
                loss.backward()
                self._opt.step()
                epoch_loss += float(loss.item()); n_batches += 1
            self.loss_history.append(epoch_loss / max(n_batches, 1))
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(self._reshape(x[None]), dtype=torch.float32, device=_DEVICE)
            logits = self._net(xt).squeeze(0).cpu().numpy()
        e = np.exp(logits - logits.max())
        return e / e.sum()

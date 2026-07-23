"""LSTM — single-layer recurrent router over the NIS window.

The NIS history is processed step by step (one of the ``window`` timesteps at
a time).  Unlike MLP's fixed concatenated window or CNN's local receptive
field, the LSTM carries hidden/cell state forward — useful when the
discriminating signal is a temporal pattern (e.g. burst timing) rather than
the instantaneous NIS level.

Plug into NeuralRouter::

    from kalman.framework.routers import NeuralRouter
    from kalman.framework.routers.nn import LSTM

    router = NeuralRouter(model=LSTM(hidden=16), window=10)
    ikf.fit(training_zk_list, labels_list)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

_DEVICE = ("mps"  if torch.backends.mps.is_available()  else
           "cuda" if torch.cuda.is_available()           else "cpu")


class LSTM:
    """Single-layer LSTM + linear softmax head.

    Parameters
    ----------
    hidden : int
        Hidden / cell state size.
    """

    def __init__(self, hidden: int = 16):
        self.hidden = int(hidden)
        self._M: int | None = None
        self._W: int | None = None
        self._lstm: nn.LSTM | None = None
        self._head: nn.Linear | None = None
        self._opt: torch.optim.Optimizer | None = None
        self.loss_history: list[float] = []

    def build(self, d_in: int, d_out: int, seed=None) -> "LSTM":
        if seed is not None:
            torch.manual_seed(seed)
        M = d_out
        W = d_in // M
        self._M, self._W = M, W
        self._lstm = nn.LSTM(M, self.hidden, batch_first=True).to(_DEVICE)
        self._head = nn.Linear(self.hidden, d_out).to(_DEVICE)
        params = list(self._lstm.parameters()) + list(self._head.parameters())
        self._opt = torch.optim.Adam(params)
        self.loss_history = []
        return self

    def _to_seq(self, X: np.ndarray) -> torch.Tensor:
        """(N, M*W) → (N, W, M) float32 tensor on device."""
        return torch.tensor(
            X.reshape(-1, self._W, self._M), dtype=torch.float32, device=_DEVICE
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 300, lr: float = 2e-3,
            batch_size: int = 64, **_) -> "LSTM":
        self._opt.param_groups[0]["lr"] = lr
        Xt = self._to_seq(X)                                       # (N, W, M)
        yt = torch.tensor(y, dtype=torch.long, device=_DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        N = len(Xt)
        for _ in range(epochs):
            idx = torch.randperm(N, device=_DEVICE)
            epoch_loss, n_batches = 0.0, 0
            for s in range(0, N, batch_size):
                b = idx[s:s + batch_size]
                self._opt.zero_grad()
                out, _ = self._lstm(Xt[b])          # (batch, W, hidden)
                logits = self._head(out[:, -1, :])  # last step → (batch, d_out)
                loss = loss_fn(logits, yt[b])
                loss.backward()
                self._opt.step()
                epoch_loss += float(loss.item()); n_batches += 1
            self.loss_history.append(epoch_loss / max(n_batches, 1))
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = self._to_seq(x[None])              # (1, W, M)
            out, _ = self._lstm(xt)
            logits = self._head(out[:, -1, :]).squeeze(0).cpu().numpy()
        e = np.exp(logits - logits.max())
        return e / e.sum()

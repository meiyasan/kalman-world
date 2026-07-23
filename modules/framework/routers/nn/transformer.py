"""TransformerNN — single-head self-attention router over the NIS window.

The NIS history (window × M) is a natural sequence for a Transformer: each
of the ``window`` timesteps is a token, each model's NIS is a feature.
Self-attention lets every position attend to every other position in one
step — no fixed receptive field (unlike the CNN) and no sequential
bottleneck (unlike the LSTM).

Architecture: per-token linear embedding → TransformerEncoderLayer (one block,
single head) → global average pool over tokens → linear softmax head.

Plug into NeuralRouter::

    from kalman.framework.routers import NeuralRouter
    from kalman.framework.routers.nn import TransformerNN

    router = NeuralRouter(model=TransformerNN(d_model=16), window=10)
    ikf.fit(training_zk_list, labels_list)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

_DEVICE = ("mps"  if torch.backends.mps.is_available()  else
           "cuda" if torch.cuda.is_available()           else "cpu")


class _Block(nn.Module):
    def __init__(self, M: int, d: int, d_ff: int, d_out: int):
        super().__init__()
        self.embed   = nn.Linear(M, d)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d, nhead=1, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True,
        )
        self.head = nn.Linear(d, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, M)
        e = self.embed(x)           # (B, W, d)
        o = self.encoder(e)         # (B, W, d)
        return self.head(o.mean(1)) # (B, d_out)


class TransformerNN:
    """Single-head self-attention block + FFN + dense softmax head.

    Parameters
    ----------
    d_model : int
        Embedding / attention dimension.
    d_ff : int, optional
        Feed-forward hidden width.  Defaults to ``2 * d_model``.
    """

    def __init__(self, d_model: int = 16, d_ff: int | None = None):
        self.d_model = int(d_model)
        self.d_ff    = int(d_ff) if d_ff is not None else 2 * self.d_model
        self._M: int | None = None
        self._W: int | None = None
        self._block: _Block | None = None
        self._opt: torch.optim.Optimizer | None = None
        self.loss_history: list[float] = []

    def build(self, d_in: int, d_out: int, seed=None) -> "TransformerNN":
        if seed is not None:
            torch.manual_seed(seed)
        M = d_out
        W = d_in // M
        self._M, self._W = M, W
        self._block = _Block(M, self.d_model, self.d_ff, d_out).to(_DEVICE)
        self._opt   = torch.optim.Adam(self._block.parameters())
        self.loss_history = []
        return self

    def _to_seq(self, X: np.ndarray) -> torch.Tensor:
        """(N, M*W) → (N, W, M) float32 tensor on device."""
        return torch.tensor(
            X.reshape(-1, self._W, self._M), dtype=torch.float32, device=_DEVICE
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 300, lr: float = 2e-3,
            batch_size: int = 64, **_) -> "TransformerNN":
        self._opt.param_groups[0]["lr"] = lr
        Xt = self._to_seq(X)
        yt = torch.tensor(y, dtype=torch.long, device=_DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        N = len(Xt)
        for _ in range(epochs):
            idx = torch.randperm(N, device=_DEVICE)
            epoch_loss, n_batches = 0.0, 0
            for s in range(0, N, batch_size):
                b = idx[s:s + batch_size]
                self._opt.zero_grad()
                loss = loss_fn(self._block(Xt[b]), yt[b])
                loss.backward()
                self._opt.step()
                epoch_loss += float(loss.item()); n_batches += 1
            self.loss_history.append(epoch_loss / max(n_batches, 1))
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = self._to_seq(x[None])
            logits = self._block(xt).squeeze(0).cpu().numpy()
        e = np.exp(logits - logits.max())
        return e / e.sum()

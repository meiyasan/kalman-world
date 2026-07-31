"""CNN forecast surrogate for :class:`~kalman.apps.weather.flow.KolmogorovFlowModel`.

Imports torch at module level, so it is **not** re-exported from
``kalman.apps`` — import it explicitly (same convention as
``kalman.framework.routers.nn``)::

    from kalman.apps.weather.surrogate import (make_trajectory, train_surrogate,
                                           as_forecaster, rollout_rmse)

The prototype that preceded this module showed that a surrogate trained on
**one-step** pairs with plain MSE diverges exponentially once cycled inside an
EnKF (nothing in the objective penalizes multi-cycle drift, and the bias
compounds every cycle).  Two rules are therefore built in here:

1. :func:`train_surrogate` uses a **rollout loss** — the network is applied
   ``rollout`` times and penalized at every intermediate cycle, which is what
   actually constrains cycle-to-cycle bias.
2. :func:`rollout_rmse` measures **open-loop** error growth against the true
   solver.  Verify it stays bounded (well below the climatological field std)
   over at least as many cycles as the DA experiment will run *before* wiring
   the surrogate into a filter — and still budget a larger inflation for the
   surrogate branch than for the true-model branch (their error statistics
   differ).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class PeriodicConv2d(nn.Module):
    """Conv2d with periodic (torus) padding — matches the solver's domain."""
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv2d(cin, cout, k)
    def forward(self, x):
        p = self.pad
        x = torch.cat([x[..., -p:, :], x, x[..., :p, :]], dim=-2)
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.c1 = PeriodicConv2d(c, c, k)
        self.c2 = PeriodicConv2d(c, c, k)
        self.act = nn.GELU()
    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.c2(h)
        return x + h


class FlowSurrogate(nn.Module):
    """Predicts the increment ``Δω = ω(t+cycle) − ω(t)``; residual formulation
    (stabilizes training and keeps the identity map as an easy fallback)."""
    def __init__(self, width=16, nblocks=3):
        super().__init__()
        self.width, self.nblocks = width, nblocks
        self.inp = PeriodicConv2d(1, width, 3)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(nblocks)])
        self.out = PeriodicConv2d(width, 1, 3)
    def forward(self, w):                     # w: (B, 1, n, n), normalized
        h = torch.tanh(self.inp(w))
        for b in self.blocks:
            h = b(h)
        return w + self.out(h)


def make_trajectory(model, n_cycles: int, burn: int = 2000) -> np.ndarray:
    """Roll the true solver and return snapshots one cycle apart, ``(T+1, n, n)``."""
    w = model.solver.advance(model.solver.random_ic(), burn)
    traj = [w.copy()]
    for _ in range(n_cycles):
        w = model.solver.advance(w, model.cycle)
        traj.append(w.copy())
    return np.stack(traj).astype(np.float32)


def train_surrogate(traj: np.ndarray, *, width=16, nblocks=3, epochs=60, lr=2e-3,
                    batch=16, rollout=4, val_frac=0.15, device="cpu", seed=0,
                    verbose=True):
    """Train :class:`FlowSurrogate` on a solver trajectory with a rollout loss.

    ``traj`` is ``(T, n, n)`` snapshots one cycle apart (from
    :func:`make_trajectory`).  Each sample is a window ``w_i … w_{i+rollout}``;
    the loss is the mean MSE of ``net^k(w_i)`` against ``w_{i+k}`` for
    ``k = 1 … rollout`` (backprop through the whole rollout).  The validation
    split is the trajectory *tail* (contiguous in time, no window leakage).

    Returns ``(net, (mu, sd), best_val)`` — normalization constants travel
    with the network (see :func:`as_forecaster` / :func:`save_surrogate`).
    """
    torch.manual_seed(seed)
    T = traj.shape[0]
    if T < rollout + 2:
        raise ValueError(f"trajectory too short ({T}) for rollout={rollout}")
    n_val = max(rollout + 1, int(val_frac * T))
    tr, va = traj[:T - n_val], traj[T - n_val:]

    mu, sd = float(tr.mean()), float(tr.std())
    trn = torch.tensor((tr - mu) / sd, device=device).unsqueeze(1)   # (Ttr, 1, n, n)
    van = torch.tensor((va - mu) / sd, device=device).unsqueeze(1)

    def rollout_loss(data, starts):
        x = data[starts]
        loss = 0.0
        for k in range(1, rollout + 1):
            x = net(x)
            loss = loss + nn.functional.mse_loss(x, data[starts + k])
        return loss / rollout

    net = FlowSurrogate(width, nblocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr_starts = np.arange(trn.shape[0] - rollout)
    va_starts = torch.arange(van.shape[0] - rollout)
    rng = np.random.default_rng(seed)

    best_val, best_state = np.inf, None
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr_starts)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = torch.tensor(perm[i:i + batch])
            opt.zero_grad()
            loss = rollout_loss(trn, b)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            vloss = rollout_loss(van, va_starts).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if verbose and (ep + 1) % max(1, epochs // 6) == 0:
            print(f"  epoch {ep+1:3d}: train={tot/len(perm):.5f}  val={vloss:.5f}")
    net.load_state_dict(best_state)
    return net, (mu, sd), best_val


def as_forecaster(net, mu, sd):
    """Wrap a trained net as a batched forecast map (d, N) → (d, N) — plug it
    into ``model.surrogate`` or ``TwinExperiment.add_method(forecast=...)``."""
    net.eval()
    def forecast(X):
        d, N = X.shape
        n = int(round(np.sqrt(d)))
        xin = torch.tensor(X.T.reshape(N, 1, n, n).astype(np.float32))
        xin = (xin - mu) / sd
        with torch.no_grad():
            out = net(xin) * sd + mu
        return out.numpy().reshape(N, d).T.astype(np.float64)
    return forecast


def rollout_rmse(forecast, model, w0=None, n_cycles=10) -> np.ndarray:
    """Open-loop rollout error of a forecaster vs the true solver, per cycle.

    The mandatory pre-DA check: if this grows past the climatological field
    std within the horizon of the planned DA experiment, the surrogate will
    drag the filter along with it — retrain (longer rollout, more data)
    before touching the filter.
    """
    w0 = model.equilibrium_state() if w0 is None else np.asarray(w0, dtype=float).ravel()
    truth, pred = w0.copy(), w0.copy()[:, None]
    errs = []
    for _ in range(n_cycles):
        truth = model.f(truth)
        pred = forecast(pred)
        errs.append(float(np.sqrt(np.mean((pred.ravel() - truth) ** 2))))
    return np.array(errs)


class ReanchoredForecast:
    """Hybrid forecaster: the surrogate runs most cycles, the true solver every
    ``every``-th cycle — the cheap first mitigation when a surrogate is not yet
    stable under cycling (its bias is flushed by the solver before it can
    compound, at ``1/every`` of the full forecast cost).  Plug into
    ``TwinExperiment.add_method(forecast=...)`` like any forecaster; call
    :meth:`reset` before reusing it in a new experiment.
    """
    def __init__(self, model, forecast, every=3):
        self.model, self.forecast, self.every = model, forecast, int(every)
        self.k = 0

    def reset(self):
        self.k = 0
        return self

    def __call__(self, X):
        self.k += 1
        if self.k % self.every == 0:
            out = np.empty_like(X)
            for i in range(X.shape[1]):
                out[:, i] = self.model.solver.advance(X[:, i], self.model.cycle)
            return out
        return self.forecast(X)


def save_surrogate(path, net, mu, sd):
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd,
                "width": net.width, "nblocks": net.nblocks}, path)


def load_surrogate(path):
    ck = torch.load(path, map_location="cpu")
    net = FlowSurrogate(width=ck["width"], nblocks=ck["nblocks"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, (ck["mu"], ck["sd"])

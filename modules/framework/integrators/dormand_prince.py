"""Dormand–Prince RK45 integrator — embedded 5(4) pair, adaptive substeps.

Advances one *outer* step ``dt`` by taking as many adaptive RK45 substeps as the
local error tolerance requires, so a stiff-ish or fast-varying ``f`` is resolved
without the caller shrinking ``dt``.  The 5th-order solution is returned; the
4th-order companion gives the error estimate that drives step-size control.
"""
from __future__ import annotations
import numpy as np

from ..integrator import KalmanIntegrator
from ..constants import KalmanIntegratorKind

# Dormand–Prince 5(4) Butcher tableau.
_C = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
_A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
]
_B5 = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])
_B4 = np.array([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])


class DormandPrinceIntegrator(KalmanIntegrator):
    """Adaptive RK45 (Dormand–Prince); ``rtol``/``atol`` control the substeps."""

    NAME = "Dormand-Prince Integrator"
    DESCRIPTION = "Adaptive RK45 (Dormand–Prince 5(4)) with error-controlled substeps."
    VERSION = "1.0"
    KIND = KalmanIntegratorKind.DORMAND_PRINCE
    # 5th-order tableau (the b5 weights) — used as a *fixed-step* RK5 by field
    # solvers (adaptive per-member step control is skipped for batched ensembles;
    # the ODE ``step`` above keeps the full adaptive 5(4) control).
    tableau = (list(_C), [list(r) for r in _A], list(_B5))

    def __init__(self, rtol: float = 1e-6, atol: float = 1e-9, max_substeps: int = 10_000):
        self.rtol, self.atol, self.max_substeps = float(rtol), float(atol), int(max_substeps)

    NATIVE = "dormand_prince"

    def _native_options(self):
        return dict(rtol=self.rtol, atol=self.atol, max_substeps=self.max_substeps)

    def _step(self, f, x, u=None, t=None, dt=1.0):
        x = np.asarray(x, dtype=float)
        t0 = 0.0 if t is None else float(t)
        target = t0 + dt
        t, h = t0, dt
        for _ in range(self.max_substeps):
            if (h > 0 and t >= target) or (h < 0 and t <= target):
                break
            h = np.sign(dt) * min(abs(h), abs(target - t))
            ks = []
            for i in range(7):
                xi = x + h * sum(_A[i][j] * ks[j] for j in range(i)) if i else x
                ks.append(np.asarray(f(xi, u, t + _C[i] * h), dtype=float))
            K = np.stack(ks)
            x5 = x + h * np.tensordot(_B5, K, axes=1)
            x4 = x + h * np.tensordot(_B4, K, axes=1)
            err = np.linalg.norm((x5 - x4) / (self.atol + self.rtol * np.maximum(np.abs(x), np.abs(x5))))
            err = err / max(np.sqrt(x.size), 1)
            if err <= 1.0 or abs(h) <= 1e-14 * abs(dt):
                x, t = x5, t + h
            # PI-free step-size update (clamped)
            h *= min(5.0, max(0.2, 0.9 * (1.0 / max(err, 1e-12)) ** 0.2))
        return x


from ..integrator import _BUILTIN_STEPS
_BUILTIN_STEPS["dormand_prince"] = DormandPrinceIntegrator._step

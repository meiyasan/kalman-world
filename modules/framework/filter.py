# kalman/framework/filter.py
from __future__ import annotations
import numpy as np

from .io.record import KalmanRecord
from ..helpers import format_shape
from .abstract.metadata import KalmanFilterMetadata
from .abstract.mixins import KalmanFilterMixin


class KalmanFilter(KalmanFilterMetadata, KalmanFilterMixin):
    """
    The Kalman **filter** — workflow (WHEN). Composes a ``model`` (WHAT) and an
    ``engine`` (HOW), owns the run state (``x/P`` + prior/posterior/transition/smooth
    logs) and drives ``forward`` / ``backward``. It implements no covariance math —
    that lives in the engine.

    Engine resolution is declarative: if no engine is passed, one is built from
    ``self.KIND`` (set by public leaves like :class:`LinearKalmanFilter`) or, failing
    that, the model's ``DEFAULT_ENGINE``; the chosen engine's ``KIND`` is validated
    against ``model.ALLOWED_ENGINES``.
    """

    NAME = "Kalman Filter"
    DESCRIPTION = "Forward/backward workflow over a composed model + engine."
    VERSION = "1.0"
    KIND = None   # public leaves (LinearKalmanFilter, ...) set this

    def __init__(self, model, engine=None, *, epsilon: float = 1e-5):
        from .engines import resolve_engine

        self._model = model
        self._engine = engine if engine is not None else resolve_engine(self.KIND or model.DEFAULT_ENGINE)

        allowed = getattr(model, "ALLOWED_ENGINES", None)
        if allowed is not None and self._engine.KIND not in allowed:
            raise TypeError(
                f"{type(model).__name__} disallows engine kind {self._engine.KIND} "
                f"(allowed: {sorted(k.name for k in allowed)})."
            )

        self.epsilon = float(epsilon)
        self.reset()

    # ------------------------------------------------------------------
    # Forward filtering
    # ------------------------------------------------------------------
    def forward(self, gk, uk=None, *, record: bool = True, **kwargs):

        indexing    = kwargs.pop("indexing", "xy")
        preprocess  = kwargs.pop("preprocess", True)
        inprocess   = kwargs.pop("inprocess", True)
        postprocess = kwargs.pop("postprocess", True)

        reset  = kwargs.pop("reset", True)
        buffer = kwargs.pop("buffer", 60)  # seconds

        if len(self.posterior) > buffer * self.sample_rate():
            self.reset_to_latest()
        elif reset:
            self.reset()

        # --- normalise measurements / control ---
        gk = gk.zk if isinstance(gk, KalmanRecord) else gk
        gk = KalmanRecord._normalize_in(gk, indexing=indexing, dtype=self.model.dtype)
        assert gk.shape[1] >= 1, f"gk must have at least 1 column, got {format_shape(gk)}."

        uk = uk.zk if isinstance(uk, KalmanRecord) else uk
        if uk is None or len(uk) < 1:
            uk = np.zeros((len(gk), self.l))
        uk = KalmanRecord._normalize_in(uk, indexing=indexing, dtype=self.model.dtype)
        if self.l > 0:
            assert uk is not None and uk.shape[1] == self.l, \
                f"Control input uk has wrong number of columns. Expected ({self.l},1), got {format_shape(uk)}."

        # --- preprocess (may alter shapes) ---
        zk, uk = self.model.preprocess(gk, uk) if preprocess else (gk, uk)
        assert zk.shape[1] == self.m, \
            f"Preprocessed zk has incorrect number of columns. Expected ({self.m},1), got {format_shape(zk)}."
        if self.l > 0:
            assert uk is not None and uk.shape[1] == self.l, \
                f"Preprocessed uk has incorrect number of columns. Expected ({self.l},1), got {format_shape(uk)}."

        # --- main loop (engine does predict + masked update) ---
        t = float(self.model.t0) if self.model.t0 is not None else 0.0
        dt = 1.0 / self.sample_rate() if self.sample_rate() else 1.0

        for z, u in zip(zk, uk):
            self.measurement.append(z)
            self.control.append(u)

            self.x, self.P, F = self.engine._predict(self.model, self.x, self.P, u=u, t=t)
            self.prior.append((self.x, self.P))
            self.transition.append(F)

            self.x, self.P = self.engine._update(self.model, self.x, self.P, z, t=t)
            self.posterior.append((self.x, self.P))

            t += dt

        return self.get_record(indexing=indexing, inprocess=inprocess, postprocess=postprocess) if record else self

    # ------------------------------------------------------------------
    # Backward smoothing (RTS) — drives engine._smooth
    # ------------------------------------------------------------------
    def backward(self, **kwargs):
        N = len(self.posterior)
        assert N > 0, "No posterior estimates available for smoothing. (run forward() first?)"
        assert len(self.prior) == N, "Prior estimates must match the number of posterior estimates."

        self.smooth = np.zeros((N, 2), dtype=object)
        self.smooth[-1] = self.posterior[-1]

        for k in reversed(range(N - 1)):
            self.smooth[k] = self.engine._smooth(
                self.model, self.posterior[k], self.prior[k + 1], self.smooth[k + 1], self.transition[k]
            )

        kwargs.pop("smooth", None)
        return self.get_record(smooth=True, **kwargs)

    # ------------------------------------------------------------------
    # Convenience aliases (mirror the model-level API: filter()/smooth())
    # ------------------------------------------------------------------
    def filter(self, z, u=None, **kwargs):
        """Alias for ``forward`` (``smooth=True`` runs forward then backward)."""
        if u is None:
            u = kwargs.pop("uk", None)
        if kwargs.pop("smooth", False):
            self.forward(z, u, record=False, **kwargs)
            return self.backward(**kwargs)
        return self.forward(z, u, **kwargs)

    def smooth(self, **kwargs):
        """Alias for ``backward`` (RTS smoother over the stored forward pass)."""
        return self.backward(**kwargs)

    # ------------------------------------------------------------------
    # Sampling convenience (uses this filter's engine for the dynamics)
    # ------------------------------------------------------------------
    def sample(self, duration, **kwargs):
        from .sampler import KalmanSampler
        return KalmanSampler(self.model, engine=self.engine).measurement(duration, **kwargs)

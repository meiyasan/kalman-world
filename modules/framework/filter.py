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

    def __init__(self, model=None, engine=None, *, epsilon: float = 1e-5,
                 n=None, m=None, F=None, H=None, Q=None, R=None,
                 fs=1.0, x0=None, P0=None, name=None, **model_kwargs):
        from .engines import resolve_engine

        # Matrix / dimension construction path: when no model is supplied, build a
        # generic linear KalmanModel from the state dimension n, measurement dimension
        # m, and the F/H/Q/R coefficients — so a filter can be created and tuned purely
        # by its matrices, with no domain-specific Model subclass required.
        if model is None:
            model = self._linear_model(n=n, m=m, F=F, H=H, Q=Q, R=R,
                                       fs=fs, x0=x0, P0=P0,
                                       name=name or self.NAME, **model_kwargs)

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

    @staticmethod
    def _linear_model(*, n=None, m=None, F=None, H=None, Q=None, R=None,
                      fs=1.0, x0=None, P0=None, name="filter", **kw):
        """Build a generic linear :class:`KalmanModel` from dimensions + coefficients.

        ``n`` = state dimension, ``m`` = measurement dimension. Any matrix left as
        ``None`` gets an editable placeholder: ``F`` = identity, ``H`` = observe the
        first ``m`` states, ``Q`` = 1e-3·identity, ``R`` = identity. ``n``/``m`` are
        inferred from ``F``/``H`` when those are given.
        """
        from .model import KalmanModel
        F = None if F is None else np.asarray(F, dtype=float)
        H = None if H is None else np.asarray(H, dtype=float)
        if n is None and F is not None: n = F.shape[0]
        if m is None and H is not None: m = H.shape[0]
        if n is None:
            raise ValueError("Provide a state dimension n (or an F matrix) to size the filter.")
        if m is None: m = n
        if F is None: F = np.eye(n)
        if H is None:
            H = np.zeros((m, n))
            for i in range(min(m, n)): H[i, i] = 1.0
        Q = np.eye(n) * 1e-3 if Q is None else np.asarray(Q, dtype=float)
        R = np.eye(m) * 1.0   if R is None else np.asarray(R, dtype=float)
        return KalmanModel(name, F=F, H=H, Q=Q, R=R, fs=fs, x0=x0, P0=P0, **kw)

    # --- matrix-coefficient accessors: read or edit the underlying linear model ---
    # (so you can do ``kf.F = [[1, dt], [0, 1]]`` after construction).
    @property
    def F(self): return self._model.F()
    @F.setter
    def F(self, M): self._model._F = np.asarray(M, dtype=self._model.dtype)
    @property
    def H(self): return self._model.H()
    @H.setter
    def H(self, M): self._model._H = np.asarray(M, dtype=self._model.dtype)
    @property
    def Q(self): return self._model.Q()
    @Q.setter
    def Q(self, M): self._model._Q = np.asarray(M, dtype=self._model.dtype)
    @property
    def R(self): return self._model.R()
    @R.setter
    def R(self, M): self._model._R = np.asarray(M, dtype=self._model.dtype)

    # initial guess — settable after construction too (``kf.x0 = [...]``, ``kf.P0 = [...]``);
    # routes through the model's initial_state() so each is normalised/validated and the
    # other is preserved. Read back through the same model attributes.
    @property
    def x0(self): return self._model.x0
    @x0.setter
    def x0(self, v): self._model.initial_state(x0=v)
    @property
    def P0(self): return self._model.P0
    @P0.setter
    def P0(self, M): self._model.initial_state(P0=M)

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

        # An explicit reset wins: start a clean run from the model's x0/P0. Only when
        # *appending* (reset=False, streaming) do we roll the window forward once it
        # grows past `buffer` seconds — that path calls reset_to_latest(), which
        # rebases the model's x0/P0 onto the latest estimate, so it must NOT fire on a
        # plain reset=True re-run (doing so silently corrupts x0/P0 across calls).
        if reset:
            self.reset()
        elif len(self.posterior) > buffer * self.sample_rate():
            self.reset_to_latest()

        # --- detect callable controller (closed-loop) before normalisation ---
        uk_fn = uk if callable(uk) else None

        # --- normalise measurements / control ---
        gk = gk.zk if isinstance(gk, KalmanRecord) else gk
        gk = KalmanRecord._normalize_in(gk, indexing=indexing, dtype=self.model.dtype)
        assert gk.shape[1] >= 1, f"gk must have at least 1 column, got {format_shape(gk)}."

        if uk_fn is None:
            uk = uk.zk if isinstance(uk, KalmanRecord) else uk
            if uk is None or len(uk) < 1:
                uk = np.zeros((len(gk), self.l))
            uk = KalmanRecord._normalize_in(uk, indexing=indexing, dtype=self.model.dtype)
            if self.l > 0:
                assert uk is not None and uk.shape[1] == self.l, \
                    f"Control input uk has wrong number of columns. Expected ({self.l},1), got {format_shape(uk)}."

        # --- preprocess (may alter shapes) ---
        if uk_fn is not None:
            zk, _ = self.model.preprocess(gk, None) if preprocess else (gk, None)
        else:
            zk, uk = self.model.preprocess(gk, uk) if preprocess else (gk, uk)
        assert zk.shape[1] == self.m, \
            f"Preprocessed zk has incorrect number of columns. Expected ({self.m},1), got {format_shape(zk)}."
        if uk_fn is None and self.l > 0:
            assert uk is not None and uk.shape[1] == self.l, \
                f"Preprocessed uk has incorrect number of columns. Expected ({self.l},1), got {format_shape(uk)}."

        # --- main loop (engine does predict + masked update) ---
        t = float(self.model.t0) if self.model.t0 is not None else 0.0
        dt = 1.0 / self.sample_rate() if self.sample_rate() else 1.0
        nu = np.zeros((self.model.m, 1))
        u = np.zeros((max(self.l, 1), 1))

        for idx, z in enumerate(zk):
            if uk_fn is not None:
                u_raw = uk_fn(self.x, self.P, t, nu=nu)
                u = np.asarray(u_raw, dtype=float).reshape(-1, 1)
            else:
                u = uk[idx]

            self.measurement.append(z)
            self.control.append(u.ravel() if np.ndim(u) > 1 else u)

            self.x, self.P, F = self.engine._predict(self.model, self.x, self.P, u=u, t=t)
            self.prior.append((self.x, self.P))
            self.transition.append(F)

            h_pred = self.model.h(self.x, t=t)
            nu = np.asarray(z, dtype=float).reshape(-1, 1) - np.asarray(h_pred, dtype=float).reshape(-1, 1)
            self.innovation.append(nu)

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

    # ==================================================================
    # Consistency diagnostics — delegated to KalmanDiagnostics
    # (diagnostics.py), all computed *post-hoc* from the stored forward
    # pass (prior/posterior/measurement) + the model's H/R/h, so no engine
    # changes are needed. Run forward()/filter() first.
    # ==================================================================
    def _diagnostics(self):
        from .diagnostics import KalmanDiagnostics
        return KalmanDiagnostics(self)

    def innovations(self):
        """Per-step innovation ν and its covariance S — see
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.innovations`."""
        return self._diagnostics().innovations()

    def gains(self):
        """Per-step Kalman gain K(t) — see
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.gains`."""
        return self._diagnostics().gains()

    def nis(self):
        """Normalized Innovation Squared (≈ χ²(m), no truth needed) — see
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.nis`."""
        return self._diagnostics().nis()

    def nees(self, truth):
        """Normalized Estimation Error Squared (≈ χ²(n), needs truth) — see
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.nees`."""
        return self._diagnostics().nees(truth)

    def diagnostics(self, sample=None, **kwargs):
        """Draw the filter-consistency dashboard (innovation, gain, NIS, and NEES if a
        ground-truth ``sample`` is given). Delegates to
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.plot`; returns the figure."""
        return self._diagnostics().plot(sample=sample, **kwargs)

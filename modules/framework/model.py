from typing import Optional
import numpy as np
from ..helpers import set_random_seed, as_col, numerical_jacobian
from .abstract.mixins import KalmanModelMixin
from .abstract.metadata import KalmanModelMetadata
from .constants import KalmanEngineKind, ALL_ENGINES, KalmanState, KalmanObservable

class KalmanModel(KalmanModelMetadata, KalmanModelMixin):
    """
    The single, flexible Kalman **model** contract (the *system*, WHAT).

    A model defines ``f/h`` and the noise covariances ``Q/R``; the Jacobians
    ``F/H/B/L/M`` are optional refinements with a numerical default. Construct it two
    ways, **same contract**, runnable by **any** engine:

      - **matrices** — pass ``F=,H=,B=,L=,M=`` (constant or callable ``M(t)``); ``f``/``h``
        then default to ``f(x)=Fx+Bu+Lw`` / ``h(x)=Hx+Mv`` and ``F()/H()/…`` return them;
      - **functions** — override ``f``/``h`` (nonlinear); ``F/H`` come from
        :func:`numerical_jacobian`.

    Engines consume only what they need: UKF/PF use ``f/h/Q/R``; EKF additionally reads
    ``F/H`` (numerical/analytic); KF reads ``F/H`` as constants.

    Engine binding is **declared, not imported**: ``DEFAULT_ENGINE`` /
    ``ALLOWED_ENGINES`` are :class:`KalmanEngineKind` values, so a model never imports an
    engine; the :class:`~kalman.framework.filter.KalmanFilter` resolves/validates them.
    """

    NAME = "Kalman Model"
    DESCRIPTION = "Generic model for Kalman-family filters; matrices (F/H/…) or override f/h."
    VERSION = "2.0"

    # Engine binding (declared via KalmanEngineKind so the model never imports an engine).
    DEFAULT_ENGINE  = KalmanEngineKind.EXTENDED      # safe universal default (falls back to linear on matrices)
    ALLOWED_ENGINES = ALL_ENGINES              # any engine

    def __init__(self, sensor_id: str | int, **kwargs):

        self.add_sensor(sensor_id)

        # --- Sampling rate ---
        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", None)
        if self.fs is not None:
            if not isinstance(self.fs, (int, float)) or self.fs <= 0:
                raise ValueError(f"Sampling frequency must be a positive number, got {self.fs!r}")

        # --- RNG seed (optional) ---
        seed = kwargs.pop("random_seed", None)
        if seed is not None:
            set_random_seed(seed)

        # --- Time base ---
        self.t0 = kwargs.pop("t0", 0.0)

        # --- Optional explicit matrices (the "linear" construction path) ---
        def _store(M):
            if M is None:
                return None
            return M if callable(M) else np.asarray(M, dtype=self.dtype)

        self._F = _store(kwargs.pop("F", None))
        self._H = _store(kwargs.pop("H", None))
        self._B = _store(kwargs.pop("B", None))
        self._L = _store(kwargs.pop("L", None))
        self._M = _store(kwargs.pop("M", None))
        self._is_linear = self._F is not None and self._H is not None

        # Infer l from B columns so _infer_l() picks it up
        if self._B is not None:
            B_mat = self._B(0) if callable(self._B) else self._B
            if B_mat is not None:
                self.l = int(np.asarray(B_mat).shape[1])

        # --- Infer dimensions ---
        self.n = self._infer_n()
        self.m = self._infer_m()
        self.l = self._infer_l()

        # --- Infer covariances ---
        self._Q = self._infer_Q(kwargs.pop("Q", None))
        self._R = self._infer_R(kwargs.pop("R", None))

        # --- Initial state ---
        x0 = kwargs.pop("x0", None)
        if x0 is None: x0 = np.zeros((self.n, 1))

        P0 = kwargs.pop("P0", None)
        self.initial_state(x0, P0)

        self._check_shapes(strict=False)

    # ============================================================
    # Engine resolution (declarative; lazy import keeps layering clean)
    # ============================================================
    def default_engine(self):
        """Instantiate the model's default engine (resolved from ``DEFAULT_ENGINE``).

        Routes through ``resolve_engine`` so the lookup survives ``KalmanEngineKind``
        identity changes from ``%autoreload`` (otherwise a reloaded ``DEFAULT_ENGINE``
        member would miss the registry and raise ``KeyError``).
        """
        from .engines import resolve_engine
        return resolve_engine(self.DEFAULT_ENGINE)

    # ============================================================
    # Core equations (matrix path; subclasses override for nonlinear f/h)
    # ============================================================
    def f(self, x, u=None, w=None, t=None):
        """State transition. Uses constant matrices if provided; else override me."""
        if self._F is None:
            raise NotImplementedError(
                "Model must define f() or be constructed with matrices (F=, H=, ...).")
        x = self.F(t=t) @ x
        if u is not None:
            B = self.B(t=t)
            x = x + (u if B is None else B @ u)
        if w is not None:
            L = self.L(t=t)
            x = x + (w if L is None else L @ w)
        return x

    def h(self, x, v=None, t=None):
        """Observation map. Uses constant matrices if provided; else override me."""
        if self._H is None:
            raise NotImplementedError(
                "Model must define h() or be constructed with matrices (F=, H=, ...).")
        z = self.H(t=t) @ x
        if v is not None:
            M = self.M(t=t)
            z = z + (v if M is None else M @ v)
        return z

    # ============================================================
    # Jacobians (stored matrices if given, else numerical)
    # ============================================================
    def F(self, x: np.ndarray = None, u: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """State-transition Jacobian ∂f/∂x (constant matrix if provided)."""
        if self._F is not None:
            return self._F(t) if callable(self._F) else self._F
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda xx: self.f(xx, u=u, w=None, t=t), as_col(x))

    def H(self, x: np.ndarray = None, t: Optional[float] = None) -> np.ndarray:
        """Observation Jacobian ∂h/∂x (constant matrix if provided)."""
        if self._H is not None:
            return self._H(t) if callable(self._H) else self._H
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda xx: self.h(xx, v=None, t=t), as_col(x))

    def B(self, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Control-input map ∂f/∂u (constant matrix if provided, numerical if u given)."""
        if self._B is not None:
            return self._B(t) if callable(self._B) else self._B
        if u is None or self._F is not None:
            return None
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda uu: self.f(as_col(x), u=uu, w=None, t=t), as_col(u))

    def L(self, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None,
           w: Optional[np.ndarray] = None, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Process-noise map ∂f/∂w (constant matrix if provided, numerical if w given)."""
        if self._L is not None:
            return self._L(t) if callable(self._L) else self._L
        if w is None or self._F is not None:
            return None
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda ww: self.f(as_col(x), u=u, w=ww, t=t), as_col(w))

    def M(self, x: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None,
           t: Optional[float] = None) -> Optional[np.ndarray]:
        """Measurement-noise map ∂h/∂v (constant matrix if provided, numerical if v given)."""
        if self._M is not None:
            return self._M(t) if callable(self._M) else self._M
        if v is None or self._H is not None:
            return None
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda vv: self.h(as_col(x), v=vv, t=t), as_col(v))

    # Unified interface returning all Jacobians
    def jacobians(self, x: np.ndarray,
                        u: Optional[np.ndarray] = None,
                        v: Optional[np.ndarray] = None,
                        w: Optional[np.ndarray] = None,
                        t: Optional[float] = None):
        """Return (F, H, B, L, M) in a consistent order."""
        return self.F(x, u, t), self.H(x, t), self.B(x, u, t), self.L(x, u, w, t), self.M(x, v, t)

    # ============================================================
    # Covariances
    # ============================================================
    def Q(self, t: Optional[float] = None) -> np.ndarray:
        """Process noise covariance."""
        return self._Q(t) if callable(self._Q) else self._Q

    def R(self, t: Optional[float] = None) -> np.ndarray:
        """Measurement noise covariance."""
        return self._R(t) if callable(self._R) else self._R

    def covariances(self, t: Optional[float] = None,
                          x: Optional[np.ndarray] = None,
                          u: Optional[np.ndarray] = None,
                          w: Optional[np.ndarray] = None,
                          v: Optional[np.ndarray] = None):
        """Return (Q, R) with optional L Q Lᵀ / M R Mᵀ coupling."""
        Q = self.Q(t)
        L = self.L(x, u, w, t)
        if L is not None:
            Q = L @ Q @ L.T
        R = self.R(t)
        M = self.M(x, v, t)
        if M is not None:
            R = M @ R @ M.T
        return Q, R

    # ============================================================
    # Generic, override-friendly defaults (so a bare model is usable)
    # ============================================================
    @property
    def states(self) -> KalmanState:
        """Generic (unlabelled) state enum; override with a domain enum."""
        return KalmanState
    @property
    def observables(self) -> KalmanObservable:
        """Generic (unlabelled) observable enum; override with a domain enum."""
        return KalmanObservable

    def control_cost(self):
        """(Qcost, Rcost, x_ref) hints for controller training. Override per domain."""
        return np.eye(self.n), np.eye(max(self.l, 1)), np.zeros((self.n, 1))

    def preprocess(self, zk, uk=None, **kwargs):
        """Identity measurement/control preprocessing (override per domain)."""
        return zk, uk
    def inprocess(self, zk_hat, **kwargs):
        """Identity state in-processing (override per domain)."""
        return zk_hat
    def postprocess(self, zk_hat, **kwargs):
        """Identity observation post-processing (override per domain)."""
        return zk_hat

    # ============================================================
    # Signature figure for this model (the app's "main plot").
    # Apps override _plot_static / _plot_interactive for a bespoke layout;
    # the generic defaults already cover any time-series model.
    # ============================================================
    def plot(self, record, sample=None, *, interactive=False, component=None, error=True, **kwargs):
        """The model's signature figure for a filtered ``record``.

        - ``model.plot(record, sample=sample)`` → static multi-panel figure.
        - ``model.plot(record, sample=sample, interactive=True)`` → the same content
          driven by one joint time slider (the standardized replacement for the old
          ``record.replay(...)``).

        ``component`` defaults to every state in ``self.states``; ``error=True`` appends
        a measurement-error panel. Apps override ``_plot_static`` / ``_plot_interactive``
        for a bespoke domain layout (see e.g. ``PowerGridStateModel``).
        """
        comps = component if component is not None else list(self.states)
        if interactive:
            return self._plot_interactive(record, sample=sample, component=comps, error=error, **kwargs)
        return self._plot_static(record, sample=sample, component=comps, error=error, **kwargs)

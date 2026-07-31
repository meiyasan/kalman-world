from typing import Optional
import numpy as np
from scipy.linalg import expm
from ..helpers import set_random_seed, as_col, numerical_jacobian, van_loan_discretize
from .abstract.mixins import KalmanModelMixin
from .abstract.metadata import KalmanModelMetadata
from .constants import (KalmanEngineKind, ALL_ENGINES, KalmanState, KalmanObservable,
                        KalmanIntegratorKind, ALL_INTEGRATORS,
                        KalmanSolverKind, ALL_SOLVERS)

class KalmanModel(KalmanModelMetadata, KalmanModelMixin):
    """
    The single, flexible Kalman **model** contract (the *system*, WHAT).

    A model defines ``f/h`` and the noise covariances ``Q/R``; the Jacobians
    ``F/H/B/L/M`` are optional refinements with a numerical default. Construct it two
    ways, **same contract**, runnable by **any** engine:

      - **matrices** — pass ``F=,H=,B=,L=,M=`` (constant or callable ``M(t)``); ``f``/``h``
        then default to ``f(x)=Fx+Bu+Lw`` / ``h(x)=Hx+Mv`` and ``F()/H()/…`` return them;
      - **continuous** — pass ``A=`` (and, for process noise, ``Qc=`` with optional
        ``Lc=``) instead of ``F=``/``Q=``; the model discretizes exactly via
        ``expm``/Van Loan (needs ``fs=``/``sample_rate=``). ``Ac()``/``Qc()``/``Lc()``
        return the continuous inputs back for inspection (e.g.
        ``helpers.symbolic_discretize(model.Ac())``); mutually exclusive with
        ``F=``/``Q=`` respectively;
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

    # Time-stepping binding (declared, not imported — same pattern as the engine).
    # An ODE model overriding ``f_continuous`` gets its discrete ``f`` by having
    # this integrator advance the derivative one filter step (``SOLVER_SUBSTEPS``
    # substeps of ``1/fs``). A PDE model instead binds a solver and overrides f().
    DEFAULT_INTEGRATOR  = KalmanIntegratorKind.RK4
    ALLOWED_INTEGRATORS = ALL_INTEGRATORS
    DEFAULT_SOLVER      = KalmanSolverKind.SPECTRAL
    ALLOWED_SOLVERS     = ALL_SOLVERS
    SOLVER_SUBSTEPS     = 1

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

        # --- Optional per-instance integrator/solver override (Kind or instance) ---
        self._integrator_override = self._validate_kind(
            kwargs.pop("integrator", None), self.ALLOWED_INTEGRATORS, "integrator")
        self._solver_override = self._validate_kind(
            kwargs.pop("solver", None), self.ALLOWED_SOLVERS, "solver")
        self._integrator_inst = None
        self._solver_inst = None

        # --- Time base ---
        self.t0 = kwargs.pop("t0", 0.0)

        # --- Optional explicit matrices (the "linear" construction path) ---
        def _store(M):
            if M is None:
                return None
            return M if callable(M) else np.asarray(M, dtype=self.dtype)

        # --- Continuous-time construction path: A (+ optional Qc/Lc) --------
        # Alternative to passing F=/Q= directly: give the continuous generator
        # ẋ=Ax(+Lw) and let the model discretize exactly (expm for F, Van Loan
        # for Q). Kept as self._Ac/_Qc/_Lc for introspection — NOT the same
        # slot as the framework's own _L (discrete process-noise map ∂f/∂w,
        # set below), which is a distinct thing left untouched here.
        A, Qc, Lc = kwargs.pop("A", None), kwargs.pop("Qc", None), kwargs.pop("Lc", None)
        self._Ac = self._Qc = self._Lc = None
        if A is not None:
            if kwargs.get("F") is not None:
                raise ValueError("Pass either F= or A=, not both.")
            if not self.fs:
                raise ValueError("A= needs fs= (or sample_rate=) to discretize via expm(A/fs).")
            A = np.asarray(A, dtype=self.dtype)
            dt = 1.0 / self.fs
            kwargs["F"] = expm(A * dt)
            self._Ac = A
            if Qc is not None:
                if kwargs.get("Q") is not None:
                    raise ValueError("Pass either Q= or Qc=, not both.")
                Lc = np.eye(A.shape[0], dtype=self.dtype) if Lc is None else np.asarray(Lc, dtype=self.dtype)
                _, kwargs["Q"] = van_loan_discretize(A, Lc, np.asarray(Qc, dtype=self.dtype), dt)
                self._Qc, self._Lc = Qc, Lc
        elif Qc is not None:
            raise ValueError("Qc= requires A= (Van Loan needs the continuous generator).")

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

    @staticmethod
    def _validate_kind(value, allowed, label):
        """Normalise an ``integrator=``/``solver=`` override: accept a Kind, a
        concrete instance (has ``.KIND``), or None; validate against ``allowed``."""
        if value is None:
            return None
        kind = getattr(value, "KIND", value)   # instance → its KIND; else the Kind itself
        if allowed is not None and kind not in allowed:
            raise TypeError(f"{label} {kind} not in ALLOWED_{label.upper()}S "
                            f"({sorted(k.name for k in allowed)}).")
        return value

    def default_integrator(self):
        """Instantiate the model's time integrator (override → ``DEFAULT_INTEGRATOR``)."""
        if self._integrator_inst is None:
            from .integrators import resolve_integrator
            from .integrator import KalmanIntegrator
            ovr = self._integrator_override
            if isinstance(ovr, KalmanIntegrator):
                self._integrator_inst = ovr
            else:
                self._integrator_inst = resolve_integrator(ovr or self.DEFAULT_INTEGRATOR)
        return self._integrator_inst

    def default_solver(self, **kwargs):
        """Instantiate the model's spatial solver (override → ``DEFAULT_SOLVER``)."""
        if self._solver_inst is None:
            from .solvers import resolve_solver
            from .solver import KalmanSolver
            ovr = self._solver_override
            if isinstance(ovr, KalmanSolver):
                self._solver_inst = ovr
            else:
                self._solver_inst = resolve_solver(ovr or self.DEFAULT_SOLVER, **kwargs)
        return self._solver_inst

    # ------------------------------------------------------------------
    # Continuous dynamics hook (opt-in): override ``f_continuous`` (the
    # derivative ẋ=f_c(x,u,t)) and the base ``f`` below integrates it with the
    # model's integrator — the ODE counterpart of binding a solver for a PDE.
    # ------------------------------------------------------------------
    def f_continuous(self, x, u=None, t=None):
        """Continuous-time derivative ẋ = f_continuous(x,u,t).  Override to opt in."""
        raise NotImplementedError

    def _uses_continuous(self) -> bool:
        return type(self).f_continuous is not KalmanModel.f_continuous

    def _integrate_continuous(self, x, u, t):
        integ = self.default_integrator()
        substeps = max(1, int(getattr(self, "SOLVER_SUBSTEPS", 1)))
        dt = (1.0 / self.fs) if self.fs else 1.0
        dt /= substeps
        fc = lambda xx, uu, tt: np.asarray(self.f_continuous(xx, u=uu, t=tt), dtype=self.dtype)
        return integ.integrate(fc, np.asarray(x, dtype=self.dtype), u=u,
                               t=(0.0 if t is None else t), dt=dt, n=substeps)

    # ============================================================
    # Core equations (matrix path; subclasses override for nonlinear f/h)
    # ============================================================
    def f(self, x, u=None, w=None, t=None):
        """State transition. Uses constant matrices if provided; else a
        continuous ``f_continuous`` integrated by the model's integrator; else
        override me."""
        if self._F is None:
            if self._uses_continuous():                 # ODE: integrate the derivative
                xk = self._integrate_continuous(x, u, t)
                if w is not None:
                    L = self.L(t=t)
                    xk = xk + (w if L is None else L @ w)
                return xk
            raise NotImplementedError(
                "Model must define f()/f_continuous or be constructed with matrices (F=, H=, ...).")
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

    def clip_u(self, u: np.ndarray) -> np.ndarray:
        """Actuator-limit hook: map a controller's raw command to what the plant can
        actually deliver. Identity by default; models with real actuator limits (e.g. a
        steering-angle clip) override this so callers — controllers logging cost, or
        anything else that needs the *deliverable* command rather than the raw one — see
        the same constraint the model enforces internally in ``f()``."""
        return u

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

    # ============================================================
    # Continuous-time introspection (only set if constructed via A=/Qc=)
    # ============================================================
    def Ac(self) -> Optional[np.ndarray]:
        """Continuous-time generator ẋ=Ax(+Lw) the model was built from, or
        None if constructed via F= directly. Feed to
        :func:`helpers.symbolic_discretize` to read F back as an equation."""
        return self._Ac

    def Qc(self) -> Optional[np.ndarray]:
        """Continuous process-noise spectral density, or None if constructed via Q= directly."""
        return self._Qc

    def Lc(self) -> Optional[np.ndarray]:
        """Continuous noise-input map (ẋ=Ax+Lw), or None if constructed via Q= directly."""
        return self._Lc

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

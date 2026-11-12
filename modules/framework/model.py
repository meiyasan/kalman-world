from typing import Optional
import numpy as np
from ..helpers import set_random_seed, as_col, numerical_jacobian, van_loan_discretize, matrix_exponential
from .abstract.mixins import KalmanModelMixin
from .abstract.metadata import KalmanModelMetadata
from .constants import KalmanDynamicsKind
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

    # What f returns (declared, like the engine): the step itself (DISCRETE),
    # or the derivative dx/dt (CONTINUOUS), which the framework decodes into
    # the step with the model's integrator, in SOLVER_SUBSTEPS steps per sample.
    # A field model binds a solver instead and calls it from f.
    DYNAMICS            = KalmanDynamicsKind.DISCRETE
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
                raise ValueError("A= needs fs= (or sample_rate=) to decode F = exp(A / fs).")
            A = np.asarray(A, dtype=self.dtype)
            dt = 1.0 / self.fs
            kwargs["F"] = matrix_exponential(A * dt)
            self._Ac = A
            if Qc is not None:
                if kwargs.get("Q") is not None:
                    raise ValueError("Pass either Q= or Qc=, not both.")
                Lc = np.eye(A.shape[0], dtype=self.dtype) if Lc is None else np.asarray(Lc, dtype=self.dtype)
                _, kwargs["Q"] = van_loan_discretize(A, Lc, np.asarray(Qc, dtype=self.dtype), dt)
                self._Qc, self._Lc = Qc, Lc
        elif Qc is not None:
            # A continuous model with no generator matrix: its noise intensity is
            # decoded in Q(), with the Jacobian of its derivative as generator.
            if self.DYNAMICS is not KalmanDynamicsKind.CONTINUOUS:
                raise ValueError("Qc= needs a continuous model: pass A=, or declare DYNAMICS = CONTINUOUS.")
            if kwargs.get("Q") is not None:
                raise ValueError("Pass either Q= or Qc=, not both.")
            if not self.fs:
                raise ValueError("Qc= needs fs= (or sample_rate=) to decode the per-step Q.")
            self._Qc = np.asarray(Qc, dtype=self.dtype)
            self._Lc = None if Lc is None else np.asarray(Lc, dtype=self.dtype)
            kwargs["Q"] = self._decode_noise
            kwargs.setdefault("L", np.eye(self.n, dtype=self.dtype))   # the coupling is inside the decoded Q
        self._decoded_noise = None

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
        """Normalise an ``integrator=``/``solver=`` override.

        Accepts a Kind, a registered name, a concrete class, a ready instance,
        or None.  A **custom** scheme — one whose ``KIND`` is not a member of the
        framework enum — is validated by its *base class* instead of by
        ``ALLOWED_*``, since a closed enum cannot describe a kind it has never
        heard of.  Names are checked by resolving them (raising on a typo) but
        returned unresolved, so ``default_solver(**kwargs)`` can still pass
        constructor arguments.
        """
        if value is None:
            return None

        from .integrator import KalmanIntegrator
        from .solver import KalmanSolver
        if label == "integrator":
            base, enum_cls = KalmanIntegrator, KalmanIntegratorKind
            from .integrators import resolve_integrator as _resolve
        else:
            base, enum_cls = KalmanSolver, KalmanSolverKind
            from .solvers import resolve_solver as _resolve

        if isinstance(value, str):
            _resolve(value)                    # raises ValueError on an unknown name
            return value                       # resolved later, with kwargs

        is_impl = isinstance(value, base) or (isinstance(value, type) and issubclass(value, base))
        kind = getattr(value, "KIND", value)   # instance/class → its KIND; else the Kind itself
        if is_impl and not isinstance(kind, enum_cls):
            return value                       # custom scheme: the base class is the contract
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
    # Continuous dynamics. A class with DYNAMICS = CONTINUOUS writes f as the
    # derivative dx/dt. When the class is created, that f is kept as
    # `derivative` and f becomes the decoded step below, so engines and
    # filters only ever see a discrete step, and the Jacobian F = df/dx is the
    # Jacobian of the integrated step.
    # ------------------------------------------------------------------
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        written = cls.__dict__.get("f")
        if (written is not None and cls.DYNAMICS is KalmanDynamicsKind.CONTINUOUS
                and written is not _decoded_step):
            cls.derivative = written
            cls.f = _decoded_step

    def derivative(self, x, u=None, w=None, t=None):
        """dx/dt of a CONTINUOUS model: the ``f`` its class wrote."""
        raise NotImplementedError(f"{type(self).__name__} is not a continuous model.")

    def substeps(self, x=None) -> int:
        """Integrator steps per sample when decoding a continuous model."""
        return max(1, int(getattr(self, "SOLVER_SUBSTEPS", 1)))

    def after_step(self, x, u=None, t=None):
        """Hook: adjust the state a continuous model lands on after one sample
        (a clamp, a slot that is not integrated). Returns the state."""
        return x

    def decode_step(self, x, u=None, t=None):
        """One sample of a CONTINUOUS model: the derivative integrated over
        ``1/fs`` by the model's integrator, then :meth:`after_step`. No noise."""
        steps = self.substeps(x)
        dt = ((1.0 / self.fs) if self.fs else 1.0) / steps
        rate = lambda xx, uu, tt: np.asarray(self.derivative(xx, u=uu, t=tt), dtype=self.dtype)
        xk = self.default_integrator().integrate(rate, np.asarray(x, dtype=self.dtype), u=u,
                                                 t=(0.0 if t is None else t), dt=dt, n=steps)
        return self.after_step(xk, u=u, t=t)

    # ============================================================
    # Core equations (matrix path; subclasses override for nonlinear f/h)
    # ============================================================
    def f(self, x, u=None, w=None, t=None):
        """State transition, the discrete step. Uses constant matrices if
        provided; otherwise a subclass writes it (the step, or with
        ``DYNAMICS = CONTINUOUS`` the derivative, which is decoded)."""
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

    def _decode_noise(self, t=None):
        """Per-step Q of a continuous model that stated a noise intensity ``Qc``:
        Van Loan over one sample, with the Jacobian of the derivative at the
        initial state as generator. Decoded once per (sample rate, initial state)."""
        x0 = np.zeros((self.n, 1)) if getattr(self, "x0", None) is None else np.asarray(self.x0, dtype=float).reshape(-1, 1)
        key = (float(self.fs), x0.tobytes())
        if self._decoded_noise is None or self._decoded_noise[0] != key:
            u = np.zeros((self.l, 1)) if getattr(self, "l", 0) else None
            A = numerical_jacobian(lambda xx: self.derivative(xx, u=u, t=t), x0)
            Lc = np.eye(self.n) if self._Lc is None else self._Lc
            _, Q = van_loan_discretize(np.asarray(A, dtype=float), Lc, self._Qc, 1.0 / self.fs)
            self._decoded_noise = (key, Q.astype(self.dtype))
        return self._decoded_noise[1]

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
        """State enum — a domain enum when declared, else one synthesized from
        ``n`` (``X0 … Xn-1``).

        The bare ``KalmanState`` has no members, which left every unlabelled
        model (anything built from matrices, including a ``KalmanSpec``) with an
        empty component list and therefore unplottable.  Synthesizing labels
        keeps the generic path usable; a domain model overriding this property
        is unaffected.
        """
        if getattr(self, "_states_enum", None) is None:
            self._states_enum = self._synthetic_labels(KalmanState, "State", "X", self.n)
        return self._states_enum

    @property
    def observables(self) -> KalmanObservable:
        """Observable enum — a domain enum when declared, else one synthesized
        from ``m`` (``Z0 … Zm-1``).  See :attr:`states`."""
        if getattr(self, "_observables_enum", None) is None:
            self._observables_enum = self._synthetic_labels(KalmanObservable, "Observable", "Z", self.m)
        return self._observables_enum

    def _synthetic_labels(self, base, suffix, prefix, count):
        """Build an unlabelled-but-indexable enum of ``count`` members."""
        return base(f"{type(self).__name__}{suffix}",
                    {f"{prefix}{i}": i for i in range(int(count or 0))})

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


def _decoded_step(self, x, u=None, w=None, t=None):
    """f of a CONTINUOUS model: the decoded step, with the process noise added."""
    xk = self.decode_step(x, u, t)
    if w is not None:
        L = self.L(t=t)
        xk = xk + (w if L is None else L @ w)
    return xk

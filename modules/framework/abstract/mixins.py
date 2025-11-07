from __future__ import annotations
from inspect import isclass

import numpy as np
from typing import Any, Optional

from ...helpers import as_col, format_shape
from ..io.record import KalmanStateRecord
from .metadata import KalmanModelMetadata, KalmanEngineMetadata
from ..constants import KalmanStateFlavor

from .contracts import KalmanModelABC, KalmanEngineABC
class KalmanProxyMixin:
    """
    Drop-in mixin for KalmanModel:
      - self.foo(...) uses self.foo if present; otherwise tries engine.foo(...)
      - self.call("foo", ...) does the same explicitly
      - self(...): forwards to engine if engine is callable
    """
    _engine: Any = None  # attach an engine object with attributes/methods to delegate to
    def attach(self: KalmanModelMetadata, engine: Any = None, **kwargs):

        if engine is None:
            raise ValueError("Engine must be provided (class or instance of KalmanEngineABC).")

        # If a class is passed, ensure it subclasses KalmanEngineABC and instantiate it
        if isclass(engine):
            if not issubclass(engine, KalmanEngineABC):
                raise TypeError(f"Engine class must subclass KalmanEngineABC, got {engine}")
            engine = engine(self, **kwargs)
        else:
            # If an instance is passed, ensure it is an instance of KalmanEngineABC
            if not isinstance(engine, KalmanEngineABC):
                raise TypeError(f"Engine must be a KalmanEngineABC instance or class, got {type(engine)}")

        self._engine = engine
        self._engine.bind(self)
        self._engine.reset()
        return self

    def _check_engine(self) -> KalmanEngineMixin:

        from .contracts import KalmanEngineABC
        if not isinstance(self._engine, KalmanEngineABC):
            raise RuntimeError("No engine class attached to model; cannot proceed.")
        return self._engine

    def filter(self, z, u=None, **kwargs):

        if kwargs.pop("smooth", False):
            return self._check_engine().forward(z, u, **kwargs).backward(**kwargs)
    
        kwargs["record"] = True
        return self._check_engine().forward(z, u, **kwargs)
    
    def smooth(self, **kwargs):
        return self._check_engine().backward(**kwargs)
    def sample(self, duration: float, **kwargs) -> KalmanStateRecord:
        from ..sampler import KalmanSampler
        return KalmanSampler(self._check_engine()).measurement(duration, **kwargs)

    def truth(self, duration: float = None):
        record = self.sample(duration, process_noise=True, measurement_noise=False)
        return record.xk, record.Pk
    def ideal(self, duration: float = None):
        record = self.sample(duration, process_noise=False, measurement_noise=False)
        return record.xk

class KalmanShapeMixin:
    """
    Mixin providing comprehensive dimension and consistency validation
    for Kalman models and engines once (n, m, l) are known.

    Expected dimensions:
        x  ∈ ℝⁿ×¹
        P, Q ∈ ℝⁿ×ⁿ
        z  ∈ ℝᵐ×¹
        R ∈ ℝᵐ×ᵐ
        F ∈ ℝⁿ×ⁿ
        H ∈ ℝᵐ×ⁿ
        L ∈ ℝⁿ×q  (usually q=n for additive noise)
        M ∈ ℝᵐ×r  (usually r=m for additive noise)
        u ∈ ℝˡ×¹  (if applicable)
    """

    def _check_matrix(self, M, expected_shape, name: str, strict=False, symmetric=False):
        """Validate matrix shape and optionally symmetry/PSD."""
        if M is None:
            raise ValueError(f"{name} is None (expected shape {expected_shape})")
        M = np.asarray(M)
        if M.shape != expected_shape:
            raise ValueError(f"{name}: got {format_shape(M)}, expected {expected_shape}")
        if symmetric:
            if not np.allclose(M, M.T, atol=1e-10):
                raise ValueError(f"{name} not symmetric.")
            eigvals = np.linalg.eigvalsh(M)
            if np.min(eigvals) < -1e-12:
                raise ValueError(f"{name} not PSD (min eigenvalue {np.min(eigvals):.2e})")
        return M

    def _check_vector(self, v, expected_length, name: str):
        """Validate vector shape (column vector form)."""
        if v is None:
            raise ValueError(f"{name} is None (expected length {expected_length})")
        v = as_col(v, name)
        if v.shape[0] != expected_length:
            raise ValueError(f"{name}: got length {v.shape[0]}, expected {expected_length}")
        return v

    # ------------------------------------------------------
    # Top-level consistency checker
    # ------------------------------------------------------
    def _check_shapes(self : KalmanModelMetadata, *, strict=False):
        """
        Validate internal model dimensions (x, P, Q, R, F, H, L, M)
        according to (n, m, l).
        """
        n, m, l = self.n, self.m, self.l

        # --- State quantities ---
        if hasattr(self, "x0"): self._check_vector(self.x0, n, "x0")
        if hasattr(self, "x"):  self._check_vector(self.x,  n, "x")
        if hasattr(self, "P"):  self._check_matrix(self.P,  (n, n), "P",  symmetric=strict)
        if hasattr(self, "Q"):  self._check_matrix(self.Q(), (n, n), "Q",  symmetric=strict)

        # --- Measurement quantities ---
        if m > 0:
            if hasattr(self, "R"):  self._check_matrix(self.R(), (m, m), "R", symmetric=strict)
            if hasattr(self, "H"):  self._check_matrix(self.H(np.zeros((n,1))), (m, n), "H")
            if hasattr(self, "M") and self.M is not None:
                M = self.M(np.zeros((n,1)))
                if M is not None:
                    if M.ndim == 2 and M.shape[0] != m:
                        raise ValueError(f"M: invalid rows {M.shape[0]}, expected {m}")

        # --- Process quantities ---
        if hasattr(self, "F"):  self._check_matrix(self.F(np.zeros((n,1))), (n, n), "F")
        if hasattr(self, "L") and self.L is not None:
            L = self.L(np.zeros((n,1)))
            if L is not None:
                if L.ndim == 2 and L.shape[0] != n:
                    raise ValueError(f"L: invalid rows {L.shape[0]}, expected {n}")

        # --- Optional control input ---
        if isinstance(l, int) and l > 0:
            self._check_vector(self.u, l, "u")

        return True

class KalmanInferenceMixin:

    # -------------------------------
    # Infer state dimensions
    # -------------------------------
    def _infer_n(self: KalmanModelMetadata):

        if self.x0 is not None:
            return self.x0.shape[0]
        if isinstance(self.n, int) and self.n > 0:
            return self.n

        F = self.F()
        if F is not None:
            return F.shape[0]

        raise ValueError(
            "Cannot infer state dimension 'n'. Provide x0, preset self.n in the subclass, or implement F"
        )

    def _infer_m(self: KalmanModelMetadata):

        if isinstance(self.m, int) and self.m >= 0:
            return self.m
        
        R = self.R()
        if R is not None:
            return R.shape[0]
    
        H = self.H()
        if H is not None:
            return H.shape[0]

        raise ValueError(
            "Cannot infer state dimension 'm'. Provide covariance R, preset self.m in the subclass, or implement H"
        )

    def _infer_l(self):

        if isinstance(getattr(self, "l", None), int) and self.l >= 0:
            return self.l

        return 0

    # ------------------------------------------------------
    # Covariance inference utilities (pure functional form)
    # ------------------------------------------------------

    def _infer_P0(self, P0=None):
        """Infer or validate initial covariance P0."""
        n = getattr(self, "n", None)
        if n is None:
            raise ValueError("State dimension 'n' must be defined before inferring P0.")

        # --- Direct argument or fallback ---
        if P0 is not None:
            P0 = np.asarray(P0)
        else:
            Q = self._infer_Q()
            Q0 = Q(0) if callable(Q) else np.asarray(Q)
            P0 = Q0.copy()

        # --- Validation ---
        if P0.shape != (n, n):
            raise ValueError(f"P0 must be ({n},{n}); got {format_shape(P0)}")

        return P0

    def _infer_Q(self, Q=None):
        """Infer or validate process noise covariance Q (matrix or callable)."""
        n = getattr(self, "n", None)
        if n is None:
            raise ValueError("State dimension 'n' must be defined before inferring Q.")

        # --- Priority: argument, attribute, or default ---
        _Q = Q
        if _Q is None:
            # Prefer self.Q if available
            _Q = self.Q() if callable(getattr(self, "Q", None)) else getattr(self, "Q", None)

        # --- Default to zero covariance if nothing found ---
        if _Q is None:
            return np.zeros((n, n))

        # --- Callable covariance (time-varying) ---
        if callable(_Q):
            Q0 = _Q(0)
            if np.shape(Q0) != (n, n):
                raise ValueError(f"Callable Q(t) must return shape ({n},{n}); got {format_shape(Q0)}")
            return _Q

        # --- Static covariance matrix ---
        Qarr = np.asarray(_Q)
        if Qarr.shape != (n, n):
            raise ValueError(f"Q must be ({n},{n}); got {format_shape(Qarr)}")
        return Qarr

    def _infer_R(self, R=None):
        """Infer or validate measurement noise covariance R (matrix or callable)."""
        m = getattr(self, "m", None)
        if m is None:
            raise ValueError("Measurement dimension 'm' must be defined before inferring R.")

        # --- Priority: argument, attribute, or default ---
        _R = R
        if _R is None:
            _R = self.R() if callable(getattr(self, "R", None)) else getattr(self, "R", None)

        # --- Default to zero covariance if nothing found ---
        if _R is None:
            return np.zeros((m, m))

        # --- Callable covariance (time-varying) ---
        if callable(_R):
            R0 = _R(0)
            if np.shape(R0) != (m, m):
                raise ValueError(f"Callable R(t) must return shape ({m},{m}); got {format_shape(R0)}")
            return _R

        # --- Static covariance matrix ---
        Rarr = np.asarray(_R)
        if Rarr.shape != (m, m):
            raise ValueError(f"R must be ({m},{m}); got {format_shape(Rarr)}")
        return Rarr

class KalmanModelMixin(KalmanModelABC, KalmanProxyMixin, KalmanShapeMixin, KalmanInferenceMixin):

    def time(self: KalmanModelMetadata, k: int | None = None) -> np.ndarray:
        return self.t0 + np.arange(0, k, 1./self.fs, dtype=float)
    def duration(self : KalmanModelMetadata, k: int) -> float:
        return (k / self.fs) if self.t0 is not None and self.fs is not None else 0.0
    def sample_rate(self : KalmanModelMetadata) -> float:
        return self.fs if self.fs is not None else 0.0

    # Initial state API
    def get_initial_state_vector(self):
        return self.x0
    def get_initial_state_covariance(self):
        return self.P0
    def get_initial_state(self):
        return self.initial_state_vector(), self.initial_state_covariance()

    def initial_state_vector(self, x0=None):
        x0, _ = self.initial_state(x0=x0)
        return x0
    def initial_state_covariance(self, P0=None):
        _, P0 = self.initial_state(P0=P0)
        return P0
    def initial_state(self, x0=None, P0=None):

        import warnings
        n = int(self.n)

        if x0 is None:
            self.x0 = np.zeros((n, 1))
        else:
            x0 = KalmanStateRecord._normalize_in(x0)
            if x0.shape != (n, 1):
                warnings.warn(
                    f"Initial state vector dimension expected {n}x1, got {format_shape(x0)}. Resizing."
                )
                x0 = np.asarray(x0, dtype=np.float64).reshape(-1, 1)
                if x0.shape[0] != n:
                    x0 = np.resize(x0, (n, 1))
            self.x0 = x0    

        P0 = P0 if P0 is not None else self.Q(0)
        P0 = KalmanStateRecord._normalize_in(P0)
        if P0.shape != (n, n):
            warnings.warn(
                f"Initial state covariance dimension expected {n}x{n}, got {format_shape(P0)}. Resizing."
            )
            P0 = np.asarray(P0, dtype=np.float64)
            if P0.ndim == 1:
                P0 = np.diag(P0)
            P0 = np.resize(P0, (n, n))
    
        self.P0 = P0

class KalmanStateMixin:

    # Latest / prior / posterior / smooth getters
    def get_latest_state(self: KalmanEngineMetadata):
        return self.x, self.P
    def get_latest_state_vector(self):
        return self.get_latest_state()[0]
    def get_latest_state_covariance(self):
        return self.get_latest_state()[1]

    def get_prior_state(self: KalmanEngineMetadata, k=None):
        return self.prior if k is None else self.prior[k]
    def get_prior_state_vector(self, k=None):
        return self._get_tuple(self.get_prior_state(k), ith=0)
    def get_prior_state_covariance(self, k=None):
        return self._get_tuple(self.get_prior_state(k), ith=1)

    def get_posterior_state(self: KalmanEngineMetadata, k=None):
        return self.posterior if k is None else self.posterior[k]
    def get_posterior_state_vector(self, k=None):
        return self._get_tuple(self.get_posterior_state(k), ith=0)
    def get_posterior_state_covariance(self, k=None):
        return self._get_tuple(self.get_posterior_state(k), ith=1)

    def get_smooth_state(self: KalmanEngineMetadata, i=None):
        return self.smooth if i is None else self.smooth[i]
    def get_smooth_state_vector(self, i=None):
        return self._get_tuple(self.get_smooth_state(i), ith=0)
    def get_smooth_state_covariance(self, i=None):
        return self._get_tuple(self.get_smooth_state(i), ith=1)

    # Measurement getters
    def get_measurement(self : KalmanEngineMetadata, k=None, indexing=None):
        return KalmanStateRecord._normalize_out(self.measurement, indexing=indexing) if k is None else self.measurement[k]

    def get_observation(self : KalmanEngineMetadata, xk, k=None, indexing=None):
        if k is None:
            return KalmanStateRecord._normalize_out([self._model.h(x) for x in xk], indexing=indexing)
        return self._model.h(xk[k])

    # Initial state API
    def get_initial_state_vector(self : KalmanEngineMetadata):
        return self._model.x0
    def get_initial_state_covariance(self : KalmanEngineMetadata):
        return self._model.P0
    def get_initial_state(self : KalmanEngineMetadata):
        return self.get_initial_state_vector(), self.get_initial_state_covariance()
    def initial_state(self : KalmanEngineMetadata, x0=None, P0=None):
        return self._model.initial_state(x0=x0, P0=P0)
    
    # Utility to extract vector/matrix from state tuple or list of tuples
    def _get_tuple(self: KalmanEngineMetadata, state, ith):
        if ith not in [0, 1]:
            raise ValueError("Component must be 0 (state vector) or 1 (covariance matrix).")
        if isinstance(state, tuple):
            return state[ith]
        return np.array([s[ith] for s in state]) if state is not None else None

class KalmanEngineMixin(KalmanEngineABC, KalmanStateMixin):

    # -------------------------------
    # Model binding and access
    # -------------------------------
    def bind(self, model: KalmanModelMixin):
        self._model = model

    # -------------------------------
    # Basic dimension properties
    # -------------------------------
    @property
    def n(self):
        return self.model._infer_n()
    @property
    def m(self):
        return self.model._infer_m()
    @property
    def l(self):
        return self.model._infer_l()

    @property
    def model(self) -> 'KalmanModelMixin | KalmanModelMetadata':
        return self._model
    @property
    def fs(self) -> float:
        return self.model.fs
    
    # -------------------------------
    # Basic engine API
    # -------------------------------
    def time(self, k: Optional[int] = None) -> float:
        return self.model.time(k if k is not None else len(self.posterior)/self.fs)
    def duration(self, k: int) -> float:
        return self.model.duration(k if k is not None else len(self.posterior)/self.fs)
    def sample_rate(self) -> float:
        return self.model.fs

    def reset_to_latest(self):
        self.reset(to_latest=True)

    def reset(self, to_latest=False):
        """
        Reset engine state. Enforce at runtime that self provides both the
        KalmanEngineMetadata and KalmanStateTupleMixin APIs.
        """
        if to_latest:
            self.model.P0 = self.get_latest_state_covariance()
            self.model.x0 = self.get_latest_state_vector()

        # Microstate
        self.P = self.model.P0
        self.x = self.model.x0

        # Macrostate
        self.prior = []
        self.posterior = []
        self.smooth = []

        self.measurement = []
        self.control = []

    def __call__(self, gk, uk=None, **kwargs):
        return self.forward(gk, uk, **kwargs)

    def __len__(self):
        return len(self.posterior)
    def __getitem__(self, idx):
        return self.posterior[idx]

    def get_record(self, *, flavor=KalmanStateFlavor.POSTERIOR, **kwargs) -> KalmanStateRecord:

        indexing    = kwargs.pop("indexing", None)
        inprocess   = kwargs.pop("inprocess", True)
        postprocess = kwargs.pop("postprocess", True)
        
        if self.prior is None or self.posterior is None:
            raise ValueError("No prior or posterior estimates available. Please run filter() first.")

        if flavor is KalmanStateFlavor.PRIOR:
            xk_hat, Pk_hat = self.get_posterior_state_vector(), self.get_posterior_state_covariance()
        elif flavor is KalmanStateFlavor.POSTERIOR:
            xk_hat, Pk_hat = self.get_posterior_state_vector(), self.get_posterior_state_covariance()
        elif flavor is KalmanStateFlavor.SMOOTHED:
            xk_hat, Pk_hat = self.get_smooth_state_vector(), self.get_smooth_state_covariance()        
        else:
            raise ValueError(f"Unexpected flavor {flavor} for KalmanStateRecord.")
            
        if inprocess:
            xk_hat = self.model.inprocess(xk_hat, **kwargs)

        zk_hat = self.get_observation(xk_hat, indexing=indexing)
        if postprocess:
            zk_hat = self.model.postprocess(zk_hat, **kwargs)

        return KalmanStateRecord(self.time(), zk=zk_hat, xk=xk_hat, Pk=Pk_hat, flavor=flavor, engine=self)

    def get_posterior_record(self) -> KalmanStateRecord:
        if not self.posterior: return KalmanStateRecord()
        return KalmanStateRecord(self, self.time(), self.get_posterior_state_vector(), self.get_posterior_state_covariance(),
                                 zk=self.get_measurement(), flavor=KalmanStateFlavor.POSTERIOR, model=self)

    def get_prior_record(self) -> KalmanStateRecord:
        if not self.prior: return KalmanStateRecord()
        return KalmanStateRecord(self, self.time(), self.get_prior_state_vector(), self.get_prior_state_covariance(),
                                 zk=self.get_measurement(), flavor=KalmanStateFlavor.PRIOR, model=self)

    def get_smooth_record(self) -> KalmanStateRecord:
        if not self.smooth: return KalmanStateRecord()
        return KalmanStateRecord(self, self.time(), self.get_smooth_state_vector(), self.get_smooth_state_covariance(),
                                 zk=self.get_measurement(), flavor=KalmanStateFlavor.SMOOTHED, model=self)

    def exclusive_mapping(self, state_idx: int|tuple, meas_idx: int|tuple) -> bool:

        """Check if measurement(s) exclusively observe state(s)."""
        # Convert to arrays
        if isinstance(state_idx, (tuple, list, np.ndarray)) or isinstance(meas_idx, (tuple, list, np.ndarray)):
            
            # Determine broadcasting length
            n = max(len(state_idx) if hasattr(state_idx, "__len__") else 1,
                    len(meas_idx) if hasattr(meas_idx, "__len__") else 1)
            
            state_idx = KalmanStateRecord._normalize_index(state_idx, n)
            meas_idx  = KalmanStateRecord._normalize_index(meas_idx, n)
            
            for i in range(n):
                if not self.exclusive_mapping(state_idx[i], meas_idx[i]):
                    return False

            return True if n else False

        else:

            s = state_idx.value if hasattr(state_idx, 'value') else int(state_idx)
            assert 0 <= s < self.n
            m = meas_idx.value  if hasattr(meas_idx,  'value') else int(meas_idx)
            assert 0 <= m < self.m

            return self.model.H()[m, s] != 0

    def inclusive_mapping(self, state_idx: int|tuple, meas_idx: int|tuple) -> bool:
        """Check if measurement(s) depend on state(s) in any way."""
        if isinstance(state_idx, (tuple, list, np.ndarray)) or isinstance(meas_idx, (tuple, list, np.ndarray)):
            # Determine broadcasting length
            n = max(len(state_idx) if hasattr(state_idx, "__len__") else 1,
                    len(meas_idx) if hasattr(meas_idx, "__len__") else 1)
            
            state_idx = KalmanStateRecord._normalize_index(state_idx, n)
            meas_idx  = KalmanStateRecord._normalize_index(meas_idx, n)

            for i in range(n):
                if self.inclusive_mapping(state_idx[i], meas_idx[i]) is False:
                    return False

            return True if n else False
            
        else:
            s = state_idx.value if hasattr(state_idx, 'value') else int(state_idx)
            m = meas_idx.value  if hasattr(meas_idx,  'value') else int(meas_idx)
            assert 0 <= s < self.n
            assert 0 <= m < self.m
            return self.H[m, s] != 0

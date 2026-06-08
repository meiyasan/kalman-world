# kalman/core/filter.py
from __future__ import annotations
import numpy as np

from .io.record import KalmanRecord
from ..models.composite import KalmanCompositeModel
from ..helpers import format_shape
from .abstract.metadata import KalmanEngineMetadata
from .abstract.mixins import KalmanEngineMixin

import warnings
warnings.simplefilter("once", RuntimeWarning)

class KalmanFilter(KalmanEngineMetadata, KalmanEngineMixin):
    
    """
    EKF wrapper (does NOT inherit from a model).
    - Accepts one or more models; multiple -> KalmanCompositeModel(*models).
    - Uses ONLY model.f, model.h, model.jacobians(x,u)->(F,H,L,M).
    - Q/R may be matrices or callables Q(x,u,t), R(x,t).
    - Non-additive noise supported via LQLᵀ and MRMᵀ if L/M provided by jacobians.
    - Hooks policy:
        * preprocess(zk, uk, **kw)   used ONLY in forward()
        * inprocess(x, **kw)         used ONLY in forward()/backward() (after predict & after update)
        * postprocess(x, **kw)       used ONLY for outputs in forward()/backward()
    """

    # -------------------- init --------------------
    def __init__(self, *models, epsilon: float = 1e-5):

        if not models:
            raise ValueError("KalmanFilter requires at least one model or model=...")
        
        model = models[0] if len(models) == 1 else KalmanCompositeModel(*models)

        self._model = model
        self.epsilon = float(epsilon)

        self.x = self.model.x0
        if self.x is None:
            self.x = np.zeros((self.model.n, 1))
        self.P = self.model.P0
        if self.P is None:
            self.P = np.eye(self.model.n)

        if self.x.shape != (self.model.n, 1):
            raise ValueError(f"x shape mismatch: expected ({self.model.n},1), got {format_shape(self.x)}")
        if self.P.shape != (self.model.n, self.model.n):
            raise ValueError(f"P shape mismatch: expected ({self.model.n},{self.model.n}), got {format_shape(self.P)}")

        # Logs / scratch
        self.prior       = [] # list of (x-, P-)
        self.posterior   = [] # list of (x+, P+)
        self.measurement = [] # list of z
        self.control     = [] # list of u (or None)
        self.transition  = [] # list of F (for RTS)
        self.F           = np.eye(self.model.n)  # latest transition matrix

    def reset(self, to_latest=False):
        """Reset engine state, including the transition-matrix log."""
        super().reset(to_latest=to_latest)
        self.transition = []
        self.F = np.eye(self.model.n)

    # -------------------- low-level EKF (NO hooks here) --------------------
    def _predict(self, x, P, **kwargs):
        """
        Basic prediction step.
        Returns predicted (x, P, F) for optional smoother use.
        """
        u = kwargs.get("u", np.zeros((self.l, 1)))
        t = kwargs.get("t", None)

        # Ensure u is a column vector (loop yields row slices of shape (l,))
        if u is not None:
            u = np.asarray(u).reshape(-1, 1)

        # Jacobian for covariance propagation (for linear models this is the
        # state-transition matrix; for EKF models it is ∂f/∂x at x).
        F = self.model.F(x=x, u=u, t=t)
        Q = self.model.Q(t=t)

        # Apply LQL^T if the model provides a process-noise coupling Jacobian L.
        # KalmanLinearModel / KalmanCompositeModel expose L(t); base KalmanModel
        # exposes L(x, u, w, t) as a numerical Jacobian ∂f/∂w evaluated at w=0.
        w0 = np.zeros((self.n, 1))
        try:
            L = self.model.L(x=x, u=u, w=w0, t=t)
        except TypeError:
            L = self.model.L(t=t)
        if L is not None:
            Q = L @ Q @ L.T

        # State propagation via the model's nonlinear (or linear) dynamics.
        # For linear models f(x,u,t) = F@x + B@u, so behaviour is unchanged.
        # For EKF models f() uses the actual nonlinear dynamics — critical for
        # correct EKF operation (using F@x here would be a first-order
        # linearisation around x=0, not the true state prediction).
        # Pass u=None when l=0 to avoid feeding a zero-sized (0,1) array into f().
        u_for_f = u if self.l > 0 else None
        x = self.model.f(x, u=u_for_f, w=None, t=t)

        P = F @ P @ F.T + Q
        return x, P, F

    def _update(self, z, t=None):
        """
        Standard Kalman update.
        """
        # Ensure z is a column vector (loop yields row slices of shape (m,))
        z = np.asarray(z).reshape(-1, 1)

        H = self.model.H(x=self.x, t=t)

        # h_pred MUST be computed before R() — time-varying models (e.g. calorimeter shower
        # models) advance an internal step counter inside R(), so calling R() first
        # would cause h() to use the wrong layer's matrix.
        # For linear models h(x) = H @ x; for EKF models h() uses actual dynamics.
        h_pred = self.model.h(self.x, v=None, t=t).reshape(-1, 1)

        R = self.model.R(t=t)
        I = np.eye(self.n)

        y = z - h_pred
        S = R + H @ self.P @ H.T

        try: 
            K = self.P @ H.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            raise RuntimeError("Kalman loop failed. Please check your model for instability and noise covariance issues.")

        self.x = self.x + K @ y
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T  # Joseph form (numerically stable)
        return self.x, self.P

    # -------------------- public API --------------------
    def forward(self, gk, uk=None, *, record: bool = False, **kwargs):

        indexing   = kwargs.pop("indexing", "xy")
        preprocess = kwargs.pop("preprocess", True)
        inprocess  = kwargs.pop("inprocess", True)
        postprocess= kwargs.pop("postprocess", True)

        reset  = kwargs.pop("reset", True)
        buffer = kwargs.pop("buffer", 60)  # in seconds

        if len(self.posterior) > buffer * self.sample_rate():
            self.reset_to_latest()
        elif reset:
            self.reset()

        # ------------------------------------------------------
        # Normalize input and control
        # ------------------------------------------------------
        gk = gk.zk if isinstance(gk, KalmanRecord) else gk
        gk = KalmanRecord._normalize_in(gk, indexing=indexing, dtype=self.model.dtype)
        # Allow preprocess to expand the column count (e.g. real signal → analytic signal).
        # The authoritative shape check happens after preprocess below.
        assert gk.shape[1] >= 1, f"gk must have at least 1 column, got {format_shape(gk)}."

        uk = uk.zk if isinstance(uk, KalmanRecord) else uk
        if uk is None or len(uk) < 1:
            uk = np.zeros((len(gk), self.l))
        uk = KalmanRecord._normalize_in(uk, indexing=indexing, dtype=self.model.dtype)
        if self.l > 0:
            if uk is not None:
                assert uk.shape[1] == self.l, f"Control input uk has wrong number of columns. Expected shape ({self.l},1), got {format_shape(uk)}."
            else:
                raise ValueError(f"Control input missing but model expects {self.l} control channels.")
        
        # ------------------------------------------------------
        # Preprocess (may alter shapes)
        # ------------------------------------------------------
        zk, uk = self.model.preprocess(gk, uk) if preprocess else (gk, uk)
        assert zk.shape[1] == self.m, f"Preprocessed zk has incorrect number of columns. Expected shape ({self.m},1), got {format_shape(zk)}."

        if self.l > 0:
            if uk is not None:
                assert uk.shape[1] == self.l, f"Preprocessed uk has incorrect number of columns. Expected shape ({self.l},1), got {format_shape(uk)}."
            else:
                raise ValueError(f"Preprocessed control input missing; expected {self.l} control channels.")

        # ------------------------------------------------------
        # Main Kalman Loop
        # ------------------------------------------------------
        t = float(self.model.t0) if self.model.t0 is not None else 0.0
        dt = 1.0 / self.sample_rate() if self.sample_rate() else 1.0

        for z, u in zip(zk, uk):
            self.measurement.append(z)
            self.control.append(u)

            self.x, self.P, self.F = self._predict(self.x, self.P, u=u, t=t)
            self.prior.append((self.x, self.P))
            self.transition.append(self.F)

            # Multi-rate support: skip update when measurement contains NaN
            if np.any(np.isnan(z)):
                self.posterior.append((self.x, self.P))
            else:
                self.x, self.P = self._update(z, t=t)
                self.posterior.append((self.x, self.P))

            t += dt

        return self.get_record(indexing=indexing, inprocess=inprocess, postprocess=postprocess) if record else self

    # Kalman backward pass (RTS smoother)
    def backward(self, **kwargs):

        """
        Rauch-Tung-Striebel (RTS) smoother.
        Requires that filter() has been run and stored prior and posterior estimates.
        """

        N = len(self.posterior)
        assert N > 0, "No posterior estimates available for smoothing. (have you run filter() yet?)"
        assert len(self.prior) == N, "Prior estimates must match the number of posterior estimates."
        
        # Initialize last smoothed state as last posterior
        self.smooth = np.zeros((N, 2), dtype=object)
        self.smooth[-1] = self.posterior[-1]

        for k in reversed(range(N - 1)):
            x, P = self.posterior[k]
            xp, Pp = self.prior[k+1]
            xs, Ps = self.smooth[k+1]
            
            # Compute smoothing gain (stable)
            if np.linalg.cond(Pp) < 1e12:
                C = P @ self.transition[k].T @ np.linalg.inv(Pp)
            else:
                C = P @ self.transition[k].T @ np.linalg.pinv(Pp)

            # Update smoothed estimates
            _xs = x + C @ (xs - xp)
            _Ps = P + C @ (Ps - Pp) @ C.T
            
            self.smooth[k] = (_xs, _Ps)
            
        kwargs.pop("smooth", True) # ensure no conflict with kwargs
        return self.get_record(smooth=True, **kwargs)

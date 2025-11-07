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

    # -------------------- low-level EKF (NO hooks here) --------------------
    def _predict(self, x, P, **kwargs):
        """
        Basic prediction step.
        Returns predicted (x, P, F) for optional smoother use.
        """
        u = kwargs.get("u", np.zeros((self.l, 1)))
        w = kwargs.get("w", np.zeros((self.n, 1)))
        t = kwargs.get("t", None)

        F = self.model.F(x, u, t)
        B = getattr(self.model, "B", None)
        Q = self.model.Q(t)

        x = F @ x
        if B is not None and self.l > 0:
            x += B @ u
        if w is not None:
            x += w

        P = F @ P @ F.T + Q
        return x, P, F

    def _update(self, z):
        """
        Standard Kalman update.
        """
        H = self.model.H(self.x)
        R = self.model.R()
        I = np.eye(self.n)

        y = z - H @ self.x
        S = R + H @ self.P @ H.T

        try: 
            K = self.P @ H.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            raise RuntimeError("Kalman loop failed. Please check your model for instability and noise covariance issues.")

        self.x = self.x + K @ y
        self.P = (I - K @ H) @ self.P
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
        assert gk.shape[1] == self.m, f"gk data has incorrect number of columns. Expected {self.m}, got {format_shape(gk)}."

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

        for z, u in zip(zk, uk):
            self.measurement.append(z)
            self.control.append(u)

            self.x, self.P, self.F = self._predict(self.x, self.P, u=u)
            self.prior.append((self.x, self.P))
            self.transition.append(self.F)

            self.x, self.P = self._update(z)
            self.posterior.append((self.x, self.P))

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

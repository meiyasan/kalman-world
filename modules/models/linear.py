import numpy as np
from ..framework.model import KalmanModel
from ..runtime.debug import get_debug


class KalmanLinearModel(KalmanModel):

    NAME = "Linear Kalman Model"
    DESCRIPTION = "Linear state-space model supporting fixed or time-varying matrices."
    VERSION = "2.1"

    """
    Linear state-space model with explicit constant or time-varying matrices.

    Equations:
        xₖ₊₁ = F xₖ + B uₖ + L wₖ
        zₖ   = H xₖ + M vₖ

    Attributes:
        F : (n,n) state transition matrix or callable F(t)
        H : (m,n) measurement matrix or callable H(t)
        B : (n,l) control matrix (optional)
        L : (n,q) process-noise mapping (optional)
        M : (m,r) measurement-noise mapping (optional)
        Q : (n,n) process noise covariance or callable Q(t)
        R : (m,m) measurement noise covariance or callable R(t)
    """

    def __init__(self, sensor_id: str, F, H, B=None, Q=None, R=None,
                 L=None, M=None, x0=None, P0=None, fs=None, **kwargs):

        # --- Store base matrices internally ---
        self._F = F if not callable(F) else np.asarray(F, dtype=np.float64) if F is not None else None
        self._H = H if not callable(H) else np.asarray(H, dtype=np.float64) if H is not None else None
        self._B = B if not callable(B) else np.asarray(B, dtype=np.float64) if B is not None else None
        self._L = L if not callable(L) else np.asarray(L, dtype=np.float64) if L is not None else None
        self._M = M if not callable(M) else np.asarray(M, dtype=np.float64) if M is not None else None

        # --- Initialize via parent ---
        super().__init__(sensor_id, x0=x0, P0=P0, Q=Q, R=R, fs=fs, **kwargs)

    # ------------------------------------------------------
    # Model Equations
    # ------------------------------------------------------
    def f(self, x, u=None, w=None, t=None):
        x = self.F(t) @ x

        if u is not None:
            B = self.B(t)
            x += u if B is None else (B @ u)

        if w is not None:
            L = self.L(t)
            x += w if L is None else (L @ w)

        return x

    def h(self, x, v=None, t=None):
        z = self.H(t) @ x
        if v is not None:
            M = self.M(t)
            z += v if M is None else (M @ v)

        return z

    # ------------------------------------------------------
    # Jacobians (constant or time-varying)
    # ------------------------------------------------------
    def F(self, t=None, x=None, u=None):
        if self._F is None:
            raise ValueError("State transition matrix F not defined.")
        return self._F(t) if callable(self._F) else self._F
    
    def H(self, t=None, x=None):
        if self._H is None:
            raise ValueError("Measurement matrix H not defined.")
        return self._H(t) if callable(self._H) else self._H

    def B(self, t=None, x=None, u=None):
        if self._B is None:
            return None
        return self._B(t) if callable(self._B) else self._B
        
    def L(self, t=None, x=None, u=None, w=None):
        if self._L is None:
            return None
        return self._L(t) if callable(self._L) else self._L

    def M(self, t=None, x=None, v=None):
        if self._M is None:
            return None
        return self._M(t) if callable(self._M) else self._M

    # ------------------------------------------------------
    # Covariances (constant or callable)
    # ------------------------------------------------------
    def Q(self, t=None):
        """Process noise covariance (Q(t) if callable)."""
        return self._Q(t) if callable(self._Q) else self._Q

    def R(self, t=None):
        """Measurement noise covariance (R(t) if callable)."""
        return self._R(t) if callable(self._R) else self._R
    
    # ------------------------------------------------------
    # String representation
    # ------------------------------------------------------
    def __str__(self):
        base = super().__str__()
        spacer = " " * 4
        t0 = getattr(self, "t0", 0.0)

        # --- Helper: safely infer matrix shape ---
        def _shape(M, args=()):
            if M is None:
                return "N/A"
            if callable(M):
                try:
                    val = M(*args)
                    return f"callable -> {tuple(np.asarray(val).shape)}"
                except Exception:
                    return "callable (eval failed)"
                
            shape = "x".join(map(str, np.asarray(M).shape))
            dims = "dim" if (M.ndim == 1 and M.shape[0] == 1) else "dims"
            return f"{shape} {dims}"


        # --- Compact summary with evaluated shapes ---
        def _desc(name, desc, M, args=()):
            return f"{2*spacer}{name:<3} ({desc:<22}) : {_shape(M, args)}"

        # --- Pretty array block for debug mode ---
        def _block(name, M, args=()):
            pad = 3 * spacer
            if M is None:
                return f"{2*spacer}{name}: N/A"
            try:
                val = M(*args) if callable(M) else M
                arr = np.asarray(val)
            except Exception:
                return f"{2*spacer}{name}: <callable (eval failed)>"
            txt = np.array2string(arr, precision=4, suppress_small=True)
            return f"{2*spacer}{name}:\n{pad}{txt.replace(chr(10), chr(10)+pad)}"

        # --- Header summary ---
        lines = [
            f"{spacer}Linear Model Structure (n={self.n}, m={self.m}, l={self.l}):",
            f"{spacer}{spacer}Jacobian Matrices:",
            _desc(f"{spacer}F (or A)", "state transition", self._F, (t0, self.x0)),
            _desc(f"{spacer}H (or C)", "measurement map", self._H, (t0, self.x0)),
            _desc(f"{spacer}B       ", "control map", self._B, (t0, self.x0, self.u0) if self.u0 is not None else (t0, self.x0)),
            _desc(f"{spacer}L       ", "proc-noise influence", self._L, (t0, self.x0, self.u0, self.w0)),
            _desc(f"{spacer}M       ", "meas-noise influence", self._M, (t0, self.x0, self.v0)),
            f"{spacer}{spacer}Covariance Matrices:",
            _desc(f"{spacer}Q (or W)", "process covariance", self._Q, (t0,)),
            _desc(f"{spacer}R (or V)", "measurement covariance", self._R, (t0,)),
        ]

        # --- Verbose block mode ---
        if get_debug():
            lines.append("")
            lines.append(f"{spacer}Expanded Matrices:")
            for name, M, args in [
                ("F", self._F, (t0, self.x0)),
                ("H", self._H, (t0, self.x0)),
                ("B", self._B, (t0, self.x0, self.u0)),
                ("L", self._L, (t0, self.x0, self.u0, self.w0)),
                ("M", self._M, (t0, self.x0, self.v0)),
                ("Q", self._Q, (t0,)),
                ("R", self._R, (t0,)),
            ]:
                lines.append(_block(name, M, args))

        return base + "\n".join(lines) + "\n"
import numpy as np

from ..framework.constants import KalmanObservation, KalmanState, KalmanStateUnits
from ..models import KalmanExtendedModel, KalmanLinearModel

# ===== State enums/units =====
class BallisticState(KalmanState):
    POSITION     = "m"
    SPEED        = "m/s"
    ACCELERATION = "m/s²"

class BallisticObservation(KalmanObservation):
    POSITION = "m"

# ===== Linear =====
class BallisticLinearModel(KalmanLinearModel):

    NAME = "Ballistic State Model"
    DESCRIPTION = "Linear constant-acceleration toy model; position measured."
    VERSION = "1.3"

    @property
    def states(self):
        return BallisticState
    @property
    def observables(self):
        return BallisticObservation

    def __init__(self, sensor_id: str | int,
                 measurement_uncertainty: float = 0.5,
                 process_uncertainty: float = 0.05,
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)

        # -------------------------------------------
        # State transition and measurement matrices
        # -------------------------------------------
        fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", None)
        assert isinstance(fs, (int, float)), "You must provide fs or sample_rate"
        dt = self.dtype(1.0 / fs)

        F = np.array([[1.0, dt, 0.0],
                      [0.0, 1.0, dt ],
                      [0.0, 0.0, 1.0]], dtype=self.dtype)

        H = np.array([[1.0, 0.0, 0.0]], dtype=self.dtype)

        # -------------------------------------------
        # Toy model for Q
        # -------------------------------------------
        q = self.dtype(process_uncertainty)
        Q = q * np.array([[1.0, 1.0, 0.0],
                          [1.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]], dtype=self.dtype)

        r = self.dtype(measurement_uncertainty)
        R = r * np.eye(1, dtype=self.dtype)

        super().__init__(sensor_id, F=F, H=H, Q=Q, R=R, fs=fs, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs): return zk, uk
    def inprocess(self, state_vector, **kwargs): return state_vector
    def postprocess(self, xk_hat, **kwargs):     return xk_hat

# ===== Nonlinear ballistic EKF (use proper Q here) =====
import numpy as np
from scipy.linalg import expm

class BallisticModel(KalmanExtendedModel):
    """
    Nonlinear 3D ballistic model with explicit gravity (G*M),
    aerodynamic drag, thrust input, and internally computed Q/R matrices.
    """

    NAME = "Ballistic Extended Model"
    DESCRIPTION = "Nonlinear 3D ballistic model with gravity, drag, and thrust."
    VERSION = "1.0"

    @property
    def states(self):
        return BallisticState
    @property
    def observables(self):
        return BallisticObservation

    def __init__(self, name, *,
                 G=6.674e-11,                 # gravitational constant [m^3/kg/s^2]
                 M=5.972e24,                  # central body mass [kg] (Earth)
                 m=1.0,                       # object mass [kg]
                 c_drag=0,                    # drag coefficient [kg/m]
                 fs=128.0,                    # sampling frequency [Hz]
                 order=3,                     # model order [p,v,a]
                 process_uncertainty=0.05,    # process noise std [m/s^3]
                 measurement_uncertainty=0.5, # measurement noise std [m],
                 f = None,                    # Transition function (optional, use default one if None)
                 h = None,                    # Measurement function (optional, use default one if None)
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)
        self.fs = fs
        self.order = order

        self["c_drag"] = c_drag
        self["G"] = G
        self["M"] = M
        self["m"] = m

        if f is not None:
            self.f = f
        if h is not None:
            self.h = h

        # ------------------------------------------------------------------
        # Noise covariances
        # ------------------------------------------------------------------
        Qd = self.Qd(process_uncertainty, order)
        R = (measurement_uncertainty**2) * np.eye(3, dtype=self.dtype)

        # Derived sizes
        self.n = 3 * order
        self.m = 3

        super().__init__(name, Q=Qd, R=R, **kwargs)
        self.clear_sensors()
        self.add_sensor("X")
        self.add_sensor("Y")
        self.add_sensor("Z")

    # ----------------------------------------------------------------------
    # Process noise covariance generator (internal)
    # ----------------------------------------------------------------------
    def Q(self, t=None, x=None):
        return self._Q
    
    def Qd(self, sigma: float, order: int) -> np.ndarray:
        """
        Build discrete process covariance Q for 3D ballistic motion.
        Supports [p,v] or [p,v,a] states (order=2 or 3).

        Parameters
        ----------
        sigma : float
            Process noise std (e.g., jerk std for order=3).
        order : int
            Model order (2 or 3).

        Returns
        -------
        Q : ndarray, shape (3*order, 3*order)
            Block-diagonal process covariance matrix.
        """
        q = self.dtype(sigma)**2
        dt = self.dtype(1.0 / self.fs)

        if order == 3:
            Q1D = np.array([
                [dt**5/20.0, dt**4/8.0, dt**3/6.0],
                [dt**4/8.0,  dt**3/3.0, dt**2/2.0],
                [dt**3/6.0,  dt**2/2.0, dt]
            ], dtype=self.dtype)
        elif order == 2:
            Q1D = np.array([
                [dt**3/3.0, dt**2/2.0],
                [dt**2/2.0, dt]
            ], dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported order={order}. Must be 2 or 3.")

        Q = np.zeros((3*order, 3*order), dtype=self.dtype)
        for i in range(3):  # fixed XYZ
            s = slice(order*i, order*(i+1))
            Q[s, s] = q * Q1D

        return Q

    # ----------------------------------------------------------------------
    # Nonlinear dynamics
    # ----------------------------------------------------------------------
    def f(self, x, u=None, w=None, t=None):
        """
        Full nonlinear motion model:
            ṗ = v
            v̇ = a_grav(p) + a_drag(v) + a_ctrl(u)
        """
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)
        dt = 1.0 / self.fs

        # Extract state components
        p = x[0:3, 0]
        v = x[3:6, 0]
        a = x[6:9, 0] if self.order >= 3 else np.zeros(3, dtype=self.dtype)

        # Gravity (depends on position, no control)
        r = np.linalg.norm(p)
        a_grav = -self["G"] * self["M"] * p / r**3 if r > 0 else np.zeros(3, dtype=self.dtype)

        # Drag (depends on velocity, without control)
        vmag = np.linalg.norm(v)
        a_drag = - (self["c_drag"] / self["m"]) * vmag * v if vmag > 0 else np.zeros(3, dtype=self.dtype)

        # Control (optional controlled thrust force)
        a_ctrl = np.zeros(3, dtype=self.dtype)
        if u is not None and u != []:
            thrust = np.asarray(u, dtype=self.dtype).reshape(3)
            a_ctrl = thrust / self["m"]

        # Total acceleration
        a_total = a_grav + a_drag + a_ctrl

        # Kinematic matrix exponential for p,v coupling
        if self.order >= 2:
            F = np.block([
                [np.zeros((3, 3)), np.eye(3)],
                [np.zeros((3, 3)), np.zeros((3, 3))]
            ])
            Phi = expm(F * dt)
            state = np.vstack((p.reshape(3, 1), v.reshape(3, 1)))
            state_next = Phi @ state
            p, v = state_next[0:3, 0], state_next[3:6, 0]

        # Apply acceleration effects
        p += 0.5 * a_total * dt**2
        v += a_total * dt
        a = a_total if self.order >= 3 else np.zeros_like(a_total)

        # Compose next state
        x = np.zeros_like(x)
        x[0:3, 0] = p
        x[3:6, 0] = v
        if self.order >= 3:
            x[6:9, 0] = a

        # Add process noise
        if w is not None:
            L = self.L(t)
            x += w if L is None else (L @ w)

        return x

    # ----------------------------------------------------------------------
    # Measurement
    # ----------------------------------------------------------------------
    def h(self, x, v=None, t=None):
        """
        Measurement model for 3D ballistic motion.
        Observes position only (nonlinear base case).
        z = p + v  (additive noise)
        """
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)
        z = x[0:3, :].copy()

        if v is not None:
            v = np.asarray(v, dtype=self.dtype).reshape(3, 1)
            z += v

        return z
    
    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk
    def inprocess(self, state_vector, **kwargs):
        return state_vector
    def postprocess(self, xk_hat, **kwargs):    
        return xk_hat
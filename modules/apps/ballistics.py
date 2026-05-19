import numpy as np

from ..framework.constants import KalmanObservable, KalmanState
from ..models import KalmanExtendedModel, KalmanLinearModel

# ===== State enums/units =====
class BallisticState(KalmanState):
    POSITION     = "m"
    SPEED        = "m/s"
    ACCELERATION = "m/s²"

class BallisticObservable(KalmanObservable):
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
        return BallisticObservable

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

        # Initial covariance: positive-definite diagonal based on measurement uncertainty.
        # Q is rank-1 (acceleration row zero), so using Q as P0 causes singular covariance
        # which leads to pathological RTS smoother gains. Use explicit per-state variances.
        if "P0" not in kwargs:
            p_var = self.dtype(measurement_uncertainty) ** 2
            v_var = (self.dtype(measurement_uncertainty) / 10.0) ** 2
            a_var = (self.dtype(measurement_uncertainty) / 100.0) ** 2
            kwargs["P0"] = np.diag([p_var, v_var, a_var]).astype(self.dtype)

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
        return BallisticObservable

    def __init__(self, name, *,
                 G=6.674e-11,                 # gravitational constant [m^3/kg/s^2]
                 M=5.972e24,                  # central body mass [kg] (Earth)
                 m=1.0,                       # object mass [kg]
                 c_drag=0,                    # drag coefficient [kg/m]
                 g=9.81,                      # surface gravity [m/s²]; None → use G*M/r³ orbital formula
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
        self["g"] = g

        if f is not None:
            self.f = f
        if h is not None:
            self.h = h

        # ------------------------------------------------------------------
        # Noise covariances
        # ------------------------------------------------------------------
        Qd = self.Qd(process_uncertainty, order)
        R = (measurement_uncertainty**2) * np.eye(3, dtype=self.dtype)

        # ------------------------------------------------------------------
        # Initial covariance P0 — unit-appropriate diagonal per axis-major
        # layout [px,vx,ax, py,vy,ay, pz,vz,az]
        # ------------------------------------------------------------------
        if kwargs.get("P0") is None:
            p_var = measurement_uncertainty ** 2
            v_var = (measurement_uncertainty / 10.0) ** 2
            a_var = (measurement_uncertainty / 100.0) ** 2
            if order == 3:
                block = [p_var, v_var, a_var]
            else:
                block = [p_var, v_var]
            kwargs["P0"] = np.diag(block * 3).astype(self.dtype)

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

        # Extract from axis-major layout: [px,vx,ax, py,vy,ay, pz,vz,az]
        p = x[[0, 3, 6], 0]  # [px, py, pz]
        v = x[[1, 4, 7], 0]  # [vx, vy, vz]
        a = x[[2, 5, 8], 0] if self.order >= 3 else np.zeros(3, dtype=self.dtype)

        # Gravity
        # Near-surface mode (g provided): constant downward acceleration in Z.
        # Orbital mode (g=None): full geocentric G*M/r³ formula — requires
        # positions in geocentric metres (origin = Earth's centre).
        if self["g"] is not None:
            a_grav = np.array([0.0, 0.0, -self["g"]], dtype=self.dtype)
        else:
            r = np.linalg.norm(p)
            a_grav = -self["G"] * self["M"] * p / r**3 if r > 0 else np.zeros(3, dtype=self.dtype)

        # Drag (depends on velocity, without control)
        vmag = np.linalg.norm(v)
        a_drag = -(self["c_drag"] / self["m"]) * vmag * v if vmag > 0 else np.zeros(3, dtype=self.dtype)

        # Control (optional controlled thrust force)
        a_ctrl = np.zeros(3, dtype=self.dtype)
        if u is not None:
            u_arr = np.asarray(u, dtype=self.dtype).ravel()
            if u_arr.size > 0:
                a_ctrl = u_arr.reshape(3) / self["m"]

        # Total acceleration
        a_total = a_grav + a_drag + a_ctrl

        # Kinematic matrix exponential for p,v coupling (per spatial axis)
        if self.order >= 2:
            F1d = np.array([[0.0, 1.0], [0.0, 0.0]])
            Phi1d = expm(F1d * dt)
            p_next = np.zeros(3, dtype=self.dtype)
            v_next = np.zeros(3, dtype=self.dtype)
            for i in range(3):
                pv = Phi1d @ np.array([p[i], v[i]])
                p_next[i] = pv[0]
                v_next[i] = pv[1]
            p, v = p_next, v_next

        # Apply acceleration contribution
        p += 0.5 * a_total * dt**2
        v += a_total * dt
        a = a_total if self.order >= 3 else np.zeros_like(a_total)

        # Compose next state in axis-major layout: [px,vx,ax, py,vy,ay, pz,vz,az]
        x_next = np.zeros_like(x)
        x_next[[0, 3, 6], 0] = p
        x_next[[1, 4, 7], 0] = v
        if self.order >= 3:
            x_next[[2, 5, 8], 0] = a

        # Add process noise
        if w is not None:
            L = self.L(t)
            x_next += w if L is None else (L @ w)

        return x_next

    # ----------------------------------------------------------------------
    # Measurement
    # ----------------------------------------------------------------------
    def h(self, x, v=None, t=None):
        """
        Measurement model: observes position only.
        Extracts [px, py, pz] from axis-major state [px,vx,ax, py,vy,ay, pz,vz,az].
        z = [px, py, pz] + v  (additive noise)
        """
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)
        # positions sit at indices 0, 3, 6 in the axis-major layout
        z = x[[0, 3, 6], :].copy()

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
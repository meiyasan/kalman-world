"""Lateral lane-keeping — the bicycle-kinematics (Euler) variant."""
from __future__ import annotations
"""
Vehicle dynamics models for the Kalman framework.

* :class:`CarSpeedModel` — a **straight-line car speed test** (the "0–100 km/h /
  standing quarter-mile" run). A single car accelerates in a straight line under a
  realistic longitudinal force balance — **traction-limited then power-limited** drive
  force, aerodynamic drag and rolling resistance — while a noisy GPS/radar measures
  position only. The EKF reconstructs the clean **speed** and **acceleration** the test
  is after, far better than naively differencing the noisy position fixes.

* :class:`CarLaneModel` — **lateral lane-keeping** with steering control.
  State ``x = [y, ψ, v]`` (lateral offset, heading, speed); control ``u = [δ]``
  (steering angle).  Bicycle kinematics; noisy lateral GPS.
"""
import numpy as np

from ...framework.models.extended import ExtendedKalmanModel
from ...framework.constants import KalmanState, KalmanObservable, KalmanIntegratorKind



# ===========================================================================
#  Lateral lane-keeping (bicycle kinematics + steering control)
# ===========================================================================
class CarLaneState(KalmanState):
    LATERAL  = "m"
    HEADING  = "rad"
    SPEED    = "m/s"


class CarLaneObservable(KalmanObservable):
    LATERAL = "m"


class CarLaneModel(ExtendedKalmanModel):
    """Lateral lane-keeping with steering control (bicycle kinematics).

    State ``x = [y, ψ, v]``:
      - ``y``  — lateral offset from lane centre [m]
      - ``ψ``  — heading angle relative to lane [rad]
      - ``v``  — longitudinal speed [m/s] (driven externally or constant)

    Control ``u = [δ]`` — steering angle [rad].

    Dynamics (discrete, Euler)::

        y_{k+1}  = y_k  + v_k · sin(ψ_k) · dt
        ψ_{k+1}  = ψ_k  + v_k · tan(δ_k) / L · dt
        v_{k+1}  = v_k   (constant speed)

    Measurement: noisy lateral GPS  ``z = y + noise``.

    ``n = 3, m = 1, l = 1``.
    """

    NAME = "Car Lane Keeping"
    DESCRIPTION = "Bicycle-kinematic lateral lane-keeping with steering control."
    VERSION = "1.0"

    @property
    def states(self):      return CarLaneState
    @property
    def observables(self): return CarLaneObservable

    def __init__(self, name="LANE", *,
                 wheelbase=2.7,
                 v0=15.0,
                 fs=50.0,
                 gps_noise=0.3,
                 q_lat=1e-2, q_head=1e-4, q_vel=1e-3,
                 steer_limit=0.5,        # max steering angle [rad] (~29°) — actuator saturation
                 **kw):
        self.dtype = np.float64
        self.wheelbase = float(wheelbase)
        self.steer_limit = float(steer_limit)
        self._dt = 1.0 / fs
        self.n, self.m, self.l = 3, 1, 1

        Q = np.diag([q_lat, q_head, q_vel]).astype(float)
        R = np.array([[gps_noise ** 2]], dtype=float)

        B = np.array([[0.0], [v0 / wheelbase * self._dt], [0.0]])

        kw.setdefault("x0", np.array([[0.0], [0.0], [v0]]))
        # f() doesn't route an additive-noise `w` through (accepted for API compatibility
        # only), so the engine's generic "numerically differentiate f w.r.t. w" fallback
        # for L would see zero sensitivity and silently multiply Q by a zero matrix every
        # predict step -- Q would have no effect on P at all. L=I says explicitly: Q
        # enters additively on every state, which is what q_lat/q_head/q_vel are meant to do.
        super().__init__(name, Q=Q, R=R, B=B, L=np.eye(self.n), fs=fs, **kw)

    def clip_u(self, u):
        """Steering-angle actuator limit: ``|delta| <= steer_limit``."""
        u_arr = np.asarray(u, float).reshape(-1).copy()
        if len(u_arr) > 0:
            u_arr[0] = np.clip(u_arr[0], -self.steer_limit, self.steer_limit)
        return u_arr.reshape(-1, 1)

    # Bicycle kinematics were historically integrated with explicit Euler; keep that
    # exact scheme by binding the Euler integrator (RK4 would shift the trajectory).
    DEFAULT_INTEGRATOR = KalmanIntegratorKind.EULER

    def f_continuous(self, x, u=None, t=None):
        """Bicycle-kinematic derivative ẋ = [v·sin ψ, v·tan δ / L, 0]."""
        x = np.asarray(x, float).reshape(-1)
        y, psi, v = x[0], x[1], x[2]
        delta = 0.0
        if u is not None:
            u_arr = self.clip_u(u).reshape(-1)
            if len(u_arr) > 0:
                delta = u_arr[0]
        return np.array([[v * np.sin(psi)], [v * np.tan(delta) / self.wheelbase], [0.0]])

    def h(self, x, v=None, t=None):
        x = np.asarray(x, float).reshape(-1)
        z = np.array([[x[0]]])
        return z if v is None else z + np.asarray(v, float).reshape(1, 1)

    def control_cost(self):
        Qcost = np.diag([100.0, 10.0, 0.0])
        Rcost = np.array([[1.0]])
        x_ref = np.zeros((self.n, 1))
        return Qcost, Rcost, x_ref

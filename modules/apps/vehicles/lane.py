"""Lateral lane-keeping with steering control (bicycle kinematics, Euler steps).

:class:`CarLaneModel`: state ``x = [y, ψ, v]`` (lateral offset, heading, speed), control
``u = [δ]`` (steering angle), and a lane camera that measures the lateral offset.
"""
from __future__ import annotations
import numpy as np

from ...framework.models.extended import ExtendedKalmanModel
from ...framework.models.native import NativeCounterpart, native_integrator_kind
from ...framework.constants import KalmanState, KalmanObservable, KalmanIntegratorKind, KalmanDynamicsKind



# ===========================================================================
#  Lateral lane-keeping (bicycle kinematics + steering control)
# ===========================================================================
class CarLaneState(KalmanState):
    LATERAL  = "m"
    HEADING  = "rad"
    SPEED    = "m/s"


class CarLaneObservable(KalmanObservable):
    LATERAL = "m"


class CarLaneModel(NativeCounterpart, ExtendedKalmanModel):
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

    Measurement: the lateral offset seen by a lane camera, ``z = y + noise``.

    ``n = 3, m = 1, l = 1``.

    Parameters
    ----------
    wheelbase : float
        Distance between the axles ``L`` [m].
    v0 : float
        Speed [m/s].
    lateral_uncertainty : float
        Standard deviation of the measured lateral offset [m] (a lane camera: a few cm).
    q_lat, q_head, q_vel : float
        Process-noise variances per step, for ``y`` [m²], ``ψ`` [rad²] and ``v`` [(m/s)²].
        The defaults are 1 cm, 1 mrad and 1 cm/s per 20 ms step (standard deviations).
    steer_limit : float
        Largest steering angle [rad]. The lateral acceleration is ``v² tan δ / L``; at
        15 m/s the default 0.1 rad gives 8.4 m/s², about what tyres hold on a dry road.
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
                 lateral_uncertainty=0.03,
                 q_lat=1e-4, q_head=1e-6, q_vel=1e-4,
                 steer_limit=0.1,
                 **kw):
        self.dtype = np.float64
        self.wheelbase = float(wheelbase)
        self.steer_limit = float(steer_limit)
        self.speed = float(v0)
        self._dt = 1.0 / fs
        self.n, self.m, self.l = 3, 1, 1

        Q = np.diag([q_lat, q_head, q_vel]).astype(float)
        R = np.array([[lateral_uncertainty ** 2]], dtype=float)

        B = np.array([[0.0], [v0 / wheelbase * self._dt], [0.0]])

        kw.setdefault("x0", np.array([[0.0], [0.0], [v0]]))
        # L = I: the process noise is added to every state.
        super().__init__(name, Q=Q, R=R, B=B, L=np.eye(self.n), fs=fs, **kw)

    # ----------------------------------------------------------------------
    # C++ counterpart (libKalmanApps: Kalman::Apps::Vehicles::CarLane). The steering
    # limit is part of the dynamics, so a subclass that changes limit_control is
    # another model. A controller function (closed loop) keeps this Python
    # model in the C++ loop; so does an integrator the C++ side does not have.
    # ----------------------------------------------------------------------
    _COUNTERPART_CONTRACT = NativeCounterpart._COUNTERPART_CONTRACT + ("limit_control",)

    def native_name(self):
        return "car-lane" if native_integrator_kind(self) is not None else None

    def native_parameters(self):
        return {"wheelbase": self.wheelbase, "steer_limit": self.steer_limit,
                "substeps": float(self.substeps()), "integrator": float(native_integrator_kind(self))}

    @property
    def steer_limit(self):
        """Largest steering angle [rad]: the command is clipped to ``|delta| <= steer_limit``
        (:attr:`control_bounds`, which the C++ side applies too)."""
        return self._steer_limit

    @steer_limit.setter
    def steer_limit(self, value):
        self._steer_limit = float(value)
        self.control_bounds = (np.array([[-self._steer_limit]]), np.array([[self._steer_limit]]))

    # Explicit Euler steps (RK4 would shift the trajectory).
    DEFAULT_INTEGRATOR = KalmanIntegratorKind.EULER
    DYNAMICS = KalmanDynamicsKind.CONTINUOUS

    def f(self, x, u=None, w=None, t=None):
        """Bicycle-kinematic derivative ẋ = [v·sin ψ, v·tan δ / L, 0]."""
        x = np.asarray(x, float).reshape(-1)
        y, psi, v = x[0], x[1], x[2]
        delta = 0.0
        if u is not None:
            u_arr = self.limit_control(u).reshape(-1)
            if len(u_arr) > 0:
                delta = u_arr[0]
        return np.array([[v * np.sin(psi)], [v * np.tan(delta) / self.wheelbase], [0.0]])

    def h(self, x, v=None, t=None):
        x = np.asarray(x, float).reshape(-1)
        z = np.array([[x[0]]])
        return z if v is None else z + np.asarray(v, float).reshape(1, 1)

    def control_cost(self):
        """Quadratic cost weights ``(Qcost, Rcost, x_ref)``: 1 per m² of lateral offset
        and 100 per rad² of steering, so 1 m of offset costs as much as 0.1 rad. The
        reference is the lane centre, heading along the lane, at the cruise speed."""
        Qcost = np.diag([1.0, 0.0, 0.0])
        Rcost = np.array([[100.0]])
        x_ref = np.array([[0.0], [0.0], [self.speed]])
        return Qcost, Rcost, x_ref

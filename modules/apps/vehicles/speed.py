"""Straight-line car speed test — the longitudinal (RK4) variant."""
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
#  Straight-line car speed test
# ===========================================================================
class CarSpeedState(KalmanState):
    DISTANCE = "m"
    SPEED    = "m/s"


class CarSpeedObservable(KalmanObservable):
    DISTANCE = "m"


class CarSpeedModel(ExtendedKalmanModel):
    """
    Straight-line acceleration ("speed") test of a single car.

    Dynamics (longitudinal force balance, divided by mass ``m``)::

        ṡ = v
        v̇ = [ F_drive(v) − F_drag(v) − F_roll − F_grade ] / m

    with

    * ``F_drive(v) = min( μ·m·g ,  P_max / v )`` — the wheels can put down at most the
      grip-limited force ``μ·m·g`` off the line; once the engine's **power** limit
      ``P_max / v`` falls below that (above ~30–40 km/h) the run becomes power-limited.
      That hand-off is exactly the kink you see in a real acceleration trace.
    * ``F_drag = ½ ρ (C_d A) v |v|`` — aerodynamic drag (caps the top speed),
    * ``F_roll = C_r m g`` — rolling resistance,
    * ``F_grade = m g sin θ`` — road grade (0 on a flat straight).

    A GPS/radar measures **position only** (``z = s + noise``); the EKF recovers the
    smooth speed and acceleration curves. ``n = 2``, ``m = 1``, ``l = 0``.
    """

    NAME = "Car Speed Test (straight line)"
    DESCRIPTION = "Longitudinal car dynamics (traction/power-limited drive + drag); GPS position measured."
    VERSION = "1.0"

    @property
    def states(self):      return CarSpeedState
    @property
    def observables(self): return CarSpeedObservable

    def __init__(self, name="CAR", *,
                 mass=1500.0,            # vehicle mass [kg]
                 power=150_000.0,        # peak engine power at the wheels [W] (~200 hp)
                 mu=1.0,                 # tyre–road grip coefficient (launch traction limit)
                 drag_area=0.66,         # C_d·A [m²]  (≈ 0.30 × 2.2 m²)
                 rho=1.225,              # air density [kg/m³]
                 c_roll=0.015,           # rolling-resistance coefficient
                 grade=0.0,              # road grade [rad] (0 = flat)
                 g=9.81,                 # gravity [m/s²]
                 fs=20.0,                # GPS/radar sample rate [Hz]
                 gps_uncertainty=0.5,    # position measurement std [m]
                 q_pos=1e-3, q_vel=5e-2, # process-noise variances (model slack)
                 v_eps=0.5,              # speed floor [m/s] so P/v stays finite at launch
                 **kw):
        self.dtype = np.float64
        self.mass = float(mass)
        self.power = float(power)
        self.mu = float(mu)
        self.drag_area = float(drag_area)
        self.rho = float(rho)
        self.c_roll = float(c_roll)
        self.grade = float(grade)
        self.g = float(g)
        self.v_eps = float(v_eps)

        self._dt = 1.0 / fs
        self.n, self.m, self.l = 2, 1, 0
        self.fs = fs

        Q = np.diag([q_pos, q_vel]).astype(float)
        R = np.array([[gps_uncertainty ** 2]], dtype=float)
        # f() doesn't route an additive-noise `w` through (it's accepted for API
        # compatibility only), so the engine's generic "numerically differentiate f
        # w.r.t. w" fallback for L would silently see zero sensitivity and multiply Q
        # by a zero matrix every predict step. L=I says explicitly: Q enters
        # additively on every state, which is what q_pos/q_vel are actually meant to do.
        super().__init__(name, Q=Q, R=R, L=np.eye(self.n), fs=fs, **kw)

    # ----------------------------------------------------------------------
    # Longitudinal dynamics
    # ----------------------------------------------------------------------
    def drive_force(self, v):
        """Wheel drive force [N]: grip-limited off the line, power-limited at speed."""
        grip  = self.mu * self.mass * self.g
        power = self.power / max(abs(v), self.v_eps)
        return min(grip, power)

    def accel(self, v):
        """Longitudinal acceleration [m/s²] at speed ``v`` (the force balance / mass)."""
        f_drive = self.drive_force(v)
        f_drag  = 0.5 * self.rho * self.drag_area * v * abs(v)
        f_roll  = self.c_roll * self.mass * self.g
        f_grade = self.mass * self.g * np.sin(self.grade)
        return (f_drive - f_drag - f_roll - f_grade) / self.mass

    def f_continuous(self, x, u=None, t=None):
        """Longitudinal derivative ẋ = [v, accel(v)] (the force balance)."""
        s = np.asarray(x, float).reshape(-1)
        return np.array([[s[1]], [self.accel(s[1])]])

    def f(self, x, u=None, w=None, t=None):
        # Integrate the continuous dynamics with the model's integrator (RK4),
        # then clamp: the car does not roll backwards in a speed run.
        s = np.asarray(self._integrate_continuous(x, u, t), float).reshape(-1)
        s[1] = max(s[1], 0.0)
        return s.reshape(-1, 1)

    def h(self, x, v=None, t=None):
        z = np.asarray(x, float).reshape(-1)[:1].reshape(1, 1)
        return z if v is None else z + np.asarray(v, float).reshape(1, 1)

    # ----------------------------------------------------------------------
    # Speed-test metrics extracted from a filtered/sampled record
    # ----------------------------------------------------------------------
    def metrics(self, record):
        """Headline figures of a run: 0–100 km/h, 0–60 mph, standing 400 m
        (≈ quarter-mile) time and trap speed, and the terminal speed reached.

        ``record`` is any record exposing ``time`` (s), ``xk[:, 0]`` (distance) and
        ``xk[:, 1]`` (speed). Times are linearly interpolated between samples; entries
        are ``None`` when the run is too short to reach a threshold.
        """
        t = np.asarray(record.time, float).reshape(-1)
        x = np.asarray(record.xk)
        dist, spd = x[:, 0], x[:, 1]

        def t_at_speed(v_target):
            i = np.argmax(spd >= v_target)
            if not (spd >= v_target).any() or i == 0:
                return None
            t0, t1, v0, v1 = t[i - 1], t[i], spd[i - 1], spd[i]
            return float(t0 + (t1 - t0) * (v_target - v0) / (v1 - v0)) if v1 > v0 else float(t1)

        def t_at_dist(d_target):
            i = np.argmax(dist >= d_target)
            if not (dist >= d_target).any() or i == 0:
                return None, None
            t0, t1, d0, d1 = t[i - 1], t[i], dist[i - 1], dist[i]
            frac = (d_target - d0) / (d1 - d0) if d1 > d0 else 1.0
            return float(t0 + (t1 - t0) * frac), float(spd[i - 1] + (spd[i] - spd[i - 1]) * frac)

        t_400, v_400 = t_at_dist(402.336)   # standing quarter-mile (402.336 m)
        return {
            "0-100 km/h [s]":   t_at_speed(100 / 3.6),
            "0-60 mph [s]":     t_at_speed(26.8224),
            "400 m time [s]":   t_400,
            "400 m trap [km/h]": None if v_400 is None else v_400 * 3.6,
            "top speed [km/h]": float(spd.max() * 3.6),
        }

    # ==================================================================
    # Signature figure — the speed run (override KalmanModel.plot).
    # ``record`` is the EKF estimate; ``sample`` the sampled truth+GPS fixes.
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     figsize=(17, 4.4), **kwargs):
        """Signature figure: speed vs time (truth, EKF estimate, and the naive
        finite-difference of the raw GPS — the mess the filter cleans up), the
        acceleration curve in g, and the EKF position-tracking error."""
        import matplotlib.pyplot as plt
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n     = min(len(truth), len(estx))
        truth, estx = truth[:n], estx[:n]
        t = np.arange(n) * self._dt

        # naive speed: difference the noisy GPS position (what you'd get without a filter)
        gps = (np.asarray(sample.zk).reshape(-1) if sample is not None
               else np.asarray(record.zk).reshape(-1))[:n]
        v_naive = np.gradient(gps, self._dt)

        a_est = np.array([self.accel(v) for v in estx[:, 1]]) / self.g
        err   = np.abs(truth[:, 0] - estx[:, 0])

        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].plot(t, v_naive * 3.6, ".", color="0.75", ms=2, label="naive ΔGPS")
        ax[0].plot(t, truth[:, 1] * 3.6, color="crimson", lw=1.6, label="truth")
        ax[0].plot(t, estx[:, 1] * 3.6, "--", color="navy", lw=1.2, label="EKF estimate")
        ax[0].axhline(100, color="0.5", ls=":", lw=0.8)
        ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("speed [km/h]")
        ax[0].set_title("Speed run"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[1].plot(t, a_est, color="darkorange")
        ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("acceleration [g]")
        ax[1].set_title("Acceleration (estimated)"); ax[1].grid(alpha=0.3)
        ax[2].plot(t, err, color="seagreen")
        ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("|pos error| [m]")
        ax[2].set_title("EKF tracking error"); ax[2].grid(alpha=0.3)
        fig.suptitle(f"{self.name()} — straight-line acceleration", y=1.02)
        fig.tight_layout()
        return fig



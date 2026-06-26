"""
Rocket science — getting off the pad and back: **takeoff** and **landing**.

This module is the *near-surface, powered-flight* half of the rocketry models (no third
bodies). The space-travel half — flying to the Moon and interplanetary transfers — lives in
:mod:`kalman.apps.space`.

The "rocket equation" family (powered flight where what differs is the **phase**):

* :class:`RocketAscentModel` — a planar **guided ascent**: gravity versus **thrust supplied
  as the control input** ``u`` (constant mass). Demonstrates control channels (``l>0``).
  *Scaled units.* Legacy alias: ``RocketLaunchModel``.
* :class:`RocketLiftoffModel` — a 1-D **vertical liftoff** whose thrust is internal and whose
  **mass burns down** (Tsiolkovsky): acceleration rises until **burnout**. *SI units.*
  Legacy alias: ``RocketTakeoffModel``.
* :class:`RocketLandingModel` — the same 1-D mass-burning dynamics run in reverse: a
  SpaceX-style **hoverslam**, free-falling then firing a closed-loop landing burn to touch
  down near zero velocity. *SI units.*

All are :class:`ExtendedKalmanModel` subclasses whose ``f()`` advances the state by one
4th-order Runge–Kutta step; the EKF linearises ``f`` with a numerical Jacobian.

The ascent model uses *scaled units* (``length = 1 Mm = 1000 km``, ``time = 1 ks = 1000 s``)
to stay well-conditioned; the near-surface liftoff/landing models use plain **SI** (m, s, kg).
Physical constants come from :mod:`kalman.runtime.astro` (astropy-sourced where available).
"""
from __future__ import annotations
import numpy as np

from ..framework.models.extended import ExtendedKalmanModel
from ..framework.constants import KalmanState, KalmanObservable
from ..runtime.astro import (LU, TU, KM, GM_EARTH, R_EARTH, H_SCALE,
                             G0, RHO0, H_ATM, rk4 as _rk4)


# ===========================================================================
#  1) Guided ascent from Earth — thrust as the control input (2-D)
# ===========================================================================
class RocketAscentState(KalmanState):
    X = "Mm"; Y = "Mm"; VX = "km/s"; VY = "km/s"


class RocketAscentObservable(KalmanObservable):
    X = "Mm"; Y = "Mm"


class RocketAscentModel(ExtendedKalmanModel):
    """
    Planar **guided** powered ascent from Earth's surface (the steering/guidance model).

    State ``x = [x, y, vx, vy]``; **control** ``u = [ax, ay]`` is the thrust acceleration
    (km/s²), so this model has ``l = 2`` control channels. Dynamics: central gravity +
    thrust (+ optional exponential-atmosphere drag). A radar measures position only.

    The distinguishing trait is the **2-D control input**: the thrust program is fed in,
    constant mass. Contrast :class:`RocketLiftoffModel` (1-D vertical liftoff whose thrust
    is internal and whose **mass burns down**). Importable as the legacy alias
    ``RocketLaunchModel``.
    """

    NAME = "Rocket Ascent (guided, thrust as control)"; VERSION = "1.0"

    @property
    def states(self):      return RocketAscentState
    @property
    def observables(self): return RocketAscentObservable

    def __init__(self, dt_s=1.0, track_km=1.0, drag=False, ballistic_coeff=2.0e-3,
                 length=40.0, width=3.7, q_pos=1e-9, q_vel=1e-9, **kw):
        self.dtype = np.float64
        self._dt   = dt_s / TU
        self.drag  = bool(drag)
        self.Cb    = float(ballistic_coeff)     # lumped 0.5*Cd*A/m (scaled)
        self.length, self.width = float(length), float(width)
        self.n, self.m, self.l = 4, 2, 2
        self.fs = 1.0 / self._dt
        Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)
        R = (track_km * KM) ** 2 * np.eye(2)
        super().__init__("LAUNCH", Q=Q, R=R, fs=self.fs, **kw)

    def acceleration(self, r, v, u):
        a = -GM_EARTH * r / np.linalg.norm(r) ** 3            # gravity
        if u is not None:
            a = a + np.asarray(u, float).reshape(2)           # thrust (control)
        if self.drag:
            alt = np.linalg.norm(r) - R_EARTH
            rho = np.exp(-max(alt, 0.0) / H_SCALE)
            a = a - self.Cb * rho * np.linalg.norm(v) * v     # quadratic drag
        return a

    def f(self, x, u=None, w=None, t=None):
        s = np.asarray(x, float).reshape(-1)
        def deriv(s, tt):
            return np.concatenate([s[2:], self.acceleration(s[:2], s[2:], u)])
        return _rk4(s, float(t or 0.0), self._dt, deriv).reshape(-1, 1)

    def h(self, x, v=None, t=None):
        z = np.asarray(x, float).reshape(-1)[:2].reshape(2, 1)
        return z if v is None else z + np.asarray(v, float).reshape(2, 1)

    # ==================================================================
    # Signature figure — powered ascent (override KalmanModel.plot).
    # ``record`` is the EKF estimate; ``sample`` the sampled truth+measurements.
    # ==================================================================
    @staticmethod
    def _pitch(vx, vy):
        """Flight-path tilt from vertical [rad] in the (downrange, altitude) plane."""
        return np.arctan2(vy, vx)

    def _orbit_frac(self, r, v):
        """Speed as a fraction of local circular-orbit speed (a 'to orbit' gauge)."""
        return float(np.linalg.norm(v)) / np.sqrt(GM_EARTH / max(np.linalg.norm(r), 1e-9))

    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     n_pose=7, rocket_frac=0.05, figsize=(15, 5.6), uk=None, **kwargs):
        """Signature figure — matches the interactive layout: the drawn rocket on the
        gravity-turn trajectory (left), altitude+speed / EKF error / thrust control
        time series (centre), and a small attitude reticle (right). Radar measurement
        positions are overlaid on the trajectory. Pass ``uk`` to show the thrust
        acceleration control programme."""
        import matplotlib.pyplot as plt
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n     = min(len(truth), len(estx))
        truth, estx = truth[:n], estx[:n]
        t_s   = np.arange(n) * self._dt * TU
        alt   = (np.linalg.norm(truth[:, :2], axis=1) - R_EARTH) * LU / 1e3
        down  = truth[:, 1] * LU / 1e3
        speed = np.linalg.norm(truth[:, 2:4], axis=1)
        err   = np.linalg.norm(truth[:, :2] - estx[:, :2], axis=1) * LU / 1e3
        uk_arr = np.asarray(uk)[:n] if uk is not None else None  # (n,2) [ax,ay] in km/s²

        n_rows = 3 if uk_arr is not None else 2
        wr = [2.2, 1.8, 0.65]
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(n_rows, 3, width_ratios=wr, wspace=0.35,
                               hspace=0.50 if n_rows == 3 else 0.45)
        axT = fig.add_subplot(gs[:, 0])           # trajectory (spans all rows)
        axA = fig.add_subplot(gs[0, 1])           # altitude & speed
        axE = fig.add_subplot(gs[1, 1])           # EKF tracking error
        axC = fig.add_subplot(gs[2, 1]) if uk_arr is not None else None  # thrust control
        axR = fig.add_subplot(gs[0, 2])           # attitude reticle (final pose)
        axGI = fig.add_subplot(gs[1:, 2]); axGI.axis("off")

        # --- trajectory with radar measurements and drawn rockets ---
        if sample is not None:
            meas = np.asarray(sample.zk)[:n]      # (n, 2): [x_Mm, y_Mm]
            axT.scatter(meas[:, 1] * LU / 1e3, (np.linalg.norm(meas, axis=1) - R_EARTH) * LU / 1e3,
                        s=4, color="0.8", label="radar", zorder=2)
        axT.plot(down, alt, color="crimson", lw=1.5, label="truth")
        axT.plot(estx[:, 1] * LU / 1e3,
                 (np.linalg.norm(estx[:, :2], axis=1) - R_EARTH) * LU / 1e3,
                 "--", color="navy", lw=1.0, label="EKF")
        span = max(down.max() - down.min(), alt.max() - alt.min(), 1.0)
        Ld = rocket_frac * span; Wd = 0.18 * Ld
        pose_idx = np.linspace(0, n - 1, n_pose).astype(int)
        for i in pose_idx:
            th = -self._pitch(truth[i, 2], truth[i, 3])
            shp = _rocket_shape(down[i], alt[i], th, Ld, Wd, anchor="com")
            _add_rocket(axT, shp)
            # thrust arrow from control vector
            if uk_arr is not None:
                u = uk_arr[i]; mag = np.linalg.norm(u)
                if mag > 0:
                    scale = 0.25 * Ld / mag
                    # thrust direction is along (ux, uy) in local frame → world (down,alt)
                    axT.annotate("", xy=(down[i] + u[0] * scale * TU / 1e3,
                                         alt[i] + u[1] * scale * TU / 1e3),
                                 xytext=(down[i], alt[i]),
                                 arrowprops=dict(arrowstyle="-|>", color="gold", lw=1.2,
                                                 shrinkA=0, shrinkB=0, mutation_scale=10))
        axT.set_xlabel("downrange [km]"); axT.set_ylabel("altitude [km]")
        axT.set_title("Powered ascent (gravity turn)"); axT.grid(alpha=0.3)
        axT.legend(loc="upper left", fontsize=8); axT.set_aspect("equal", adjustable="datalim")

        # --- time series ---
        axA.plot(t_s, alt,   color="steelblue",  lw=1.5, label="altitude [km]")
        axA.plot(t_s, speed, color="darkorange",  lw=1.3, label="speed [km/s]")
        axA.legend(fontsize=8); axA.grid(alpha=0.3); axA.set_title("Altitude & speed")

        axE.plot(t_s, err, color="seagreen", lw=1.5)
        if axC is None:
            axE.set_xlabel("time [s]")
        axE.set_ylabel("|pos error| [km]")
        axE.set_title("EKF tracking error"); axE.grid(alpha=0.3)

        if axC is not None and uk_arr is not None:
            thrust_mag = np.linalg.norm(uk_arr, axis=1)
            pitch_deg  = np.degrees(np.arctan2(uk_arr[:, 1], uk_arr[:, 0]))
            axC.plot(t_s, thrust_mag, color="gold",   lw=1.4, label="thrust |u| [km/s²]")
            axC2 = axC.twinx()
            axC2.plot(t_s, pitch_deg, color="0.5", lw=1.0, ls="--", label="pitch [°]")
            axC2.set_ylabel("pitch [°]", fontsize=8)
            axC.set_xlabel("time [s]"); axC.set_ylabel("|u| [km/s²]", fontsize=8)
            axC.set_title("Thrust control"); axC.grid(alpha=0.3)
            lines = axC.get_lines() + axC2.get_lines()
            axC.legend(lines, [l.get_label() for l in lines], fontsize=7)

        # --- attitude reticle showing final rocket orientation ---
        th_final = -self._pitch(truth[-1, 2], truth[-1, 3])
        reticle = _make_reticle(axR)
        _set_reticle(reticle, th_final)

        fig.suptitle(f"{self.name()} — powered ascent", y=1.01)
        fig.tight_layout()
        return fig

    def _plot_interactive(self, record, sample=None, *, nmax=None, rocket_frac=0.05,
                          frames=100, interval=70, figsize=(15, 5.6), uk=None, **kwargs):
        """Interactive ascent — same 3-col layout as the static: animated rocket on the
        gravity-turn trajectory (left), growing altitude+speed / EKF error / optional thrust
        control time series (centre), compact attitude reticle + live readout (narrow right).
        Radar measurement positions are overlaid on the trajectory. Pass ``uk`` to show the
        thrust acceleration control programme."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from ..helpers import animate

        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n = min(len(truth), len(estx)) if nmax is None else int(nmax)
        truth, estx = truth[:n], estx[:n]
        t_s   = np.arange(n) * self._dt * TU
        alt   = (np.linalg.norm(truth[:, :2], axis=1) - R_EARTH) * LU / 1e3
        down  = truth[:, 1] * LU / 1e3
        speed = np.linalg.norm(truth[:, 2:4], axis=1)
        err   = np.linalg.norm(truth[:, :2] - estx[:, :2], axis=1) * LU / 1e3
        uk_arr = np.asarray(uk)[:n] if uk is not None else None
        thrust_mag = np.linalg.norm(uk_arr, axis=1) if uk_arr is not None else None
        pitch_deg  = np.degrees(np.arctan2(uk_arr[:, 1], uk_arr[:, 0])) \
                     if uk_arr is not None else None

        span = max(down.max() - down.min(), alt.max() - alt.min(), 1.0)
        Ld = rocket_frac * span; Wd = 0.18 * Ld

        n_rows = 3 if uk_arr is not None else 2
        wr = [2.2, 1.8, 0.65]
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(n_rows, 3, width_ratios=wr, wspace=0.35,
                               hspace=0.50 if n_rows == 3 else 0.45)
        ax  = fig.add_subplot(gs[:, 0])        # trajectory
        axA = fig.add_subplot(gs[0, 1])        # altitude & speed
        axE = fig.add_subplot(gs[1, 1])        # EKF error
        axC = fig.add_subplot(gs[2, 1]) if uk_arr is not None else None  # control
        axR = fig.add_subplot(gs[0, 2])        # attitude reticle
        axI = fig.add_subplot(gs[1:, 2]); axI.axis("off")  # live readout

        # --- trajectory: full guide (faint) + animated rocket ---
        ax.plot(down, alt, color="0.85", lw=1.0, label="trajectory")
        ax.plot(estx[:, 1] * LU / 1e3,
                (np.linalg.norm(estx[:, :2], axis=1) - R_EARTH) * LU / 1e3,
                "--", color="navy", lw=0.8, alpha=0.4, label="EKF")
        # measurements
        meas_xy = None
        if sample is not None:
            meas_raw = np.asarray(sample.zk)[:n]   # (n, 2): [x_Mm, y_Mm]
            meas_d = meas_raw[:, 1] * LU / 1e3
            meas_a = (np.linalg.norm(meas_raw, axis=1) - R_EARTH) * LU / 1e3
            meas_xy = np.column_stack([meas_d, meas_a])
            ax.scatter(meas_d, meas_a, s=4, color="0.8", label="radar", zorder=2)
        body  = Polygon(np.zeros((4, 2)), closed=True, fc="0.25", ec="k", lw=0.8, zorder=4)
        nose  = Polygon(np.zeros((3, 2)), closed=True, fc="0.25", ec="k", lw=0.8, zorder=4)
        flame = Polygon(np.zeros((3, 2)), closed=True, fc="orange", ec="orangered", alpha=0.9, zorder=3)
        thrust_arr = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                  arrowprops=dict(arrowstyle="-|>", color="gold", lw=1.4,
                                                  shrinkA=0, shrinkB=0, mutation_scale=10)) \
                     if uk_arr is not None else None
        for p in (flame, body, nose):
            ax.add_patch(p)
        mx = max(down.max(), 1.0); my = max(alt.max(), 1.0)
        ax.set_xlim(-0.12 * mx, 1.15 * mx); ax.set_ylim(-0.05 * my, 1.12 * my)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("downrange [km]"); ax.set_ylabel("altitude [km]")
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)

        # --- time-series: faint guide + growing lines ---
        axA.plot(t_s, alt,   color="steelblue",  lw=0.7, alpha=0.15)
        axA.plot(t_s, speed, color="darkorange",  lw=0.7, alpha=0.15)
        axE.plot(t_s, err,   color="seagreen",    lw=0.7, alpha=0.15)
        la_a, = axA.plot([], [], color="steelblue",  lw=1.5, label="altitude [km]")
        la_s, = axA.plot([], [], color="darkorange",  lw=1.3, label="speed [km/s]")
        le,   = axE.plot([], [], color="seagreen",    lw=1.5)
        cA = axA.axvline(0, color="0.5", lw=0.8, ls=":")
        cE = axE.axvline(0, color="0.5", lw=0.8, ls=":")
        axA.set_xlim(0, t_s[-1]); axA.set_ylim(0, max(alt.max(), speed.max()) * 1.12)
        axA.legend(fontsize=7); axA.grid(alpha=0.3); axA.set_title("Altitude & speed")
        axE.set_xlim(0, t_s[-1]); axE.set_ylim(0, err.max() * 1.12)
        if axC is None:
            axE.set_xlabel("time [s]")
        axE.set_ylabel("|pos error| [km]")
        axE.set_title("EKF tracking error"); axE.grid(alpha=0.3)

        lc_m = lc_p = cC = axC2 = None
        if axC is not None and uk_arr is not None:
            axC.plot(t_s, thrust_mag, color="gold",  lw=0.7, alpha=0.15)
            axC2 = axC.twinx()
            axC2.plot(t_s, pitch_deg, color="0.5",  lw=0.7, alpha=0.15, ls="--")
            lc_m, = axC.plot([], [], color="gold",   lw=1.4, label="|u| [km/s²]")
            lc_p, = axC2.plot([], [], color="0.5",  lw=1.0, ls="--", label="pitch [°]")
            cC    = axC.axvline(0, color="0.5", lw=0.8, ls=":")
            axC.set_xlim(0, t_s[-1]); axC2.set_xlim(0, t_s[-1])
            axC.set_xlabel("time [s]"); axC.set_ylabel("|u| [km/s²]", fontsize=8)
            axC2.set_ylabel("pitch [°]", fontsize=8)
            axC.set_title("Thrust control"); axC.grid(alpha=0.3)
            lines = axC.get_lines() + axC2.get_lines()
            axC.legend(lines, [l.get_label() for l in lines], fontsize=7)

        # --- reticle showing current attitude ---
        reticle = _make_reticle(axR)
        txt = axI.text(0.05, 0.95, "", transform=axI.transAxes, va="top",
                       family="monospace", fontsize=8.5)

        step = max(1, n // frames); frame_idx = list(range(0, n, step)) + [n - 1]

        def _draw(k):
            th = -self._pitch(truth[k, 2], truth[k, 3])
            shp = _rocket_shape(down[k], alt[k], th, Ld, Wd, flame=0.7 * Ld, anchor="com")
            body.set_xy(shp["body"]); nose.set_xy(shp["nose"]); flame.set_xy(shp["flame"])
            if thrust_arr is not None and uk_arr is not None:
                u = uk_arr[k]; mag = np.linalg.norm(u)
                scale = 0.25 * Ld / max(mag, 1e-9)
                thrust_arr.set_position((down[k], alt[k]))
                thrust_arr.xy = (down[k] + u[0] * scale * TU / 1e3,
                                 alt[k]  + u[1] * scale * TU / 1e3)
            la_a.set_data(t_s[:k+1], alt[:k+1])
            la_s.set_data(t_s[:k+1], speed[:k+1])
            le.set_data(t_s[:k+1], err[:k+1])
            cA.set_xdata([t_s[k], t_s[k]]); cE.set_xdata([t_s[k], t_s[k]])
            if lc_m is not None and uk_arr is not None:
                lc_m.set_data(t_s[:k+1], thrust_mag[:k+1])
                lc_p.set_data(t_s[:k+1], pitch_deg[:k+1])
                cC.set_xdata([t_s[k], t_s[k]])
            _set_reticle(reticle, th)
            spd = np.linalg.norm(truth[k, 2:4])
            ax.set_title("Powered ascent    t = %.0f s" % t_s[k])
            txt.set_text("alt   %6.1f km\ndown  %6.1f km\nspeed %5.2f km/s\npitch %5.0f°" %
                         (alt[k], down[k], spd, np.degrees(self._pitch(truth[k, 2], truth[k, 3]))))
            return body, nose, flame, la_a, la_s, le, txt

        return animate(fig, _draw, frames=frame_idx, interval=interval)


# ===========================================================================
#  2) Vertical takeoff from the pad — propellant burn (the rocket equation)
# ===========================================================================
# This model is in PLAIN SI UNITS (m, s, kg), unlike the scaled-unit ascent model
# above: a takeoff lives near the surface (altitude ~0–100 km, speed ~0–2 km/s,
# mass ~10³ kg), which SI already keeps well-conditioned. G0/RHO0/H_ATM come from
# kalman.runtime.astro.
class RocketLiftoffState(KalmanState):
    ALTITUDE = "m"      # altitude of the rocket's base (on the pad at 0)
    VELOCITY = "m/s"
    MASS     = "kg"


class RocketLiftoffObservable(KalmanObservable):
    ALTITUDE = "m"


class RocketLiftoffModel(ExtendedKalmanModel):
    """
    Vertical **liftoff** from the launch pad with a finite propellant load (1-D).

    The point a liftoff makes that a fixed-mass model cannot: the vehicle **burns mass**,
    so even at constant thrust the acceleration *grows* as the rocket lightens, then turns
    negative at **burnout** — Tsiolkovsky's rocket equation in flight. Mass is therefore a
    state, not a parameter::

        ḣ = v
        v̇ = T(m)/m − g − ½ ρ(h) (C_d A) v|v| / m
        ṁ = −ṁ_fuel        (while propellant remains; 0 after burnout)

    with thrust ``T = v_e · ṁ_fuel`` while ``m > m_dry`` (then the engine cuts off and the
    rocket coasts). ``ρ(h) = ρ₀ e^{−h/H}`` is the exponential atmosphere. A ground radar /
    altimeter measures **altitude only** (``z = h + noise``); the EKF reconstructs the
    unmeasured climb rate and the remaining mass. ``n = 3``, ``m = 1``, ``l = 0``.

    Contrast :class:`RocketAscentModel` (2-D guided ascent, thrust as control, constant
    mass) and :class:`RocketLandingModel` (the same 1-D mass-burning dynamics run in
    reverse to touch down). Importable as the legacy alias ``RocketTakeoffModel``.
    """

    NAME = "Rocket Liftoff (vertical, fuel burns)"; VERSION = "1.0"

    @property
    def states(self):      return RocketLiftoffState
    @property
    def observables(self): return RocketLiftoffObservable

    def __init__(self, name="LIFTOFF", *,
                 m0=1000.0,              # wet (fuelled) mass at liftoff [kg]
                 m_dry=400.0,            # dry/structural mass [kg] (burnout mass)
                 ve=2500.0,             # effective exhaust velocity [m/s] (Isp·g₀)
                 mdot=8.0,               # propellant mass-flow rate [kg/s]
                 drag_area=0.25,         # C_d·A [m²]
                 drag=True,              # include atmospheric drag
                 g=G0,                   # gravity [m/s²] (near-surface constant)
                 length=40.0, width=3.7, # body size [m] (for the drawn rocket)
                 fs=10.0,                # radar/altimeter sample rate [Hz]
                 track_m=20.0,           # altitude measurement std [m]
                 q_alt=1e-3, q_vel=1e-2, q_mass=1e-4,   # process-noise variances (model slack)
                 **kw):
        self.dtype = np.float64
        self.m0 = float(m0)
        self.m_dry = float(m_dry)
        self.ve = float(ve)
        self.mdot = float(mdot)
        self.drag_area = float(drag_area)
        self.drag = bool(drag)
        self.g = float(g)
        self.length, self.width = float(length), float(width)

        self._dt = 1.0 / fs
        self.n, self.m, self.l = 3, 1, 0
        self.fs = fs

        Q = np.diag([q_alt, q_vel, q_mass]).astype(float)
        R = np.array([[track_m ** 2]], dtype=float)
        super().__init__(name, Q=Q, R=R, fs=fs, **kw)

    # --- handy figures of merit -------------------------------------------
    @property
    def thrust(self):
        """Engine thrust while burning [N] = exhaust velocity × mass-flow."""
        return self.ve * self.mdot

    @property
    def burn_time(self):
        """Propellant burn duration [s] = propellant mass / mass-flow."""
        return (self.m0 - self.m_dry) / self.mdot

    def twr(self, mass=None):
        """Thrust-to-weight ratio at a given mass (defaults to liftoff mass)."""
        mass = self.m0 if mass is None else mass
        return self.thrust / (mass * self.g)

    def accel(self, x):
        """Net vertical acceleration [m/s²] for state ``x = [h, v, m]``."""
        h, v, m = np.asarray(x, float).reshape(-1)[:3]
        return self._deriv(np.array([h, v, m]), 0.0)[1]

    # --- dynamics ----------------------------------------------------------
    def _deriv(self, s, t):
        h, v, m = s
        burning = m > self.m_dry
        mdot = self.mdot if burning else 0.0
        thrust = self.ve * mdot
        m_eff = max(m, self.m_dry)                 # keep the division safe and physical
        a = thrust / m_eff - self.g
        if self.drag:
            rho = RHO0 * np.exp(-max(h, 0.0) / H_ATM)
            a -= 0.5 * rho * self.drag_area * v * abs(v) / m_eff
        return np.array([v, a, -mdot])

    def f(self, x, u=None, w=None, t=None):
        s = _rk4(np.asarray(x, float).reshape(-1), float(t or 0.0), self._dt, self._deriv)
        s[2] = max(s[2], self.m_dry)               # mass never drops below the dry mass
        return s.reshape(-1, 1)

    def h(self, x, v=None, t=None):
        z = np.asarray(x, float).reshape(-1)[:1].reshape(1, 1)
        return z if v is None else z + np.asarray(v, float).reshape(1, 1)

    # --- flight metrics from a record -------------------------------------
    def metrics(self, record):
        """Headline flight figures from a sampled/filtered ``record`` (states
        ``[h, v, m]``): burnout time & velocity, apogee, peak acceleration and peak
        dynamic pressure (max-Q)."""
        t = np.asarray(record.time, float).reshape(-1)
        x = np.asarray(record.xk)
        h, v, m = x[:, 0], x[:, 1], x[:, 2]
        i_bo = int(np.argmax(m <= self.m_dry + 1e-6))
        if not (m <= self.m_dry + 1e-6).any():
            i_bo = len(t) - 1
        rho = RHO0 * np.exp(-np.clip(h, 0, None) / H_ATM)
        q = 0.5 * rho * v ** 2
        a = np.array([self.accel(xi) for xi in x])
        return {
            "burnout time [s]":     float(t[i_bo]),
            "burnout speed [m/s]":  float(v[i_bo]),
            "apogee [km]":          float(h.max() / 1e3),
            "max accel [g]":        float(np.nanmax(a) / G0),
            "max-Q [kPa]":          float(q.max() / 1e3),
            "liftoff TWR":          float(self.twr()),
        }

    def _fuel_frac(self, m):
        return (float(m) - self.m_dry) / max(self.m0 - self.m_dry, 1e-9)

    # ==================================================================
    # Signature figure — the liftoff (override KalmanModel.plot).
    # ``record`` is the EKF estimate; ``sample`` the sampled truth+altitude fixes.
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     n_pose=6, figsize=(15, 6.2), **kwargs):
        """Signature figure — matches the interactive layout: ghost rocket poses climbing
        along a vertical equal-aspect track (left), altitude / climb-rate+mass / TWR time
        series (centre) and a small propellant gauge (right)."""
        import matplotlib.pyplot as plt
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n     = min(len(truth), len(estx))
        truth, estx = truth[:n], estx[:n]
        t = np.arange(n) * self._dt
        meas = (np.asarray(sample.zk).reshape(-1) if sample is not None
                else np.asarray(record.zk).reshape(-1))
        meas = meas[:n] if meas.size else None
        twr = self.thrust / np.clip(estx[:, 2], self.m_dry, None) / self.g
        twr = np.where(estx[:, 2] > self.m_dry + 1e-6, twr, 0.0)

        alt_max = max(truth[:, 0].max(), 1.0)
        Ld = 0.10 * alt_max; Wd = 0.12 * Ld   # slender rocket: 12 % width/length

        # 4-col layout: rocket track | altitude | climb+mass | TWR | gauge (narrow)
        wr = [1.6, 1.8, 1.8, 0.75]
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(3, 4, width_ratios=wr, wspace=0.38, hspace=0.50)
        axR  = fig.add_subplot(gs[:, 0])    # rocket ghost track
        axA  = fig.add_subplot(gs[0, 1:3]) # altitude
        axV  = fig.add_subplot(gs[1, 1:3]) # climb rate & mass
        axT  = fig.add_subplot(gs[2, 1:3]) # TWR
        axG  = fig.add_subplot(gs[0, 3])   # propellant gauge
        axGI = fig.add_subplot(gs[1:, 3]); axGI.axis("off")

        # --- ghost poses fading in along the vertical track ---
        idx = np.linspace(0, n - 1, n_pose).astype(int)
        axR.axhline(0.0, color="0.4", lw=2)
        for jj, i in enumerate(idx):
            alpha = 0.20 + 0.80 * jj / max(n_pose - 1, 1)
            burning = truth[i, 2] > self.m_dry + 1e-6
            fl = 0.6 * Ld if burning else 0.0
            shp = _rocket_shape(0.0, truth[i, 0], 0.0, Ld, Wd, flame=fl, anchor="base")
            _add_rocket(axR, shp, alpha=alpha)
            axR.text(Wd * 2.2, truth[i, 0] + 0.12 * Ld, "%.0fs" % t[i],
                     fontsize=7, color="0.4", va="bottom")
        axR.set_xlim(-2.5 * Ld, 2.5 * Ld)
        axR.set_ylim(-0.08 * alt_max, 1.15 * alt_max)
        axR.set_aspect("equal", adjustable="datalim")
        axR.set_xticks([]); axR.set_ylabel("altitude [m]")
        axR.set_title("Liftoff — rocket climbing (%.0f%% propellant left)"
                      % (100 * self._fuel_frac(truth[-1, 2])))
        axR.grid(alpha=0.2, axis="y")

        _fuel_gauge(axG, self._fuel_frac(truth[-1, 2]))

        # --- time series ---
        if meas is not None:
            axA.scatter(t, meas / 1e3, s=4, color="0.8", label="radar", zorder=2)
        axA.plot(t, truth[:, 0] / 1e3, color="crimson", lw=1.6, label="truth")
        axA.plot(t, estx[:, 0] / 1e3, "--", color="navy", lw=1.1, label="EKF")
        axA.set_ylim(0, alt_max / 1e3 * 1.12)
        axA.set_ylabel("altitude [km]"); axA.legend(fontsize=8); axA.grid(alpha=0.3)
        axA.set_title("Altitude")

        axV.plot(t, truth[:, 1], color="crimson", lw=1.5, label="climb rate [m/s]")
        axV.plot(t, truth[:, 2], color="darkorange", lw=1.3, label="mass [kg]")
        axV.axhline(self.m_dry, color="0.6", ls=":", lw=0.8)
        axV.legend(fontsize=8); axV.grid(alpha=0.3); axV.set_title("Climb rate & mass")

        axT.plot(t, twr, color="seagreen"); axT.axhline(1.0, color="0.6", ls=":", lw=0.8)
        axT.set_xlabel("time [s]"); axT.set_ylabel("T / W")
        axT.set_title("TWR (rises as fuel burns)"); axT.grid(alpha=0.3)

        fig.suptitle(f"{self.name()} — vertical liftoff & burnout", y=1.01)
        fig.tight_layout()
        return fig

    def _plot_interactive(self, record, sample=None, *, nmax=None, frames=100, interval=70,
                          figsize=(15, 6.2), **kwargs):
        """Interactive liftoff — same 4-col layout as the static: animated rocket track
        (left), growing altitude / climb-rate+mass / TWR time series (centre), compact
        propellant gauge + live readout (narrow right column)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from ..helpers import animate

        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n = min(len(truth), len(estx)) if nmax is None else int(nmax)
        truth, estx = truth[:n], estx[:n]
        t = np.arange(n) * self._dt
        meas = (np.asarray(sample.zk).reshape(-1)[:n] if sample is not None else None)
        twr_all = self.thrust / np.clip(truth[:, 2], self.m_dry, None) / self.g
        twr_all = np.where(truth[:, 2] > self.m_dry + 1e-6, twr_all, 0.0)

        alt_max = max(truth[:, 0].max(), 1.0)
        Ld = 0.10 * alt_max; Wd = 0.12 * Ld   # slender — matches static ghost poses

        wr = [1.6, 1.8, 1.8, 0.75]
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(3, 4, width_ratios=wr, wspace=0.38, hspace=0.50)
        ax  = fig.add_subplot(gs[:, 0])          # animated rocket track
        axA = fig.add_subplot(gs[0, 1:3])        # altitude
        axV = fig.add_subplot(gs[1, 1:3])        # climb rate & mass
        axT = fig.add_subplot(gs[2, 1:3])        # TWR
        axG = fig.add_subplot(gs[0, 3])          # propellant gauge
        axI = fig.add_subplot(gs[1:, 3]); axI.axis("off")  # live readout

        # --- rocket track (equal aspect, same as static) ---
        ax.axhline(0.0, color="0.4", lw=2)
        body  = Polygon(np.zeros((4, 2)), closed=True, fc="0.25", ec="k", lw=0.8, zorder=4)
        nose  = Polygon(np.zeros((3, 2)), closed=True, fc="0.25", ec="k", lw=0.8, zorder=4)
        flame = Polygon(np.zeros((3, 2)), closed=True, fc="orange", ec="orangered", alpha=0.9, zorder=3)
        for p in (flame, body, nose):
            ax.add_patch(p)
        ax.set_xlim(-2.5 * Ld, 2.5 * Ld)
        ax.set_ylim(-0.18 * alt_max, 1.25 * alt_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_ylabel("altitude [m]"); ax.grid(alpha=0.2, axis="y")

        # --- time-series: faint full-range guide + growing animated lines ---
        axA.plot(t, truth[:, 0] / 1e3, color="crimson", lw=0.7, alpha=0.15)
        axA.plot(t, estx[:, 0] / 1e3, color="navy",   lw=0.7, alpha=0.15, ls="--")
        axV.plot(t, truth[:, 1], color="crimson",   lw=0.7, alpha=0.15)
        axV.plot(t, truth[:, 2], color="darkorange", lw=0.7, alpha=0.15)
        axT.plot(t, twr_all,     color="seagreen",   lw=0.7, alpha=0.15)
        if meas is not None:
            axA.scatter(t, meas / 1e3, s=3, color="0.75", alpha=0.35, zorder=2)

        la_t, = axA.plot([], [], color="crimson",    lw=1.6, label="truth")
        la_e, = axA.plot([], [], "--", color="navy", lw=1.1, label="EKF")
        la_z  = axA.scatter([], [], s=5, color="0.8", label="radar", zorder=3) \
                if meas is not None else None
        lv_v, = axV.plot([], [], color="crimson",    lw=1.5, label="climb rate [m/s]")
        lv_m, = axV.plot([], [], color="darkorange", lw=1.3, label="mass [kg]")
        lt_t, = axT.plot([], [], color="seagreen", lw=1.5)
        cA = axA.axvline(0, color="0.5", lw=0.8, ls=":")
        cV = axV.axvline(0, color="0.5", lw=0.8, ls=":")
        cT = axT.axvline(0, color="0.5", lw=0.8, ls=":")

        axA.set_xlim(0, t[-1]); axA.set_ylim(0, alt_max / 1e3 * 1.12)
        axA.set_ylabel("altitude [km]"); axA.legend(fontsize=7); axA.grid(alpha=0.3)
        axA.set_title("Altitude")
        v_top = max(truth[:, 1].max(), truth[:, 2].max()) * 1.08
        axV.set_xlim(0, t[-1]); axV.set_ylim(0, v_top)
        axV.axhline(self.m_dry, color="0.6", ls=":", lw=0.8)
        axV.legend(fontsize=7); axV.grid(alpha=0.3); axV.set_title("Climb rate & mass")
        axT.set_xlim(0, t[-1]); axT.set_ylim(-0.05, twr_all.max() * 1.12)
        axT.axhline(1.0, color="0.6", ls=":", lw=0.8)
        axT.set_xlabel("time [s]"); axT.set_ylabel("T / W")
        axT.set_title("TWR (rises as fuel burns)"); axT.grid(alpha=0.3)

        gbar = _fuel_gauge(axG, 1.0)
        txt  = axI.text(0.05, 0.95, "", transform=axI.transAxes, va="top",
                        family="monospace", fontsize=8.5)

        step = max(1, n // frames); frame_idx = list(range(0, n, step)) + [n - 1]

        def _draw(k):
            h, v, m = truth[k]
            burning = m > self.m_dry + 1e-6
            fl = 0.7 * Ld if burning else 0.0
            shp = _rocket_shape(0.0, h, 0.0, Ld, Wd, flame=fl, anchor="base")
            body.set_xy(shp["body"]); nose.set_xy(shp["nose"])
            flame.set_xy(shp.get("flame", np.zeros((3, 2))))
            la_t.set_data(t[:k+1], truth[:k+1, 0] / 1e3)
            la_e.set_data(t[:k+1], estx[:k+1, 0] / 1e3)
            if la_z is not None:
                la_z.set_offsets(np.column_stack([t[:k+1], meas[:k+1] / 1e3]))
            lv_v.set_data(t[:k+1], truth[:k+1, 1])
            lv_m.set_data(t[:k+1], truth[:k+1, 2])
            lt_t.set_data(t[:k+1], twr_all[:k+1])
            for cur in (cA, cV, cT): cur.set_xdata([t[k], t[k]])
            _set_gauge(axG, gbar, self._fuel_frac(m))
            ax.set_title("Liftoff    t = %.1f s" % t[k])
            txt.set_text("alt   %7.0f m\nspeed %6.0f m/s\nmass  %6.0f kg\nTWR   %5.2f" %
                         (h, v, m, self.thrust / max(m, self.m_dry) / self.g if burning else 0.0))
            return body, nose, flame, la_t, la_e, lv_v, lv_m, lt_t, txt

        return animate(fig, _draw, frames=frame_idx, interval=interval)


# ===========================================================================
#  3) Vertical landing burn — the SpaceX-style hoverslam (powered descent)
# ===========================================================================
class RocketLandingState(KalmanState):
    ALTITUDE = "m"
    VELOCITY = "m/s"
    MASS     = "kg"


class RocketLandingObservable(KalmanObservable):
    ALTITUDE = "m"


class RocketLandingModel(ExtendedKalmanModel):
    """
    Vertical **propulsive landing** of a booster — the SpaceX "hoverslam" / suicide burn.

    Same 1-D mass-burning dynamics as :class:`RocketLiftoffModel`, run the other way: the
    stage falls engine-off (``v < 0``), then a **closed-loop landing burn** ignites late
    and throttles to arrive at the pad near zero velocity — the late, single-shot burn
    that *is* a hoverslam (there is no fuel to hover)::

        ḣ = v
        v̇ = a_T(h,v,m) − g − ½ ρ(h) (C_d A) v|v| / m
        ṁ = −a_T·m / v_e            (propellant spent ∝ thrust; 0 once tanks are dry)

    The onboard guidance ``a_T`` (see :meth:`guidance`) holds the descent to a reference
    profile ``v_ref(h) = −√(2·decel·h) − v_touch`` that vanishes at the pad: above it the
    engine stays off and the booster free-falls; once it catches the profile the engine
    fires and tracks it down. A ground radar measures **altitude only**; the EKF recovers
    the unmeasured descent rate and the dwindling propellant. ``n = 3``, ``m = 1``,
    ``l = 0``. Seed it descending, e.g. ``initial_state([h0, -v0, m0])``.
    """

    NAME = "Rocket Landing (hoverslam / powered descent)"; VERSION = "1.0"

    @property
    def states(self):      return RocketLandingState
    @property
    def observables(self): return RocketLandingObservable

    def __init__(self, name="LANDING", *,
                 m0=1000.0,              # mass at the start of descent [kg] (incl. landing fuel)
                 m_dry=700.0,            # dry mass [kg] (no propellant left below this)
                 ve=2500.0,             # effective exhaust velocity [m/s]
                 thrust_max=30_000.0,    # maximum engine thrust [N]
                 drag_area=0.25,         # C_d·A [m²]
                 drag=True,              # include atmospheric drag
                 g=G0,                   # gravity [m/s²]
                 decel=12.0,             # reference deceleration of the descent profile [m/s²]
                 v_touch=1.0,            # target touchdown speed [m/s]
                 kp=1.5,                 # descent-rate tracking gain [1/s]
                 fs=10.0,                # radar sample rate [Hz]
                 track_m=20.0,           # altitude measurement std [m]
                 q_alt=1e-3, q_vel=1e-2, q_mass=1e-4,   # process-noise variances
                 **kw):
        self.dtype = np.float64
        self.m0 = float(m0)
        self.m_dry = float(m_dry)
        self.ve = float(ve)
        self.thrust_max = float(thrust_max)
        self.drag_area = float(drag_area)
        self.drag = bool(drag)
        self.g = float(g)
        self.decel = float(decel)
        self.v_touch = float(v_touch)
        self.kp = float(kp)

        self._dt = 1.0 / fs
        self.n, self.m, self.l = 3, 1, 0
        self.fs = fs

        Q = np.diag([q_alt, q_vel, q_mass]).astype(float)
        R = np.array([[track_m ** 2]], dtype=float)
        super().__init__(name, Q=Q, R=R, fs=fs, **kw)

    # --- atmosphere / guidance --------------------------------------------
    def _rho(self, h):
        return RHO0 * np.exp(-max(h, 0.0) / H_ATM)

    def guidance(self, x):
        """Commanded thrust acceleration ``a_T`` [m/s²] (≥ 0, upward) for state
        ``x = [h, v, m]``: zero while above the reference descent profile (free fall)
        or out of propellant, otherwise throttled to track it down to the pad."""
        h, v, m = np.asarray(x, float).reshape(-1)[:3]
        if m <= self.m_dry or h <= 0.0:
            return 0.0
        m_eff = max(m, self.m_dry)
        a_max = self.thrust_max / m_eff
        v_ref = -np.sqrt(2.0 * self.decel * max(h, 0.0)) - self.v_touch
        a_drag = -0.5 * self._rho(h) * self.drag_area * v * abs(v) / m_eff
        a_cmd = self.kp * (v_ref - v) + self.g - a_drag      # so that v̇ → kp·(v_ref − v)
        return float(np.clip(a_cmd, 0.0, a_max))

    def accel(self, x):
        """Net vertical acceleration [m/s²] for state ``x = [h, v, m]``."""
        return self._deriv(np.asarray(x, float).reshape(-1)[:3], 0.0)[1]

    # --- dynamics ----------------------------------------------------------
    def _deriv(self, s, t):
        h, v, m = s
        if h <= 0.0:                                # on the pad: frozen (landed)
            return np.zeros(3)
        a_T = self.guidance(s)
        m_eff = max(m, self.m_dry)
        a = a_T - self.g
        if self.drag:
            a -= 0.5 * self._rho(h) * self.drag_area * v * abs(v) / m_eff
        mdot = a_T * m_eff / self.ve if m > self.m_dry else 0.0
        return np.array([v, a, -mdot])

    def f(self, x, u=None, w=None, t=None):
        s = _rk4(np.asarray(x, float).reshape(-1), float(t or 0.0), self._dt, self._deriv)
        if s[0] <= 0.0:                             # touchdown: clamp to the pad
            s[0] = 0.0
            s[1] = 0.0
        s[2] = max(s[2], self.m_dry)                # never burn past the dry mass
        return s.reshape(-1, 1)

    def h(self, x, v=None, t=None):
        z = np.asarray(x, float).reshape(-1)[:1].reshape(1, 1)
        return z if v is None else z + np.asarray(v, float).reshape(1, 1)

    # --- landing metrics from a record ------------------------------------
    def metrics(self, record):
        """Headline landing figures from a sampled/filtered ``record`` (states
        ``[h, v, m]``): ignition time & altitude, touchdown speed, propellant burned and
        the peak deceleration of the landing burn."""
        t = np.asarray(record.time, float).reshape(-1)
        x = np.asarray(record.xk)
        h, v, mass = x[:, 0], x[:, 1], x[:, 2]
        burning = mass < self.m0 - 1e-6
        i_ig = int(np.argmax(burning)) if burning.any() else len(t) - 1
        i_td = int(np.argmin(np.abs(h)))            # closest to the pad
        a = np.array([self.accel(xi) for xi in x])
        return {
            "ignition time [s]":     float(t[i_ig]),
            "ignition altitude [m]": float(h[i_ig]),
            "touchdown speed [m/s]": float(abs(v[i_td])),
            "propellant burned [kg]": float(self.m0 - mass.min()),
            "max decel [g]":         float(np.nanmax(a) / G0),
        }

    # ==================================================================
    # Signature figure — the landing burn (override KalmanModel.plot).
    # ``record`` is the EKF estimate; ``sample`` the sampled truth+altitude fixes.
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     figsize=(17, 4.4), **kwargs):
        """Signature figure: altitude vs time (truth, EKF, noisy radar fixes), the descent
        rate flaring to zero at the pad alongside the dwindling mass, and the throttle
        (thrust/weight) showing the single late landing burn."""
        import matplotlib.pyplot as plt
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n     = min(len(truth), len(estx))
        truth, estx = truth[:n], estx[:n]
        t = np.arange(n) * self._dt
        meas = (np.asarray(sample.zk).reshape(-1) if sample is not None
                else np.asarray(record.zk).reshape(-1))
        meas = meas[:n] if meas.size else None
        # throttle = thrust / weight, recomputed from the estimated state
        a_T = np.array([self.guidance(xi) for xi in estx])
        throttle = a_T * np.clip(estx[:, 2], self.m_dry, None) / self.thrust_max

        fig, ax = plt.subplots(1, 3, figsize=figsize)
        if meas is not None:
            ax[0].scatter(t, meas, s=4, color="0.8", label="radar")
        ax[0].plot(t, truth[:, 0], color="crimson", lw=1.6, label="truth")
        ax[0].plot(t, estx[:, 0], "--", color="navy", lw=1.2, label="EKF")
        ax[0].axhline(0.0, color="0.6", ls=":", lw=0.8)
        ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("altitude [m]")
        ax[0].set_title("Powered descent"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[1].plot(t, truth[:, 1], color="crimson", lw=1.6, label="descent rate [m/s]")
        ax[1].plot(t, estx[:, 1], "--", color="navy", lw=1.0)
        ax[1].plot(t, truth[:, 2], color="darkorange", lw=1.4, label="mass [kg]")
        ax[1].axhline(self.m_dry, color="0.6", ls=":", lw=0.8)
        ax[1].set_xlabel("time [s]"); ax[1].set_title("Descent rate & mass")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        ax[2].plot(t, throttle, color="seagreen")
        ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("throttle (T / T_max)")
        ax[2].set_title("Landing burn (ignites late)"); ax[2].grid(alpha=0.3)
        fig.suptitle(f"{self.name()} — hoverslam to the pad", y=1.02)
        fig.tight_layout()
        return fig


# ===========================================================================
#  4) Planar powered descent — a rigid booster that steers itself to the pad
# ===========================================================================
def _rocket_shape(x, y, th, length, width, *, flame=0.0, nose=None, anchor="base"):
    """Corner arrays for a rocket drawn at ``(x, y)`` tilted ``th`` rad from vertical.

    Returns a dict of ``(N, 2)`` point arrays — ``body`` (rectangle), ``nose`` (triangle) and,
    when ``flame > 0``, a ``flame`` triangle out of the base. The body axis (nose direction)
    maps to ``(−sin θ, cos θ)``, so positive ``θ`` leans the vehicle toward +x. ``anchor="base"``
    (default) plants the **base** of the vehicle at ``(x, y)`` so it stands on the ground/pad at
    that point (the physical offset from the centre of mass); ``anchor="com"`` centres it instead.
    """
    hw = 0.5 * width
    nose = 1.5 * hw if nose is None else nose
    y0, y1 = (0.0, length) if anchor == "base" else (-0.5 * length, 0.5 * length)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    def world(pts):
        P = (R @ np.asarray(pts, float).T)
        return np.column_stack([P[0] + x, P[1] + y])
    parts = {
        "body": world([(-hw, y0), (hw, y0), (hw, y1), (-hw, y1)]),
        "nose": world([(-hw, y1), (hw, y1), (0.0, y1 + nose)]),
    }
    if flame > 0:
        parts["flame"] = world([(-0.6 * hw, y0), (0.6 * hw, y0), (0.0, y0 - flame)])
    return parts


def _add_rocket(ax, shape, *, color="0.25", zorder=4, alpha=1.0):
    """Add body/nose/(flame) polygons for ``shape`` to ``ax``; return the patch handles."""
    from matplotlib.patches import Polygon
    arts = []
    if "flame" in shape:
        arts.append(ax.add_patch(Polygon(shape["flame"], closed=True, fc="orange",
                                         ec="orangered", lw=0.6, alpha=0.9 * alpha, zorder=zorder - 1)))
    arts.append(ax.add_patch(Polygon(shape["body"], closed=True, fc=color, ec="k", lw=0.8,
                                      alpha=alpha, zorder=zorder)))
    arts.append(ax.add_patch(Polygon(shape["nose"], closed=True, fc=color, ec="k", lw=0.8,
                                      alpha=alpha, zorder=zorder)))
    return arts


def _fuel_gauge(ax, frac, *, label="Propellant"):
    """Render a vertical bar gauge on the **dedicated** axes ``ax`` (a side panel, not an inset
    over the trajectory); return the bar patch so its height can be updated each frame."""
    import matplotlib.pyplot as plt
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="0.92", ec="k", lw=1.0))
    frac = float(np.clip(frac, 0.0, 1.0))
    col = "limegreen" if frac > 0.5 else "orange" if frac > 0.2 else "crimson"
    bar = ax.add_patch(plt.Rectangle((0, 0), 1, frac, fc=col, ec="none"))
    ax.set_title("%s\n%.0f%%" % (label, 100 * frac), fontsize=7)
    ax._gauge_label = label
    return bar


def _set_gauge(ax, bar, frac):
    """Update a bar gauge on axes ``ax`` to ``frac`` (0..1), colour-coded by level."""
    frac = float(np.clip(frac, 0.0, 1.0))
    bar.set_height(frac)
    bar.set_color("limegreen" if frac > 0.5 else "orange" if frac > 0.2 else "crimson")
    ax.set_title("%s\n%.0f%%" % (getattr(ax, "_gauge_label", ""), 100 * frac), fontsize=7)


def _make_reticle(ax):
    """Configure the **dedicated** axes ``ax`` as a circular **attitude reticle** (artificial
    horizon): a fixed vehicle mark over a sky/ground horizon that rolls with the tilt. Drawn in
    its own side panel (never over the trajectory). Returns a handle dict for :func:`_set_reticle`."""
    import matplotlib.pyplot as plt
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    clip = plt.Circle((0, 0), 1.0, transform=ax.transData)
    sky = ax.add_patch(plt.Polygon([[0, 0]], closed=True, fc="#7ec8e3", ec="none"))
    gnd = ax.add_patch(plt.Polygon([[0, 0]], closed=True, fc="#9c6b3f", ec="none"))
    sky.set_clip_path(clip); gnd.set_clip_path(clip)
    hor = ax.plot([], [], color="white", lw=1.6)[0]
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec="k", lw=1.3, zorder=5))
    ax.plot([-0.55, -0.16], [0, 0], color="yellow", lw=2.6, zorder=6)   # fixed wings
    ax.plot([0.16, 0.55], [0, 0], color="yellow", lw=2.6, zorder=6)
    ax.plot([0], [0], marker="o", color="yellow", ms=3, zorder=6)
    return dict(ax=ax, sky=sky, gnd=gnd, hor=hor)


def _set_reticle(R, theta):
    """Roll the attitude reticle to tilt ``theta`` (rad from vertical)."""
    a = -float(theta)                                  # vehicle-frame: world rolls opposite
    c, s = np.cos(a), np.sin(a)
    d = np.array([c, s]) * 1.6                          # along-horizon (clipped to the circle)
    n = np.array([-s, c]) * 2.0                         # toward the sky
    R["sky"].set_xy([d, -d, -d + n, d + n])
    R["gnd"].set_xy([d, -d, -d - n, d - n])
    R["hor"].set_data([-d[0], d[0]], [-d[1], d[1]])
    R["ax"].set_title("tilt %+.0f°" % np.degrees(theta), fontsize=8)


class RocketDescentState(KalmanState):
    X     = "m"      # horizontal position (pad at 0)
    Y     = "m"      # altitude of the rocket's BASE (bottom) — touchdown is base at y = 0
    VX    = "m/s"
    VY    = "m/s"
    THETA = "rad"    # tilt from vertical (0 = upright)
    OMEGA = "rad/s"  # tilt rate
    MASS  = "kg"


class RocketDescentObservable(KalmanObservable):
    X     = "m"      # downrange radar
    Y     = "m"      # altimeter
    THETA = "rad"    # inertial-measurement-unit attitude


class RocketDescentModel(ExtendedKalmanModel):
    """
    Planar **powered descent & landing** of a rigid booster — the accurate, *steerable*
    cousin of :class:`RocketLandingModel` (which is 1-D vertical only).

    The vehicle is a **rigid body** (a rectangle of length ``L`` and width ``W``) with a
    **gimballed** engine. Mass burns as it fires. The state carries horizontal position and
    **orientation**, so the booster can tilt to kill sideways drift and must arrive **upright**
    on the pad::

        ẋ = vx                     v̇x = −(T/m) sinθ        − ½ρ C_dA |v| vx / m
        ẏ = vy                     v̇y =  (T/m) cosθ − g    − ½ρ C_dA |v| vy / m
        θ̇ = ω                      ω̇  =  T sinδ · ℓ / I        (engine gimbal torque)
        ṁ = −T / v_e

    An onboard **control system** (see :meth:`guidance`) closes three loops every step:
    a vertical throttle loop tracking the descent profile ``v_ref(y)``; an outer horizontal
    loop that commands a target tilt ``θ_des`` to null cross-range position/velocity; and an
    inner attitude loop that gimbals the engine (``δ``) to track ``θ_des`` and right the vehicle
    for touchdown. A radar/altimeter + IMU measure ``[x, y, θ]``; the EKF reconstructs the
    velocities, tilt rate and remaining propellant. ``n = 7``, ``m = 3``.

    **Control input.** The two actuators — engine **throttle** and **gimbal** (the side
    adjustment that steers the booster) — are exposed as the control vector
    ``u = [throttle ∈ [0,1], gimbal δ ∈ rad]`` when constructed with ``control=True``
    (``l = 2``): supply your own ``uk`` program, or call :meth:`autopilot` to roll the guidance
    law out into one (just like the Apollo ascent's thrust program). With the default
    ``control=False`` (``l = 0``) the same guidance runs **closed-loop inside** the model, so a
    plain ``sample()`` / ``filter()`` flies and lands it.
    """

    NAME = "Rocket Descent (2-D, gimballed, lands upright)"; VERSION = "1.0"

    @property
    def states(self):      return RocketDescentState
    @property
    def observables(self): return RocketDescentObservable

    def __init__(self, name="DESCENT", *,
                 m0=1000.0,              # mass at start of descent [kg]
                 m_dry=700.0,            # dry mass [kg]
                 ve=2800.0,             # exhaust velocity [m/s]
                 thrust_max=40_000.0,    # max engine thrust [N]  (TWR ≈ 4 at liftoff mass)
                 length=40.0,            # body length [m]  (rectangle, for inertia & drawing)
                 width=3.7,              # body width [m]
                 drag_area=2.0,          # C_d·A [m²]
                 drag=True,
                 g=G0,
                 decel=8.0,              # deceleration of the near-ground flare [m/s²]
                 v_cruise=55.0,          # capped descent speed away from the ground [m/s]
                 v_touch=1.5,            # target touchdown descent speed [m/s]
                 theta_max=0.35,         # max commanded tilt [rad] (~20°)
                 gimbal_max=0.20,        # max engine gimbal [rad] (~11°)
                 kp_v=0.7,               # vertical descent-rate gain [1/s]
                 kx=6.0e-3, kvx=4.5e-2,  # cross-range → desired-tilt gains (ωn≈0.24, ζ≈0.8)
                 kth=8.0, kom=4.0,       # attitude (tilt-tracking) gains
                 fs=20.0,                # sensor sample rate [Hz]
                 control=False,          # expose u=[throttle, gimbal] as a control input (l=2)
                 track_m=15.0, track_th=0.01,   # position [m] / attitude [rad] noise std
                 q_pos=1e-3, q_vel=1e-2, q_ang=1e-5, q_rate=1e-4, q_mass=1e-3,
                 **kw):
        self.dtype = np.float64
        self.m0, self.m_dry, self.ve = float(m0), float(m_dry), float(ve)
        self.thrust_max = float(thrust_max)
        self.length, self.width = float(length), float(width)
        self.drag_area, self.drag, self.g = float(drag_area), bool(drag), float(g)
        self.decel, self.v_cruise, self.v_touch = float(decel), float(v_cruise), float(v_touch)
        self.theta_max, self.gimbal_max = float(theta_max), float(gimbal_max)
        self.kp_v, self.kx, self.kvx = float(kp_v), float(kx), float(kvx)
        self.kth, self.kom = float(kth), float(kom)
        self.lever = 0.5 * self.length          # base-to-CoM lever arm [m]

        self._dt = 1.0 / fs
        self.n, self.m, self.l = 7, 3, (2 if control else 0)
        self.fs = fs
        Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_ang, q_rate, q_mass]).astype(float)
        R = np.diag([track_m ** 2, track_m ** 2, track_th ** 2]).astype(float)
        super().__init__(name, Q=Q, R=R, fs=fs, **kw)

    # --- atmosphere / inertia ---------------------------------------------
    def _rho(self, y):
        return RHO0 * np.exp(-max(y, 0.0) / H_ATM)

    def inertia(self, m):
        """Moment of inertia of the booster (slender rectangle) [kg·m²]."""
        return m * self.length ** 2 / 12.0

    # --- the onboard control system ---------------------------------------
    def guidance(self, s):
        """Closed-loop commands for state ``s = [x, y, vx, vy, θ, ω, m]``:
        returns ``(T, δ)`` — engine thrust [N] and gimbal angle [rad]."""
        x, y, vx, vy, th, om, m = np.asarray(s, float).reshape(-1)[:7]
        if m <= self.m_dry or y <= 0.0:
            return 0.0, 0.0
        # vertical loop: hold a capped descent speed that tapers to v_touch near the pad, so the
        # engine stays lit through the descent and attitude control is available throughout.
        v_desc = min(self.v_cruise, np.sqrt(2.0 * self.decel * max(y, 0.0)) + self.v_touch)
        v_ref = -v_desc
        a_vert = self.g + self.kp_v * (v_ref - vy)              # needed vertical accel
        T = m * a_vert / max(np.cos(th), 0.3)
        T = float(np.clip(T, 0.0, self.thrust_max))
        # outer horizontal loop: desired tilt to null cross-range (x>0 ⇒ lean +θ ⇒ thrust −x)
        th_des = float(np.clip(self.kx * x + self.kvx * vx, -self.theta_max, self.theta_max))
        # inner attitude loop: gimbal to track θ_des and damp the tilt rate
        delta = float(np.clip(self.kth * (th_des - th) - self.kom * om,
                              -self.gimbal_max, self.gimbal_max))
        return T, delta

    def _control_from_u(self, u):
        """Map a control input ``u = [throttle, gimbal]`` to physical ``(T, δ)`` (clamped)."""
        thr, delta = np.asarray(u, float).reshape(-1)[:2]
        return (float(np.clip(thr, 0.0, 1.0)) * self.thrust_max,
                float(np.clip(delta, -self.gimbal_max, self.gimbal_max)))

    # --- dynamics ----------------------------------------------------------
    def _deriv(self, s, t, ctrl=None):
        x, y, vx, vy, th, om, m = s
        if y <= 0.0:                                   # landed: frozen on the pad
            return np.zeros(7)
        T, delta = self.guidance(s) if ctrl is None else ctrl
        m_eff = max(m, self.m_dry)
        ax = -(T / m_eff) * np.sin(th)
        ay =  (T / m_eff) * np.cos(th) - self.g
        if self.drag:
            v = np.hypot(vx, vy)
            k = 0.5 * self._rho(y) * self.drag_area * v / m_eff
            ax -= k * vx
            ay -= k * vy
        domega = T * np.sin(delta) * self.lever / self.inertia(m_eff)
        mdot = -T / self.ve if m > self.m_dry else 0.0
        return np.array([vx, vy, ax, ay, om, domega, mdot])

    def f(self, x, u=None, w=None, t=None):
        # u supplied (control=True) ⇒ fly that throttle/gimbal command (zero-order hold over the
        # step); u absent ⇒ the autopilot closes the loop inside _deriv at each RK4 substage.
        ctrl = self._control_from_u(u) if (self.l > 0 and u is not None) else None
        deriv = lambda ss, tt: self._deriv(ss, tt, ctrl)
        s = _rk4(np.asarray(x, float).reshape(-1), float(t or 0.0), self._dt, deriv)
        if s[1] <= 0.0:                                # touchdown: clamp to the pad
            s[1] = 0.0
            s[2] = s[3] = s[5] = 0.0                   # zero velocities & tilt rate
        s[6] = max(s[6], self.m_dry)
        return s.reshape(-1, 1)

    def autopilot(self, x0, n, *, t0=0.0):
        """Roll the guidance law out into an explicit **control program**.

        Flies the closed-loop controller from ``x0`` for ``n`` steps and returns
        ``(uk, truth)`` where ``uk[i] = [throttle, gimbal]`` is the command applied at step ``i``
        (ready to pass to ``sample(uk=uk)`` / ``filter(zk, uk)`` on a ``control=True`` model) and
        ``truth`` is the resulting ``(n, 7)`` state history."""
        x = np.asarray(x0, float).reshape(-1).copy()
        uk = np.zeros((n, 2)); truth = np.zeros((n, self.n))
        for i in range(n):
            truth[i] = x
            T, delta = self.guidance(x)
            uk[i] = (T / self.thrust_max, delta)
            x = self.f(x, u=uk[i], t=t0 + i * self._dt).reshape(-1)
        return uk, truth

    def h(self, x, v=None, t=None):
        s = np.asarray(x, float).reshape(-1)
        z = np.array([[s[0]], [s[1]], [s[4]]])         # [x, y, θ]
        return z if v is None else z + np.asarray(v, float).reshape(3, 1)

    # --- landing metrics ---------------------------------------------------
    def metrics(self, record):
        """Headline landing figures: touchdown miss distance & speed, final tilt, ignition
        time/altitude and propellant burned, from a sampled/filtered ``record``."""
        t = np.asarray(record.time, float).reshape(-1)
        s = np.asarray(record.xk)
        y = s[:, 1]
        i_td = int(np.argmin(np.abs(y)))
        T = np.array([self.guidance(si)[0] for si in s])
        burning = T > 1e-6
        i_ig = int(np.argmax(burning)) if burning.any() else 0
        return {
            "touchdown miss [m]":    float(abs(s[i_td, 0])),
            "touchdown speed [m/s]": float(np.hypot(s[i_td, 2], s[i_td, 3])),
            "final tilt [deg]":      float(np.degrees(abs(s[i_td, 4]))),
            "ignition time [s]":     float(t[i_ig]),
            "ignition altitude [m]": float(s[i_ig, 1]),
            "propellant burned [kg]": float(self.m0 - s[:, 6].min()),
        }

    # ==================================================================
    # Signature figure — the booster drawn along its descent (static).
    # ==================================================================
    def _fuel_frac(self, m):
        return (float(m) - self.m_dry) / max(self.m0 - self.m_dry, 1e-9)

    @staticmethod
    def _touchdown_index(truth, n):
        """First index (≤ ``n``, +3 settle frames) at which the booster has reached the pad, so
        plots end at touchdown instead of trailing post-landing process-noise chatter."""
        landed = np.where(np.asarray(truth)[:n, 1] <= 0.5)[0]
        return min(n, int(landed[0]) + 3) if len(landed) else n

    def _draw_rocket(self, ax, s, *, scale, color="0.25"):
        x, y, _, _, th, _, _ = np.asarray(s, float).reshape(-1)[:7]
        T = self.guidance(s)[0]
        flame = scale * self.length * 0.8 * (T / self.thrust_max)
        flame = min(flame, max(y, 0.0))            # exhaust splashes on the pad, never below it
        _add_rocket(ax, _rocket_shape(x, y, th, scale * self.length, scale * self.width,
                                      flame=flame, anchor="base"), color=color)

    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     nmax=None, scale=6.0, n_pose=7, figsize=(15, 6.6), **kwargs):
        """Signature figure — matches the interactive layout: booster trajectory (left),
        position / attitude+gimbal / throttle time series (centre), compact propellant
        gauge + attitude reticle (narrow right column)."""
        import matplotlib.pyplot as plt
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n = min(len(truth), len(estx)) if nmax is None else int(nmax)
        n = self._touchdown_index(truth, n)
        truth, estx = truth[:n], estx[:n]
        t = np.arange(n) * self._dt
        meas = (np.asarray(sample.zk) if sample is not None
                else (np.asarray(record.zk) if np.size(record.zk) else None))
        meas = meas[:n] if meas is not None else None
        cmd = np.array([self.guidance(si) for si in truth])
        throttle, gimbal = cmd[:, 0] / self.thrust_max, cmd[:, 1]

        # 4-col layout: trajectory | position | att+gimbal | throttle | gauge+reticle(narrow)
        wr = [3.0, 1.8, 1.8, 0.75]
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(3, 4, width_ratios=wr, wspace=0.38, hspace=0.50)
        axT = fig.add_subplot(gs[:, 0])      # trajectory
        axA = fig.add_subplot(gs[0, 1:3])   # position
        axO = fig.add_subplot(gs[1, 1:3])   # attitude & gimbal
        axU = fig.add_subplot(gs[2, 1:3])   # throttle
        axG = fig.add_subplot(gs[0, 3])     # propellant gauge
        axR = fig.add_subplot(gs[1, 3])     # attitude reticle (final)
        axGI = fig.add_subplot(gs[2, 3]); axGI.axis("off")

        # --- trajectory ---
        if meas is not None:
            axT.scatter(meas[:, 0], meas[:, 1], s=5, color="0.8", label="radar", zorder=2)
        axT.plot(truth[:, 0], truth[:, 1], color="crimson", lw=1.6, label="truth")
        axT.plot(estx[:, 0], estx[:, 1], "--", color="navy", lw=1.0, label="EKF")
        for i in np.linspace(0, n - 1, n_pose).astype(int):
            self._draw_rocket(axT, truth[i], scale=scale)
        axT.axhline(0.0, color="0.4", lw=2)
        axT.plot([0], [0], "v", color="seagreen", ms=12, label="pad")
        axT.set_aspect("equal", adjustable="datalim"); axT.grid(alpha=0.3)
        axT.set_xlabel("cross-range x [m]"); axT.set_ylabel("altitude y [m]")
        axT.legend(loc="upper left", fontsize=8, framealpha=0.9)
        axT.set_title(f"{self.name()} — booster drawn ×{scale:g}")

        # --- time series ---
        axA.plot(t, truth[:, 1], color="crimson", lw=1.5, label="altitude [m]")
        axA.plot(t, truth[:, 0], color="purple",  lw=1.0, label="cross-range [m]")
        axA.axhline(0, color="0.6", ls=":", lw=0.8)
        axA.set_ylim(bottom=0)
        axA.legend(fontsize=8); axA.grid(alpha=0.3); axA.set_title("Position")

        axO.plot(t, np.degrees(truth[:, 4]), color="teal",       lw=1.5, label="tilt θ [°]")
        axO.plot(t, np.degrees(gimbal),       color="darkorange", lw=1.0, label="gimbal δ [°]")
        axO.axhline(0, color="0.6", ls=":", lw=0.8)
        axO.legend(fontsize=8); axO.grid(alpha=0.3); axO.set_title("Attitude & gimbal")

        axU.plot(t, throttle, color="seagreen"); axU.set_ylim(-0.05, 1.05)
        axU.grid(alpha=0.3); axU.set_title("Throttle"); axU.set_xlabel("time [s]")

        # --- gauge at final propellant fraction + reticle at final tilt ---
        _fuel_gauge(axG, self._fuel_frac(truth[-1, 6]))
        reticle = _make_reticle(axR)
        _set_reticle(reticle, truth[-1, 4])

        fig.tight_layout()
        return fig

    # ==================================================================
    # Interactive replay — the booster flies the descent live.
    # ==================================================================
    def _plot_interactive(self, record, sample=None, *, component=None, error=True,
                          nmax=None, scale=6.0, frames=120, interval=70,
                          figsize=(15, 6.6), **kwargs):
        """Interactive descent — same 4-col layout and figsize as the static: booster
        trajectory (left), growing position / attitude+gimbal / throttle time series
        (centre), compact attitude reticle + propellant gauge + live readout (narrow
        right column). Radar measurement positions are overlaid on the trajectory."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from ..helpers import animate

        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n = min(len(truth), len(estx)) if nmax is None else int(nmax)
        n = self._touchdown_index(truth, n)
        truth, estx = truth[:n], estx[:n]
        meas = (np.asarray(sample.zk) if sample is not None
                else (np.asarray(record.zk) if np.size(record.zk) else None))
        meas = meas[:n] if meas is not None else None
        t = np.arange(n) * self._dt
        cmd = np.array([self.guidance(si) for si in truth])
        throttle_all = cmd[:, 0] / self.thrust_max
        gimbal_all   = np.degrees(cmd[:, 1])

        xspan = max(np.abs(truth[:, 0]).max(), scale * self.length) * 1.3 + 30
        ytop  = truth[:, 1].max() * 1.08 + scale * self.length

        # Same 4-col layout as static
        wr = [3.0, 1.8, 1.8, 0.75]
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(3, 4, width_ratios=wr, wspace=0.38, hspace=0.50)
        ax  = fig.add_subplot(gs[:, 0])      # trajectory
        axA = fig.add_subplot(gs[0, 1:3])   # position
        axO = fig.add_subplot(gs[1, 1:3])   # attitude & gimbal
        axU = fig.add_subplot(gs[2, 1:3])   # throttle
        axR = fig.add_subplot(gs[0, 3])     # attitude reticle
        axG = fig.add_subplot(gs[1, 3])     # propellant gauge
        axI = fig.add_subplot(gs[2, 3]); axI.axis("off")  # live readout

        # --- trajectory ---
        ax.axhline(0.0, color="0.4", lw=2)
        ax.plot([0], [0], "v", color="seagreen", ms=12, label="pad")
        path_t = ax.plot([], [], color="crimson", lw=1.3, label="truth")[0]
        path_e = ax.plot([], [], "--", color="navy", lw=1.0, label="EKF")[0]
        meas_sc = ax.scatter([], [], s=6, color="0.8", label="radar") \
                  if meas is not None else None
        body  = Polygon(np.zeros((4, 2)), closed=True, fc="0.25", ec="k", lw=0.8, zorder=4)
        nose  = Polygon(np.zeros((3, 2)), closed=True, fc="0.25", ec="k", lw=0.8, zorder=4)
        flame = Polygon(np.zeros((3, 2)), closed=True, fc="orange", ec="orangered", alpha=0.9, zorder=3)
        for p in (flame, body, nose):
            ax.add_patch(p)
        ax.set_xlim(-xspan, xspan); ax.set_ylim(-0.04 * ytop, ytop)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("cross-range x [m]"); ax.set_ylabel("altitude y [m]")
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        # --- time-series: faint guide + growing animated lines ---
        axA.plot(t, truth[:, 1], color="crimson", lw=0.7, alpha=0.15)
        axA.plot(t, truth[:, 0], color="purple",  lw=0.7, alpha=0.15)
        axO.plot(t, np.degrees(truth[:, 4]), color="teal",       lw=0.7, alpha=0.15)
        axO.plot(t, gimbal_all,               color="darkorange", lw=0.7, alpha=0.15)
        axU.plot(t, throttle_all, color="seagreen", lw=0.7, alpha=0.15)
        if meas is not None:
            axA.scatter(t, meas[:, 1], s=3, color="0.8", alpha=0.35, zorder=2)  # altitude meas guide

        la1, = axA.plot([], [], color="crimson",    lw=1.5, label="altitude [m]")
        la2, = axA.plot([], [], color="purple",     lw=1.0, label="cross-range [m]")
        lo1, = axO.plot([], [], color="teal",       lw=1.5, label="tilt θ [°]")
        lo2, = axO.plot([], [], color="darkorange", lw=1.0, label="gimbal δ [°]")
        lu,  = axU.plot([], [], color="seagreen",   lw=1.5)
        for axi in (axA, axO): axi.axhline(0, color="0.6", ls=":", lw=0.8)
        cA = axA.axvline(0, color="0.5", lw=0.8, ls=":")
        cO = axO.axvline(0, color="0.5", lw=0.8, ls=":")
        cU = axU.axvline(0, color="0.5", lw=0.8, ls=":")

        axA.set_xlim(0, t[-1]); axA.set_ylim(bottom=0)
        axA.legend(fontsize=7); axA.grid(alpha=0.3); axA.set_title("Position")
        axO.set_xlim(0, t[-1]); axO.legend(fontsize=7); axO.grid(alpha=0.3)
        axO.set_title("Attitude & gimbal")
        axU.set_xlim(0, t[-1]); axU.set_ylim(-0.05, 1.05)
        axU.grid(alpha=0.3); axU.set_title("Throttle"); axU.set_xlabel("time [s]")

        # --- side column ---
        reticle = _make_reticle(axR)
        gbar    = _fuel_gauge(axG, 1.0)
        txt = axI.text(0.05, 0.95, "", transform=axI.transAxes, va="top",
                       family="monospace", fontsize=8.5)

        step = max(1, n // frames); frame_idx = list(range(0, n, step)) + [n - 1]

        def _draw(k):
            path_t.set_data(truth[:k+1, 0], truth[:k+1, 1])
            path_e.set_data(estx[:k+1, 0], estx[:k+1, 1])
            if meas_sc is not None:
                meas_sc.set_offsets(meas[:k+1, :2])
            x, y, vx, vy, th, om, mss = truth[k]
            T, delta = cmd[k]
            fl = min(scale * self.length * 0.8 * (T / self.thrust_max), max(y, 0.0))
            shp = _rocket_shape(x, y, th, scale * self.length, scale * self.width,
                                flame=fl, anchor="base")
            body.set_xy(shp["body"]); nose.set_xy(shp["nose"])
            flame.set_xy(shp.get("flame", np.zeros((3, 2))))
            la1.set_data(t[:k+1], truth[:k+1, 1])
            la2.set_data(t[:k+1], truth[:k+1, 0])
            lo1.set_data(t[:k+1], np.degrees(truth[:k+1, 4]))
            lo2.set_data(t[:k+1], gimbal_all[:k+1])
            lu.set_data(t[:k+1], throttle_all[:k+1])
            for cur in (cA, cO, cU): cur.set_xdata([t[k], t[k]])
            _set_reticle(reticle, th)
            _set_gauge(axG, gbar, self._fuel_frac(mss))
            ax.set_title("Powered descent    t = %.1f s" % t[k])
            txt.set_text("alt   %7.1f m\nx     %7.1f m\nspeed %6.1f m/s\n"
                         "tilt  %6.1f°\nthrot %6.0f%%" %
                         (y, x, np.hypot(vx, vy), np.degrees(th), 100 * T / self.thrust_max))
            return path_t, path_e, flame, body, nose, la1, la2, lo1, lo2, lu, txt

        return animate(fig, _draw, frames=frame_idx, interval=interval)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (the model family was renamed for clarity:
# Launch→Ascent, Takeoff→Liftoff). Existing notebooks/scripts keep working.
# ---------------------------------------------------------------------------
RocketLaunchModel = RocketAscentModel
RocketTakeoffModel = RocketLiftoffModel

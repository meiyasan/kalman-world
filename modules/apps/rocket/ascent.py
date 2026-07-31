"""Rocket ascent — powered gravity-turn climb to orbit."""
from __future__ import annotations
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
import numpy as np

from ...framework.models.extended import ExtendedKalmanModel
from ...framework.constants import KalmanState, KalmanObservable
from ...runtime.astro import (LU, TU, KM, GM_EARTH, R_EARTH, H_SCALE,
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
                          frames=30, interval=70, figsize=(15, 5.6), uk=None, **kwargs):
        """Interactive ascent — same 3-col layout as the static: animated rocket on the
        gravity-turn trajectory (left), growing altitude+speed / EKF error / optional thrust
        control time series (centre), compact attitude reticle + live readout (narrow right).
        Radar measurement positions are overlaid on the trajectory. Pass ``uk`` to show the
        thrust acceleration control programme."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from ...helpers import animate

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



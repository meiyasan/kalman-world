"""Rocket liftoff — the first seconds off the pad."""
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

    def _plot_interactive(self, record, sample=None, *, nmax=None, frames=30, interval=70,
                          figsize=(15, 6.2), **kwargs):
        """Interactive liftoff — same 4-col layout as the static: animated rocket track
        (left), growing altitude / climb-rate+mass / TWR time series (centre), compact
        propellant gauge + live readout (narrow right column)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from ...helpers import animate

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



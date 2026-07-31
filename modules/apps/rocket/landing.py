"""Rocket landing — powered descent and touchdown."""
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



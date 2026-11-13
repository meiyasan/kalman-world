"""Rocket descent — re-entry and guided descent (autopilot)."""
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

from ....framework.models.extended import ExtendedKalmanModel
from ....framework.models.native import NativeTwin, native_integrator_kind
from ....framework.constants import KalmanState, KalmanObservable, KalmanDynamicsKind
from ....runtime.astro import (LU, TU, KM, GM_EARTH, R_EARTH, H_SCALE,
                             G0, RHO0, H_ATM, rk4 as _rk4)



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


class RocketDescentModel(NativeTwin, ExtendedKalmanModel):
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
        # L = I: Q enters additively on every state. f() does not route an
        # additive `w` through, so without it the engine would difference f with
        # respect to w, find no sensitivity, and multiply Q by a zero matrix at
        # every prediction: the covariance then collapses to zero.
        # Start fuelled. The base default is a state of zeros, which for this
        # model means no mass at all: the engine would cut off in the first step.
        if kw.get("x0") is None:
            x0 = np.zeros((7, 1))
            x0[6, 0] = self.m0
            kw["x0"] = x0
        super().__init__(name, Q=Q, R=R, L=np.eye(self.n), fs=fs, **kw)

    # --- C++ twin (libKalmanApps: Kalman::Apps::Astro::RocketDescent) ------
    # Both configurations have one: the guidance flying the booster (l = 0,
    # "rocket-descent") and the throttle/gimbal program as the control input
    # (control=True, l = 2, "rocket-descent-control"). The atmosphere, the
    # inertia, the guidance, the command mapping and _deriv are the dynamics:
    # a subclass that changes one of them is another model.
    _TWIN_CONTRACT = NativeTwin._TWIN_CONTRACT + ("_deriv", "_rho", "inertia", "guidance", "_control_from_u")

    def native_name(self):
        if native_integrator_kind(self) is None:
            return None
        return "rocket-descent-control" if self.l > 0 else "rocket-descent"

    def native_parameters(self):
        return {"m_dry": self.m_dry, "ve": self.ve, "thrust_max": self.thrust_max,
                "length": self.length, "lever": self.lever, "drag_area": self.drag_area,
                "drag": float(self.drag), "g": self.g, "decel": self.decel, "v_cruise": self.v_cruise,
                "v_touch": self.v_touch, "theta_max": self.theta_max, "gimbal_max": self.gimbal_max,
                "kp_v": self.kp_v, "kx": self.kx, "kvx": self.kvx, "kth": self.kth, "kom": self.kom,
                "rho0": RHO0, "h_atm": H_ATM,
                "substeps": float(self.substeps()), "integrator": float(native_integrator_kind(self))}

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
        # vertical loop: hold a capped descent speed, then brake at the constant deceleration
        # `decel` along v² = v_touch² + 2·decel·y, which reaches v_touch at the pad. On that
        # braking arc the rate of change of the reference, dv_ref/dt = decel·(−vy)/v_brake,
        # is fed forward: feedback on the speed error alone lags the profile and would hit
        # the pad at a large fraction of v_cruise.
        v_brake = np.sqrt(self.v_touch ** 2 + 2.0 * self.decel * max(y, 0.0))
        braking = v_brake < self.v_cruise
        v_ref = -(v_brake if braking else self.v_cruise)
        a_ff = self.decel * max(-vy, 0.0) / v_brake if braking else 0.0
        a_vert = self.g + a_ff + self.kp_v * (v_ref - vy)       # needed vertical accel
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

    DYNAMICS = KalmanDynamicsKind.CONTINUOUS      # f is ds/dt; the framework integrates it (RK4)

    def f(self, x, u=None, w=None, t=None):
        # ds/dt. u supplied (control=True) ⇒ fly that throttle/gimbal command (zero-order hold over the
        # step); u absent ⇒ the autopilot closes the loop inside _deriv at each RK4 substage.
        ctrl = self._control_from_u(u) if (self.l > 0 and u is not None) else None
        return self._deriv(np.asarray(x, float).reshape(-1), float(t or 0.0), ctrl).reshape(np.shape(x))

    def after_step(self, x, u=None, t=None):
        s = np.asarray(x, float).reshape(-1)
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
        s = np.asarray(record.xk).reshape(len(t), -1)
        y = s[:, 1]
        # Touchdown is the last sample still in the air. Once on the pad, f() zeroes the
        # velocities, and process noise can relight the engine, so later samples would
        # report a zero touchdown speed and propellant burned after landing.
        landed = np.where(y <= 0.5)[0]
        i_td = max(int(landed[0]) - 1, 0) if len(landed) else int(np.argmin(np.abs(y)))
        T = np.array([self.guidance(si)[0] for si in s])
        burning = T > 1e-6
        i_ig = int(np.argmax(burning)) if burning.any() else 0
        return {
            "touchdown miss [m]":    float(abs(s[i_td, 0])),
            "touchdown speed [m/s]": float(np.hypot(s[i_td, 2], s[i_td, 3])),
            "final tilt [deg]":      float(np.degrees(abs(s[i_td, 4]))),
            "ignition time [s]":     float(t[i_ig]),
            "ignition altitude [m]": float(s[i_ig, 1]),
            "propellant burned [kg]": float(s[0, 6] - s[i_td, 6]),
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
                          nmax=None, scale=6.0, frames=30, interval=70,
                          figsize=(15, 6.6), **kwargs):
        """Interactive descent — same 4-col layout and figsize as the static: booster
        trajectory (left), growing position / attitude+gimbal / throttle time series
        (centre), compact attitude reticle + propellant gauge + live readout (narrow
        right column). Radar measurement positions are overlaid on the trajectory."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from ....helpers import animate

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

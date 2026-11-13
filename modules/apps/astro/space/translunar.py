"""Trans-lunar injection and cislunar cruise (Earth-Moon)."""
from __future__ import annotations
"""
Space science — travelling through space (the companion of :mod:`kalman.apps.rocket`,
which handles takeoff and landing near the pad).

Two regimes, both tracked with an Extended Kalman Filter:

* **Fly to the Moon** — :class:`TransLunarModel`: a geocentric restricted four-body coast
  (Earth + Moon + Sun), the Apollo free-return and its lunar **gravitational catapult**.
  With ``use_ephemeris=True`` the Moon/Sun starting geometry is taken from the real
  ephemeris (astropy) at a chosen ``epoch``.

* **Interplanetary travel** — :class:`InterplanetaryModel`: a heliocentric **N-body** cruise
  (Sun + planetary perturbations) general over any *departure → target* planet (Mars is the
  worked example). Built up **gradually**:

    1. :func:`hohmann_transfer` — analytic two-impulse estimate (Δv, time of flight, phasing,
       synodic period, next ideal departure);
    2. :func:`lambert` / :func:`porkchop` / :func:`best_departure` — Lambert targeting and a
       porkchop scan over the live ephemeris to find the **minimum-Δv launch window**, refined
       to a time-of-day at a real launch site (Earth rotation);
    3. :class:`InterplanetaryModel` — the full N-body EKF cruise, seeded by the Lambert
       solution, with impulsive mid-course Δv maneuvers as the control input.

Positions are in scaled units (Mm), velocities in km/s; physical constants and the ephemeris
come from :mod:`kalman.runtime.astro`.
"""
import numpy as np

from ....framework.models.extended import ExtendedKalmanModel
from ....framework.constants import KalmanState, KalmanObservable, KalmanDynamicsKind
from ....framework.control import KalmanController
from ....framework.models.native import NativeTwin
from ....runtime import astro as _astro
from ....runtime.astro import (
    LU, TU, KM, DAY, GM_EARTH, GM_MOON, GM_SUN, R_EARTH, R_MOON, D_MOON, D_SUN,
    G0, gm, radius, rk4 as _rk4, body_state, planet_positions, ephemeris_samples,
    times_from_epoch, launch_site_state, moon_position, sun_position, _as_time,
)

_PLANETS = ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")

# (colour, marker size) per planet — used by plot_solar_system for a non-naive layout.
_PLANET_STYLE = {
    "mercury": ("#9c9c9c", 5), "venus": ("#d9a066", 7), "earth": ("#3b6fb6", 7),
    "mars": ("#c1440e", 6), "jupiter": ("#caa472", 12), "saturn": ("#e0c47a", 11),
    "uranus": ("#9fd6e0", 9), "neptune": ("#5b7fd6", 9),
}
_BODY_COLOR = {
    "sun": "gold", "mercury": "#9c9c9c", "venus": "#d9a066", "earth": "steelblue",
    "mars": "#c1440e", "jupiter": "#caa472", "saturn": "#e0c47a",
    "uranus": "#9fd6e0", "neptune": "#5b7fd6", "moon": "#aaa",
}
_BODY_MS = {
    "sun": 20, "mercury": 5, "venus": 9, "earth": 9, "moon": 5,
    "mars": 7, "jupiter": 16, "saturn": 14, "uranus": 11, "neptune": 11,
}

# Leg colours for multi-leg trajectories (cycles if more than 6 legs)
_LEG_COLORS = ["royalblue", "darkorange", "limegreen", "crimson", "violet", "gold"]


def _kep_arc(r0, v0, tof_ks, n=200, mu=None):
    """Two-body RK4 arc: returns (n+1, 3) position array in Mm."""
    if mu is None:
        mu = GM_SUN
    def _deriv(s, t): return np.r_[s[3:], -mu * s[:3] / np.linalg.norm(s[:3]) ** 3]
    s = np.concatenate([r0, v0]); dt = tof_ks / n
    pts = [s[:3].copy()]
    for _ in range(n):
        s = _rk4(s, 0, dt, _deriv)
        pts.append(s[:3].copy())
    return np.array(pts)


def _orbit_ellipse(body, t_ref=None, n=300):
    """True heliocentric orbit path for *body* (one full period, two-body RK4).

    Returns an (n+1, 2) array of (x, y) in AU, or ``None`` for hyperbolic orbits.
    Using the actual ephemeris position+velocity avoids the circular-orbit approximation
    and gives the correct elliptical shape (eccentricity, inclination projection).
    """
    r0, v0 = body_state(body, t_ref)
    r0_mag = np.linalg.norm(r0)
    v0_sq  = float(np.dot(v0, v0))
    denom  = 2.0 * GM_SUN / r0_mag - v0_sq
    if denom <= 0:
        return None
    a    = GM_SUN / denom
    T_ks = 2.0 * np.pi * np.sqrt(a ** 3 / GM_SUN)
    pts  = _kep_arc(r0, v0, T_ks, n=n)
    return pts[:, :2] / D_SUN


# ===========================================================================
#  Geocentric gravity helpers (shared by the trans-lunar model)
# ===========================================================================
def _third_body(r, r_body, GM):
    """Third-body perturbation in a frame centred on the primary: ``GM[(r_b-r)/|r_b-r|³ −
    r_b/|r_b|³]``. The first term is the body's direct pull on the spacecraft; the second
    (indirect) subtracts its pull on the central body, so a distant body contributes only its
    true (tiny) tidal effect rather than a spurious bulk force."""
    d = r_body - r
    return GM * (d / np.linalg.norm(d) ** 3 - r_body / np.linalg.norm(r_body) ** 3)


def gravity(r, t, moon_phase=0.0, sun_phase=0.0, bodies=("earth", "moon", "sun")):
    """Geocentric acceleration on a spacecraft at ``r`` (scaled): Earth's central pull plus
    the **third-body perturbations** of the Moon and Sun (analytic circular orbits)."""
    r = np.asarray(r, dtype=float).reshape(3)
    a = np.zeros(3)
    if "earth" in bodies:
        a += -GM_EARTH * r / np.linalg.norm(r) ** 3
    if "moon" in bodies:
        a += _third_body(r, moon_position(t, moon_phase), GM_MOON)
    if "sun" in bodies:
        a += _third_body(r, sun_position(t, sun_phase), GM_SUN)
    return a


# The C++ twins (libKalmanApps) integrate with the built-in Euler, RK4 or default
# Dormand-Prince; any other scheme, or a subclass of one, keeps the Python physics.
_NATIVE_INTEGRATORS = {"euler": 0, "rk4": 1, "dormand_prince": 2}

# What a continuous space model's twin stands in for: the framework's contract plus
# the hooks its decoded step is made of.
_SPACE_TWIN_CONTRACT = NativeTwin._TWIN_CONTRACT + (
    "derivative", "acceleration", "substeps", "after_step", "decode_step")


def _native_integrator(model):
    """The C++ IntegratorKind of ``model``'s integrator, or None when it has no twin."""
    from ....framework.integrator import KalmanIntegrator, _BUILTIN_STEPS
    integrator = model.default_integrator()
    name = getattr(integrator, "NATIVE", None)
    kind = _NATIVE_INTEGRATORS.get(name)
    if kind is None or type(integrator)._step is not _BUILTIN_STEPS.get(name) \
            or type(integrator).step is not KalmanIntegrator.step:
        return None
    if kind == 2 and (integrator.rtol, integrator.atol, integrator.max_substeps) != (1e-6, 1e-9, 10_000):
        return None
    return kind


# ===========================================================================
#  1) Trans-lunar coast — the gravitational catapult (Earth + Moon + Sun)
# ===========================================================================

class TransLunarState(KalmanState):
    X = "Mm"; Y = "Mm"; Z = "Mm"; VX = "km/s"; VY = "km/s"; VZ = "km/s"


class TransLunarObservable(KalmanObservable):
    X = "Mm"; Y = "Mm"; Z = "Mm"


class TransLunarModel(NativeTwin, ExtendedKalmanModel):
    """
    Restricted four-body trans-lunar model (Earth + Moon + Sun).

    State ``x = [rx, ry, rz, vx, vy, vz]`` (position in Mm, velocity in km/s); only the
    position is measured (ground-station tracking). ``f()`` is one RK4 step of the multi-body
    gravity ODE; the Moon and Sun ride on circular orbits. Pass ``use_ephemeris=True`` with an
    ``epoch`` (anything :func:`astropy.time.Time` accepts, or ``None`` = now) to seed the Moon
    and Sun **phases from the real ephemeris** at that date, so the flyby geometry matches a
    genuine launch window while keeping the fast analytic propagation.
    """

    NAME = "Trans-Lunar (Earth+Moon+Sun)"; VERSION = "1.0"

    @property
    def states(self):      return TransLunarState
    @property
    def observables(self): return TransLunarObservable

    def __init__(self, moon_phase_deg=125.0, sun_phase_deg=60.0, dt_s=120.0,
                 track_km=5.0, bodies=("earth", "moon", "sun"), n_sub=1,
                 use_ephemeris=False, epoch=None, q_pos=1e-6, q_vel=1e-7, **kw):
        self.dtype = np.float64
        if use_ephemeris:                              # real Moon/Sun geometry at the epoch
            rm, _ = body_state("moon", epoch, origin="geocentric")
            rs, _ = body_state("sun",  epoch, origin="geocentric")
            moon_phase_deg = np.degrees(np.arctan2(rm[1], rm[0]))
            sun_phase_deg  = np.degrees(np.arctan2(rs[1], rs[0]))
        self.use_ephemeris = bool(use_ephemeris)
        self.epoch = epoch
        self.moon_phase = np.deg2rad(moon_phase_deg)
        self.sun_phase  = np.deg2rad(sun_phase_deg)
        self.bodies     = tuple(bodies)
        self._dt        = dt_s / TU
        self.n_sub      = max(1, int(n_sub))   # internal RK4 substeps per f() call
        self.n, self.m, self.l = 6, 3, 0
        self.fs = 1.0 / self._dt
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(float)
        R = (track_km * KM) ** 2 * np.eye(3)
        super().__init__("ROCKET", Q=Q, R=R, fs=self.fs, **kw)

    # ------------------------------------------------------------------
    # C++ twin (libKalmanApps: Kalman::Apps::Astro::TransLunar). The Moon and the
    # Sun ride on circular orbits given by a formula, so every configuration has
    # one, ephemeris-seeded phases included; another integrator stays in Python.
    # ------------------------------------------------------------------
    _TWIN_CONTRACT = _SPACE_TWIN_CONTRACT

    def native_name(self):
        if _native_integrator(self) is None:
            return None
        return "translunar"

    def native_parameters(self):
        return {"gm_earth": GM_EARTH, "gm_moon": GM_MOON, "gm_sun": GM_SUN,
                "d_moon": D_MOON, "d_sun": D_SUN,
                "omega_moon": _astro.OMEGA_MOON, "omega_sun": _astro.OMEGA_SUN,
                "moon_phase": self.moon_phase, "sun_phase": self.sun_phase,
                "earth": float("earth" in self.bodies), "moon": float("moon" in self.bodies),
                "sun": float("sun" in self.bodies),
                "substeps": float(self.n_sub), "integrator": float(_native_integrator(self))}

    # --- dynamics ---
    def acceleration(self, r, t):
        return gravity(r, t, self.moon_phase, self.sun_phase, self.bodies)

    DYNAMICS = KalmanDynamicsKind.CONTINUOUS      # f is ds/dt; the framework integrates it (RK4)

    def f(self, x, u=None, w=None, t=None):
        s = np.asarray(x, float).reshape(-1)
        return np.concatenate([s[3:], self.acceleration(s[:3], float(t or 0.0))]).reshape(np.shape(x))

    def substeps(self, x=None):
        # n_sub integrator steps per measurement, so a close lunar flyby (steep gravity
        # gradient) is resolved although the filter only sees a position fix every dt_s.
        return self.n_sub

    def h(self, x, v=None, t=None):
        z = np.asarray(x, float).reshape(-1)[:3].reshape(3, 1)
        return z if v is None else z + np.asarray(v, float).reshape(3, 1)

    # --- helpers for analysis / plotting ---
    def moon_xyz(self, t):  return moon_position(t, self.moon_phase)
    def sun_xyz(self, t):   return sun_position(t, self.sun_phase)

    def moon_track(self, times):
        """Moon (x,y,z) over an array of times (scaled units)."""
        return np.array([self.moon_xyz(t) for t in np.atleast_1d(times)])

    def body_accelerations(self, r, t):
        """Geocentric acceleration magnitudes at ``r``, ``t`` (m/s²): Earth's direct pull,
        and the Moon's / Sun's **third-body perturbation** (the Sun's is its true tidal value)."""
        r = np.asarray(r, float).reshape(3)
        return {
            "earth": np.linalg.norm(GM_EARTH * r / np.linalg.norm(r) ** 3),
            "moon":  np.linalg.norm(_third_body(r, self.moon_xyz(t), GM_MOON)),
            "sun":   np.linalg.norm(_third_body(r, self.sun_xyz(t), GM_SUN)),
        }

    # ==================================================================
    # Signature figure — geocentric trajectory (override KalmanModel.plot).
    # ``record`` is the EKF estimate; ``sample`` the sampled truth+measurements.
    # Reached via ``model.plot(record, sample=sample[, interactive=True, nmax=K])``.
    # ==================================================================
    def _trajectory_arrays(self, record, sample, nmax):
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        meas  = (np.asarray(sample.zk) if sample is not None
                 else (np.asarray(record.zk) if np.size(record.zk) else None))
        K = int(nmax or len(truth))
        meas = meas[:K] if meas is not None else None
        return truth[:K], estx[:K], meas, K

    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     nmax=None, scale=12.0, kepler=None, figsize=(8.6, 8.4), **kwargs):
        """Signature figure: the geocentric trajectory in the x–y plane — truth, EKF
        reconstruction and noisy fixes, with Earth (exaggerated ×``scale``), the Moon's
        orbit and the **lunar flyby** (closest approach) starred. Pass ``kepler`` — an
        Earth-only counterfactual trajectory (e.g. from a ``bodies=("earth",)`` twin model)
        — to overlay where the craft *would* drift **without the Moon**, making the
        gravitational catapult visible. ``nmax`` truncates the path (e.g. at re-entry)."""
        import matplotlib.pyplot as plt
        truth, estx, meas, K = self._trajectory_arrays(record, sample, nmax)
        moon  = self.moon_track(np.arange(K) * self._dt)
        i_fly = int(np.linalg.norm(truth[:, :3] - moon, axis=1).argmin())  # closest approach
        kep   = None if kepler is None else np.asarray(getattr(kepler, "xk", kepler))
        th    = np.linspace(0, 2 * np.pi, 240)
        reach = [np.abs(truth[:, :2]).max(), D_MOON]
        if kep is not None:
            reach.append(np.abs(kep[:, :2]).max())
        lim = 1.08 * max(reach)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(D_MOON * np.cos(th), D_MOON * np.sin(th), ":", color="0.6", lw=0.7, label="Moon orbit")
        ax.add_patch(plt.Circle((0, 0), R_EARTH * scale, color="steelblue"))
        if meas is not None:
            ax.scatter(meas[:, 0], meas[:, 1], s=3, color="0.8", label="measured")
        if kep is not None:
            ax.plot(kep[:, 0], kep[:, 1], "-", color="0.55", lw=1.4, label="without Moon (Keplerian)")
        ax.plot(truth[:, 0], truth[:, 1], color="crimson", lw=1.7,
                label=("with Moon (truth)" if kep is not None else "truth"))
        ax.plot(estx[:, 0], estx[:, 1], "--", color="navy", lw=1.0, label="EKF reconstruction")
        ax.plot([moon[i_fly, 0]], [moon[i_fly, 1]], "o", color="dimgray", ms=13, label="Moon at flyby")
        ax.plot([truth[i_fly, 0]], [truth[i_fly, 1]], "*", color="crimson", ms=16, zorder=5)
        ax.annotate("lunar flyby\n(slingshot)", xy=(truth[i_fly, 0], truth[i_fly, 1]),
                    xytext=(truth[i_fly, 0] - 0.34 * lim, truth[i_fly, 1] + 0.15 * lim),
                    fontsize=9.5, arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("x [Mm = 1000 km]"); ax.set_ylabel("y [Mm]"); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8.5)
        ax.set_title(f"{self.name()} — lunar gravitational catapult")
        return fig

    def _plot_interactive(self, record, sample=None, *, component=None, error=True,
                          nmax=None, frames=30, interval=100, scale=12.0,
                          kepler=None, figsize=(8.4, 8.2), **kwargs):
        """Interactive **lunar-catapult** replay: the spacecraft flies its free return while the
        per-body **perturbing** accelerations (Earth central pull + Moon/Sun third-body tidal
        terms) are drawn as live vectors with a numeric read-out. Pass ``kepler`` (the Earth-only
        twin, e.g. from a ``bodies=("earth",)`` model) to fly the **without-Moon** path in lock-step
        beside it — you watch the two diverge at the flyby, which *is* the catapult."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from ....helpers import animate
        truth, estx, meas, K = self._trajectory_arrays(record, sample, nmax)
        kep    = None if kepler is None else np.asarray(getattr(kepler, "xk", kepler))
        # Auto-extend the Keplerian (Earth-only) path until it returns to periapsis, capped at
        # 30 days.  The trans-lunar ellipse has a period of ~18-20 days (apoapsis ≫ D_MOON),
        # so the caller's sampled length is almost always shorter than the return trip.
        if kep is not None and np.linalg.norm(kep[-1, :3]) > 1.4 * R_EARTH:
            _deriv_E = lambda s, tt: np.concatenate(
                [s[3:], -GM_EARTH * s[:3] / np.linalg.norm(s[:3]) ** 3])
            s = kep[-1].copy(); n0 = len(kep)
            extras = []
            for i in range(max(0, int(30 * DAY / self._dt) - n0)):
                s = _rk4(s, (n0 + i) * self._dt, self._dt, _deriv_E)
                extras.append(s.copy())
                if np.linalg.norm(s[:3]) < 1.4 * R_EARTH:    # periapsis / re-entry
                    break
            if extras:
                kep = np.vstack([kep, np.array(extras)])
        # Run the replay over the LONGER of the two flights: the catapult craft is home at K,
        # but the without-Moon ellipse loops out and back for much longer — keep animating so its
        # complete reference trajectory is drawn while the real craft already sits at Earth.
        Kanim  = max(K, len(kep)) if kep is not None else K
        t      = np.arange(Kanim) * self._dt
        moon   = self.moon_track(t)                                          # Moon orbits the whole time
        i_fly  = int(np.linalg.norm(truth[:, :3] - moon[:K], axis=1).argmin())   # closest approach
        t_days = t / (86400.0 / TU)
        th     = np.linspace(0, 2 * np.pi, 200)
        reach  = [np.abs(truth[:, :2]).max(), D_MOON]
        if kep is not None:
            reach.append(np.abs(kep[:, :2]).max())
        lim    = 1.08 * max(reach)
        step   = max(1, Kanim // frames)
        frame_idx = list(range(0, Kanim, step)) + [Kanim - 1]

        # Perturbing geocentric accelerations on the catapult craft: Earth central + Moon/Sun
        # third-body (only defined while it is in flight, i.e. up to splashdown at K).
        r   = truth[:, :3]
        sun = np.array([self.sun_xyz(s) for s in t[:K]])
        rE  = np.linalg.norm(r, axis=1)
        vM  = np.array([_third_body(r[i], moon[i], GM_MOON) for i in range(K)])
        vS  = np.array([_third_body(r[i], sun[i],  GM_SUN ) for i in range(K)])
        gE  = GM_EARTH / rE ** 2;           uE = -r[:, :2] / rE[:, None]
        gM  = np.linalg.norm(vM, axis=1);   uM = vM[:, :2] / gM[:, None]
        gS  = np.linalg.norm(vS, axis=1);   uS = vS[:, :2] / gS[:, None]
        Lmax = 0.20 * lim

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(D_MOON * np.cos(th), D_MOON * np.sin(th), ":", color="0.6", lw=0.7, label="Moon orbit")
        ax.add_patch(plt.Circle((0, 0), R_EARTH * scale, color="steelblue"))
        meas_sc  = ax.scatter([], [], s=3, color="0.8", label="measured")
        kep_ln = kep_pt = None
        if kep is not None:
            kep_ln = ax.plot([], [], "-", color="0.55", lw=1.4, label="without Moon (Keplerian)")[0]
            kep_pt = ax.plot([], [], "o", color="0.55", ms=5)[0]
        truth_ln = ax.plot([], [], color="crimson", lw=1.7,
                           label=("with Moon (truth)" if kep is not None else "truth"))[0]
        est_ln   = ax.plot([], [], "--", color="navy", lw=1.0, label="EKF reconstruction")[0]
        sc_t     = ax.plot([], [], "o", color="crimson", ms=6)[0]
        moon_pt  = ax.plot([], [], "o", color="dimgray", ms=9)[0]
        star_fly = ax.plot([], [], "*", color="crimson", ms=16, zorder=6)[0]   # flyby (once reached)
        arr_E = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=2.6))
        arr_M = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="seagreen",  lw=2.6))
        arr_S = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="goldenrod", lw=2.6))
        g_txt = ax.text(0.015, 0.985, "", transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("x [Mm = 1000 km]"); ax.set_ylabel("y [Mm]"); ax.grid(alpha=0.3)
        proxies = [Line2D([0], [0], color="steelblue", lw=2.6, label="Earth (central)"),
                   Line2D([0], [0], color="seagreen",  lw=2.6, label="Moon (3rd-body)"),
                   Line2D([0], [0], color="goldenrod", lw=2.6, label="Sun (tidal)")]
        h, l = ax.get_legend_handles_labels()
        ax.legend(h + proxies, l + [p.get_label() for p in proxies], loc="lower left", fontsize=8)

        def _draw(k):
            km = min(k, K - 1)               # catapult craft freezes at splashdown (K-1)
            inflight = k < K
            if meas is not None:
                meas_sc.set_offsets(meas[:km + 1, :2])
            if kep is not None:
                kk = min(k, len(kep) - 1)
                kep_ln.set_data(kep[:kk + 1, 0], kep[:kk + 1, 1])
                kep_pt.set_data([kep[kk, 0]], [kep[kk, 1]])
            truth_ln.set_data(truth[:km + 1, 0], truth[:km + 1, 1])
            est_ln.set_data(estx[:km + 1, 0], estx[:km + 1, 1])
            p = truth[km, :2]
            sc_t.set_data([p[0]], [p[1]]); moon_pt.set_data([moon[k, 0]], [moon[k, 1]])
            if km >= i_fly:
                star_fly.set_data([truth[i_fly, 0]], [truth[i_fly, 1]])
            if inflight:                      # live gravity vectors only while the craft flies
                gmax = max(gE[km], gM[km], gS[km]); gtot = gE[km] + gM[km] + gS[km]
                for ar, u, g in [(arr_E, uE[km], gE[km]), (arr_M, uM[km], gM[km]), (arr_S, uS[km], gS[km])]:
                    Lk = Lmax * g / gmax
                    ar.set_position((p[0], p[1])); ar.xy = (p[0] + u[0] * Lk, p[1] + u[1] * Lk)
                # geocentric pull on the craft + each body's % share of the total (Moon's share
                # surges through the flyby — that is the catapult; the Sun's stays a sliver).
                g_txt.set_text("geocentric pull on craft [m/s²]   share\n"
                               "Earth %.3e   %5.1f%%\n"
                               "Moon  %.3e   %5.1f%%\n"
                               "Sun   %.3e   %5.2f%%"
                               % (gE[km], 100 * gE[km] / gtot, gM[km], 100 * gM[km] / gtot,
                                  gS[km], 100 * gS[km] / gtot))
                ax.set_title("Apollo free-return — lunar catapult    t = %.2f days" % t_days[k])
            else:                             # after splashdown: hide vectors, keep the Keplerian flying
                for ar in (arr_E, arr_M, arr_S):
                    ar.set_position((p[0], p[1])); ar.xy = (p[0], p[1])
                g_txt.set_text("catapult craft: splashed down @ %.1f d\n"
                               "Keplerian (no Moon): still en route" % t_days[K - 1])
                ax.set_title("Apollo free-return — the Moon got the crew home %.1f days sooner    t = %.2f days"
                             % (t_days[-1] - t_days[K - 1], t_days[k]))
            return meas_sc, truth_ln, est_ln, sc_t, moon_pt, star_fly, arr_E, arr_M, arr_S, g_txt

        return animate(fig, _draw, frames=frame_idx, interval=interval)


# ===========================================================================
#  2) Interplanetary transfers — Hohmann, Lambert, porkchop, launch windows
# ===========================================================================
def _stumpff_C(z):
    if z > 1e-6:
        return (1.0 - np.cos(np.sqrt(z))) / z
    if z < -1e-6:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    return 0.5


def _stumpff_S(z):
    if z > 1e-6:
        sz = np.sqrt(z); return (sz - np.sin(sz)) / sz ** 3
    if z < -1e-6:
        sz = np.sqrt(-z); return (np.sinh(sz) - sz) / sz ** 3
    return 1.0 / 6.0



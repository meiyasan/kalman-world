"""Interplanetary cruise with perturbers + trajectory-control maneuvers."""
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
from ....framework.constants import KalmanState, KalmanObservable
from ....framework.control import KalmanController
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


# ===========================================================================
#  1) Trans-lunar coast — the gravitational catapult (Earth + Moon + Sun)
# ===========================================================================

class _PerturberField:
    """Cheap interpolated heliocentric positions of the perturbing planets over a span, so the
    N-body integrator can sample them at any ``t`` without re-querying the ephemeris."""

    def __init__(self, bodies, epoch, span_days, step_days=2.0):
        self.bodies = tuple(bodies)
        self.mu = {b: gm(b) for b in self.bodies}
        self.t_grid = np.arange(0.0, span_days * DAY + step_days * DAY, step_days * DAY)
        times = times_from_epoch(epoch, self.t_grid)
        samp = ephemeris_samples(self.bodies, times)
        self.R = {b: samp[b][0] for b in self.bodies}

    def positions(self, t):
        return {b: np.array([np.interp(t, self.t_grid, self.R[b][:, k]) for k in range(3)])
                for b in self.bodies}


class InterplanetaryState(KalmanState):
    X = "Mm"; Y = "Mm"; Z = "Mm"; VX = "km/s"; VY = "km/s"; VZ = "km/s"


class InterplanetaryObservable(KalmanObservable):
    X = "Mm"; Y = "Mm"; Z = "Mm"


class InterplanetaryModel(ExtendedKalmanModel):
    """
    Heliocentric **N-body** interplanetary cruise, general over any *departure → target*
    planet (Mars is the worked example).

    State ``x = [rx, ry, rz, vx, vy, vz]`` (Sun-centred, position in Mm, velocity in km/s);
    only the position is tracked (deep-space network). ``f()`` integrates the Sun's central
    pull plus the **third-body perturbations of the planets** (from the live ephemeris,
    interpolated over the mission span) with RK4. With ``control=True`` the input ``u`` is an
    impulsive mid-course **Δv** [km/s] applied at the start of a step (``l = 3``). Seed it from
    a Lambert solution, e.g. ``model.initial_state(best["x0"])`` where ``best`` comes from
    :func:`best_departure`. If ``epoch`` is ``None`` the perturbers are dropped and the model
    reduces to a fast two-body Sun cruise (handy for tests without the ephemeris).

    **Spacecraft defaults** are representative of an MRO / InSight class probe at departure:

    * dry mass  ~600 kg  (structure + instruments + avionics)
    * wet mass ~1200 kg  (+ ~600 kg hydrazine / bipropellant)
    * cruise speed ~29–33 km/s heliocentric (seeded by the Lambert solution)

    Pass these to :class:`TrajectoryControlManeuver` to get realistic burn durations and
    propellant accounting.
    """

    NAME = "Interplanetary cruise (heliocentric N-body)"; VERSION = "1.0"

    # Representative spacecraft mass for an MRO/InSight-class Mars probe [kg]
    SPACECRAFT_DRY_MASS_KG = 600.0
    SPACECRAFT_WET_MASS_KG = 1200.0

    @property
    def states(self):      return InterplanetaryState
    @property
    def observables(self): return InterplanetaryObservable

    def __init__(self, departure="earth", target="mars", *, epoch=None, span_days=400.0,
                 dt_s=43200.0, track_Mm=0.05, n_sub=1, control=False,
                 perturbers=("mercury", "venus", "earth", "mars", "jupiter", "saturn"),
                 q_pos=1e-4, q_vel=1e-8, **kw):
        self.dtype = np.float64
        self.departure = departure
        self.target = target
        self.epoch = epoch
        self._dt = dt_s / TU
        self.n_sub = max(1, int(n_sub))
        self.n, self.m, self.l = 6, 3, (3 if control else 0)
        self.fs = 1.0 / self._dt
        # Patched conic: the departure/target planets are handled by the hyperbolic legs, not as
        # heliocentric point masses the probe would sit on (which would diverge at the endpoints).
        pert = tuple(b for b in perturbers if b not in (departure, target))
        self._field = (_PerturberField(pert, epoch, span_days)
                       if (epoch is not None and pert) else None)
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(float)
        R = (track_Mm) ** 2 * np.eye(3)
        super().__init__("PROBE", Q=Q, R=R, fs=self.fs, **kw)

    # --- dynamics ---
    def acceleration(self, r, t):
        r = np.asarray(r, float).reshape(3)
        a = -GM_SUN * r / np.linalg.norm(r) ** 3            # Sun central pull
        if self._field is not None:
            pos = self._field.positions(t)
            for b in self._field.bodies:
                a += _third_body(r, pos[b], self._field.mu[b])
        return a

    def f(self, x, u=None, w=None, t=None):
        s = np.asarray(x, float).reshape(-1).copy()
        if u is not None and self.l > 0:                    # impulsive mid-course burn
            s[3:6] = s[3:6] + np.asarray(u, float).reshape(3)
        deriv = lambda s, tt: np.concatenate([s[3:], self.acceleration(s[:3], tt)])
        t0, h = float(t or 0.0), self._dt / self.n_sub
        for j in range(self.n_sub):
            s = _rk4(s, t0 + j * h, h, deriv)
        return s.reshape(-1, 1)

    def h(self, x, v=None, t=None):
        z = np.asarray(x, float).reshape(-1)[:3].reshape(3, 1)
        return z if v is None else z + np.asarray(v, float).reshape(3, 1)

    # --- planet positions for plotting ---
    def planet_track(self, body, K):
        """Heliocentric (x,y,z) of ``body`` over the first ``K`` steps (scaled)."""
        t = np.arange(K) * self._dt
        if self._field is not None and body in self._field.bodies:
            return np.array([self._field.positions(tt)[body] for tt in t])
        times = times_from_epoch(self.epoch, t)
        return ephemeris_samples([body], times)[body][0]

    # ==================================================================
    # Signature figure — heliocentric cruise (override KalmanModel.plot).
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     nmax=None, figsize=(8.6, 8.4), xlim=None, ylim=None, **kwargs):
        """Signature figure: heliocentric transfer in the ecliptic (x–y) plane, axes in AU."""
        import matplotlib.pyplot as plt
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        meas  = (np.asarray(sample.zk) if sample is not None
                 else (np.asarray(record.zk) if np.size(record.zk) else None))
        K = int(nmax or min(len(truth), len(estx)))
        truth, estx = truth[:K], estx[:K]
        meas = meas[:K] if meas is not None else None
        AU = D_SUN
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0], [0], "o", color="gold", ms=14, label="Sun")
        for body, col in [(self.departure, "steelblue"), (self.target, "indianred")]:
            try:
                trk = self.planet_track(body, K) / AU
                ax.plot(trk[:, 0], trk[:, 1], ":", color=col, lw=0.9)
                ax.plot([trk[0, 0]], [trk[0, 1]], "o", color=col, ms=7, label=f"{body} (depart)")
                ax.plot([trk[-1, 0]], [trk[-1, 1]], "*", color="white",
                        ms=12, mew=1.5, mec=col, label=f"{body} (arrival)")
            except Exception:
                pass
        if meas is not None:
            ax.scatter(meas[:, 0] / AU, meas[:, 1] / AU, s=3, color="0.8", label="DSN fixes")
        ax.plot(truth[:, 0] / AU, truth[:, 1] / AU, color="crimson", lw=1.7, label="truth")
        ax.plot(estx[:, 0] / AU, estx[:, 1] / AU, "--", color="navy", lw=1.0, label="EKF reconstruction")
        lim = 1.1 * np.abs(truth[:, :2] / AU).max()
        ax.autoscale(False)
        ax.set_xlim(*(xlim if xlim is not None else (-lim, lim)))
        ax.set_ylim(*(ylim if ylim is not None else (-lim, lim)))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]"); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8.5)
        ax.set_title(f"{self.name()} — {self.departure} → {self.target} cruise")
        return fig

    def _plot_interactive(self, record, sample=None, *, component=None, error=True,
                          nmax=None, frames=30, interval=90, figsize=(8.4, 8.2),
                          burns=None, no_burn=None, **kwargs):
        """Interactive heliocentric cruise replay with per-body gravity readout and optional
        mid-course burn visualisation.

        Per-body gravity arrows are log-scaled so planetary perturbations remain visible
        alongside the Sun. The text box shows every body's magnitude [m/s²] and % share.
        **Departure and target planets** are shown in a separate "patched-conic excl." section
        with faint arrows — they are excluded from the N-body dynamics (patched-conic
        convention) but their third-body influence on the probe is still displayed.

        Parameters
        ----------
        burns : list of (step_idx, dv_kms_vector) | None
            Optional mid-course manoeuvres. Each entry ``(step, dv)`` draws a static orange
            diamond at the burn position and flashes a Δv arrow on the exact animation frame.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from ....helpers import animate
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        meas  = (np.asarray(sample.zk) if sample is not None
                 else (np.asarray(record.zk) if np.size(record.zk) else None))
        K = int(nmax or min(len(truth), len(estx)))
        truth, estx = truth[:K], estx[:K]
        meas = meas[:K] if meas is not None else None
        AU = D_SUN

        dep_trk_Mm = self.planet_track(self.departure, K)   # (K,3) in Mm — for accel
        tgt_trk_Mm = self.planet_track(self.target,    K)
        dep_trk = dep_trk_Mm / AU   # (K,3) in AU — for plotting
        tgt_trk = tgt_trk_Mm / AU
        t_days  = np.arange(K) * self._dt / DAY
        _date_strs = None
        if self.epoch is not None:
            from astropy.time import TimeDelta as _TD
            _t0 = _as_time(self.epoch)
            _date_strs = [(_t0 + _TD(float(d) * 86400, format="sec")).iso[:10] for d in t_days]
        lim = 1.1 * max(np.abs(truth[:, :2] / AU).max(),
                        np.abs(dep_trk[:, :2]).max(), np.abs(tgt_trk[:, :2]).max())
        burns = list(burns or [])
        no_burn_xk = None
        if no_burn is not None:
            nb_arr = np.asarray(getattr(no_burn, "xk", no_burn))
            no_burn_xk = nb_arr[:K]            # clip to animation length
        step = max(1, K // frames)
        frame_idx = sorted(set(list(range(0, K, step)) + [K - 1]
                               + [min(int(bs), K - 1) for bs, _ in burns]))

        # ---- pre-compute heliocentric accelerations for all K steps ----
        # 1 Mm/ks² = 1 m/s² (LU/TU² = 1e6 m / (1e3 s)² = 1 m/s²).
        t_arr = np.arange(K) * self._dt
        r = truth[:, :3]                                      # (K,3) in Mm
        rN = np.linalg.norm(r, axis=1)
        gSun_mag = GM_SUN / rN ** 2
        uSun = -r[:, :2] / rN[:, None]

        pert_bodies = list(self._field.bodies) if self._field is not None else []
        pert_g = {}; pert_u = {}; pert_trk = {}   # pert_trk[b] = (K,3) in AU
        if self._field is not None:
            for b in pert_bodies:
                pos_b = np.column_stack([
                    np.interp(t_arr, self._field.t_grid, self._field.R[b][:, c])
                    for c in range(3)])                       # (K,3) in Mm
                pert_trk[b] = pos_b / AU                    # (K,3) in AU — for plotting
                d  = pos_b - r
                dN = np.linalg.norm(d,     axis=1, keepdims=True)
                pN = np.linalg.norm(pos_b, axis=1, keepdims=True)
                acc = self._field.mu[b] * (d / dN ** 3 - pos_b / pN ** 3)
                mag = np.linalg.norm(acc, axis=1)
                pert_g[b] = mag
                pert_u[b] = acc[:, :2] / np.where(mag[:, None] > 0, mag[:, None], 1.0)

        # Departure and target: display-only (patched-conic excludes them from dynamics).
        # A minimum-distance clamp avoids the singularity at t=0 when spacecraft ≈ Earth.
        _D_MIN = 5.0 * R_EARTH
        dep_tgt = []
        for b, pos_Mm in [(self.departure, dep_trk_Mm), (self.target, tgt_trk_Mm)]:
            d  = pos_Mm - r
            dN = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), _D_MIN)
            pN = np.linalg.norm(pos_Mm, axis=1, keepdims=True)
            acc = gm(b) * (d / dN ** 3 - pos_Mm / pN ** 3)
            mag = np.linalg.norm(acc, axis=1)
            pert_g[b] = mag
            pert_u[b] = acc[:, :2] / np.where(mag[:, None] > 0, mag[:, None], 1.0)
            dep_tgt.append(b)
        all_disp = pert_bodies + dep_tgt

        # lim stays mission-scoped (spacecraft + dep/tgt planets).
        # Perturber orbit tracks are drawn beyond lim — visible on manual unzoom.

        # Log-scaled arrow lengths — 6-decade span keeps planet arrows visible beside Sun
        all_g = np.concatenate([gSun_mag] + [v for v in pert_g.values()])
        log_max = np.log10(np.nanmax(all_g) + 1e-30)
        log_min = log_max - 6
        Lmax = 0.18 * lim

        def _alen(g):
            if g <= 0: return 0.0
            return Lmax * np.clip((np.log10(g + 1e-30) - log_min) / max(log_max - log_min, 1e-6),
                                  0.05, 1.0)

        max_dv = max((np.linalg.norm(np.asarray(dv)) for _, dv in burns), default=1.0)
        burn_steps = {min(int(bs), K - 1): np.asarray(bdv, float).reshape(-1)
                      for bs, bdv in burns}

        # ---- figure setup ----
        fig, ax = plt.subplots(figsize=figsize)
        # Fix limits BEFORE any ax.plot() so outer-planet orbit tracks (Saturn ±9.5 AU)
        # never trigger autoscale — users can still pan/zoom the widget to reveal them.
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale(False)
        ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]"); ax.grid(alpha=0.3)

        sun_ms = _BODY_MS.get("sun", 14)
        ax.plot([0], [0], "o", color="gold", ms=sun_ms, label="Sun", zorder=5)
        # Full elliptical orbit paths (replaces K-step partial arcs)
        t_epoch = _as_time(self.epoch) if self.epoch is not None else None
        _orbit_bodies = sorted(set([self.departure, self.target] + list(pert_trk.keys())))
        for b in _orbit_bodies:
            col = ("steelblue" if b == self.departure else
                   "indianred" if b == self.target else
                   _BODY_COLOR.get(b, "gray"))
            oc = _orbit_ellipse(b, t_epoch)
            if oc is not None:
                ax.plot(oc[:, 0], oc[:, 1], ":", color=col, lw=0.8, alpha=0.4)
        dep_ms  = _BODY_MS.get(self.departure, 8)
        tgt_ms  = _BODY_MS.get(self.target, 8)
        dep_pt  = ax.plot([], [], "o", color="steelblue", ms=dep_ms, label=self.departure)[0]
        tgt_pt  = ax.plot([], [], "o", color="indianred", ms=tgt_ms, label=self.target)[0]
        # Animated dots for perturbing planets
        pert_pts = {b: ax.plot([], [], "o", color=_BODY_COLOR.get(b, "gray"),
                               ms=_BODY_MS.get(b, 6), label=b)[0] for b in pert_trk}
        meas_sc  = ax.scatter([], [], s=4, color="0.8", label="DSN fixes")
        truth_ln = ax.plot([], [], color="crimson", lw=1.7, label="truth")[0]
        est_ln   = ax.plot([], [], "--", color="navy", lw=1.0, label="EKF")[0]
        probe_pt = ax.plot([], [], "o", color="crimson", ms=6)[0]
        arr_sun  = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                               arrowprops=dict(arrowstyle="-|>", color="gold", lw=2.4))
        # Dynamics perturbers: solid arrows; patched-conic bodies: thin/faint arrows
        pert_arrs = {}
        for b in all_disp:
            col = _BODY_COLOR.get(b, "gray")
            lw, alpha = (1.6, 1.0) if b in pert_bodies else (1.0, 0.5)
            pert_arrs[b] = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                       arrowprops=dict(arrowstyle="-|>", color=col,
                                                       lw=lw, alpha=alpha))
        g_txt = ax.text(0.015, 0.985, "", transform=ax.transAxes, fontsize=7.2, va="top",
                        family="monospace",
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        # No-burn counterfactual (grows with animation frame)
        no_burn_ln = None
        if no_burn_xk is not None:
            no_burn_ln = ax.plot([], [], "--", color="0.55", lw=1.3, label="no TCM")[0]
        # Static burn crosses + per-burn persistent labels
        burn_label_artists = {}   # bs -> text artist
        for bs, bdv_arr in burn_steps.items():
            bp = truth[bs, :2] / AU
            ax.plot([bp[0]], [bp[1]], "x", color="darkorange", ms=11, mew=2.2, zorder=8)
            dv_mag = float(np.linalg.norm(bdv_arr))
            # Label anchored clearly above the diamond; white bbox ensures it's always readable
            lbl = ax.text(bp[0], bp[1] + 0.06 * lim,
                          "TCM  %.0f m/s" % (dv_mag * 1e3),
                          fontsize=8.5, color="darkorange", fontweight="bold",
                          ha="center", va="bottom", zorder=10,
                          bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                    ec="darkorange", alpha=0.88),
                          visible=False)
            burn_label_artists[bs] = lbl
        # Transient Δv arrow (only on the exact burn frame)
        burn_arr = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                               arrowprops=dict(arrowstyle="-|>", color="orange", lw=3.0))

        # Departure event annotation (always visible — departure is at t=0)
        dep_pos = truth[0, :2] / AU
        ax.annotate("Dep  " + (_date_strs[0] if _date_strs else ""),
                    (dep_pos[0], dep_pos[1]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7.5, color="steelblue", fontweight="bold", zorder=9)
        # Arrival event annotation (pre-created, becomes visible near end)
        arr_pos = truth[-1, :2] / AU
        arr_ann = ax.annotate("Arr  " + (_date_strs[-1] if _date_strs else ""),
                              (arr_pos[0], arr_pos[1]),
                              xytext=(6, -12), textcoords="offset points",
                              fontsize=7.5, color="indianred", fontweight="bold",
                              zorder=9, visible=False)
        # Only add legend entries for bodies that reach ≥0.1% of total accel at any frame
        # (peak share), so Mercury / distant bodies don't crowd the legend with 0.00% lines.
        _peak_share = {}
        _gtot_max = np.max(gSun_mag) + sum(np.max(v) for v in pert_g.values())
        for b in all_disp:
            _peak_share[b] = np.max(pert_g[b]) / max(_gtot_max, 1e-30)
        proxies = [Line2D([0], [0], color="gold", lw=2.4, label="Sun (central)")]
        for b in pert_bodies:
            if _peak_share[b] >= 1e-3:   # ≥0.1% peak share → show in legend
                proxies.append(Line2D([0], [0], color=_BODY_COLOR.get(b, "gray"),
                                      lw=1.6, label=f"{b} (N-body)"))
        for b in dep_tgt:
            if _peak_share[b] >= 1e-3:
                proxies.append(Line2D([0], [0], color=_BODY_COLOR.get(b, "gray"),
                                      lw=1.0, alpha=0.5, label=f"{b} (SOI approach)"))
        if burns:
            proxies.append(Line2D([0], [0], marker="x", color="darkorange",
                                  ms=8, mew=2.0, lw=0, label="TCM burn"))
        first_burn_step = min(burn_steps.keys()) if burn_steps else K
        # Pre-compute distance (in AU) from spacecraft to each dep/tgt planet for arrow clamping
        _dep_tgt_trk = {b: (dep_trk_Mm if b == self.departure else tgt_trk_Mm) / AU
                        for b in dep_tgt}
        h, l = ax.get_legend_handles_labels()
        ax.legend(h + proxies, l + [p.get_label() for p in proxies],
                  loc="lower right", fontsize=7.2)
        title = ax.set_title("")

        def _draw(k):
            k = min(k, K - 1)
            p = truth[k, :2] / AU
            dep_pt.set_data([dep_trk[k, 0]], [dep_trk[k, 1]])
            tgt_pt.set_data([tgt_trk[k, 0]], [tgt_trk[k, 1]])
            for b, pt in pert_pts.items():
                pt.set_data([pert_trk[b][k, 0]], [pert_trk[b][k, 1]])
            truth_ln.set_data(truth[:k + 1, 0] / AU, truth[:k + 1, 1] / AU)
            est_ln.set_data(estx[:k + 1, 0] / AU, estx[:k + 1, 1] / AU)
            probe_pt.set_data([p[0]], [p[1]])
            if meas is not None:
                meas_sc.set_offsets(meas[:k + 1, :2] / AU)
            # Sun arrow
            gS = gSun_mag[k]; Ls = _alen(gS)
            arr_sun.set_position((p[0], p[1]))
            arr_sun.xy = (p[0] + uSun[k, 0] * Ls, p[1] + uSun[k, 1] * Ls)
            # Perturber arrows (dynamics + patched-conic display)
            for b, arr in pert_arrs.items():
                gP = pert_g[b][k]; LP = _alen(gP)
                # Hide arrows for bodies whose peak share is negligible (avoids clutter)
                if _peak_share.get(b, 0) < 1e-3:
                    arr.set_position((0, 0)); arr.xy = (0, 0); continue
                # Clamp patched-conic planet arrows so tip never overshoots the planet
                if b in _dep_tgt_trk:
                    planet_p = _dep_tgt_trk[b][k, :2]
                    dist_to_planet = np.linalg.norm(planet_p - p)
                    LP = min(LP, dist_to_planet * 0.85)
                arr.set_position((p[0], p[1]))
                arr.xy = (p[0] + pert_u[b][k, 0] * LP, p[1] + pert_u[b][k, 1] * LP)
            # Text readout
            gtot = gS + sum(pert_g[b][k] for b in all_disp)
            txt = "heliocentric accel [m/s²]    share\n"
            txt += "Sun      %.3e       %5.1f%%\n" % (gS, 100 * gS / max(gtot, 1e-30))
            for b in pert_bodies:
                gP = pert_g[b][k]
                txt += "%-8s %.3e       %6.2f%%\n" % (b[:8], gP, 100 * gP / max(gtot, 1e-30))
            if dep_tgt:
                txt += "── patched-conic excl. ──\n"
                for b in dep_tgt:
                    gP = pert_g[b][k]
                    txt += "%-8s %.3e       %6.2f%%\n" % (b[:8], gP, 100 * gP / max(gtot, 1e-30))
            g_txt.set_text(txt.rstrip())
            # TCM labels: explicitly set visibility every frame so jshtml state is correct
            for bs, lbl_art in burn_label_artists.items():
                lbl_art.set_visible(k >= bs)
            # Transient Δv arrow: only on the exact burn frame
            if k in burn_steps:
                bdv = burn_steps[k]; dv_mag = np.linalg.norm(bdv)
                bu = bdv[:2] / max(np.linalg.norm(bdv[:2]), 1e-10)
                Lb = Lmax * 0.5 * dv_mag / max_dv
                burn_arr.set_position((p[0], p[1]))
                burn_arr.xy = (p[0] + bu[0] * Lb, p[1] + bu[1] * Lb)
            else:
                burn_arr.set_position((0, 0)); burn_arr.xy = (0, 0)
            # Arrival label: explicit visibility every frame
            arr_ann.set_visible(k >= int(0.90 * (K - 1)))
            # No-TCM counterfactual: only visible from the burn step onward
            if no_burn_ln is not None:
                if k >= first_burn_step:
                    no_burn_ln.set_data(no_burn_xk[first_burn_step:k + 1, 0] / AU,
                                        no_burn_xk[first_burn_step:k + 1, 1] / AU)
                else:
                    no_burn_ln.set_data([], [])
            date_part = ("   " + _date_strs[k]) if _date_strs else ""
            title.set_text("%s → %s cruise    t = %.0f days%s" % (self.departure, self.target, t_days[k], date_part))
            return (dep_pt, tgt_pt, truth_ln, est_ln, probe_pt, meas_sc, title,
                    arr_sun, g_txt, burn_arr, arr_ann,
                    *burn_label_artists.values(),
                    *pert_arrs.values(), *pert_pts.values(),
                    *([no_burn_ln] if no_burn_ln is not None else []))

        return animate(fig, _draw, frames=frame_idx, interval=interval)


# ===========================================================================
#  4) Trajectory Correction Manoeuvre (TCM) controller
# ===========================================================================
class TrajectoryControlManeuver(KalmanController):
    """Realistic finite-burn Trajectory Correction Manoeuvre (TCM) controller.

    Inherits from :class:`~kalman.framework.KalmanController`.  At each
    scheduled epoch the controller:

    1. Solves **Lambert's problem** from the current EKF position to the
       target arrival position to find the velocity vector needed.
    2. Computes **Δv** = v_needed − v_current.
    3. Applies the **Tsiolkovsky rocket equation** to derive propellant
       consumed and **burn duration** (seconds to minutes, not an instant).
    4. Returns the velocity impulse; the EKF applies it as a state update.

    **Early TCMs amplify corrections geometrically.**  A 1 m/s Δv fired
    175 days before Mars arrival shifts the arrival point by ≈ 15 000 km,
    whereas the same burn fired 5 days out moves it only ≈ 430 km.  The
    default schedule (day 5, 30, 90) exploits this leverage:

    * **TCM-1 (day 5)** — dominant burn, corrects launch-injection error.
    * **TCM-2 (day 30)** — mid-course trim.
    * **TCM-3 (day 90)** — optional fine-tune; often < 1 m/s.

    Default spacecraft: MRO / InSight class (~600 kg dry, 1200 kg wet,
    22 N hydrazine thrusters, Isp 220 s).  Pass ``dry_mass_kg`` /
    ``wet_mass_kg`` / ``Isp_s`` / ``thrust_N`` to override.

    Parameters
    ----------
    model : InterplanetaryModel
        Cruise model.  ``control=True`` is enforced automatically.
    r_target : (3,) array_like
        Heliocentric arrival position [Mm] (e.g. ``best["x0"][:3]`` for
        the Lambert best solution at the *target* date, or compute via
        ``body_state(target, arrival_date)[0]``).
    t_arrival_ks : float
        Mission-elapsed arrival time [ks]  (``tof_days * DAY``).
    tcm_days : list of float, optional
        Mission-elapsed days at which TCMs fire.  Default: [5, 30, 90].
    dry_mass_kg, wet_mass_kg : float
        Spacecraft dry / wet mass [kg].
    Isp_s : float
        Specific impulse [s].
    thrust_N : float
        Total thruster thrust [N].
    max_dv_kms : float
        Per-TCM Δv cap [km/s] (default 0.05 = 50 m/s).  Guards against
        overburn when the Lambert solution diverges from the current state
        by more than a realistic injection error.
    mu : float, optional
        Gravitational parameter for Lambert solve [Mm³/ks²].  Defaults to
        ``GM_SUN`` (heliocentric cruise).
    """

    DRY_MASS_KG      = 600.0    # kg  — MRO/InSight class dry mass
    WET_MASS_KG      = 1200.0   # kg  — including ~600 kg hydrazine
    ISP_S            = 220.0    # s   — MR-106 class monoprop thrusters
    THRUST_N         = 22.0     # N   — Aerojet MR-106 (×1 thruster; real missions use clusters)
    MAX_DV_KMS       = 0.050    # km/s — 50 m/s per-TCM cap
    TCM_DAYS_DEFAULT = [5.0, 30.0, 90.0]

    def __init__(self, model, r_target, t_arrival_ks, *,
                 tcm_days=None, dry_mass_kg=None, wet_mass_kg=None,
                 Isp_s=None, thrust_N=None, max_dv_kms=None, mu=None):
        model.l = 3                              # ensure model accepts control input
        super().__init__(model, Rcost=np.eye(3))
        self.r_target      = np.asarray(r_target, float).reshape(3)
        self.t_arrival     = float(t_arrival_ks)
        tcm_days           = list(tcm_days or self.TCM_DAYS_DEFAULT)
        self._tcm_times    = sorted(float(d) * DAY for d in tcm_days)   # [TU = ks]
        self.dry_mass      = float(dry_mass_kg  if dry_mass_kg  is not None else self.DRY_MASS_KG)
        self._wet_mass_init = float(wet_mass_kg if wet_mass_kg  is not None else self.WET_MASS_KG)
        self.current_mass  = self._wet_mass_init
        self.Isp           = float(Isp_s        if Isp_s        is not None else self.ISP_S)
        self.thrust        = float(thrust_N     if thrust_N     is not None else self.THRUST_N)
        self.max_dv        = float(max_dv_kms   if max_dv_kms   is not None else self.MAX_DV_KMS)
        self.mu            = float(mu if mu is not None else GM_SUN)
        self.burns         = []       # list of burn metadata dicts (populated at each TCM)
        self._fired        = set()    # indices of TCMs already executed
        self._dt_half      = 0.5 * model._dt   # match tolerance: half an EKF step

    @property
    def propellant_remaining_kg(self):
        """Propellant left in the tanks [kg]."""
        return max(self.current_mass - self.dry_mass, 0.0)

    @property
    def propellant_consumed_kg(self):
        """Total propellant consumed across all TCMs [kg]."""
        return sum(b["prop_kg"] for b in self.burns)

    def _policy(self, x_hat, P, t, *, nu=None):
        t = float(t)
        for i, t_tcm in enumerate(self._tcm_times):
            if i in self._fired:
                continue
            if abs(t - t_tcm) <= self._dt_half:
                return self._execute_tcm(i, x_hat, t)
        return np.zeros(3)

    def _execute_tcm(self, idx, x_hat, t):
        """Compute and log the burn for scheduled TCM *idx*."""
        self._fired.add(idx)
        r_now = x_hat[:3].flatten()
        v_now = x_hat[3:6].flatten()
        tof   = self.t_arrival - t          # remaining time-of-flight [ks]

        if tof <= 0.0 or self.propellant_remaining_kg <= 0.0:
            return np.zeros(3)

        # Lambert targeting: what velocity do we need NOW to reach r_target in tof?
        try:
            v_needed, _ = lambert(r_now, self.r_target, tof, mu=self.mu)
        except Exception:
            return np.zeros(3)

        dv_vec = v_needed - v_now           # required Δv [km/s]
        dv_mag = float(np.linalg.norm(dv_vec))

        # Cap: prevent overburn if injection error is unrealistically large
        if dv_mag > self.max_dv:
            dv_vec = dv_vec * (self.max_dv / dv_mag)
            dv_mag = self.max_dv

        # Tsiolkovsky rocket equation (SI: Isp in s, g0 in m/s², Δv in m/s)
        dv_ms       = dv_mag * 1e3          # km/s → m/s
        max_dv_ms   = self.Isp * G0 * np.log(self.current_mass / max(self.dry_mass, 1.0))
        if dv_ms > max_dv_ms > 0.0:        # out of propellant — burn what's left
            dv_ms   = max_dv_ms
            dv_vec  = dv_vec * (dv_ms / max(dv_mag * 1e3, 1e-10))
            dv_mag  = dv_ms * 1e-3

        prop_kg = self.current_mass * (1.0 - np.exp(-dv_ms / (self.Isp * G0)))
        prop_kg = min(prop_kg, self.propellant_remaining_kg)

        # Burn duration: Δm × Isp × g0 / F  (constant-thrust, isentropic approximation)
        burn_s  = prop_kg * self.Isp * G0 / max(self.thrust, 1e-6)

        m_before          = self.current_mass
        self.current_mass -= prop_kg

        self.burns.append({
            "tcm_idx":        idx + 1,
            "t_ks":           t,
            "t_days":         t / DAY,
            "dv_kms":         float(dv_mag),
            "dv_ms":          float(dv_ms),
            "dv_vec_kms":     dv_vec.copy(),
            "prop_kg":        float(prop_kg),
            "mass_before_kg": float(m_before),
            "mass_after_kg":  float(self.current_mass),
            "burn_s":         float(burn_s),
            "tof_rem_days":   float(tof / DAY),
        })
        return dv_vec.reshape(3)

    def reset(self):
        """Reset to the pre-launch state (full tanks, no burns fired).

        Call this between a truth-propagation pass and the EKF filter pass so
        the controller can re-fire the same TCM schedule on the filter estimate.
        """
        super().reset()
        if not hasattr(self, "_wet_mass_init"):
            return   # called by KalmanController.__init__ before our attrs exist
        self.current_mass = self._wet_mass_init
        self.burns        = []
        self._fired       = set()

    def summary(self):
        """Print a formatted table of all TCMs executed so far."""
        if not self.burns:
            print("No TCMs executed.")
            return
        print("TCM Summary")
        print("─" * 76)
        for b in self.burns:
            print("  TCM-%-2d  t=%6.1f d  Δv=%8.2f m/s  burn=%6.1f s  "
                  "Δm=%6.2f kg  m_after=%7.2f kg  TOF_rem=%5.0f d"
                  % (b["tcm_idx"], b["t_days"], b["dv_ms"], b["burn_s"],
                     b["prop_kg"], b["mass_after_kg"], b["tof_rem_days"]))
        print("─" * 76)
        total_dv   = sum(b["dv_ms"]   for b in self.burns)
        total_prop = sum(b["prop_kg"] for b in self.burns)
        print("  TOTAL   Δv=%8.2f m/s   Δm=%6.2f kg   propellant left: %.2f kg"
              % (total_dv, total_prop, self.propellant_remaining_kg))


# ===========================================================================
#  Plotting helpers (module-level): solar-system map + porkchop
# ===========================================================================
def plot_solar_system(time=None, *, figsize=(13, 6.2)):
    """A clean live solar-system map at ``time``: the Sun, each planet's (near-circular)
    orbit and its current position (labelled), in two panels — the full system and a zoom on
    the inner planets — both to scale in AU. Returns the figure."""
    import matplotlib.pyplot as plt
    t = _as_time(time)
    pos = planet_positions(t, bodies=_PLANETS)
    # Split the system around Mars (Mars shown in both as the shared reference orbit), so the
    # inner planets aren't crushed into the origin by Neptune's scale.
    inner = ("mercury", "venus", "earth", "mars")
    outer = ("jupiter", "saturn", "uranus", "neptune")
    th = np.linspace(0, 2 * np.pi, 361)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, group, title in [(axes[0], inner, "Inner system  (≤ Mars)"),
                             (axes[1], outer, "Outer system  (> Mars)")]:
        ax.plot(0, 0, "o", color="gold", ms=15, zorder=5)
        ax.annotate("Sun", (0, 0), color="goldenrod", fontsize=8, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")
        for b in group:
            r = pos[b] / D_SUN
            col, ms = _PLANET_STYLE[b]
            rad = float(np.hypot(r[0], r[1]))
            ax.plot(rad * np.cos(th), rad * np.sin(th), "-", color=col, lw=0.8, alpha=0.45)
            ax.plot(r[0], r[1], "o", color=col, ms=ms, zorder=4)
            ax.annotate(b, (r[0], r[1]), color=col, fontsize=8, ha="left", va="bottom",
                        xytext=(4, 3), textcoords="offset points")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]"); ax.set_title(title)
    fig.suptitle("Solar system — %s UTC" % t.iso[:16], fontweight="bold")
    fig.tight_layout()
    return fig


def plot_porkchop(grid, *, ax=None, dv_max=None, figsize=(9.5, 5.8)):
    """Render a :func:`porkchop` ``grid`` as a proper porkchop plot: filled Δv contours
    focused on the launch-window basin (optimum + ``dv_max`` km/s), labelled contour lines, a
    colour bar, and the minimum-Δv node starred. Returns the figure."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from astropy.time import Time
    x = mdates.date2num(Time(grid["dep_mjd"], format="mjd").to_datetime())   # departure calendar dates
    y, DV = grid["tof_days"], grid["dv_total"]
    # Lambert can fail to converge for awkward transfer geometries and silently return NaN
    # (sqrt of a negative number) rather than raising — contourf then renders those NaN cells
    # as gaps (the "white diamond" artefacts). Treat them as "very large Δv", same as the
    # excluded/failed nodes already marked np.inf, so they blend into the dark-red background.
    DV = np.where(np.isfinite(DV), DV, np.inf)
    DV_fin = DV[np.isfinite(DV)]
    vmin = float(np.min(DV_fin))
    if dv_max is None:
        # Compress the colour range to the lower 60th-percentile spread so the basin
        # structure is visible; cap at vmin+4 for very wide distributions.
        spread = float(np.percentile(DV_fin, 60)) - vmin
        dv_max = vmin + max(spread * 2.5, 1.0)
        dv_max = min(dv_max, vmin + 4.0)
    else:
        dv_max = float(dv_max)
    # contourf uses clipped Z (clean colors); contour uses NaN-masked Z so line labels
    # only appear inside the launch-window basin.
    Z_fill = np.clip(DV, vmin, dv_max).T                      # (n_tof, n_dep)
    Z_line = np.where(DV.T <= dv_max, DV.T, np.nan)
    n_levels = 20
    levels = np.linspace(vmin, dv_max, n_levels + 1)
    fig = ax.figure if ax is not None else plt.subplots(figsize=figsize)[1].figure
    if ax is None:
        ax = fig.axes[0]
    cf = ax.contourf(x, y, Z_fill, levels=levels, cmap="turbo", extend="max")
    cl = ax.contour(x, y, Z_line, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
    ax.clabel(cl, fmt="%.1f", fontsize=7.5, inline=True)
    best = grid["best"]
    bx, by = mdates.date2num(best["dep_date"].to_datetime()), best["tof_days"]
    ax.plot(bx, by, "*", color="white", mec="k", ms=20, zorder=6, label="min Δv")
    # Place annotation on the less-crowded side of the star
    x_frac = (bx - x.min()) / max(x.max() - x.min(), 1e-9)
    y_frac = (by - y.min()) / max(y.max() - y.min(), 1e-9)
    dx = -80 if x_frac > 0.55 else 9
    dy = -32 if y_frac > 0.65 else 9
    ha = "right" if x_frac > 0.55 else "left"
    ax.annotate("min Δv = %.2f km/s\n%s,  %.0f d"
                % (best["dv_total_kms"], best["dep_date"].iso[:10], best["tof_days"]),
                (bx, by), color="white", fontsize=8.5, fontweight="bold",
                xytext=(dx, dy), textcoords="offset points", ha=ha)
    cb = fig.colorbar(cf, ax=ax, label="total Δv  [km/s]", pad=0.02)
    cb.ax.tick_params(labelsize=8)
    ax.xaxis_date()
    span_days = x.max() - x.min()          # matplotlib date numbers are in days
    if span_days < 30:
        locator  = mdates.DayLocator(interval=max(1, int(span_days // 6)))
        formatter = mdates.DateFormatter("%d %b %Y")
    elif span_days < 90:
        locator  = mdates.WeekdayLocator(byweekday=0, interval=max(1, int(span_days // (7 * 6))))
        formatter = mdates.DateFormatter("%d %b %Y")
    elif span_days < 400:
        locator  = mdates.MonthLocator(interval=max(1, int(span_days // (30 * 6))))
        formatter = mdates.DateFormatter("%b %Y")
    elif span_days < 1100:
        locator  = mdates.MonthLocator(interval=max(1, int(span_days // (30 * 7))))
        formatter = mdates.DateFormatter("%b %Y")
    else:
        locator  = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    rotation = 0 if span_days > 1100 else 30
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rotation); lbl.set_ha("right" if rotation else "center")
    ax.set_xlabel("departure date")
    ax.set_ylabel("time of flight [days]")
    ax.set_title("%s → %s porkchop  (departure + arrival Δv)" % (best["departure"], best["target"]))

    # ── Lunar-specific: mark perigee dates and annotate basin physics ──────
    if best.get("target") == "moon" and grid.get("perigee_dates"):
        y_top = float(np.max(y))
        y_bot = float(np.min(y))
        y_span = y_top - y_bot
        first_peri = True
        for pdate, pdist_km in zip(grid["perigee_dates"], grid["perigee_dists_km"]):
            px = mdates.date2num(pdate.to_datetime())
            if px < x.min() or px > x.max():
                continue
            lbl = "perigee" if first_peri else None
            ax.axvline(px, color="white", lw=0.9, ls=":", alpha=0.7, label=lbl, zorder=4)
            ax.annotate(f"{pdist_km/1e3:.0f} Mm", (px, y_top - 0.05 * y_span),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=6.5, color="white", alpha=0.85, va="top", rotation=90)
            first_peri = False
        # Synodic-period annotation: measure spacing between first two perigees
        visible = [p for p in grid["perigee_dates"]
                   if x.min() <= mdates.date2num(p.to_datetime()) <= x.max()]
        if len(visible) >= 2:
            T_syn = (visible[1] - visible[0]).to_value("jd")
            ax.text(0.01, 0.01,
                    f"Lunar synodic period ≈ {T_syn:.1f} d\n"
                    "Deeper basin = Moon closer to perigee\n"
                    "(apsidal precession, 8.85-yr cycle)",
                    transform=ax.transAxes, fontsize=7, color="white",
                    va="bottom", alpha=0.85,
                    bbox=dict(boxstyle="round", fc="0.15", ec="none", alpha=0.6))

    ax.legend(loc="upper left" if x_frac > 0.55 else "upper right", fontsize=8)
    fig.tight_layout()
    return fig

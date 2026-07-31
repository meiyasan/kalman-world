"""Orbital-transfer toolkit — Lambert, Hohmann, porkchop, gravity assist."""
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

from ...framework.models.extended import ExtendedKalmanModel
from ...framework.constants import KalmanState, KalmanObservable
from ...framework.control import KalmanController
from ...runtime.astro import (
    LU, TU, KM, DAY, GM_EARTH, GM_MOON, GM_SUN, R_EARTH, R_MOON, D_MOON, D_SUN,
    G0, gm, radius, rk4 as _rk4, body_state, planet_positions, ephemeris_samples,
    times_from_epoch, launch_site_state, moon_position, sun_position, _as_time,
)
from .translunar import _stumpff_C, _stumpff_S

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

def lambert(r1, r2, tof, mu=GM_SUN, prograde=True):
    """Solve Lambert's problem (single revolution) by the universal-variable method.

    ``r1``, ``r2`` are scaled position vectors [Mm], ``tof`` the time of flight in scaled
    units [ks]. Returns ``(v1, v2)`` velocity vectors [km/s] at the two endpoints. Raises
    ``ValueError`` if it fails to converge.
    """
    r1 = np.asarray(r1, float).reshape(3); r2 = np.asarray(r2, float).reshape(3)
    r1n = np.linalg.norm(r1); r2n = np.linalg.norm(r2)
    cross = np.cross(r1, r2)
    cosd = np.clip(np.dot(r1, r2) / (r1n * r2n), -1.0, 1.0)
    dnu = np.arccos(cosd)
    if prograde and cross[2] < 0:
        dnu = 2 * np.pi - dnu
    elif not prograde and cross[2] >= 0:
        dnu = 2 * np.pi - dnu
    A = np.sin(dnu) * np.sqrt(r1n * r2n / (1.0 - cosd))
    if A == 0.0:
        raise ValueError("Lambert: transfer angle gives A = 0")

    def yz(z):
        return r1n + r2n + A * (z * _stumpff_S(z) - 1.0) / np.sqrt(_stumpff_C(z))

    # When A < 0 (long-way transfer), dF(z=0) < 0, so Newton diverges to negative z.
    # Scan upward to find a positive z where F > 0, then Newton converges from above.
    if A < 0:
        z = 0.0
        sqmu_tof = np.sqrt(mu) * tof
        for _ in range(300):
            C_ = _stumpff_C(z); S_ = _stumpff_S(z); y_ = yz(z)
            if y_ > 0:
                chi_ = np.sqrt(y_ / C_)
                if chi_ ** 3 * S_ + A * np.sqrt(y_) > sqmu_tof:
                    break
            z += 0.5
    else:
        z = 0.0
    for _ in range(80):
        C = _stumpff_C(z); S = _stumpff_S(z)
        y = yz(z)
        if A > 0.0 and y < 0.0:                      # keep y positive
            z += 0.1
            continue
        chi = np.sqrt(y / C)
        F = chi ** 3 * S + A * np.sqrt(y) - np.sqrt(mu) * tof
        if abs(z) < 1e-6:
            dF = (np.sqrt(2.0) / 40.0) * y ** 1.5 + (A / 8.0) * (np.sqrt(y) + A * np.sqrt(0.5 / y))
        else:
            dF = (y / C) ** 1.5 * ((0.5 / z) * (C - 1.5 * S / C) + 0.75 * S ** 2 / C) \
                 + (A / 8.0) * (3.0 * S / C * np.sqrt(y) + A * np.sqrt(C / y))
        dz = F / dF
        z -= dz
        if abs(dz) < 1e-10:
            break
    y = yz(z)
    if not np.isfinite(y) or y <= 0:
        raise ValueError("Lambert: failed to converge (non-physical y)")
    f = 1.0 - y / r1n
    g = A * np.sqrt(y / mu)
    gdot = 1.0 - y / r2n
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        raise ValueError("Lambert: non-finite solution")
    return v1, v2


def hohmann_transfer(departure="earth", target="mars", epoch=None, mu=GM_SUN):
    """Analytic two-impulse Hohmann estimate between two (near-circular, coplanar) planet
    orbits — the quick first cut. Returns a dict with departure/arrival Δv [km/s], total Δv,
    time of flight [days], the required phase angle [deg], the synodic period [days] and the
    next ideal departure date (astropy ``Time``). Orbit radii are taken from the live
    ephemeris at ``epoch``.
    """
    r1 = np.linalg.norm(body_state(departure, epoch)[0])
    r2 = np.linalg.norm(body_state(target,   epoch)[0])
    a_t = 0.5 * (r1 + r2)
    v1c, v2c = np.sqrt(mu / r1), np.sqrt(mu / r2)
    vp = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))            # transfer speed at r1
    va = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))            # transfer speed at r2
    dv1, dv2 = abs(vp - v1c), abs(v2c - va)
    tof = np.pi * np.sqrt(a_t ** 3 / mu)                 # half-ellipse [ks]
    w1, w2 = np.sqrt(mu / r1 ** 3), np.sqrt(mu / r2 ** 3)
    phase = np.pi - w2 * tof                             # target lead angle at departure
    t_syn = 2 * np.pi / abs(w1 - w2)
    # current angular separation (target − departure) and wait time to the window
    th1 = np.arctan2(body_state(departure, epoch)[0][1], body_state(departure, epoch)[0][0])
    th2 = np.arctan2(body_state(target,   epoch)[0][1], body_state(target,   epoch)[0][0])
    sep = (th2 - th1) % (2 * np.pi)
    t_wait = ((sep - phase) % (2 * np.pi)) / abs(w1 - w2)
    from astropy.time import TimeDelta
    next_dep = _as_time(epoch) + TimeDelta(t_wait * TU, format="sec")
    return {
        "departure": departure, "target": target,
        "r1_AU": r1 / D_SUN, "r2_AU": r2 / D_SUN,
        "dv_depart_kms": float(dv1), "dv_arrive_kms": float(dv2), "dv_total_kms": float(dv1 + dv2),
        "tof_days": float(tof / DAY), "phase_angle_deg": float(np.degrees(phase % (2 * np.pi))),
        "synodic_period_days": float(t_syn / DAY), "next_departure": next_dep,
    }


def porkchop(departure="earth", target="mars", start=None, end=None, *,
             n_dep=40, tof_days=(150.0, 300.0), n_tof=40, mu=GM_SUN, prograde=True):
    """Scan a departure-date × time-of-flight grid, solving Lambert against the live ephemeris
    at every node, to build a porkchop plot. Returns a dict of 2-D arrays (``dv_depart``,
    ``dv_arrive``, ``dv_total``, ``c3`` and the axes ``dep_mjd``/``tof_days``) plus ``best`` —
    the minimum-total-Δv node ready to seed :class:`InterplanetaryModel`.
    """
    from astropy.time import Time, TimeDelta
    start = _as_time(start) if start is not None else Time.now()
    end = _as_time(end) if end is not None else (start + TimeDelta(1.2 * 365.25, format="jd"))
    dep_times = start + TimeDelta(np.linspace(0.0, (end - start).to_value("jd"), n_dep), format="jd")
    tofs = np.linspace(tof_days[0], tof_days[1], n_tof)

    # Two vectorised ephemeris calls: departure planet on the dep grid, target on every arrival.
    Rdep, Vdep = ephemeris_samples([departure], dep_times)[departure]
    arr_times = (dep_times[:, None] + TimeDelta(tofs[None, :] * 86400.0, format="sec")).ravel()
    Rtarg, Vtarg = ephemeris_samples([target], arr_times)[target]

    DVd = np.full((n_dep, n_tof), np.inf)
    DVa = np.full((n_dep, n_tof), np.inf)
    for i in range(n_dep):
        for j in range(n_tof):
            k = i * n_tof + j
            try:
                v1, v2 = lambert(Rdep[i], Rtarg[k], tofs[j] * DAY, mu=mu, prograde=prograde)
            except Exception:
                continue
            DVd[i, j] = np.linalg.norm(v1 - Vdep[i])
            DVa[i, j] = np.linalg.norm(v2 - Vtarg[k])
    DVt = DVd + DVa
    i0, j0 = np.unravel_index(np.nanargmin(DVt), DVt.shape)
    k0 = i0 * n_tof + j0
    v1b, v2b = lambert(Rdep[i0], Rtarg[k0], tofs[j0] * DAY, mu=mu, prograde=prograde)
    best = {
        "departure": departure, "target": target,
        "dep_date": dep_times[i0], "arrival_date": arr_times[k0], "tof_days": float(tofs[j0]),
        "dv_depart_kms": float(DVd[i0, j0]), "dv_arrive_kms": float(DVa[i0, j0]),
        "dv_total_kms": float(DVt[i0, j0]), "c3_km2s2": float(DVd[i0, j0] ** 2),
        "v_inf_kms": float(DVd[i0, j0]),
        "r0": Rdep[i0].copy(), "v0": v1b.copy(), "x0": np.concatenate([Rdep[i0], v1b]),
    }
    return {
        "dep_mjd": dep_times.mjd, "tof_days": tofs,
        "dv_depart": DVd, "dv_arrive": DVa, "dv_total": DVt, "c3": DVd ** 2,
        "best": best,
    }


def best_departure(departure="earth", target="mars", start=None, end=None, *,
                   site=None, **kw):
    """Minimum-Δv departure for an Earth→target transfer (wraps :func:`porkchop`).

    If ``site`` (a launch-site name, e.g. ``"Kennedy Space Center"``) is given, the best
    **time of day** is also returned: the hour whose Earth-rotation velocity is most aligned
    with the departure hyperbolic-excess direction, plus the eastward rotation assist [km/s]
    it contributes (it is *added* to the inertial launch velocity, so it lowers the Δv the
    vehicle must provide). Uses online IERS Earth-orientation data via astropy.
    """
    grid = porkchop(departure, target, start, end, **kw)
    best = grid["best"]
    if site is not None:
        from astropy.time import TimeDelta
        vinf_hat = best["v0"] - body_state(departure, best["dep_date"])[1]
        vinf_hat = vinf_hat / np.linalg.norm(vinf_hat)
        hours = np.arange(0.0, 24.0, 0.5)
        times = best["dep_date"] + TimeDelta(hours * 3600.0, format="sec")
        proj = np.array([np.dot(launch_site_state(site, t)[1], vinf_hat) for t in times])
        h_best = int(np.argmax(proj))
        best = dict(best)
        best["site"] = site
        best["launch_time"] = times[h_best]
        best["rotation_assist_kms"] = float(proj[h_best])
    return best


def gravity_assist(v_in_helio, v_planet_helio, planet, h_periapsis_km=300.0):
    """Patched-conic gravity-assist flyby.

    Given the spacecraft's heliocentric velocity ``v_in_helio`` [km/s] and the flyby
    planet's heliocentric velocity ``v_planet_helio`` [km/s], computes the
    **maximum-energy-gain** exit velocity achievable at a periapsis altitude
    ``h_periapsis_km`` [km].

    The planetocentric v∞ magnitude is conserved; the flyby bends its direction by at most::

        δ = 2 arcsin(μ_p / (μ_p + r_p v∞²))

    The optimal bending rotates v∞ toward the planet's heliocentric velocity — this
    maximises the dot product of the exit v∞ with the planet's orbital velocity, which
    transfers the most of the planet's orbital momentum to the spacecraft.

    Returns a dict:

    * ``v_inf_mag``      — hyperbolic-excess speed entering the SOI [km/s]
    * ``turn_angle_deg`` — maximum achievable flyby turn angle [°]
    * ``v_out_helio``    — best-case heliocentric exit velocity [km/s, 3-vector]
    * ``v_helio_in``     — heliocentric speed before flyby [km/s]
    * ``v_helio_out``    — heliocentric speed after flyby [km/s]
    * ``dv_gain_kms``    — heliocentric speed gained (positive = energy gain) [km/s]
    """
    v_in  = np.asarray(v_in_helio,     float).reshape(3)
    v_plt = np.asarray(v_planet_helio, float).reshape(3)
    v_inf     = v_in - v_plt
    v_inf_mag = np.linalg.norm(v_inf)

    r_p  = radius(planet) + h_periapsis_km * KM      # periapsis radius [Mm]
    mu_p = gm(planet)                                 # [Mm³ ks⁻²]
    sin_half = 1.0 / (1.0 + r_p * v_inf_mag**2 / mu_p)
    delta    = 2.0 * np.arcsin(np.clip(sin_half, 0.0, 1.0))   # max turn angle [rad]

    # Rotation axis: rotate v_inf toward +v_planet (perp component of v_planet w.r.t. v_inf).
    v_inf_hat = v_inf / max(v_inf_mag, 1e-30)
    vp_perp   = v_plt - np.dot(v_plt, v_inf_hat) * v_inf_hat
    n_vp      = np.linalg.norm(vp_perp)
    if n_vp > 1e-12:
        perp_hat = vp_perp / n_vp
    else:
        cand = np.array([1., 0., 0.]) if abs(v_inf_hat[0]) < 0.9 else np.array([0., 1., 0.])
        perp_hat = cand - np.dot(cand, v_inf_hat) * v_inf_hat
        perp_hat /= np.linalg.norm(perp_hat)

    v_inf_out = v_inf_mag * (np.cos(delta) * v_inf_hat + np.sin(delta) * perp_hat)
    v_out     = v_inf_out + v_plt

    return {
        "v_inf_mag":      float(v_inf_mag),
        "turn_angle_deg": float(np.degrees(delta)),
        "v_out_helio":    v_out,
        "v_helio_in":     float(np.linalg.norm(v_in)),
        "v_helio_out":    float(np.linalg.norm(v_out)),
        "dv_gain_kms":    float(np.linalg.norm(v_out) - np.linalg.norm(v_in)),
    }


def hohmann_sequence(planets, epoch=None, mu=GM_SUN):
    """Analytic multi-leg Hohmann estimate with free intermediate gravity assists.

    ``planets`` is an ordered list of body names, e.g.
    ``["earth", "venus", "jupiter"]``.  The function chains
    :func:`hohmann_transfer` leg by leg and treats every *intermediate* planet as a
    **free gravity-assist** waypoint (no Δv cost at the flyby — an optimistic lower bound
    assuming the intermediate v∞ turns are geometrically feasible).  Only the *first*
    departure burn and the *final* arrival burn count toward ``dv_total``.

    Returns a dict with:

    * ``legs``           — list of :func:`hohmann_transfer` dicts (one per leg)
    * ``tof_days``       — total time of flight summed over all legs [days]
    * ``dv_total_kms``   — departure Δv + final-arrival Δv (free flybys at waypoints)
    * ``dv_direct_kms``  — direct single-leg Hohmann from first to last planet (for comparison)
    """
    if len(planets) < 2:
        raise ValueError("hohmann_sequence needs at least two planets")
    legs = [hohmann_transfer(planets[i], planets[i + 1], epoch=epoch, mu=mu)
            for i in range(len(planets) - 1)]
    direct = hohmann_transfer(planets[0], planets[-1], epoch=epoch, mu=mu)
    return {
        "planets":        list(planets),
        "legs":           legs,
        "tof_days":       float(sum(h["tof_days"]    for h in legs)),
        "dv_total_kms":   float(legs[0]["dv_depart_kms"] + legs[-1]["dv_arrive_kms"]),
        "dv_direct_kms":  float(direct["dv_total_kms"]),
    }


def flyby_porkchop(departure, flyby, target, dep_date, *,
                   tof1_range=(80.0, 300.0), tof2_range=(300.0, 1200.0),
                   n_tof1=30, n_tof2=30, h_min_km=300.0, mu=GM_SUN):
    """Two-leg gravity-assist porkchop: ``departure`` → ``flyby`` → ``target``.

    For a **fixed departure date** scans a grid of (time-of-flight to flyby planet) ×
    (time-of-flight from flyby planet to target), solving two Lambert arcs at every node
    and checking the patched-conic flyby constraint:

    * **v∞ magnitude mismatch** → powered periapsis burn Δv (included in total)
    * **required turn angle > δ_max** at ``h_min_km`` altitude → node rejected

    Returns a dict with 2-D arrays (``dv_total``, ``dv_depart``, ``dv_arrive``,
    ``dv_flyby``) over the (n_tof1 × n_tof2) grid, axes ``tof1_days`` / ``tof2_days``,
    the fixed departure date, and ``best`` — the minimum-Δv node with state vectors
    ready to seed :class:`InterplanetaryModel` for leg 1.

    Typical ranges for Earth → Venus → Jupiter: ``tof1_range=(80, 220)``,
    ``tof2_range=(500, 1300)``.
    """
    from astropy.time import TimeDelta
    dep_date     = _as_time(dep_date)
    tofs1 = np.linspace(tof1_range[0], tof1_range[1], int(n_tof1))
    tofs2 = np.linspace(tof2_range[0], tof2_range[1], int(n_tof2))

    dep_arr      = dep_date + TimeDelta([0.0], format="sec")       # 1-element Time array
    Rdep, Vdep   = ephemeris_samples([departure], dep_arr)[departure]
    flyby_times  = dep_date + TimeDelta(tofs1 * 86400.0, format="sec")
    Rfly, Vfly   = ephemeris_samples([flyby], flyby_times)[flyby]
    all_tgt_t    = (flyby_times[:, None]
                    + TimeDelta(tofs2[None, :] * 86400.0, format="sec")).ravel()
    Rtgt, Vtgt   = ephemeris_samples([target], all_tgt_t)[target]

    r_p   = radius(flyby) + h_min_km * KM
    mu_p  = gm(flyby)

    DVd = np.full((n_tof1, n_tof2), np.inf)
    DVa = np.full((n_tof1, n_tof2), np.inf)
    DVf = np.full((n_tof1, n_tof2), np.inf)

    for i in range(n_tof1):
        try:
            vd, vf_in = lambert(Rdep[0], Rfly[i], tofs1[i] * DAY, mu=mu)
        except Exception:
            continue
        dv_dep    = np.linalg.norm(vd - Vdep[0])
        v_inf_in  = vf_in - Vfly[i]
        vi_mag    = np.linalg.norm(v_inf_in)
        e_in      = 1.0 + r_p * vi_mag**2 / mu_p

        for j in range(n_tof2):
            k = i * n_tof2 + j
            try:
                vf_out, vt = lambert(Rfly[i], Rtgt[k], tofs2[j] * DAY, mu=mu)
            except Exception:
                continue
            v_inf_out = vf_out - Vfly[i]
            vo_mag    = np.linalg.norm(v_inf_out)
            # Required turn angle between in/out v∞ vectors
            cos_a     = np.clip(np.dot(v_inf_in, v_inf_out) / (vi_mag * max(vo_mag, 1e-30)),
                                -1.0, 1.0)
            req_angle = np.arccos(cos_a)
            # Achievable turn for a powered flyby at periapsis r_p:
            # each hyperbola half-deflects by arccos(−1/e)−π/2; total = sum of both halves.
            e_out     = 1.0 + r_p * vo_mag**2 / mu_p
            delta_eff = np.arccos(-1.0 / e_in) + np.arccos(-1.0 / e_out) - np.pi
            if req_angle > delta_eff:
                continue                       # turn angle not achievable → skip
            # Powered periapsis burn to equalise v∞ magnitudes (zero for a free flyby)
            v_peri_in  = np.sqrt(vi_mag**2 + 2.0 * mu_p / r_p)
            v_peri_out = np.sqrt(vo_mag**2 + 2.0 * mu_p / r_p)
            DVd[i, j] = dv_dep
            DVa[i, j] = np.linalg.norm(vt - Vtgt[k])
            DVf[i, j] = abs(v_peri_out - v_peri_in)

    DVt = np.where(DVd < np.inf, DVd + DVa + DVf, np.inf)
    if not np.any(np.isfinite(DVt)):
        raise ValueError(
            f"flyby_porkchop: no feasible {departure}→{flyby}→{target} trajectory found; "
            "try wider tof ranges or a smaller h_min_km")
    i0, j0 = np.unravel_index(np.nanargmin(DVt), DVt.shape)
    k0     = i0 * n_tof2 + j0
    v1, v1f = lambert(Rdep[0],  Rfly[i0],  tofs1[i0] * DAY, mu=mu)
    v2f, v2  = lambert(Rfly[i0], Rtgt[k0], tofs2[j0] * DAY, mu=mu)
    fly_time = flyby_times[i0]
    arr_time = fly_time + TimeDelta(tofs2[j0] * 86400.0, format="sec")
    best = {
        "departure": departure, "flyby": flyby, "target": target,
        "dep_date": dep_date, "fly_date": fly_time, "arrival_date": arr_time,
        "tof1_days":      float(tofs1[i0]), "tof2_days": float(tofs2[j0]),
        "dv_depart_kms":  float(DVd[i0, j0]),
        "dv_arrive_kms":  float(DVa[i0, j0]),
        "dv_flyby_kms":   float(DVf[i0, j0]),
        "dv_total_kms":   float(DVt[i0, j0]),
        "r0": Rdep[0].copy(), "v0": v1.copy(),
        "x0": np.concatenate([Rdep[0], v1]),
        "r_flyby": Rfly[i0].copy(), "v_flyby_in": v1f.copy(), "v_flyby_out": v2f.copy(),
    }
    return {
        "tof1_days": tofs1, "tof2_days": tofs2,
        "dv_depart": DVd, "dv_arrive": DVa, "dv_flyby": DVf, "dv_total": DVt,
        "dep_date": dep_date, "flyby": flyby, "best": best,
    }


def plot_flyby_porkchop(grid, *, ax=None, dv_max=None, figsize=(9.5, 5.8)):
    """Render a :func:`flyby_porkchop` ``grid`` as a filled contour map over
    (tof-to-flyby) × (tof-flyby-to-target).  The minimum-Δv node is starred.
    Returns the figure."""
    import matplotlib.pyplot as plt
    DVt  = np.where(np.isfinite(grid["dv_total"]), grid["dv_total"], np.inf)
    vmin = float(np.min(DVt[np.isfinite(DVt)]))
    dv_max = vmin + 3.0 if dv_max is None else float(dv_max)
    t1, t2 = grid["tof1_days"], grid["tof2_days"]
    Z_fill = np.clip(DVt, vmin, dv_max).T
    Z_line = np.where(DVt.T <= dv_max, DVt.T, np.nan)
    levels = np.linspace(vmin, dv_max, 21)
    fig = ax.figure if ax is not None else plt.subplots(figsize=figsize)[1].figure
    if ax is None:
        ax = fig.axes[0]
    cf = ax.contourf(t1, t2, Z_fill, levels=levels, cmap="turbo", extend="max")
    cl = ax.contour(t1, t2, Z_line, levels=levels[::3], colors="k", linewidths=0.4, alpha=0.4)
    ax.clabel(cl, fmt="%.2f", fontsize=7, inline=True)
    best = grid["best"]
    ax.plot(best["tof1_days"], best["tof2_days"],
            "*", color="white", mec="k", ms=20, zorder=6, label="min Δv")
    ax.annotate("min Δv = %.2f km/s\ntof₁ %.0f d  |  tof₂ %.0f d"
                % (best["dv_total_kms"], best["tof1_days"], best["tof2_days"]),
                (best["tof1_days"], best["tof2_days"]),
                color="white", fontsize=8.5, fontweight="bold",
                xytext=(10, 10), textcoords="offset points")
    fig.colorbar(cf, ax=ax, label="total Δv  [km/s]")
    ax.set_xlabel("time to %s flyby [days]" % best["flyby"])
    ax.set_ylabel("time %s → %s [days]" % (best["flyby"], best["target"]))
    ax.set_title("%s → %s → %s  (Δv grid, fixed departure %s)"
                 % (best["departure"], best["flyby"], best["target"],
                    best["dep_date"].iso[:10]))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_gravity_assist(ga, planet="jupiter", *, n_pts=200, figsize=(8, 7),
                        frames=30, interval=90):
    """Animated patched-conic flyby in the planet's rest frame.

    The plot is **zoomed to the periapsis region** so the planet and the hyperbolic
    bend are both clearly visible.  The full asymptotic extent (millions of km) is
    accessible by panning/zooming the figure after the animation stops.

    ``ga`` is the dict returned by :func:`gravity_assist`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Arc
    from ...helpers import animate

    v_inf = ga["v_inf_mag"]
    r_p   = radius(planet) + 300.0 * KM
    mu_p  = gm(planet)
    e     = 1.0 + r_p * v_inf**2 / mu_p
    a_hyp = r_p / (e - 1.0)

    th_inf = np.arccos(-1.0 / e)
    th = np.linspace(-th_inf * 0.98, th_inf * 0.98, n_pts)
    r_h = a_hyp * (e**2 - 1.0) / (1.0 + e * np.cos(th))
    x_h = r_h * np.cos(th)
    y_h = r_h * np.sin(th)

    R_planet = radius(planet)      # Mm  (1 Mm = 10³ km)

    # Display in Mm (megameters = 10³ km): avoids six-digit km numbers for giant planets
    # while remaining more intuitive than AU (which is far too large for a flyby region).
    lim_zoom = 8.0 * r_p          # Mm — ±8× periapsis keeps planet at ~12% of axis width
    lim_full = float(r_h.max() * 1.15)   # Mm

    # Periapsis arc (zoomed sub-set for the dashed reference)
    mask_zoom = np.abs(r_h) <= lim_full
    th_z = th[mask_zoom]
    r_z  = r_h[mask_zoom]
    x_z, y_z = r_z * np.cos(th_z), r_z * np.sin(th_z)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-lim_zoom, lim_zoom); ax.set_ylim(-lim_zoom, lim_zoom)
    ax.set_aspect("equal", adjustable="box"); ax.autoscale(False)

    # Planet body — draw at physical size; guaranteed visible at ×8 periapsis zoom
    planet_col = _BODY_COLOR.get(planet, "gray")
    ax.add_patch(Circle((0, 0), R_planet, color=planet_col, alpha=0.75, zorder=3))
    ax.text(0, R_planet * 1.12, planet.capitalize(), fontsize=10, color=planet_col,
            ha="center", va="bottom", fontweight="bold", zorder=4)

    # Dashed reference: full hyperbolic path (clipped to axes by matplotlib)
    ax.plot(x_z, y_z, ":", color="0.65", lw=0.8, label="hyperbolic path")

    # Asymptote lines (full extent, clipped)
    L_asym = lim_full
    ax.plot([-L_asym, 0], [L_asym * np.tan(np.pi - th_inf), 0],
            "--", color="steelblue", lw=0.9, alpha=0.7, label="in asymptote")
    out_angle = th_inf
    ax.plot([0, L_asym * np.cos(out_angle)], [0, L_asym * np.sin(out_angle)],
            "--", color="darkorange", lw=0.9, alpha=0.7, label="out asymptote")

    trail,  = ax.plot([], [], color="crimson", lw=2.2, label="spacecraft")
    sc_pt,  = ax.plot([], [], "o", color="white", ms=7, mec="crimson", zorder=6)
    v_arr   = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.0))
    title   = ax.set_title("")

    # Physics info box — r_p shown in km for physical clarity
    info = (
        "Gravity assist — planet rest frame\n"
        "v∞ = %.2f km/s   (hyperbolic excess)\n"
        "e  = %.4f   rp = %.0f km\n"
        "Turn δ = %.0f°   Δv_helio = +%.2f km/s"
        % (v_inf, e, r_p * 1e3, ga["turn_angle_deg"], ga["dv_gain_kms"])
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8, va="top",
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.88))

    ax.set_xlabel("x [Mm]"); ax.set_ylabel("y [Mm]"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Velocity at each point (vis-viva + tangent direction)
    v_mag = np.sqrt(v_inf**2 + 2.0 * mu_p / r_h)
    dx = np.gradient(x_h); dy = np.gradient(y_h)
    ds = np.sqrt(dx**2 + dy**2)
    tx, ty = dx / ds, dy / ds

    v_scale = lim_zoom * 0.22 / v_mag.max()
    frame_idx = np.linspace(0, n_pts - 1, frames).astype(int)

    def _draw(k):
        i = min(int(k), n_pts - 1)
        trail.set_data(x_h[:i+1], y_h[:i+1])
        px, py = x_h[i], y_h[i]
        sc_pt.set_data([px], [py])
        vx = tx[i] * v_mag[i] * v_scale
        vy = ty[i] * v_mag[i] * v_scale
        v_arr.set_position((px, py)); v_arr.xy = (px + vx, py + vy)
        phase = "approach" if i < n_pts // 2 else "departure"
        title.set_text("%s flyby  |  v = %.1f km/s  |  turn δ = %.0f°  |  %s"
                       % (planet, v_mag[i], ga["turn_angle_deg"], phase))
        return trail, sc_pt, v_arr, title

    return animate(fig, _draw, frames=frame_idx, interval=interval)


def plot_two_leg(best, *, planets=("earth", "mars", "venus"),
                 n_arc=200, figsize=(7.5, 7.5), frames=30, interval=90):
    """Animated two-leg gravity-assist trajectory from a :func:`flyby_porkchop` ``best`` dict.

    Shows both Lambert arcs with the spacecraft moving through the flyby and onward,
    planet positions updating in real time, and event dates annotated.
    """
    import matplotlib.pyplot as plt
    from astropy.time import TimeDelta
    from ...helpers import animate

    dep_name, fly_name, tgt_name = best["departure"], best["flyby"], best["target"]
    t_dep = best["dep_date"]
    t_fly = best["fly_date"]
    t_arr = best["arrival_date"]

    r_dep = best["r0"]
    r_fly = best["r_flyby"]
    v1    = best["v0"]
    v2    = best["v_flyby_out"]

    # Target position at arrival
    r_tgt = body_state(tgt_name, t_arr)[0]

    arc1 = _kep_arc(r_dep, v1, best["tof1_days"] * DAY, n_arc)
    arc2 = _kep_arc(r_fly, v2, best["tof2_days"] * DAY, n_arc)
    traj = np.vstack([arc1, arc2[1:]])
    N_total = len(traj) - 1
    t_total = best["tof1_days"] + best["tof2_days"]
    t_sc = np.concatenate([
        np.linspace(0, best["tof1_days"], n_arc + 1),
        np.linspace(best["tof1_days"], t_total, n_arc + 1)[1:],
    ])

    # Planet positions sampled once per animation frame (direct 1:1 mapping — no interpolation)
    frame_idx = np.linspace(0, N_total, frames).astype(int)
    frame_times = [t_dep + TimeDelta(float(t_sc[i]) * 86400, format="sec") for i in frame_idx]
    all_bodies = sorted(set([dep_name, fly_name, tgt_name] + list(planets)))
    planet_pos = {b: np.array([body_state(b, t)[0] for t in frame_times]) / D_SUN
                  for b in all_bodies}

    AU = D_SUN
    fig, ax = plt.subplots(figsize=figsize)
    lim = 1.1 * max(np.linalg.norm(r_fly) / AU, np.linalg.norm(r_tgt) / AU)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box"); ax.autoscale(False)

    # True elliptical orbit paths (computed from ephemeris at departure epoch)
    for b in all_bodies:
        oc = _orbit_ellipse(b, t_dep)
        if oc is not None:
            ax.plot(oc[:, 0], oc[:, 1], ":", color=_BODY_COLOR.get(b, "gray"), lw=0.9, alpha=0.4)

    ln1, = ax.plot([], [], color="royalblue",  lw=2.2,
                   label="Leg 1  %s→%s  (Lambert)" % (dep_name, fly_name))
    ln2, = ax.plot([], [], color="darkorange", lw=2.2,
                   label="Leg 2  %s→%s  (Lambert)" % (fly_name, tgt_name))
    sc_pt, = ax.plot([], [], "o", color="white", ms=7, mec="k", zorder=6)

    ppts = {}
    for b in all_bodies:
        ppts[b] = ax.plot([], [], "o", color=_BODY_COLOR.get(b, "gray"),
                          ms=_BODY_MS.get(b, 6), mec="none",
                          zorder=5, label=b)[0]

    ax.plot(0, 0, "o", color="gold", ms=_BODY_MS["sun"], zorder=6)
    # Static event markers
    for r_vec, marker, col, lbl in [
        (r_dep, "o", _BODY_COLOR.get(dep_name, "gray"),
         "%s dep  %s" % (dep_name, t_dep.iso[:10])),
        (r_fly, "+", _BODY_COLOR.get(fly_name, "gray"),
         "%s flyby  %s" % (fly_name, t_fly.iso[:10])),
        (r_tgt, "*", _BODY_COLOR.get(tgt_name, "gray"),
         "%s arr  %s" % (tgt_name, t_arr.iso[:10])),
    ]:
        if marker == "+":
            ax.plot(r_vec[0] / AU, r_vec[1] / AU, marker, color=col,
                    ms=12, mew=2.2, mec=col, zorder=7)
        elif marker == "*":
            # white star with planet-coloured edge — distinct from the filled planet dot
            ax.plot(r_vec[0] / AU, r_vec[1] / AU, marker, color="white",
                    ms=14, mew=1.5, mec=col, zorder=7)
        else:
            ax.plot(r_vec[0] / AU, r_vec[1] / AU, marker, color=col,
                    ms=12, mew=0, mec="none", zorder=7)
        ax.annotate(lbl, (r_vec[0] / AU, r_vec[1] / AU),
                    xytext=(8, 6), textcoords="offset points", fontsize=7.5, color="0.3")

    title = ax.set_title("")
    ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]"); ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    # FuncAnimation calls _draw(i) where i is the ENUMERATION index (0..frames-1),
    # NOT the value from frame_idx.  Planet positions are already sampled at each frame.
    def _draw(fi):
        fi = min(int(fi), frames - 1)
        k  = frame_idx[fi]          # trajectory index (0..N_total)
        t_now = t_sc[k]
        if k <= n_arc:
            ln1.set_data(arc1[:k+1, 0] / D_SUN, arc1[:k+1, 1] / D_SUN)
            ln2.set_data([], [])
        else:
            ln1.set_data(arc1[:, 0] / D_SUN, arc1[:, 1] / D_SUN)
            ln2.set_data(arc2[:k-n_arc+1, 0] / D_SUN, arc2[:k-n_arc+1, 1] / D_SUN)
        sc_pt.set_data([traj[k, 0] / D_SUN], [traj[k, 1] / D_SUN])
        for b, pt in ppts.items():
            pt.set_data([planet_pos[b][fi, 0]], [planet_pos[b][fi, 1]])
        leg = "Leg 1: %s → %s" % (dep_name, fly_name) if k <= n_arc \
              else "Leg 2: %s → %s" % (fly_name, tgt_name)
        date = (t_dep + TimeDelta(float(t_now) * 86400, format="sec")).iso[:10]
        title.set_text("%s → %s → %s   t = %.0f d   %s   %s"
                       % (dep_name, fly_name, tgt_name, t_now, date, leg))
        return ln1, ln2, sc_pt, title, *ppts.values()

    return animate(fig, _draw, frames=frames, interval=interval)


def chain_lambert(sequence, dates, *, mu=GM_SUN):
    """Chain Lambert solves for a multi-leg gravity-assist trajectory.

    Parameters
    ----------
    sequence : list of str
        Ordered body names, e.g. ``["earth","venus","venus","earth","jupiter","saturn"]``.
        Length N.
    dates : list of astropy Time
        One timestamp per body (length N).  The i-th entry is the date the spacecraft
        is at ``sequence[i]``.

    Returns
    -------
    list of dicts, one per leg.  Each dict contains:

    * ``departure``, ``arrival``  — body name strings
    * ``dep_date``, ``arr_date``  — astropy Time
    * ``tof_days``                — float, time of flight
    * ``r_dep``, ``r_arr``        — (3,) position vectors [Mm]
    * ``v_dep``, ``v_arr``        — (3,) heliocentric velocity vectors [km/s]
    * ``dv_flyby_kms``            — Δv at the intermediate flyby (0 for a free slingshot)
    """
    from astropy.time import TimeDelta
    if len(sequence) != len(dates):
        raise ValueError("sequence and dates must have the same length")
    legs = []
    for i in range(len(sequence) - 1):
        r_dep, _ = body_state(sequence[i],     dates[i])
        r_arr, _ = body_state(sequence[i + 1], dates[i + 1])
        tof_days  = float((dates[i + 1] - dates[i]).to_value("jd"))
        v_dep, v_arr = lambert(r_dep, r_arr, tof_days * DAY, mu=mu)
        dv_flyby = 0.0
        if i > 0:
            dv_flyby = float(np.linalg.norm(v_dep - legs[-1]["v_arr"]))
        legs.append({
            "departure":    sequence[i],
            "arrival":      sequence[i + 1],
            "dep_date":     dates[i],
            "arr_date":     dates[i + 1],
            "tof_days":     tof_days,
            "r_dep": r_dep, "v_dep": v_dep,
            "r_arr": r_arr, "v_arr": v_arr,
            "dv_flyby_kms": dv_flyby,
        })
    return legs


def plot_multi_leg(legs, *, title=None, extra_planets=("mars",),
                   n_arc=150, figsize=(8.5, 8.5), frames=30, interval=80,
                   zoom_lim=None):
    """Animated N-leg gravity-assist trajectory from a :func:`chain_lambert` leg list.

    Parameters
    ----------
    legs : list of dicts returned by :func:`chain_lambert`.
    title : str | None   Figure title (auto-generated if None).
    extra_planets : tuple of str   Additional bodies to show (beyond those in the legs).
    n_arc : int          RK4 steps per leg arc.
    frames : int         Number of animation frames.
    zoom_lim : float | None
        When given, adds a second panel (right) showing the inner solar system zoomed
        to ±zoom_lim AU — useful for multi-flyby trajectories where close encounters
        are invisible at the full-mission scale.
    """
    import matplotlib.pyplot as plt
    from astropy.time import TimeDelta
    from ...helpers import animate

    AU = D_SUN

    # ── Build arcs ──────────────────────────────────────────────────────────
    arcs  = [_kep_arc(lg["r_dep"], lg["v_dep"], lg["tof_days"] * DAY, n_arc) for lg in legs]
    traj  = np.vstack([arcs[0]] + [a[1:] for a in arcs[1:]])
    N_total = len(traj) - 1
    t0_ = 0.0
    t_sc_parts = []
    for lg in legs:
        t_sc_parts.append(np.linspace(t0_, t0_ + lg["tof_days"], n_arc + 1))
        t0_ += lg["tof_days"]
    t_sc = np.concatenate([t_sc_parts[0]] + [p[1:] for p in t_sc_parts[1:]])

    t_dep_global = legs[0]["dep_date"]

    # ── Planet position sampling ─────────────────────────────────────────────
    frame_idx   = np.linspace(0, N_total, frames).astype(int)
    frame_times = [t_dep_global + TimeDelta(float(t_sc[fi]) * 86400, format="sec")
                   for fi in frame_idx]
    all_bodies = sorted(set(
        [lg["departure"] for lg in legs] + [legs[-1]["arrival"]] + list(extra_planets)
    ))
    planet_pos = {b: np.array([body_state(b, t)[0] for t in frame_times]) / AU
                  for b in all_bodies}

    # ── Axis limits ──────────────────────────────────────────────────────────
    lim = 1.15 * max(np.linalg.norm(lg["r_arr"]) / AU for lg in legs)

    # ── Static event geometry ────────────────────────────────────────────────
    event_bodies = [legs[0]["departure"]] + [lg["arrival"] for lg in legs]
    event_dates  = [legs[0]["dep_date"]]  + [lg["arr_date"] for lg in legs]
    event_r      = [legs[0]["r_dep"]]     + [lg["r_arr"]    for lg in legs]

    # ── Shared title string ──────────────────────────────────────────────────
    ttl_str = title or (" → ".join(dict.fromkeys(event_bodies)) +
                        "  (%s – %s)" % (legs[0]["dep_date"].iso[:10],
                                         legs[-1]["arr_date"].iso[:10]))

    # ── Helper: populate one axis with all static and animated elements ───────
    def _populate(ax, ax_lim, show_legend):
        ax.set_xlim(-ax_lim, ax_lim); ax.set_ylim(-ax_lim, ax_lim)
        ax.set_aspect("equal", adjustable="box"); ax.autoscale(False)
        ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]"); ax.grid(alpha=0.3)

        for b in all_bodies:
            oc = _orbit_ellipse(b, t_dep_global)
            if oc is not None:
                ax.plot(oc[:, 0], oc[:, 1], ":", color=_BODY_COLOR.get(b, "gray"),
                        lw=0.8, alpha=0.35)

        lns = []
        for i, lg in enumerate(legs):
            col = _LEG_COLORS[i % len(_LEG_COLORS)]
            lbl = "Leg %d  %s→%s" % (i + 1, lg["departure"], lg["arrival"])
            ln, = ax.plot([], [], color=col, lw=2.2, label=lbl)
            lns.append(ln)

        sc, = ax.plot([], [], "o", color="black", ms=4, mec="none", zorder=7)
        ax.plot(0, 0, "o", color="gold", ms=_BODY_MS["sun"], zorder=6)

        pp = {b: ax.plot([], [], "o", color=_BODY_COLOR.get(b, "gray"),
                         ms=_BODY_MS.get(b, 6), mec="none", zorder=5, label=b)[0]
              for b in all_bodies}

        for j, (b, d, r) in enumerate(zip(event_bodies, event_dates, event_r)):
            col = _BODY_COLOR.get(b, "gray")
            if j == 0:
                mark, ms_, mew_ = "o", 11, 0
            elif j == len(event_bodies) - 1:
                mark, ms_, mew_ = "*", 12, 1.2
            else:
                mark, ms_, mew_ = "+", 12, 2.2   # flyby → cross
            if mark == "*":
                # white star with planet-coloured edge — distinct from the filled planet dot
                ax.plot(r[0] / AU, r[1] / AU, mark, color="white",
                        ms=ms_, mew=mew_, mec=col, zorder=8)
            elif mark == "+":
                ax.plot(r[0] / AU, r[1] / AU, mark, color=col,
                        ms=ms_, mew=mew_, mec=col, zorder=8)
            else:
                ax.plot(r[0] / AU, r[1] / AU, mark, color=col,
                        ms=ms_, mew=0, mec="none", zorder=8)
            lbl_ev = ("Dep" if j == 0 else ("Arr" if j == len(event_bodies) - 1 else "Flyby"))
            ax.annotate("%s  %s  %s" % (lbl_ev, b, d.iso[:10]),
                        (r[0] / AU, r[1] / AU),
                        xytext=(7, 6), textcoords="offset points", fontsize=7, color="0.3",
                        annotation_clip=True)

        if show_legend:
            ax.legend(loc="lower left", fontsize=7.5)
        return lns, sc, pp

    # ── Auto-compute zoom_lim from inner arcs when requested ─────────────────
    if zoom_lim is True or zoom_lim == "auto":
        _inner_ext = [np.abs(arc[:, :2]).max() / AU for arc in arcs
                      if np.linalg.norm(arc[:, :2], axis=1).max() / AU < 4.0]
        zoom_lim = (max(_inner_ext) * 1.25) if _inner_ext else None

    # Pre-compute per-leg endpoint distances (AU) for inner-panel clip filtering.
    # Suppress only legs where BOTH endpoints are outside the inner panel — those arcs
    # lie entirely in the outer solar system and would clutter the inner view.
    # Legs that START inside (e.g. earth→jupiter) clip naturally at the axis boundary.
    _r_dep_AU = [np.linalg.norm(lg["r_dep"][:2]) / AU for lg in legs]
    _r_arr_AU = [np.linalg.norm(lg["r_arr"][:2]) / AU for lg in legs]

    # ── Figure ───────────────────────────────────────────────────────────────
    if zoom_lim is not None:
        fw, fh = figsize
        fig, (ax_inner, ax_outer) = plt.subplots(1, 2, figsize=(fw * 2, fh))
        fig.subplots_adjust(wspace=0.08)
        lns_inner, sc_inner, pp_inner = _populate(ax_inner, zoom_lim, show_legend=False)
        ax_inner.set_title("inner solar system  (±%.2g AU)" % zoom_lim, fontsize=9)
        lns_outer, sc_outer, pp_outer = _populate(ax_outer, lim, show_legend=True)
        fig.suptitle(ttl_str, fontsize=9)
        ax_outer.set_ylabel("")
        frame_title = ax_outer.text(0.5, 1.02, "", transform=ax_outer.transAxes,
                                    ha="center", fontsize=8.5, color="0.35")
    else:
        fig, ax_outer = plt.subplots(figsize=figsize)
        ax_inner = None
        lns_inner = sc_inner = pp_inner = None
        lns_outer, sc_outer, pp_outer = _populate(ax_outer, lim, show_legend=True)
        fig.suptitle(ttl_str, fontsize=9)
        frame_title = ax_outer.text(0.5, 1.02, "", transform=ax_outer.transAxes,
                                    ha="center", fontsize=8.5, color="0.35")

    # ── Leg boundary indices ──────────────────────────────────────────────────
    leg_end_idx = []
    k_ = 0
    for lg in legs:
        k_ += n_arc
        leg_end_idx.append(k_)

    def _update_ax(lns, sc, pp, k, fi, clip_lim=None):
        k0 = 0
        for i, (ln, end) in enumerate(zip(lns, leg_end_idx)):
            # Inner panel: suppress only legs where BOTH endpoints are outside the zoom box.
            # Legs with one endpoint inside (e.g. earth→jupiter) are drawn and clipped by axes.
            if clip_lim is not None and _r_dep_AU[i] > clip_lim and _r_arr_AU[i] > clip_lim:
                ln.set_data([], [])
                k0 = end
                continue
            if k >= end:
                ln.set_data(arcs[i][:, 0] / AU, arcs[i][:, 1] / AU)
            elif k >= k0:
                ln.set_data(arcs[i][:k - k0 + 1, 0] / AU, arcs[i][:k - k0 + 1, 1] / AU)
            else:
                ln.set_data([], [])
            k0 = end
        sc_x, sc_y = traj[k, 0] / AU, traj[k, 1] / AU
        if clip_lim is not None and (abs(sc_x) > clip_lim or abs(sc_y) > clip_lim):
            sc.set_data([], [])   # hide dot when spacecraft is outside the inner panel
        else:
            sc.set_data([sc_x], [sc_y])
        for b, pt in pp.items():
            pt.set_data([planet_pos[b][fi, 0]], [planet_pos[b][fi, 1]])

    def _draw(fi):
        fi = min(int(fi), frames - 1)
        k  = frame_idx[fi]
        leg_i = next((i for i, end in enumerate(leg_end_idx) if k <= end), len(legs) - 1)
        _update_ax(lns_outer, sc_outer, pp_outer, k, fi)
        if ax_inner is not None:
            _update_ax(lns_inner, sc_inner, pp_inner, k, fi, clip_lim=zoom_lim)
        lg_ = legs[leg_i]
        date = (t_dep_global + TimeDelta(float(t_sc[k]) * 86400, format="sec")).iso[:10]
        frame_title.set_text("t = %.0f d   %s   Leg %d: %s → %s"
                             % (t_sc[k], date, leg_i + 1, lg_["departure"], lg_["arrival"]))
        artists = (*lns_outer, sc_outer, frame_title, *pp_outer.values())
        if ax_inner is not None:
            artists += (*lns_inner, sc_inner, *pp_inner.values())
        return artists

    return animate(fig, _draw, frames=frames, interval=interval)


def lunar_porkchop(start=None, end=None, *, n_dep=40, tof_days=(3.0, 6.0), n_tof=30,
                   r_park_km=400.0):
    """Geocentric departure-date × time-of-flight scan for Earth–Moon transfers.

    Solves Lambert's problem in the **geocentric frame** (Earth = central body, Moon = target)
    at every grid node to build a trans-lunar porkchop plot.  The Moon's exact position comes
    from the network-free ``builtin`` ephemeris (astropy).

    ``dv_depart`` is the **TLI burn** (perigee impulsive Δv from a circular parking orbit at
    ``r_park_km`` altitude); ``dv_arrive`` is the arrival speed relative to the Moon (the LOI
    burn needed to capture). Returns the same dict structure as :func:`porkchop` so
    :func:`plot_porkchop` works unchanged.
    """
    from astropy.time import Time, TimeDelta
    start = _as_time(start) if start is not None else Time.now()
    end   = _as_time(end)   if end is not None else (start + TimeDelta(60.0, format="jd"))
    dep_times = start + TimeDelta(
        np.linspace(0.0, (end - start).to_value("jd"), n_dep), format="jd")
    tofs = np.linspace(tof_days[0], tof_days[1], n_tof)

    # Moon geocentric positions at departure and arrival (Mm, km/s)
    moon_dep_R, _ = ephemeris_samples(["moon"], dep_times, origin="geocentric")["moon"]
    arr_times = (dep_times[:, None]
                 + TimeDelta(tofs[None, :] * 86400.0, format="sec")).ravel()
    moon_arr_R, moon_arr_V = ephemeris_samples(["moon"], arr_times, origin="geocentric")["moon"]

    # Parking orbit: spacecraft departs from radius r_park at periapsis
    r_park = R_EARTH + r_park_km * KM
    v_circ = np.sqrt(GM_EARTH / r_park)
    # Fixed departure point (r_park, 0, 0) — a good approximation because r_park << D_MOON
    r1 = np.array([r_park, 0.0, 0.0])

    DVd = np.full((n_dep, n_tof), np.inf)
    DVa = np.full((n_dep, n_tof), np.inf)
    for i in range(n_dep):
        for j in range(n_tof):
            k = i * n_tof + j
            r2 = moon_arr_R[k, :3]
            if np.linalg.norm(r2) < R_MOON:
                continue                              # Moon too close to Earth (bad ephemeris)
            try:
                v1, v2 = lambert(r1, r2, tofs[j] * DAY, mu=GM_EARTH)
            except Exception:
                continue
            DVd[i, j] = abs(np.linalg.norm(v1) - v_circ)   # TLI Δv from parking orbit
            DVa[i, j] = np.linalg.norm(v2 - moon_arr_V[k, :3])   # LOI speed vs Moon

    DVt = DVd + DVa
    i0, j0 = np.unravel_index(np.nanargmin(DVt), DVt.shape)
    k0 = i0 * n_tof + j0
    v1b, v2b = lambert(r1, moon_arr_R[k0, :3], tofs[j0] * DAY, mu=GM_EARTH)
    best = {
        "departure": "earth", "target": "moon",
        "dep_date": dep_times[i0], "arrival_date": arr_times[k0], "tof_days": float(tofs[j0]),
        "dv_depart_kms": float(DVd[i0, j0]), "dv_arrive_kms": float(DVa[i0, j0]),
        "dv_total_kms": float(DVt[i0, j0]), "c3_km2s2": float(DVd[i0, j0] ** 2),
        "r0": r1.copy(), "v0": v1b.copy(),
    }
    # Find lunar perigee dates within the scan window (local distance minima).
    # Sample Moon distance at 6-hour resolution and pick local minima.
    _fine = start + TimeDelta(
        np.linspace(0.0, (end - start).to_value("jd"), int((end - start).to_value("jd") * 4)),
        format="jd")
    _moon_R_fine, _ = ephemeris_samples(["moon"], _fine, origin="geocentric")["moon"]
    _dist_fine = np.linalg.norm(_moon_R_fine, axis=1)
    _peri_idx = [i for i in range(1, len(_dist_fine) - 1)
                 if _dist_fine[i] < _dist_fine[i-1] and _dist_fine[i] < _dist_fine[i+1]]
    perigee_dates = [_fine[i] for i in _peri_idx]
    perigee_dists_km = [float(_dist_fine[i]) * 1e3 for i in _peri_idx]   # Mm → km
    return {
        "dep_mjd": dep_times.mjd, "tof_days": tofs,
        "dv_depart": DVd, "dv_arrive": DVa, "dv_total": DVt, "c3": DVd ** 2,
        "best": best,
        "perigee_dates": perigee_dates,        # list[astropy.time.Time]
        "perigee_dists_km": perigee_dists_km,  # list[float] — Moon distance at each perigee
    }


# ===========================================================================
#  3) Interplanetary cruise — heliocentric N-body EKF (the end goal)
# ===========================================================================

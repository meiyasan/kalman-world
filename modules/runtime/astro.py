"""
Astrodynamics runtime services for the Kalman framework.

**Physical constants** are sourced from :mod:`astropy.constants` / :mod:`astropy.units`
wherever astropy ships them (Earth, Sun, Jupiter, the astronomical unit, *G*, standard
gravity ``g0``).  The Moon and the remaining planets — which astropy does *not* expose as
named constants — use cited IAU 2015 / JPL DE440 GM and mean-radius values.  Everything is
then expressed in the framework's **scaled units** (``length = 1 Mm = 1e6 m``,
``time = 1 ks = 1e3 s``) that keep the orbital/ascent models well-conditioned.

This module also provides the shared services both app modules build on:

* :func:`rk4` — the 4th-order Runge–Kutta step used by every integrator;
* :func:`body_state` / :func:`planet_positions` — **exact** ephemeris position & velocity of
  any solar-system body at a given time (astropy's network-free ``builtin`` ephemeris), with
  an analytic circular-orbit fallback when astropy is unavailable;
* :func:`launch_site` / :func:`launch_site_state` — launch-pad coordinates resolved from a
  **library** (``EarthLocation.of_address`` online geocoding) and their Earth-rotation state.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
#  Physical constants — astropy where available, cited literals as a fallback
# ---------------------------------------------------------------------------
try:                                                   # pragma: no cover - import guard
    from astropy import constants as _const
    _G_SI        = _const.G.value
    _GM_EARTH_SI = _const.GM_earth.value
    _GM_SUN_SI   = _const.GM_sun.value
    _GM_JUP_SI   = _const.GM_jup.value
    _R_EARTH_SI  = _const.R_earth.value
    _R_SUN_SI    = _const.R_sun.value
    _AU_SI       = _const.au.value
    _G0_SI       = _const.g0.value
    HAVE_ASTROPY = True
except Exception:                                      # astropy absent -> CODATA/IAU literals
    _G_SI, _GM_EARTH_SI, _GM_SUN_SI, _GM_JUP_SI = 6.67430e-11, 3.986004418e14, 1.32712440018e20, 1.26686534e17
    _R_EARTH_SI, _R_SUN_SI, _AU_SI, _G0_SI      = 6.3781e6, 6.957e8, 1.495978707e11, 9.80665
    HAVE_ASTROPY = False

# Bodies astropy does not ship as named constants -> JPL DE440 / IAU 2015 GM [m^3 s^-2].
_GM_SI = {
    "sun":     _GM_SUN_SI,
    "mercury": 2.2031868551e13,
    "venus":   3.2485859200e14,
    "earth":   _GM_EARTH_SI,
    "moon":    4.9028001184e12,
    "mars":    4.2828375816e13,
    "jupiter": _GM_JUP_SI,
    "saturn":  3.7940584841e16,
    "uranus":  5.7945563999e15,
    "neptune": 6.8365271006e15,
}
# Mean (volumetric) radii [m].
_R_SI = {
    "sun": _R_SUN_SI, "mercury": 2.4397e6, "venus": 6.0518e6, "earth": _R_EARTH_SI,
    "moon": 1.7374e6, "mars": 3.3895e6, "jupiter": 6.9911e7, "saturn": 5.8232e7,
    "uranus": 2.5362e7, "neptune": 2.4622e7,
}

# ---------------------------------------------------------------------------
#  Scaled units (1 Mm, 1 ks) and conversions
# ---------------------------------------------------------------------------
LU, TU = 1.0e6, 1.0e3            # 1 Mm (1e6 m), 1 ks (1e3 s)
KM  = 1.0e3 / LU                 # 1 km in LU            (= 1e-3)
DAY = 86400.0 / TU               # 1 day in TU           (= 86.4)
VU  = LU / TU                    # 1 scaled velocity unit = 1 km/s
_GM_SCALE = TU**2 / LU**3        # SI GM (m^3 s^-2) -> scaled


def gm(body: str) -> float:
    """Standard gravitational parameter ``GM`` of ``body`` in **scaled** units."""
    return _GM_SI[body.lower()] * _GM_SCALE


def radius(body: str) -> float:
    """Mean radius of ``body`` in **scaled** units (Mm)."""
    return _R_SI[body.lower()] / LU


# Named scaled constants kept for the existing models and their importers.
GM_EARTH = gm("earth"); GM_MOON = gm("moon"); GM_SUN = gm("sun")
R_EARTH  = radius("earth"); R_MOON = radius("moon"); R_SUN = radius("sun")
D_SUN    = _AU_SI / LU                              # Earth–Sun mean distance (1 au)
D_MOON   = 3.844e8 / LU                             # Moon semimajor axis (analytic fallback)
OMEGA_MOON = 2 * np.pi / (27.321661  * DAY)         # sidereal month  (analytic fallback)
OMEGA_SUN  = 2 * np.pi / (365.256363 * DAY)         # sidereal year   (analytic fallback)

# Atmosphere / surface gravity for the near-surface launch & landing models.
G0      = _G0_SI                 # standard gravity [m/s^2]
RHO0    = 1.225                  # sea-level air density [kg/m^3] (ISA)
H_ATM   = 8500.0                 # density scale height [m]  (SI liftoff/landing models)
H_SCALE = 8.5 * KM               # density scale height (scaled ascent model)


# ---------------------------------------------------------------------------
#  Numerical integrator (shared) — thin shim over the single RK4Integrator so
#  there is one home for the RK4 math. Signature kept (``deriv(s, t)``) for the
#  rocket/space callers; the integrator's step is ``f(x, u, t)``, adapted below.
# ---------------------------------------------------------------------------
_RK4 = None

def rk4(s, t, dt, deriv):
    """One 4th-order Runge–Kutta step of ``ds/dt = deriv(s, t)`` (delegates to
    :class:`~kalman.framework.integrators.RK4Integrator`)."""
    global _RK4
    if _RK4 is None:
        from ..framework.integrators import RK4Integrator
        _RK4 = RK4Integrator()
    return _RK4.step(lambda x, u, tt: deriv(x, tt), s, t=t, dt=dt)


# ---------------------------------------------------------------------------
#  Exact ephemeris (astropy builtin — no download) with analytic fallback
# ---------------------------------------------------------------------------
EPHEMERIS = "builtin"            # network-free; set to 'de440' for the JPL kernel


def _as_time(time):
    """Coerce ``time`` (``astropy.time.Time``, ISO string, ``datetime`` or ``None``) to Time."""
    from astropy.time import Time
    if time is None:
        return Time.now()
    if isinstance(time, Time):
        return time
    return Time(time)


def moon_position(t, phase=0.0):
    """Analytic Moon position (scaled, geocentric) on a circular orbit — fallback only."""
    a = OMEGA_MOON * t + phase
    return D_MOON * np.array([np.cos(a), np.sin(a), 0.0])


def sun_position(t, phase=0.0):
    """Analytic Sun position (scaled, geocentric) on a circular orbit — fallback only."""
    a = OMEGA_SUN * t + phase
    return D_SUN * np.array([np.cos(a), np.sin(a), 0.0])


def body_state(name, time=None, *, origin="heliocentric"):
    """Exact position **and** velocity of a solar-system body in *scaled units*.

    Returns ``(r, v)`` with position in Mm and velocity in km/s (= scaled ``LU/TU``).
    ``origin`` is ``'heliocentric'`` (default, Sun-centred), ``'bary'`` (solar-system
    barycentre) or ``'geocentric'`` (Earth-centred — used for the trans-lunar frame).
    Uses astropy's network-free ``builtin`` ephemeris; if astropy is unavailable, falls
    back to the analytic circular orbits for the Sun/Moon.
    """
    try:
        from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
        import astropy.units as u
    except Exception:                                  # no astropy -> analytic fallback
        r = (moon_position(0.0) if name.lower() == "moon" else
             sun_position(0.0) if name.lower() == "sun" else np.zeros(3))
        return r, np.zeros(3)

    t = _as_time(time)
    with solar_system_ephemeris.set(EPHEMERIS):
        r_b, v_b = get_body_barycentric_posvel(name, t)
        if origin in ("heliocentric", "sun"):
            r0, v0 = get_body_barycentric_posvel("sun", t)
            r_b, v_b = r_b - r0, v_b - v0
        elif origin in ("geocentric", "earth"):
            r0, v0 = get_body_barycentric_posvel("earth", t)
            r_b, v_b = r_b - r0, v_b - v0
    r = r_b.xyz.to_value(u.m) / LU
    v = v_b.xyz.to_value(u.m / u.s) / VU
    return r, v


def planet_positions(time=None, bodies=None, *, origin="heliocentric"):
    """Scaled positions (Mm) of the planets (and Moon) at ``time`` as a ``{name: r}`` dict."""
    if bodies is None:
        bodies = ("mercury", "venus", "earth", "mars", "jupiter",
                  "saturn", "uranus", "neptune", "moon")
    return {b: body_state(b, time, origin=origin)[0] for b in bodies}


def ephemeris_samples(bodies, times, *, origin="heliocentric"):
    """**Vectorised** ephemeris over an array of times.

    Returns ``{body: (R, V)}`` where ``R``/``V`` are ``(len(times), 3)`` arrays of scaled
    position [Mm] and velocity [km/s]. astropy evaluates the whole time array at once, so this
    is the fast path for building interpolators that an N-body integrator can sample cheaply.
    """
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
    import astropy.units as u
    out = {}
    with solar_system_ephemeris.set(EPHEMERIS):
        r0 = v0 = None
        if origin in ("heliocentric", "sun"):
            r0, v0 = get_body_barycentric_posvel("sun", times)
        elif origin in ("geocentric", "earth"):
            r0, v0 = get_body_barycentric_posvel("earth", times)
        for b in bodies:
            rb, vb = get_body_barycentric_posvel(b, times)
            if r0 is not None:
                rb, vb = rb - r0, vb - v0
            R = (rb.xyz.to_value(u.m) / LU).T          # (N, 3)
            V = (vb.xyz.to_value(u.m / u.s) / VU).T     # (N, 3)
            out[b] = (np.atleast_2d(R), np.atleast_2d(V))
    return out


def times_from_epoch(epoch, t_scaled):
    """astropy ``Time`` array at ``epoch + t_scaled`` (``t_scaled`` in scaled TU = ks)."""
    from astropy.time import TimeDelta
    epoch = _as_time(epoch)
    return epoch + TimeDelta(np.atleast_1d(t_scaled) * TU, format="sec")


# ---------------------------------------------------------------------------
#  Launch sites (resolved from a library) and Earth-rotation state
# ---------------------------------------------------------------------------
# Convenience presets — *names* only; coordinates come from the geocoder.
LAUNCH_SITES = [
    "Kennedy Space Center",
    "Cape Canaveral Space Force Station",
    "Guiana Space Centre, Kourou",
    "Baikonur Cosmodrome",
    "Vandenberg Space Force Base",
    "Starbase, Boca Chica",
    "Tanegashima Space Center",
    "Satish Dhawan Space Centre, Sriharikota",
    "Jiuquan Satellite Launch Center",
    "Wenchang Space Launch Site",
    "Plesetsk Cosmodrome",
]

# Offline fallback coordinates (lat_deg, lon_deg, height_m) for a few well-known pads.
_SITE_FALLBACK = {
    "kennedy space center":               (28.5729, -80.6490,  3.0),
    "cape canaveral space force station": (28.4889, -80.5778,  3.0),
    "guiana space centre, kourou":        ( 5.2360, -52.7750,  0.0),
    "baikonur cosmodrome":                (45.9650,  63.3050, 90.0),
    "vandenberg space force base":        (34.7420,-120.5724,100.0),
    "starbase, boca chica":               (25.9970, -97.1560,  5.0),
}

_site_cache: dict[str, object] = {}


def launch_site(name):
    """Return an ``astropy.coordinates.EarthLocation`` for a launch site.

    Coordinates are resolved **from a library**: ``EarthLocation.of_address`` (online
    OpenStreetMap geocoding); any address string works. Results are cached. When offline,
    a small built-in coordinate table covers the most common pads.
    """
    key = name.strip().lower()
    if key in _site_cache:
        return _site_cache[key]
    from astropy.coordinates import EarthLocation
    import astropy.units as u
    try:
        loc = EarthLocation.of_address(name)
    except Exception:
        if key not in _SITE_FALLBACK:
            raise
        lat, lon, h = _SITE_FALLBACK[key]
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=h * u.m)
    _site_cache[key] = loc
    return loc


def launch_site_state(name, time=None):
    """Scaled GCRS position & velocity of a launch site, **including Earth's rotation**.

    Returns ``(r, v)`` with position in Mm and velocity in km/s. The velocity is the
    inertial (GCRS) velocity of the rotating pad — ~0.46 km/s eastward at the equator —
    via ``EarthLocation.get_gcrs_posvel`` (astropy uses online IERS Earth-orientation data
    when available, else the bundled IERS-B table).
    """
    import astropy.units as u
    loc = launch_site(name)
    t = _as_time(time)
    r_gcrs, v_gcrs = loc.get_gcrs_posvel(t)
    r = r_gcrs.xyz.to_value(u.m) / LU
    v = v_gcrs.xyz.to_value(u.m / u.s) / VU
    return r, v

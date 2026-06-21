"""
Rocket / astrodynamics models for the Kalman framework.

Two domain models (the "two issues"):

* :class:`TransLunarModel` — a restricted **four-body** problem: a massless spacecraft
  coasting under **Earth + Moon + Sun** gravity, with the Moon and Sun on prescribed
  circular orbits. This is the Apollo trans-lunar trajectory and its lunar **gravitational
  catapult** (flyby / slingshot).

* :class:`RocketLaunchModel` — a planar **powered ascent** from Earth's surface: gravity
  versus **thrust supplied as the control input** ``u`` (optionally exponential-atmosphere
  drag). Demonstrates a model with control channels (``l>0``).

Both are :class:`ExtendedKalmanModel` subclasses whose ``f()`` advances the state by one
4th-order Runge–Kutta step; the EKF linearises ``f`` with a numerical Jacobian.

Scaled units keep the filter well-conditioned:
``length = 1 Mm = 1000 km``, ``time = 1 ks = 1000 s`` (so speed is in km/s,
acceleration in m/s², and Apollo's trans-lunar injection ≈ 10.9 km/s).
"""
from __future__ import annotations
import numpy as np

from ..framework.models.extended import ExtendedKalmanModel
from ..framework.constants import KalmanState, KalmanObservable

# --- scaled units --------------------------------------------------------------
LU, TU = 1.0e6, 1.0e3            # 1 Mm (1000 km), 1 ks (1000 s)
KM = 1.0e3 / LU                  # 1 km in LU
DAY = 86400.0 / TU               # 1 day in TU

# --- standard gravitational parameters GM = G·M (scaled) ----------------------
GM_EARTH = 3.986004418e14   * TU**2 / LU**3     # 398.6
GM_MOON  = 4.9048695e12     * TU**2 / LU**3     # 4.905
GM_SUN   = 1.32712440018e20 * TU**2 / LU**3     # 1.327e8

# --- body sizes / orbits (scaled) ---------------------------------------------
R_EARTH = 6.371e6  / LU                          # 6.371 Mm
R_MOON  = 1.7374e6 / LU                           # 1.737 Mm  (~0.27 R_earth)
R_SUN   = 6.9634e8 / LU
D_MOON  = 3.844e8  / LU                           # 384.4 Mm
D_SUN   = 1.496e11 / LU                           # 149600 Mm
OMEGA_MOON = 2 * np.pi / (27.321661 * DAY)
OMEGA_SUN  = 2 * np.pi / (365.25    * DAY)

# atmosphere (for the launch model)
H_SCALE = 8.5 * KM                               # density scale height ~8.5 km


def moon_position(t, phase=0.0):
    """Moon position (scaled) on its circular orbit at time ``t``."""
    a = OMEGA_MOON * t + phase
    return D_MOON * np.array([np.cos(a), np.sin(a), 0.0])


def sun_position(t, phase=0.0):
    """Sun position (scaled) on its circular orbit at time ``t``."""
    a = OMEGA_SUN * t + phase
    return D_SUN * np.array([np.cos(a), np.sin(a), 0.0])


def _third_body(r, r_body, GM):
    """Geocentric third-body perturbation: GM[(r_b - r)/|r_b - r|^3 - r_b/|r_b|^3].

    The first term is the body's direct pull on the spacecraft; the second (indirect)
    term subtracts its pull on the Earth itself — required in an Earth-centred frame.
    Without it, a distant body (the Sun) would inject a large spurious force instead of
    its true (tiny) tidal effect.
    """
    d = r_body - r
    return GM * (d / np.linalg.norm(d) ** 3 - r_body / np.linalg.norm(r_body) ** 3)


def gravity(r, t, moon_phase=0.0, sun_phase=0.0, bodies=("earth", "moon", "sun")):
    """Geocentric acceleration on a spacecraft at ``r`` (scaled units): Earth's central
    pull plus the **third-body perturbations** of the Moon and Sun."""
    r = np.asarray(r, dtype=float).reshape(3)
    a = np.zeros(3)
    if "earth" in bodies:
        a += -GM_EARTH * r / np.linalg.norm(r) ** 3
    if "moon" in bodies:
        a += _third_body(r, moon_position(t, moon_phase), GM_MOON)
    if "sun" in bodies:
        a += _third_body(r, sun_position(t, sun_phase), GM_SUN)
    return a


def _rk4(s, t, dt, deriv):
    k1 = deriv(s, t)
    k2 = deriv(s + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = deriv(s + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = deriv(s + dt * k3, t + dt)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# ===========================================================================
#  1) Trans-lunar coast — the gravitational catapult (Earth + Moon + Sun)
# ===========================================================================
class TransLunarState(KalmanState):
    X = "Mm"; Y = "Mm"; Z = "Mm"; VX = "km/s"; VY = "km/s"; VZ = "km/s"


class TransLunarObservable(KalmanObservable):
    X = "Mm"; Y = "Mm"; Z = "Mm"


class TransLunarModel(ExtendedKalmanModel):
    """
    Restricted four-body trans-lunar model (Earth + Moon + Sun).

    State ``x = [rx, ry, rz, vx, vy, vz]`` (position in Mm, velocity in km/s); only the
    position is measured (ground-station tracking). ``f()`` is one RK4 step of the
    multi-body gravity ODE; the Moon and Sun ride on prescribed circular orbits.
    """

    NAME = "Trans-Lunar (Earth+Moon+Sun)"; VERSION = "1.0"

    @property
    def states(self):      return TransLunarState
    @property
    def observables(self): return TransLunarObservable

    def __init__(self, moon_phase_deg=125.0, sun_phase_deg=60.0, dt_s=120.0,
                 track_km=5.0, bodies=("earth", "moon", "sun"), n_sub=1,
                 q_pos=1e-6, q_vel=1e-7, **kw):
        self.dtype = np.float64
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

    # --- dynamics ---
    def acceleration(self, r, t):
        return gravity(r, t, self.moon_phase, self.sun_phase, self.bodies)

    def f(self, x, u=None, w=None, t=None):
        # Advance one measurement step with ``n_sub`` internal RK4 substeps, so a close
        # lunar flyby (steep gravity gradient) is resolved accurately even though the EKF
        # only sees a position fix every ``dt_s``. n_sub=1 reproduces the plain one-step f.
        s = np.asarray(x, float).reshape(-1)
        deriv = lambda s, tt: np.concatenate([s[3:], self.acceleration(s[:3], tt)])
        t0, h = float(t or 0.0), self._dt / self.n_sub
        for j in range(self.n_sub):
            s = _rk4(s, t0 + j * h, h, deriv)
        return s.reshape(-1, 1)

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
                          nmax=None, frames=120, interval=100, scale=12.0,
                          kepler=None, figsize=(8.4, 8.2), **kwargs):
        """Interactive **lunar-catapult** replay: the spacecraft flies its free return while the
        per-body **perturbing** accelerations (Earth central pull + Moon/Sun third-body tidal
        terms) are drawn as live vectors with a numeric read-out. Pass ``kepler`` (the Earth-only
        twin, e.g. from a ``bodies=("earth",)`` model) to fly the **without-Moon** path in lock-step
        beside it — you watch the two diverge at the flyby, which *is* the catapult."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from ..helpers import animate
        truth, estx, meas, K = self._trajectory_arrays(record, sample, nmax)
        kep    = None if kepler is None else np.asarray(getattr(kepler, "xk", kepler))
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
#  2) Powered launch from Earth — thrust as the control input
# ===========================================================================
class RocketLaunchState(KalmanState):
    X = "Mm"; Y = "Mm"; VX = "km/s"; VY = "km/s"


class RocketLaunchObservable(KalmanObservable):
    X = "Mm"; Y = "Mm"


class RocketLaunchModel(ExtendedKalmanModel):
    """
    Planar powered ascent from Earth's surface.

    State ``x = [x, y, vx, vy]``; **control** ``u = [ax, ay]`` is the thrust acceleration
    (km/s²), so this model has ``l = 2`` control channels. Dynamics: central gravity +
    thrust (+ optional exponential-atmosphere drag). A radar measures position only.
    """

    NAME = "Rocket Launch (powered ascent)"; VERSION = "1.0"

    @property
    def states(self):      return RocketLaunchState
    @property
    def observables(self): return RocketLaunchObservable

    def __init__(self, dt_s=1.0, track_km=1.0, drag=False, ballistic_coeff=2.0e-3,
                 q_pos=1e-9, q_vel=1e-9, **kw):
        self.dtype = np.float64
        self._dt   = dt_s / TU
        self.drag  = bool(drag)
        self.Cb    = float(ballistic_coeff)     # lumped 0.5*Cd*A/m (scaled)
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
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     figsize=(17, 4.2), **kwargs):
        """Signature figure: ascent trajectory (down-range vs altitude), altitude & speed
        versus time, and the EKF position-tracking error."""
        import matplotlib.pyplot as plt
        truth = np.asarray(sample.xk) if sample is not None else np.asarray(record.xk)
        estx  = np.asarray(record.xk)
        n     = min(len(truth), len(estx))
        truth, estx = truth[:n], estx[:n]
        t_s   = np.arange(n) * self._dt * TU                       # seconds
        alt   = (np.linalg.norm(truth[:, :2], axis=1) - R_EARTH) * LU / 1e3
        down  = truth[:, 1] * LU / 1e3
        speed = np.linalg.norm(truth[:, 2:4], axis=1)
        err   = np.linalg.norm(truth[:, :2] - estx[:, :2], axis=1) * LU / 1e3

        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].plot(down, alt, color="crimson")
        ax[0].set_xlabel("downrange [km]"); ax[0].set_ylabel("altitude [km]")
        ax[0].set_title("Ascent trajectory"); ax[0].grid(alpha=0.3)
        ax[1].plot(t_s, alt, color="steelblue", label="altitude [km]")
        ax[1].plot(t_s, speed, color="darkorange", label="speed [km/s]")
        ax[1].set_xlabel("time [s]"); ax[1].set_title("Altitude & speed")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        ax[2].plot(t_s, err, color="seagreen")
        ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("|pos error| [km]")
        ax[2].set_title("EKF tracking error"); ax[2].grid(alpha=0.3)
        fig.suptitle(f"{self.name()} — powered ascent", y=1.02)
        fig.tight_layout()
        return fig

"""Reduced storm models — typhoon track/intensity and tornado-environment regime.

Two low-dimensional models built to sit *on top of* the field-scale EnKF
(``kalman.apps.weather.flow``): the EnKF estimates the environment, these estimate the
local phenomenon and — crucially — its **covariance**.

- :class:`StormModel` — a reduced tropical-cyclone tracker (typhoon, hurricane
  and cyclone are regional names for one storm, so one model covers all).  State
  ``[lon, lat, v_e, v_n, I]`` (centre, translation velocity, intensity); the
  dynamics are the textbook **steering flow + beta drift + coordinated turn**,
  so the motion has genuine *regimes* (straight-running vs. recurving).  That
  regime structure is exactly what an :class:`~kalman.framework.filters.InteractiveKalmanFilter`
  (IKF) is for — recurvature is the "manoeuvre".  Use
  :meth:`StormModel.ikf_bank` to get the ready-made model bank + transition
  matrix.  The ensemble/covariance gives the track **cone of uncertainty**
  (:meth:`forecast_cone`) and an intensity error bar for free.

- :class:`TornadoEnvironmentModel` — an honest tornado model is *not* a
  deterministic tracker (a tornado is sub-grid and minutes-predictable); the
  actionable quantity is the **probability the environment is in the tornadic
  regime**.  This is a mean-reverting significant-tornado-parameter latent with
  a benign and a tornadic regime; IKF returns the regime probability, and the
  **seasonal climatology enters as the IKF transition matrix** (``season=``).

Neither needs training data; both deliver calibrated covariance/probability
from physics + the filter, which is the point (see the notebook 13 discussion
of EnKF-vs-ML).
"""
from __future__ import annotations
import numpy as np

from ...framework.models.extended import ExtendedKalmanModel
from ...framework.models.native import NativeTwin
from ...framework.constants import KalmanState, KalmanObservable

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320   # × cos(lat)


# ======================================================================
# Tropical-cyclone tracker
# ======================================================================
class StormState(KalmanState):
    LON = "°E"      # centre longitude
    LAT = "°N"      # centre latitude
    VE  = "km/h"    # eastward translation speed
    VN  = "km/h"    # northward translation speed
    I   = "kt"      # intensity (max sustained wind)


class StormObservable(KalmanObservable):
    LON = "°E"
    LAT = "°N"
    I   = "kt"




class StormModel(NativeTwin, ExtendedKalmanModel):
    """
    Reduced tropical-cyclone track + intensity model (steering + beta + turn).

    A typhoon, a hurricane and a cyclone are the same phenomenon under
    different regional names, so this one model covers all of them; the only
    thing that differs between basins is the ``hemisphere``, which sets the
    beta-drift and recurvature directions.

    State ``x = [lon, lat, v_e, v_n, I]``.  Over one step ``dt`` (hours):

    .. code-block:: text

        (v_e, v_n)  rotate at turn rate ω        (coordinated turn: ω=0 straight)
        centre advected by (v_e+β_e, v_n+β_n)    (steering velocity + beta drift)
        lon += displacement_east / (111.32 cos lat),  lat += disp_north / 110.574
        I    persists (random walk; intensity change is Q-driven — RI is hard)

    **Beta drift** ``β`` is the quasi-steady poleward-westward self-advection
    from the planetary-vorticity gradient (~1–2 m/s toward WNW in the NH); its
    sign flips by ``hemisphere``.  The **turn rate** ``ω`` is what distinguishes
    the motion regimes an IKF switches between (``ω=0`` straight vs.
    ``ω=ω_recurve`` recurving) — see :meth:`ikf_bank`.

    Observation: a centre fix ``[lon, lat]`` (best-track / satellite), plus
    intensity ``I`` when ``observe_intensity`` (default True).  ``n=5``.

    Parameters
    ----------
    omega : float
        Turn rate [rad/h]; 0 = straight-running.  ``ikf_bank`` sets this per
        regime.  Sign convention: negative = clockwise in the (east, north)
        plane = NH recurvature (poleward then eastward).
    beta_drift : float
        Beta-drift speed [km/h] (default ~5 km/h ≈ 1.4 m/s).
    hemisphere : {"N", "S"}
        Sets the beta-drift direction (WNW in NH, WSW in SH).
    fix_noise : float
        Centre-fix std [deg] (default 0.3° ≈ 33 km, a realistic satellite fix).
    intensity_noise : float
        Intensity measurement std [kt] (default 10).
    """

    NAME = "Storm Tracker"
    DESCRIPTION = ("Reduced tropical-cyclone track + intensity model "
                   "(steering flow + beta drift + coordinated turn) — "
                   "typhoon, hurricane or cyclone alike.")
    VERSION = "3.0"

    @property
    def states(self):      return StormState
    @property
    def observables(self): return StormObservable

    def __init__(self, sensor_id="TC", *,
                 omega=0.0,
                 beta_drift=5.0,
                 hemisphere="N",
                 dt_hours=6.0,
                 observe_intensity=True,
                 fix_noise=0.3,
                 intensity_noise=10.0,
                 q_pos=0.25, q_vel=6.0, q_intensity=8.0,
                 steering=None, intensity_fn=None,
                 x0=None, **kw):
        self.dtype = np.float64
        self.omega = float(omega)
        self.beta_drift = float(beta_drift)
        self.hemisphere = str(hemisphere).upper()
        if self.hemisphere not in ("N", "S"):
            raise ValueError(f"hemisphere must be 'N' or 'S', not {hemisphere!r}.")
        self._dt = float(dt_hours)
        self.observe_intensity = bool(observe_intensity)
        # Optional physics couplings (both None → the pure kinematic model):
        #   steering(lon, lat, t) -> (v_e, v_n) km/h   — environmental deep-layer
        #     flow the vortex is advected by (from the field EnKF, or analytic).
        #     When set, recurvature EMERGES from the flow, so ω is unused.
        #   intensity_fn(lon, lat, I, t) -> dI/dt kt/h  — intensity tendency
        #     (physics leaves this None = persistence; the ML elaboration is a
        #     LEARNED intensity_fn, mirroring operational statistical intensity).
        self.steering = steering
        self.intensity_fn = intensity_fn
        self.n = 5
        self.m = 3 if self.observe_intensity else 2
        self.l = 0

        # Beta drift vector [km/h]: WNW in NH (poleward+westward), WSW in SH.
        pole = 1.0 if self.hemisphere == "N" else -1.0
        ang = np.deg2rad(20.0)                       # ~20° off due-west, toward the pole
        self.beta_e = -self.beta_drift * np.cos(ang)
        self.beta_n =  pole * self.beta_drift * np.sin(ang)

        Q = np.diag([q_pos**2, q_pos**2, q_vel**2, q_vel**2, q_intensity**2]).astype(float)
        if self.observe_intensity:
            R = np.diag([fix_noise**2, fix_noise**2, intensity_noise**2]).astype(float)
        else:
            R = np.diag([fix_noise**2, fix_noise**2]).astype(float)

        if x0 is None:
            x0 = np.array([[135.0], [15.0], [-20.0], [6.0], [80.0]])
        super().__init__(sensor_id, Q=Q, R=R, L=np.eye(self.n), fs=1.0 / self._dt,
                         x0=x0, **kw)

    # ------------------------------------------------------------------
    # C++ twin (libKalmanApps: Kalman::Apps::Weather::Storm). A steering flow or
    # an intensity tendency is a Python callable, which has no C++ counterpart.
    # ------------------------------------------------------------------
    def native_name(self):
        if self.steering is not None or self.intensity_fn is not None:
            return None
        return "storm" if self.observe_intensity else "storm-track-only"

    def native_parameters(self):
        return {"omega": self.omega, "beta_drift": self.beta_drift,
                "hemisphere": 1.0 if self.hemisphere == "N" else -1.0,
                "dt_hours": self._dt}

    # ------------------------------------------------------------------
    def f(self, x, u=None, w=None, t=None):
        x = np.asarray(x, float).reshape(-1)
        lon, lat, ve, vn, I = x[0], x[1], x[2], x[3], x[4]
        dt, w_ = self._dt, self.omega

        if self.steering is not None:
            # Beta-advection model: translation = environmental steering + beta.
            # Recurvature is not a prescribed turn rate — it emerges when the
            # storm drifts poleward into a steering flow that reverses. This is
            # the physically realistic predictor (no ML, no fitted turn rate).
            se, sn = self.steering(lon, lat, t)
            vte, vtn = se + self.beta_e, sn + self.beta_n
            de, dn = vte * dt, vtn * dt
            ve_n, vn_n = se, sn                            # velocity state = diagnosed steering
        elif abs(w_) < 1e-9:
            vte, vtn = ve + self.beta_e, vn + self.beta_n  # advecting velocity (+ beta)
            de, dn = vte * dt, vtn * dt
            ve_n, vn_n = ve, vn
        else:
            vte, vtn = ve + self.beta_e, vn + self.beta_n
            s, c = np.sin(w_ * dt), np.cos(w_ * dt)
            de = (vte * s - vtn * (1 - c)) / w_            # curved displacement [km]
            dn = (vte * (1 - c) + vtn * s) / w_
            ve_n = ve * c - vn * s                         # rotate steering velocity
            vn_n = ve * s + vn * c
        lat_n = lat + dn / KM_PER_DEG_LAT
        lon_n = lon + de / (KM_PER_DEG_LON * np.cos(np.deg2rad(lat)))
        I_n = I if self.intensity_fn is None else I + self.intensity_fn(lon, lat, I, t) * dt
        return np.array([[lon_n], [lat_n], [ve_n], [vn_n], [I_n]])

    def h(self, x, v=None, t=None):
        x = np.asarray(x, float).reshape(-1)
        z = np.array([[x[0]], [x[1]], [x[4]]]) if self.observe_intensity else np.array([[x[0]], [x[1]]])
        return z if v is None else z + np.asarray(v, float).reshape(-1, 1)

    # ------------------------------------------------------------------
    # IKF bank: straight-running vs recurving (vs stalling) motion regimes
    # ------------------------------------------------------------------
    @classmethod
    def ikf_bank(cls, sensor_id="TC", *, omega_recurve=0.022, stall=True,
                 stay=0.90, hemisphere="N", **kw):
        """Build the IKF model bank + Markov transition matrix.

        Returns ``(models, transition, initial_probs)`` ready for
        :class:`~kalman.framework.filters.InteractiveKalmanFilter`.  Regimes:

        - **straight** — ``ω=0`` (trade-wind steering, WNW motion);
        - **recurve**  — ``ω`` set for the hemisphere (poleward-then-eastward);
        - **stall**    — ``ω=0`` with inflated velocity process noise (erratic /
          slow-moving), included when ``stall=True``.

        ``stay`` is the self-transition probability (regime persistence);
        off-diagonal mass is split among the other regimes.  Feed a season
        prior by adjusting ``stay`` or passing your own transition later.
        """
        hemi = str(hemisphere).upper()
        pole_sign = -1.0 if hemi == "N" else 1.0
        w_rec = pole_sign * abs(omega_recurve)
        models = [
            cls(sensor_id, omega=0.0,   hemisphere=hemi, **kw),
            cls(sensor_id, omega=w_rec, hemisphere=hemi, **kw),
        ]
        if stall:
            models.append(cls(sensor_id, omega=0.0, hemisphere=hemi,
                              q_vel=18.0, **kw))
        M = len(models)
        off = (1.0 - stay) / (M - 1)
        Pi = np.full((M, M), off) + (stay - off) * np.eye(M)
        probs = np.full(M, 1.0 / M)
        return models, Pi, probs

    imm_bank = ikf_bank      # the literature's name (Interacting Multiple Model)

    # ------------------------------------------------------------------
    # Forecast cone: propagate mean + covariance forward with NO updates
    # ------------------------------------------------------------------
    def forecast_cone(self, x0, P0, steps, *, k_sigma=1.0):
        """Free-run the model ``steps`` cycles from ``(x0, P0)`` and return the
        track **cone of uncertainty**.

        Pure prediction (``engine._predict`` only, no observations), so the
        position covariance grows exactly as the model says it should.  Returns
        a dict with ``lon``/``lat`` (mean track), ``I``/``I_sigma`` (intensity ±σ),
        and ``ellipses`` — per-step ``(cx, cy, a, b, angle_deg)`` at ``k_sigma``
        for the lon/lat error ellipse.
        """
        from ...framework.engines.extended import ExtendedKalmanEngine
        eng = ExtendedKalmanEngine()
        x = np.asarray(x0, float).reshape(-1, 1)
        P = np.asarray(P0, float)
        lon, lat, Ivals, Isig, ell = [], [], [], [], []
        t = 0.0
        for _ in range(int(steps)):
            x, P, _ = eng._predict(self, x, P, t=t)
            t += self._dt
            xr = x.ravel()
            lon.append(xr[0]); lat.append(xr[1])
            Ivals.append(xr[4]); Isig.append(float(np.sqrt(max(P[4, 4], 0.0))))
            ell.append(self._ellipse(xr[0], xr[1], P[:2, :2], k_sigma))
        return {"lon": np.array(lon), "lat": np.array(lat),
                "I": np.array(Ivals), "I_sigma": np.array(Isig),
                "ellipses": ell}

    def ensemble_forecast(self, x0, P0, steps, *, n_samples=300, seed=0):
        """Monte-Carlo forecast: draw ``n_samples`` initial states from
        ``N(x0, P0)`` and propagate each through the (nonlinear) dynamics with
        per-step process-noise draws — the honest **spaghetti ensemble** the
        Gaussian :meth:`forecast_cone` summarises.  Returns ``(n_samples, steps,
        5)``.  Feed to :func:`landfall_probability` for a strike probability."""
        rng = np.random.default_rng(seed)
        x0 = np.asarray(x0, float).reshape(-1)
        P0 = np.asarray(P0, float)
        Qc = np.linalg.cholesky(self.Q() + 1e-12 * np.eye(self.n))
        X = rng.multivariate_normal(x0, 0.5 * (P0 + P0.T) + 1e-9 * np.eye(self.n), size=n_samples)
        out = np.empty((n_samples, int(steps), self.n))
        for i in range(n_samples):
            x = X[i].reshape(-1, 1)
            for k in range(int(steps)):
                x = np.asarray(self.f(x), float) + (Qc @ rng.standard_normal((self.n, 1)))
                out[i, k] = x.ravel()
        return out

    @staticmethod
    def _ellipse(cx, cy, P2, k_sigma):
        vals, vecs = np.linalg.eigh(0.5 * (P2 + P2.T))
        vals = np.maximum(vals, 0.0)
        order = np.argsort(vals)[::-1]
        a, b = k_sigma * np.sqrt(vals[order])
        ang = np.degrees(np.arctan2(vecs[1, order[0]], vecs[0, order[0]]))
        return (float(cx), float(cy), float(a), float(b), float(ang))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Signature figures — reached via the standard ``model.plot(record)`` /
    # ``model.plot(record, interactive=True)``, so a notebook needs no
    # bespoke plotting code to reproduce the canonical storm figure.
    # ------------------------------------------------------------------
    def _draw_track(self, ax, record, *, sample=None, fixes=None, cone=None,
                    ens=None, k_sigma=1.0, pad=5.0):
        """Draw the track map onto ``ax`` (shared by static + interactive)."""
        from matplotlib.patches import Ellipse
        xk = np.asarray(record.xk); Pk = np.asarray(record.Pk)
        lon, lat = xk[:, 0], xk[:, 1]
        lons, lats = [lon], [lat]
        if sample is not None:
            st = np.asarray(sample.xk); lons.append(st[:, 0]); lats.append(st[:, 1])
        if cone is not None:
            lons.append(np.asarray(cone["lon"])); lats.append(np.asarray(cone["lat"]))
        allx, ally = np.concatenate(lons), np.concatenate(lats)
        ax.set_xlim(allx.min() - pad, allx.max() + pad)
        ax.set_ylim(ally.min() - pad, ally.max() + pad)
        try:                                          # coastlines if geo data is present
            from .geo import earth_geometry
            for seg in earth_geometry("coastline"):
                ax.plot(seg[:, 0], seg[:, 1], color="0.6", lw=0.6, zorder=0)
        except Exception:
            pass
        if ens is not None:                           # spaghetti ensemble, if supplied
            for tr in np.asarray(ens):
                ax.plot(tr[:, 0], tr[:, 1], "-", color="tab:orange", lw=0.4, alpha=0.25, zorder=1)
        if sample is not None:
            st = np.asarray(sample.xk)
            ax.plot(st[:, 0], st[:, 1], "k-", lw=2, label="truth track", zorder=3)
        if fixes is not None:
            fx = np.asarray(fixes)
            ax.plot(fx[:, 0], fx[:, 1], "x", color="tab:red", ms=5, alpha=0.6,
                    label="fixes", zorder=4)
        ax.plot(lon, lat, "-", color="tab:blue", lw=1.8, label="analysis track", zorder=5)
        for k in range(0, len(lon), max(1, len(lon) // 20)):
            cx, cy, a, b, ang = self._ellipse(lon[k], lat[k], Pk[k, :2, :2], k_sigma)
            ax.add_patch(Ellipse((cx, cy), 2*a, 2*b, angle=ang, fill=False,
                                 edgecolor="tab:blue", alpha=0.4, lw=0.8, zorder=5))
        if cone is not None:
            ax.plot(cone["lon"], cone["lat"], "--", color="tab:orange", lw=1.5,
                    label="forecast track", zorder=4)
            for (cx, cy, a, b, ang) in cone["ellipses"]:
                ax.add_patch(Ellipse((cx, cy), 2*a, 2*b, angle=ang, fill=True,
                                     facecolor="tab:orange", alpha=0.12,
                                     edgecolor="none", zorder=2))
        ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(lat))))
        ax.set_xlabel("longitude [°E]"); ax.set_ylabel("latitude [°N]")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
        ax.set_title(f"Storm track — ±{k_sigma:g}σ position covariance")
        return ax

    def _draw_intensity(self, ax, record, *, sample=None, fixes=None, cone=None,
                        k_sigma=1.0):
        """Draw the intensity panel with its ±σ band onto ``ax``."""
        xk = np.asarray(record.xk); Pk = np.asarray(record.Pk)
        t = np.arange(len(xk)) * self._dt
        I = xk[:, 4]
        sig = np.sqrt(np.maximum(Pk[:, 4, 4], 0.0))
        if sample is not None:
            st = np.asarray(sample.xk)
            ax.plot(t[:len(st)], st[:, 4], "k-", lw=2, label="truth")
        if fixes is not None:
            fx = np.asarray(fixes)
            if fx.shape[1] > 2:
                ax.plot(t[:len(fx)], fx[:, 2], "x", color="tab:red", ms=4, alpha=0.6,
                        label="fixes")
        ax.plot(t, I, "-", color="tab:blue", lw=1.8, label="analysis")
        ax.fill_between(t, I - k_sigma * sig, I + k_sigma * sig, color="tab:blue",
                        alpha=0.2, label=f"±{k_sigma:g}σ")
        if cone is not None:
            tf = t[-1] + np.arange(1, len(cone["I"]) + 1) * self._dt
            ax.plot(tf, cone["I"], "--", color="tab:orange", lw=1.5, label="forecast")
            ax.fill_between(tf, cone["I"] - k_sigma * cone["I_sigma"],
                            cone["I"] + k_sigma * cone["I_sigma"],
                            color="tab:orange", alpha=0.15)
        ax.set_xlabel("time [h]"); ax.set_ylabel("intensity [kt]")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
        ax.set_title("Intensity with uncertainty")
        return ax

    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     fixes=None, cone=None, ens=None, k_sigma=1.0,
                     figsize=(13, 5.6), **kwargs):
        """The model's signature figure: the track map with ±σ position
        ellipses (plus optional truth, raw ``fixes``, a :meth:`forecast_cone`
        and an :meth:`ensemble_forecast` spaghetti) beside the intensity trace
        with its ±σ band — the model's two deliverables, both with covariance."""
        import matplotlib.pyplot as plt
        fig, (ax_map, ax_int) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.25, 1]})
        self._draw_track(ax_map, record, sample=sample, fixes=fixes, cone=cone,
                         ens=ens, k_sigma=k_sigma)
        self._draw_intensity(ax_int, record, sample=sample, fixes=fixes, cone=cone,
                             k_sigma=k_sigma)
        fig.suptitle(f"{self.name()} — tropical cyclone ({self.hemisphere}H)",
                     fontweight="bold")
        fig.tight_layout()
        return fig

    def _plot_interactive(self, record, sample=None, *, fixes=None, cone=None,
                          k_sigma=1.0, frames=30, interval=150, figsize=(13, 5.6),
                          **kwargs):
        """Interactive replay: the track grows cycle by cycle with its covariance
        ellipse while the intensity band fills in.  Returns the framework's
        standard Jupyter play/scrub player (see ``helpers.animate``)."""
        import matplotlib.pyplot as plt
        from ...helpers import animate
        xk = np.asarray(record.xk)
        n = len(xk)
        steps = sorted(set(np.linspace(2, n, min(frames, max(n - 1, 2)), dtype=int)))

        fig, (ax_map, ax_int) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.25, 1]})

        class _Slice:                       # a record view truncated to k cycles
            def __init__(self, rec, k):
                self.xk = np.asarray(rec.xk)[:k]
                self.Pk = np.asarray(rec.Pk)[:k]

        def _draw(k):
            ax_map.clear(); ax_int.clear()
            part = _Slice(record, k)
            smp = _Slice(sample, k) if sample is not None else None
            fx = None if fixes is None else np.asarray(fixes)[:k]
            self._draw_track(ax_map, part, sample=smp, fixes=fx,
                             cone=cone if k >= n else None, k_sigma=k_sigma)
            self._draw_intensity(ax_int, part, sample=smp, fixes=fx,
                                 cone=cone if k >= n else None, k_sigma=k_sigma)
            fig.suptitle(f"{self.name()}  "
                         f"(cycle {k}/{n}, t = {k * self._dt:.0f} h)", fontweight="bold")
            return (ax_map, ax_int)

        fig.tight_layout()
        return animate(fig, _draw, frames=steps, interval=interval)




def synthetic_track(n_steps=40, *, recurve_at=18, dt_hours=6.0, hemisphere="N",
                    speed=22.0, omega_recurve=0.022, fix_noise=0.3,
                    intensity_noise=10.0, peak_intensity=120.0, seed=0):
    """Generate a synthetic best-track: straight WNW run, then recurvature, with
    a rise-and-decay intensity — plus noisy centre/intensity fixes.

    Returns ``(truth (T, 5), fixes (T, 3))`` — feed ``truth`` as the OSSE truth
    and ``fixes`` as observations to an IKF tracker.  Exercises the regime
    switch at ``recurve_at``.
    """
    rng = np.random.default_rng(seed)
    pole = 1.0 if str(hemisphere).upper() == "N" else -1.0
    lon, lat = 138.0, 12.0
    ang = np.deg2rad(20.0)
    ve, vn = -speed * np.cos(ang), pole * speed * np.sin(ang)
    w = pole * -abs(omega_recurve)
    truth, fixes = [], []
    for k in range(n_steps):
        frac = k / max(1, n_steps - 1)
        I = peak_intensity * np.sin(np.pi * min(frac / 0.7, 1.0)) + 40.0
        truth.append([lon, lat, ve, vn, I])
        fixes.append([lon + rng.normal(0, fix_noise),
                      lat + rng.normal(0, fix_noise),
                      I + rng.normal(0, intensity_noise)])
        turn = w if k >= recurve_at else 0.0
        if abs(turn) < 1e-9:
            de, dn = ve * dt_hours, vn * dt_hours
        else:
            s, c = np.sin(turn*dt_hours), np.cos(turn*dt_hours)
            de = (ve*s - vn*(1-c))/turn; dn = (ve*(1-c) + vn*s)/turn
            ve, vn = ve*c - vn*s, ve*s + vn*c
        lat += dn / KM_PER_DEG_LAT
        lon += de / (KM_PER_DEG_LON * np.cos(np.deg2rad(lat)))
    return np.array(truth), np.array(fixes)


# ======================================================================
# Realistic physics coupling: environmental steering + statistical intensity
# ======================================================================
def analytic_steering(lon, lat, t=None, *, ridge_lat=27.0, easterly=-26.0,
                      westerly=32.0, poleward=3.0, poleward_mid=12.0,
                      hemisphere="N", width=5.0):
    """Idealised deep-layer steering flow [km/h] for a beta-advection tracker.

    Trade-wind **easterlies** equatorward of the subtropical ridge
    (``ridge_lat``) smoothly giving way to mid-latitude **westerlies** poleward
    of it (``tanh`` transition of scale ``width``).  The westerlies steer
    **poleward-eastward** (``poleward_mid``), so a recurving storm accelerates
    toward higher latitudes — the NE sweep that carries Northwest-Pacific
    typhoons toward Japan.  A storm born in the tropics runs west, drifts
    poleward (beta), crosses the ridge into the westerlies and **recurves** —
    recurvature as physics, not a fitted turn rate.  In the real system this
    callable is replaced by the deep-layer mean flow diagnosed from the field
    EnKF (``kalman.apps.weather.flow``); the signature is identical.
    """
    pole = 1.0 if str(hemisphere).upper() == "N" else -1.0
    frac = 0.5 * (1.0 + np.tanh(pole * (lat - ridge_lat) / width))   # 0 tropics → 1 mid-lat
    ve = (1 - frac) * easterly + frac * westerly
    vn = pole * (poleward * (1 - frac) + poleward_mid * frac)         # NE sweep in the westerlies
    return float(ve), float(vn)


def analytic_environment(lon, lat, t=None):
    """Idealised intensity environment at a point → ``(sst_warmth, shear)``.

    ``sst_warmth`` is a 2-D **warm pool** (peaks near 135°E, 10°N and falls off
    in both lon and lat), ``shear`` grows toward mid-latitudes.  The two vary
    along different directions, so they are *not* collinear along a track — a
    learned intensity model can separate their effects, as a real one must.
    Stand-in for the satellite SST / analysis wind-shear predictors that
    operational statistical intensity schemes consume.
    """
    sst_warmth = float(30.0 * np.exp(-((lon - 135.0) / 45.0) ** 2
                                     - ((lat - 10.0) / 14.0) ** 2))
    shear = float(3.0 + 0.9 * max(abs(lat) - 15.0, 0.0))
    return sst_warmth, shear


def _true_intensity_tendency(lon, lat, I, t=None, env=analytic_environment):
    """Ground-truth intensity physics used to synthesise data: relaxation toward
    a maximum potential intensity set by SST, penalised by shear."""
    sst_warmth, shear = env(lon, lat, t)
    mpi = 25.0 + 3.0 * sst_warmth                       # max potential intensity [kt]
    # Slow relaxation (~1.5-day e-folding) so intensification is a multi-day
    # process a persistence forecast visibly fails to capture — realistic RI.
    return 0.025 * (mpi - I) - 0.30 * shear


class LearnedIntensityModel:
    """Data-driven intensity-tendency model — the ML elaboration on the physics.

    Ridge regression of ``dI/dt`` on environmental predictors
    ``[sst_warmth, shear, I, 1]`` (the toy analogue of the operational **SHIPS**
    statistical intensity scheme, which regresses on ~20 predictors).  The
    physics tracker leaves intensity as persistence; this *learns* the
    SST/shear→intensification relationship from past storms and plugs into
    :class:`StormModel` via :meth:`as_intensity_fn`.

    Deliberately dependency-free (numpy least squares) so it runs anywhere; a
    torch MLP is the drop-in scale-up when the relationship is nonlinear.
    """

    def __init__(self, ridge=1e-3, env=analytic_environment):
        self.ridge = float(ridge)
        self.env = env
        self.coef_ = None
        self._mu = self._sd = None                      # feature standardisation

    @staticmethod
    def _features(sst_warmth, shear, I):
        return np.array([sst_warmth, shear, I])

    def fit(self, tracks, dt_hours=6.0):
        """Fit on a list of truth tracks ``(T, 5)`` — builds ``(features, dI/dt)``
        pairs from consecutive intensities and their environment.  Features are
        **standardised** before the ridge solve (they span very different
        scales: SST ~0–30, shear ~3–30, intensity ~15–120), so no predictor is
        implicitly penalised more than another."""
        X, y = [], []
        for tr in tracks:
            tr = np.asarray(tr, float)
            for k in range(len(tr) - 1):
                lon, lat, I = tr[k, 0], tr[k, 1], tr[k, 4]
                sst, shear = self.env(lon, lat)
                X.append(self._features(sst, shear, I))
                y.append((tr[k + 1, 4] - I) / dt_hours)
        X, y = np.asarray(X), np.asarray(y)
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.column_stack([(X - self._mu) / self._sd, np.ones(len(X))])
        A = Xs.T @ Xs + self.ridge * np.eye(Xs.shape[1])
        self.coef_ = np.linalg.solve(A, Xs.T @ y)
        return self

    def tendency(self, lon, lat, I, t=None):
        sst, shear = self.env(lon, lat, t)
        fs = (self._features(sst, shear, I) - self._mu) / self._sd
        return float(np.append(fs, 1.0) @ self.coef_)

    def as_intensity_fn(self):
        """Return the ``intensity_fn(lon, lat, I, t)`` closure for StormModel."""
        if self.coef_ is None:
            raise RuntimeError("fit() the LearnedIntensityModel first.")
        return self.tendency


def synthetic_steered_track(n_steps=44, *, lon0=165.0, lat0=9.0, hemisphere="N",
                            steering=analytic_steering, intensity_env=analytic_environment,
                            eddy=6.0, fix_noise=0.4, intensity_noise=8.0,
                            dt_hours=6.0, I0=35.0, seed=0):
    """Truth track advected by a **steering field** (recurvature emerges) with
    environment-driven intensity, plus noisy fixes.

    Unlike :func:`synthetic_track` (which prescribes a turn), motion here is
    ``steering + beta + eddy`` and intensity follows
    :func:`_true_intensity_tendency` — so a steering-coupled physics tracker can
    predict the recurvature, and a learned intensity model has a real
    SST/shear signal to capture.  Returns ``(truth (T,5), fixes (T,3))``.
    """
    rng = np.random.default_rng(seed)
    pole = 1.0 if str(hemisphere).upper() == "N" else -1.0
    beta = 5.0
    be, bn = -beta * np.cos(np.deg2rad(20)), pole * beta * np.sin(np.deg2rad(20))
    lon, lat, I = lon0, lat0, I0
    truth, fixes = [], []
    for k in range(n_steps):
        se, sn = steering(lon, lat, hemisphere=hemisphere) if steering.__name__ == "analytic_steering" else steering(lon, lat)
        ve, vn = se, sn
        truth.append([lon, lat, ve, vn, I])
        fixes.append([lon + rng.normal(0, fix_noise), lat + rng.normal(0, fix_noise),
                      I + rng.normal(0, intensity_noise)])
        vte = se + be + rng.normal(0, eddy)
        vtn = sn + bn + rng.normal(0, eddy)
        lat += vtn * dt_hours / KM_PER_DEG_LAT
        lon += vte * dt_hours / (KM_PER_DEG_LON * np.cos(np.deg2rad(lat)))
        I = max(15.0, I + _true_intensity_tendency(lon, lat, I, env=intensity_env) * dt_hours)
    return np.array(truth), np.array(fixes)


# ======================================================================
# Japan / Northwest-Pacific configuration + probabilistic landfall
# ======================================================================
JAPAN_BBOX = (128.0, 146.0, 30.0, 46.0)   # lon_min, lon_max, lat_min, lat_max
_LAND_PATH = None


def japan_steering(lon, lat, t=None):
    """Northwest-Pacific steering tuned so a tropical-genesis storm recurves
    into the Japan region (subtropical ridge ~25°N; the westerlies sweep it NE
    toward Honshu).  Drop-in ``steering`` for :class:`StormModel`."""
    return analytic_steering(lon, lat, ridge_lat=27.0, easterly=-15.0,
                             westerly=27.0, hemisphere="N")


def field_steering(velocity_sampler, *, base=japan_steering, weight=1.0):
    """Compose a climatological steering with a **field-diagnosed flow anomaly**.

    ``velocity_sampler`` is a ``(lon, lat) → (v_e, v_n)`` callable — in practice
    ``KolmogorovFlowModel.velocity_sampler(field, region)`` built from the
    field-EnKF analysis (or one ensemble member) of notebook 13.  The returned
    ``steering(lon, lat, t)`` adds ``weight × anomaly`` to the mean ridge
    ``base``, so:

    - the **mean** flow still gives physical recurvature, and
    - flow-dependent steering **anomalies from the field EnKF perturb the
      track** — feed one sampler per field-ensemble member to turn the
      *environmental* uncertainty into **track-cone** uncertainty, the real
      coupling between the field filter and the vortex tracker.
    """
    def steering(lon, lat, t=None):
        be, bn = base(lon, lat, t)
        ae, an = velocity_sampler(lon, lat)
        return be + weight * ae, bn + weight * an
    return steering


def _land_path():
    """Cached matplotlib Path of the Natural Earth land polygons (for pointwise
    land tests); None if the coastline data / matplotlib is unavailable."""
    global _LAND_PATH
    if _LAND_PATH is None:
        try:
            from matplotlib.path import Path as MPath
            from .geo import earth_geometry
            verts, codes = [], []
            for ring in earth_geometry("land"):
                verts.append(ring)
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 1)
            _LAND_PATH = MPath(np.vstack(verts), np.asarray(codes, dtype=np.uint8))
        except Exception:
            _LAND_PATH = False
    return _LAND_PATH or None


def japan_region(lon, lat):
    """True where ``(lon, lat)`` is land inside the Japan bounding box.

    Uses the Natural Earth land polygons (shipped with ``reanalysis``) so a
    strike means actually crossing the coastline, not just entering a box; if
    that data is unavailable it degrades to the bounding box alone.  Accepts
    scalars or arrays.
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    lo_min, lo_max, la_min, la_max = JAPAN_BBOX
    inbox = (lon >= lo_min) & (lon <= lo_max) & (lat >= la_min) & (lat <= la_max)
    path = _land_path()
    if path is None:
        return inbox
    pts = np.column_stack([np.ravel(lon), np.ravel(lat)])
    on_land = path.contains_points(pts).reshape(np.shape(lon))
    return inbox & on_land


def landfall_probability(tracks, region=japan_region):
    """Fraction of forecast ensemble tracks that make landfall in ``region``.

    ``tracks`` is ``(n_samples, steps, ≥2)`` from
    :meth:`StormModel.ensemble_forecast`.  Returns ``(probability,
    hit_per_track)`` — the operational **strike probability** read straight off
    the forecast covariance, plus the boolean per-member landfall flag.
    """
    tracks = np.asarray(tracks, float)
    hit = np.array([np.any(region(tr[:, 0], tr[:, 1])) for tr in tracks])
    return float(hit.mean()), hit


# ======================================================================
# Tornado-environment regime model (probabilistic, IKF)
# ======================================================================
class TornadoState(KalmanState):
    STP = "STP"     # significant-tornado-parameter proxy (dimensionless)


class TornadoObservable(KalmanObservable):
    STP = "STP"


# Seasonal IKF transition priors: (stay_benign, stay_tornadic).  Higher
# stay_tornadic and easier switch-in during the local severe season.
_SEASON_PI = {
    "peak":   (0.90, 0.85),   # local tornado season (SH spring–summer for S. Brazil)
    "shoulder": (0.95, 0.60),
    "off":    (0.985, 0.30),  # rare, short-lived excursions
}


class TornadoEnvironmentModel(NativeTwin, ExtendedKalmanModel):
    """
    Mean-reverting tornado-environment latent — one regime of an IKF.

    A single latent ``x = [STP]`` (a significant-tornado-parameter proxy —
    combine CAPE × shear × storm-relative helicity from local mesonet/radar)
    that relaxes toward a regime equilibrium::

        x_{k+1} = μ + a (x_k − μ) + noise      (a = persistence, μ = regime mean)

    A tornado is not deterministically trackable, so the deliverable is the
    **probability the environment sits in the tornadic regime** — obtained by
    running a benign and a tornadic instance of this model in an
    :class:`~kalman.framework.filters.InteractiveKalmanFilter`; the IKF mode probability
    *is* that probability, and it comes with the latent's covariance.
    ``ikf_bank(season=...)`` sets the Markov transition matrix from the local
    **seasonal climatology** — the physically-honest way "seasoning" enters.

    Parameters
    ----------
    mu : float
        Regime equilibrium STP (benign ≈ 0.3, tornadic ≈ 3).
    persistence : float
        AR(1) coefficient ``a`` (0–1); higher = longer regime memory.
    q_std, obs_noise : float
        Process / observation std.
    """

    NAME = "Tornado Environment"
    DESCRIPTION = "Mean-reverting significant-tornado-parameter latent (one IKF regime)."
    VERSION = "1.0"

    @property
    def states(self):      return TornadoState
    @property
    def observables(self): return TornadoObservable

    def __init__(self, sensor_id="ENV", *, mu=0.3, persistence=0.85,
                 q_std=0.15, obs_noise=0.4, dt_hours=1.0, x0=None, **kw):
        self.dtype = np.float64
        self.mu = float(mu)
        self.a = float(persistence)
        self.n = self.m = 1
        self.l = 0
        Q = np.array([[q_std**2]], float)
        R = np.array([[obs_noise**2]], float)
        if x0 is None:
            x0 = np.array([[mu]])
        super().__init__(sensor_id, Q=Q, R=R, L=np.eye(1), fs=1.0 / float(dt_hours),
                         x0=x0, **kw)

    # C++ twin (libKalmanApps: Kalman::Apps::Weather::Tornado).
    def native_name(self):
        return "tornado-environment"

    def native_parameters(self):
        return {"mu": self.mu, "persistence": self.a}

    def f(self, x, u=None, w=None, t=None):
        x = np.asarray(x, float).reshape(-1)
        return np.array([[self.mu + self.a * (x[0] - self.mu)]])

    def h(self, x, v=None, t=None):
        x = np.asarray(x, float).reshape(-1)
        z = np.array([[x[0]]])
        return z if v is None else z + np.asarray(v, float).reshape(1, 1)

    @classmethod
    def ikf_bank(cls, sensor_id="ENV", *, season="shoulder",
                 mu_benign=0.3, mu_tornadic=3.0, **kw):
        """Benign + tornadic instances + the **seasonal** transition matrix.

        Returns ``(models, transition, initial_probs)`` for the IKF; the
        transition matrix is drawn from ``_SEASON_PI[season]`` so the local
        severe-weather season raises both the tornadic-regime persistence and
        the prior probability of switching into it.
        """
        stay_b, stay_t = _SEASON_PI.get(season, _SEASON_PI["shoulder"])
        benign   = cls(sensor_id, mu=mu_benign,   persistence=0.7, q_std=0.10, **kw)
        tornadic = cls(sensor_id, mu=mu_tornadic, persistence=0.9, q_std=0.5,  **kw)
        Pi = np.array([[stay_b, 1 - stay_b], [1 - stay_t, stay_t]])
        probs = np.array([0.9, 0.1])
        return [benign, tornadic], Pi, probs

    imm_bank = ikf_bank      # the literature's name (Interacting Multiple Model)


def synthetic_environment(n_steps=200, *, onsets=((70, 110),), season="shoulder",
                          mu_benign=0.3, mu_tornadic=3.0, obs_noise=0.4, seed=0):
    """Synthetic STP time series that spends ``onsets`` cycle-windows in the
    tornadic regime, plus noisy observations and the true regime labels.

    Returns ``(obs (T, 1), labels (T,))`` — feed ``obs`` to the IKF tracker and
    compare ``mode_probabilities[:, 1]`` against ``labels``.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_steps, dtype=int)
    for lo, hi in onsets:
        labels[lo:hi] = 1
    x = mu_benign
    obs = []
    for k in range(n_steps):
        mu, a, q = ((mu_tornadic, 0.9, 0.5) if labels[k] else (mu_benign, 0.7, 0.10))
        x = mu + a * (x - mu) + rng.normal(0, q)
        obs.append([x + rng.normal(0, obs_noise)])
    return np.array(obs), labels


# ----------------------------------------------------------------------
# Brazil (southern) tornado configuration — SH severe-weather calendar
# ----------------------------------------------------------------------
# Southern Brazil (Rio Grande do Sul, Santa Catarina, Paraná) is a genuine
# tornado hot-spot: the South American Low-Level Jet feeds warm moist Amazonian
# air into cold fronts sweeping up from the south, spiking shear + instability
# in the SH spring–summer.  The *season* is the physically-honest prior — it
# sets the IKF transition matrix, not a hand-tuned threshold.
_BRAZIL_SEASON = {9: "peak", 10: "peak", 11: "peak", 12: "peak", 1: "peak", 2: "peak",
                  3: "shoulder", 8: "shoulder",
                  4: "off", 5: "off", 6: "off", 7: "off"}


def brazil_season(month):
    """Season key ('peak' | 'shoulder' | 'off') for a calendar ``month`` (1–12)
    over southern Brazil — SH spring–summer (Sep–Feb) is peak tornado season."""
    return _BRAZIL_SEASON[int(month) % 12 or 12]


def brazil_tornado_scenario(n_days=360, *, start_month=7, seed=0,
                            mu_benign=0.3, mu_tornadic=3.0, obs_noise=0.4,
                            base_rate=0.006, peak_rate=0.05):
    """A year of daily tornado-environment observations over southern Brazil.

    Tornadic episodes (2–4 days each, front passages) are sampled with a
    **seasonally varying rate** — frequent in the Sep–Feb peak, rare off-season
    — so the same STP excursion carries a very different climatological prior
    depending when it lands.  Returns ``(obs (T,1), labels (T,), months (T,))``.
    """
    rng = np.random.default_rng(seed)
    months = ((start_month - 1 + np.arange(n_days) * 12 // n_days) % 12) + 1
    rate = {"peak": peak_rate, "shoulder": 0.5 * (peak_rate + base_rate), "off": base_rate}
    labels = np.zeros(n_days, dtype=int)
    k = 0
    while k < n_days:
        if rng.random() < rate[brazil_season(months[k])]:
            dur = rng.integers(2, 5)
            labels[k:k + dur] = 1
            k += dur
        else:
            k += 1
    x = mu_benign
    obs = []
    for k in range(n_days):
        mu, a, q = ((mu_tornadic, 0.9, 0.5) if labels[k] else (mu_benign, 0.7, 0.10))
        x = mu + a * (x - mu) + rng.normal(0, q)
        obs.append([x + rng.normal(0, obs_noise)])
    return np.array(obs), labels, months


def seasonal_tornado_probability(obs, months, *, season_fn=brazil_season,
                                 model_cls=None, **ikf_kw):
    """Per-step P(tornadic) with a **season-dependent prior**.

    Runs the IKF once per season (each with that season's transition matrix from
    :meth:`TornadoEnvironmentModel.ikf_bank`) and, at every step, returns the
    probability from the filter whose season matches that step's month.  So an
    identical environmental signal reads as a confident warning in the peak
    season and a hedged one off-season — the Bayesian-correct way to fold the
    Brazil climatology into a nowcast.  Returns the ``(T,)`` probability array.
    """
    from ...framework.filters import InteractiveKalmanFilter
    cls = model_cls or TornadoEnvironmentModel
    obs = np.asarray(obs, float)
    months = np.asarray(months)
    seasons = np.array([season_fn(mo) for mo in months])
    prob = np.zeros(len(obs))
    for s in np.unique(seasons):
        models, Pi, p0 = cls.ikf_bank(season=str(s), **ikf_kw)
        ikf = InteractiveKalmanFilter(models, Pi, p0)
        ikf.forward(obs)
        p_s = ikf.mode_probabilities[:, 1]
        prob[seasons == s] = p_s[seasons == s]
    return prob

"""
GPS positioning with general (and special) relativity.

A GPS receiver solves for its position **and clock** from pseudoranges to several
satellites. The satellites carry atomic clocks that, relative to a ground clock, run
fast by a net **+38.6 µs/day** — the sum of special-relativistic time dilation
(−7.2 µs/day, they move fast) and the general-relativistic blueshift (+45.8 µs/day, they
sit higher in Earth's potential well). Left uncorrected that is **≈ 11.6 km/day** of
range error. On top of the constant rate, each satellite's eccentric orbit adds a
*periodic* relativistic term

    Δt_rel(t) = F · e · √a · sin E(t),   F = −2√μ / c² ≈ −4.443×10⁻¹⁰ s/√m,

which is the correction a receiver actually applies per satellite (up to ~14 m).

Idealisations: the satellite clocks are *not* detuned before launch (real ones are, which
removes the secular rate), the receiver clock is perfect, the Earth does not rotate (no
Sagnac term) and there is no atmosphere.

`GPSModel` is an EKF on the state ``x = [x, y, z, b, ḃ]`` (ECEF position, receiver clock
bias and clock drift, all in metres / m·s⁻¹). The measurement is the per-satellite
pseudorange ``ρ_i = |s_i(t) − p| + b`` (nonlinear → analytic line-of-sight Jacobian). The
relativistic terms are baked into the simulated pseudoranges; toggling
``model.apply_relativistic`` decides whether the receiver removes them in
:meth:`GPSModel.preprocess`, giving a clean **before/after** comparison.

The famous secular rate is *common-mode* and is absorbed by the estimated clock drift
``ḃ`` (which is exactly why a 4th satellite is needed — to solve for the clock); the part
that genuinely corrupts **position** is the per-satellite eccentricity term.
"""
from __future__ import annotations

import numpy as np

from ...framework.constants import KalmanObservable, KalmanState
from ...framework.models.extended import ExtendedKalmanModel
from ...framework.models.native import NativeTwin

# ---- physical constants (SI) ----
C_LIGHT = 299_792_458.0            # speed of light [m/s]
MU_EARTH = 3.986004418e14          # Earth gravitational parameter [m^3/s^2]
R_EARTH = 6_371_000.0              # mean Earth radius [m]
GPS_A = 26_560_000.0               # GPS orbit semi-major axis [m] (period ~11h58m)
F_REL = -2.0 * np.sqrt(MU_EARTH) / C_LIGHT**2          # relativistic factor [s/sqrt(m)]
SECULAR_RATE = 38.6e-6 / 86400.0   # net SR+GR satellite clock rate [s/s] (+38.6 us/day)


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class GPSConstellation:
    """A small Keplerian GPS constellation (vectorised positions + relativistic offsets)."""

    def __init__(self, a, e, inc, raan, argp, M0):
        self.a = np.asarray(a, float)
        self.e = np.asarray(e, float)
        self.inc = np.asarray(inc, float)
        self.raan = np.asarray(raan, float)
        self.argp = np.asarray(argp, float)
        self.M0 = np.asarray(M0, float)
        self.n_mean = np.sqrt(MU_EARTH / self.a**3)                     # mean motion [rad/s]
        # Per-satellite perifocal→ECI rotation R = Rz(Ω) Rx(i) Rz(ω).
        self.R = np.stack([_Rz(O) @ _Rx(i) @ _Rz(w)
                           for O, i, w in zip(self.raan, self.inc, self.argp)])

    @property
    def size(self):
        return self.a.shape[0]

    def _eccentric_anomaly(self, M):
        """Solve Kepler's equation E − e sinE = M (Newton), vectorised over any shape."""
        M = np.mod(M + np.pi, 2.0 * np.pi) - np.pi
        E = M.copy()
        for _ in range(12):
            E = E - (E - self.e * np.sin(E) - M) / (1.0 - self.e * np.cos(E))
        return E

    def _anomalies(self, t):
        t1 = np.atleast_1d(np.asarray(t, float))
        M = self.M0[None, :] + self.n_mean[None, :] * t1[:, None]       # (N, S)
        return t1, self._eccentric_anomaly(M)

    def positions(self, t):
        """Satellite ECEF positions: ``(S,3)`` for scalar ``t`` or ``(N,S,3)`` for array ``t``."""
        scalar = np.ndim(t) == 0
        _, E = self._anomalies(t)
        nu = 2.0 * np.arctan2(np.sqrt(1 + self.e) * np.sin(E / 2),
                              np.sqrt(1 - self.e) * np.cos(E / 2))
        r = self.a * (1.0 - self.e * np.cos(E))                         # (N, S)
        pf = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(r)], axis=-1)  # (N,S,3)
        pos = np.einsum("sij,nsj->nsi", self.R, pf)                     # rotate to ECI/ECEF
        return pos[0] if scalar else pos

    def relativistic_seconds(self, t):
        """Per-satellite eccentricity relativistic clock offset Δt_rel [s]."""
        scalar = np.ndim(t) == 0
        _, E = self._anomalies(t)
        off = F_REL * self.e * np.sqrt(self.a) * np.sin(E)              # (N, S)
        return off[0] if scalar else off


# ===== State / observable enums =====
class GPSState(KalmanState):
    X           = "ECEF x [m]"
    Y           = "ECEF y [m]"
    Z           = "ECEF z [m]"
    CLOCK_BIAS  = "clock bias [m]"
    CLOCK_DRIFT = "clock drift [m/s]"


class GPSObservable(KalmanObservable):
    PSEUDORANGE = "m"


class GPSModel(NativeTwin, ExtendedKalmanModel):
    """EKF GPS positioning with togglable relativistic clock corrections.

    Parameters
    ----------
    sensor_id : str
        Receiver name.
    receiver_lat, receiver_lon : float
        Receiver geodetic latitude / longitude [deg] (spherical Earth).
    n_sat : int
        Number of (highest-elevation, visible) satellites to use.
    fs : float
        Measurement rate [Hz].
    pseudorange_uncertainty : float
        Pseudorange noise std [m].
    process_uncertainty : float
        Position random-walk std [m/sqrt(s)] (the receiver is nearly static).
    clock_drift_uncertainty : float
        Random-walk std of the clock drift [m/s per sqrt(s)]. The simulated receiver clock
        is ideal, so the default is small; a quartz receiver clock needs about 0.1.
    apply_relativistic : bool
        If True, :meth:`preprocess` removes the relativistic terms from the pseudoranges
        before filtering (toggle this to compare before/after).
    """

    NAME = "GPS Relativistic Positioning Model"
    DESCRIPTION = ("EKF over [x,y,z,clock-bias,clock-drift] from satellite pseudoranges, "
                   "with special/general-relativistic clock corrections.")
    VERSION = "1.0"

    @property
    def states(self):
        return GPSState

    @property
    def observables(self):
        return GPSObservable

    def __init__(self, sensor_id: str | int,
                 receiver_lat: float = 45.0,
                 receiver_lon: float = 10.0,
                 n_sat: int = 8,
                 fs: float = 1.0,
                 pseudorange_uncertainty: float = 1.0,
                 process_uncertainty: float = 0.2,
                 clock_drift_uncertainty: float = 1e-3,
                 apply_relativistic: bool = True,
                 secular: bool = True,
                 elevation_mask: float | None = None,
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)
        self.fs = float(fs)
        self.apply_relativistic = bool(apply_relativistic)
        self._sigma_rho = float(pseudorange_uncertainty)
        self._sigma_process = float(process_uncertainty)
        self._sigma_drift = float(clock_drift_uncertainty)
        self._secular_on = bool(secular)
        # Elevation-cutoff [deg] below which a satellite is "out of sight". When set, the model
        # tracks the FULL constellation and masks each satellite epoch-by-epoch as it rises/sets
        # (the EKF skips the masked pseudoranges); when None, a fixed set of satellites is used.
        self.elevation_mask = None if elevation_mask is None else float(elevation_mask)

        # Receiver truth (ECEF, spherical) + local ENU basis.
        lat, lon = np.radians(receiver_lat), np.radians(receiver_lon)
        self.receiver_ecef = R_EARTH * np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=self.dtype)
        self._enu = np.array([
            [-np.sin(lon),               np.cos(lon),              0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [ np.cos(lat) * np.cos(lon),  np.cos(lat) * np.sin(lon), np.sin(lat)],
        ], dtype=self.dtype)

        # Build a 24-satellite pool. With an elevation mask we keep the WHOLE pool and decide
        # visibility dynamically (satellites set and rise); otherwise we keep the n_sat highest
        # above the horizon at t0.
        pool = self._build_pool()
        self.constellation = pool if self.elevation_mask is not None \
            else self._select_visible(pool, n_sat)
        M = self.constellation.size

        # State [x,y,z, b, b_dot]; continuous generator: static receiver
        # (ẋ=ẏ=ż=0), clock bias driven by drift (ḃ=b_dot), drift itself
        # constant (ḃ_dot=0) — only entry is the b<-b_dot coupling. Passed as
        # A= (not F=) below: the base KalmanModel discretizes exactly via
        # expm(A/fs) into self._F, which f() below reads directly.
        A = np.zeros((5, 5), dtype=self.dtype)
        A[3, 4] = 1.0

        # Continuous noise: direct random-walk disturbance on x,y,z (receiver
        # isn't perfectly static) plus a white-noise disturbance driving b_dot's
        # own rate (clock-drift wander) — same Lc/Qc-per-column pattern as the
        # v̇ disturbance elsewhere; passed as Qc=/Lc= for exact Van Loan discretization.
        Lc = np.zeros((5, 4), dtype=self.dtype)
        Lc[0, 0] = Lc[1, 1] = Lc[2, 2] = 1.0
        Lc[4, 3] = 1.0
        qp = process_uncertainty ** 2
        qd = clock_drift_uncertainty ** 2
        Qc = np.diag([qp, qp, qp, qd]).astype(self.dtype)
        R = (self._sigma_rho ** 2) * np.eye(M, dtype=self.dtype)

        self.n = 5
        self.m = M
        if kwargs.get("x0") is None:
            kwargs["x0"] = np.concatenate([self.receiver_ecef, [0.0, 0.0]]).reshape(-1, 1)
        if kwargs.get("P0") is None:
            kwargs["P0"] = np.diag([100.0 ** 2, 100.0 ** 2, 100.0 ** 2,
                                    100.0 ** 2, 1.0 ** 2]).astype(self.dtype)

        super().__init__(sensor_id, A=A, Qc=Qc, Lc=Lc, R=R, fs=fs, **kwargs)

    # ------------------------------------------------------------------
    # The C++ twin (Kalman::Apps::GPS::Receiver): the filtering pass runs
    # there, on the constellation of this instance. preprocess() (the
    # relativistic corrections) stays here: it prepares the data.
    # ------------------------------------------------------------------
    _TWIN_DECODES = ("_F",)                # F = exp(A / fs): the twin decodes the same generator

    def native_name(self):
        return "gps"

    def native_parameters(self):
        return {"pseudorange_uncertainty": self._sigma_rho, "process_uncertainty": self._sigma_process,
                "clock_drift_uncertainty": self._sigma_drift}

    def native_arrays(self):
        c = self.constellation
        return {"semi_major_axis": c.a, "eccentricity": c.e, "inclination": c.inc, "raan": c.raan,
                "arg_perigee": c.argp, "mean_anomaly": c.M0}

    # ------------------------------------------------------------------
    # Constellation construction / visibility
    # ------------------------------------------------------------------
    def _build_pool(self, n_planes=6, per_plane=4):
        rng = np.random.default_rng(2024)
        a, e, inc, raan, argp, M0 = [], [], [], [], [], []
        for p in range(n_planes):
            for s in range(per_plane):
                a.append(GPS_A)
                e.append(rng.uniform(0.002, 0.020))             # eccentricity, as in the real constellation
                inc.append(np.radians(55.0))
                raan.append(2.0 * np.pi * p / n_planes)
                argp.append(rng.uniform(0.0, 2.0 * np.pi))
                M0.append(2.0 * np.pi * s / per_plane + 0.5 * p)
        return GPSConstellation(a, e, inc, raan, argp, M0)

    def _select_visible(self, pool, n_sat):
        pos0 = pool.positions(0.0)                              # (S,3) at t0
        up = self.receiver_ecef / np.linalg.norm(self.receiver_ecef)
        los = pos0 - self.receiver_ecef
        elev = np.arcsin(np.clip((los @ up) / np.linalg.norm(los, axis=1), -1, 1))
        order = np.argsort(elev)[::-1]                          # highest elevation first
        keep = order[:n_sat]
        return GPSConstellation(pool.a[keep], pool.e[keep], pool.inc[keep],
                                pool.raan[keep], pool.argp[keep], pool.M0[keep])

    # ------------------------------------------------------------------
    # Dynamics (linear) + measurement (nonlinear pseudorange)
    # ------------------------------------------------------------------
    def f(self, x, u=None, w=None, t=None):
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)
        xn = self._F @ x
        if w is not None:
            xn = xn + np.asarray(w, dtype=self.dtype).reshape(-1, 1)
        return xn

    def h(self, x, v=None, t=None):
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)
        sat = self.constellation.positions(float(t) if t is not None else 0.0)   # (M,3)
        rng = np.linalg.norm(sat - x[:3, 0], axis=1)            # geometric range
        z = (rng + x[3, 0]).reshape(-1, 1)                      # + receiver clock bias
        if v is not None:
            z = z + np.asarray(v, dtype=self.dtype).reshape(-1, 1)
        return z

    def H(self, x=None, t=None):
        x = (self.x0 if x is None else np.asarray(x, dtype=self.dtype)).reshape(-1, 1)
        sat = self.constellation.positions(float(t) if t is not None else 0.0)   # (M,3)
        diff = x[:3, 0] - sat                                   # ∂range/∂p = (p − s)/|p − s|
        rng = np.linalg.norm(diff, axis=1, keepdims=True)
        H = np.zeros((self.m, 5), dtype=self.dtype)
        H[:, :3] = diff / rng
        H[:, 3] = 1.0                                           # ∂ρ/∂(clock bias)
        return H

    # ------------------------------------------------------------------
    # Relativistic correction (applied to the raw pseudoranges)
    # ------------------------------------------------------------------
    def _correction_meters(self, N):
        """Total relativistic range correction c·(Δt_rel + secular), shape (N, M)."""
        t = self.t0 + np.arange(N) / self.fs
        corr = self.constellation.relativistic_seconds(t)      # (N, M) eccentricity
        if self._secular_on:
            corr = corr + (SECULAR_RATE * t)[:, None]          # common-mode secular
        return C_LIGHT * corr

    def preprocess(self, zk, uk=None, **kwargs):
        if not self.apply_relativistic:
            return zk, uk
        zk = np.asarray(zk, dtype=self.dtype)
        return zk - self._correction_meters(zk.shape[0]), uk

    def inprocess(self, state_vector, **kwargs): return state_vector
    def postprocess(self, zk_hat, **kwargs):     return zk_hat

    # ------------------------------------------------------------------
    # Synthetic measurements (reality includes relativity)
    # ------------------------------------------------------------------
    def simulate(self, duration: float, seed=None) -> np.ndarray:
        """Generate ``(N, n_sat)`` pseudoranges from the true receiver position, with the
        relativistic terms (eccentricity + secular) and Gaussian noise baked in."""
        if seed is not None:
            np.random.seed(seed)
        N = int(round(duration * self.fs))
        t = self.t0 + np.arange(N) / self.fs
        pos = self.constellation.positions(t)                  # (N, M, 3)
        rng = np.linalg.norm(pos - self.receiver_ecef[None, None, :], axis=2)    # (N, M)
        pr = rng + self._correction_meters(N)                  # cb_true = 0
        pr = pr + self._sigma_rho * np.random.randn(*pr.shape)
        if self.elevation_mask is not None:                    # blank out-of-sight satellites
            pr = np.where(self.visible_mask(t), pr, np.nan)    # NaN → EKF skips them
        return pr

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def enu_error(self, record) -> np.ndarray:
        """East/North/Up position error [m] of a filtered record vs. the true receiver."""
        d = np.asarray(record.xk)[:, :3] - self.receiver_ecef[None, :]
        return d @ self._enu.T

    def position_error(self, record) -> np.ndarray:
        """3-D position error magnitude [m] over time."""
        return np.linalg.norm(np.asarray(record.xk)[:, :3] - self.receiver_ecef[None, :], axis=1)

    def elevations(self, t=0.0) -> np.ndarray:
        """Satellite elevation angles [deg] from the receiver at time ``t``."""
        return self.azel(t)[1]

    def azel(self, t=0.0):
        """Satellite (azimuth, elevation) in degrees from the receiver at time ``t``."""
        pos = self.constellation.positions(float(t))           # (M,3)
        los = pos - self.receiver_ecef
        enu = los @ self._enu.T                                 # East, North, Up
        rng = np.linalg.norm(los, axis=1)
        az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0
        el = np.degrees(np.arcsin(np.clip(enu[:, 2] / rng, -1, 1)))
        return az, el

    # ------------------------------------------------------------------
    # Visibility over time (satellites rise & set) + geometry quality (DOP)
    # ------------------------------------------------------------------
    def azel_track(self, t):
        """Vectorised (azimuth, elevation) in degrees over a time array ``t`` → each ``(N, M)``."""
        t = np.atleast_1d(np.asarray(t, float))
        pos = self.constellation.positions(t)                  # (N, M, 3)
        los = pos - self.receiver_ecef[None, None, :]
        enu = los @ self._enu.T                                 # (N, M, 3) East/North/Up
        rng = np.linalg.norm(los, axis=2)
        az = np.degrees(np.arctan2(enu[..., 0], enu[..., 1])) % 360.0
        el = np.degrees(np.arcsin(np.clip(enu[..., 2] / rng, -1, 1)))
        return az, el

    def visible_mask(self, t):
        """Boolean ``(N, M)`` — True where a satellite is above the elevation cutoff at each epoch
        (uses ``elevation_mask`` if set, else the horizon, 0°)."""
        cutoff = 0.0 if self.elevation_mask is None else self.elevation_mask
        return self.azel_track(t)[1] >= cutoff

    def n_visible(self, t):
        """Number of satellites in view at each epoch → ``(N,)``."""
        return self.visible_mask(t).sum(axis=1)

    def dop(self, t):
        """Geometric/position/horizontal/vertical **dilution of precision** over time from the
        *visible* satellites' geometry → dict of ``(N,)`` arrays (NaN when < 4 are in view).

        DOP multiplies the per-satellite ranging error into position error: e.g. the expected
        horizontal error is ``HDOP · σ_pseudorange``. It is the single number that says how good
        the current sky geometry is for a GPS user — and it spikes whenever satellites set."""
        t = np.atleast_1d(np.asarray(t, float))
        pos = self.constellation.positions(t)                  # (N, M, 3)
        los = pos - self.receiver_ecef[None, None, :]
        e = los / np.linalg.norm(los, axis=2, keepdims=True)   # unit line-of-sight (N, M, 3)
        vis = self.visible_mask(t)
        out = {k: np.full(len(t), np.nan) for k in ("gdop", "pdop", "hdop", "vdop")}
        for k in range(len(t)):
            m = vis[k]
            if m.sum() < 4:
                continue
            G = np.column_stack([e[k, m], np.ones(int(m.sum()))])   # rows [ex,ey,ez,1]
            try:
                Cov = np.linalg.inv(G.T @ G)                   # in local ENU? -> rotate diag
            except np.linalg.LinAlgError:
                continue
            # rotate the position block into East/North/Up to split H/V DOP
            Cenu = self._enu @ Cov[:3, :3] @ self._enu.T
            out["gdop"][k] = np.sqrt(np.trace(Cov))
            out["pdop"][k] = np.sqrt(np.trace(Cov[:3, :3]))
            out["hdop"][k] = np.sqrt(Cenu[0, 0] + Cenu[1, 1])
            out["vdop"][k] = np.sqrt(Cenu[2, 2])
        return out

    # ==================================================================
    # Signature figure — one filtered record: ENU error + clock estimate
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     figsize=(14, 5), **kwargs):
        import matplotlib.pyplot as plt
        t = np.asarray(record.time)
        enu = self.enu_error(record)
        xk = np.asarray(record.xk)

        fig, (axE, axC) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(wspace=0.28))
        for k, lbl, col in ((0, "East", "steelblue"), (1, "North", "darkorange"), (2, "Up", "seagreen")):
            axE.plot(t, enu[:, k], color=col, lw=1.6, label=f"{lbl}  (RMS {np.sqrt(np.mean(enu[:,k]**2)):.1f} m)")
        axE.axhline(0, color="gray", lw=0.8)
        axE.set_xlabel("Time (s)"); axE.set_ylabel("position error [m]")
        axE.legend(fontsize=8); axE.grid(True, alpha=0.3)
        axE.set_title(f"{self.name()} — position error"
                      f"  ({'GR-corrected' if self.apply_relativistic else 'uncorrected'})")

        axC.plot(t, xk[:, 3], color="crimson", lw=1.8, label="clock bias b [m]")
        axC2 = axC.twinx()
        axC2.plot(t, xk[:, 4], color="purple", lw=1.4, alpha=0.8, label="clock drift ḃ [m/s]")
        axC.set_xlabel("Time (s)"); axC.set_ylabel("clock bias [m]", color="crimson")
        axC2.set_ylabel("clock drift [m/s]", color="purple")
        axC.grid(True, alpha=0.3); axC.set_title("Estimated receiver clock")
        return fig

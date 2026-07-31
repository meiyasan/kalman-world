"""2D turbulence data assimilation — Kolmogorov flow + ensemble OSSE harness.

The geophysical-DA testbed: a pseudo-spectral 2D incompressible Navier–Stokes
solver (vorticity form, Kolmogorov forcing) wrapped as a :class:`KalmanModel`
whose state is the full vorticity *field* (n = grid²), observed at ``p`` sparse
point probes.  The natural filter is the :class:`EnsembleKalmanFilter` — the
covariance lives in the ensemble, never densely — with Gaspari–Cohn
localization on the periodic torus supplied by the model itself.

:class:`TwinExperiment` is the OSSE (observing-system simulation experiment)
scaffold: truth from the real solver, synthetic observations at known probe
locations, cycled analysis, RMSE/spread diagnostics against a free-running
control.  It is the standard validation harness for any learned component
(surrogate forecast, learned observation operator, …) before it is trusted in
a filter; see ``kalman.apps.weather.surrogate`` for the CNN forecast surrogate.

**RNG discipline** (hard-won): the harness draws the *observation* noise from
one shared stream so every method sees literally the same measurement
realization, while each method's engine consumes its **own** stream for the
perturbed-obs draws — so adding or removing a method can never silently change
another method's results.
"""
from __future__ import annotations
import time
import numpy as np

from ...framework.model import KalmanModel
from ...framework.constants import KalmanEngineKind, KalmanObservable
from ...framework.engines.ensemble import EnsembleKalmanEngine, gaspari_cohn
from ...framework.solvers.spectral import SpectralSolver


# ======================================================================
# Pseudo-spectral solver (the "truth" dynamics)
# ======================================================================
class NS2D(SpectralSolver):
    """Pseudo-spectral 2D incompressible Navier–Stokes (vorticity form).

    .. code-block:: text

        ∂ω/∂t = −u·∇ω + ν ∇²ω + F,    F = χ cos(k_f y)   (Kolmogorov forcing)

    Domain ``[0, 2π]²`` periodic, 2/3-rule dealiasing, RK4 time stepping.
    State vector = vorticity field flattened, shape ``(n²,)``.

    A concrete :class:`~kalman.framework.solvers.SpectralSolver` — the spectral
    grid, ``gradient``/``velocity`` operators, RK4 ``advance`` and numpy/torch
    backends live in the base; this subclass only fixes the **Kolmogorov
    physics** (``rhs_hat``: advection − viscosity + forcing).
    """

    NAME = "NS2D (Kolmogorov)"
    DESCRIPTION = "Pseudo-spectral 2D Navier–Stokes, vorticity form, Kolmogorov forcing."

    def __init__(self, n=64, nu=1/70.0, kf=4, chi=1.0, dt=0.01, seed=0,
                 *, backend=None, device=None, integrator=None):
        super().__init__(n=n, dt=dt, backend=backend, device=device, seed=seed,
                         integrator=integrator)
        self.nu, self.kf, self.chi = nu, kf, chi
        ny, nx = self.shape
        y = self.Ly * np.arange(ny) / ny                     # forcing varies with y
        self.F = chi * np.cos(kf * y)[:, None] * np.ones((1, nx))
        self.Fh = np.fft.fft2(self.F)
        if self.backend == "torch":
            self._Fh_t = self._torch.as_tensor(self.Fh, dtype=self._cdtype, device=self._dev)

    def rhs_hat(self, wh, t=None):
        u, v = self._vel_spectral(wh)
        wx, wy = self._grad_spectral(wh)
        adv = np.fft.fft2(u*wx + v*wy) * self.dealias
        return -adv - self.nu*self.k2*wh + self.Fh

    def _rhs_hat_torch(self, wh):
        t, tt = self._torch, self._tt
        psih = wh * tt["k2inv"]
        u = t.fft.ifft2(1j * tt["ky"] * psih).real
        v = t.fft.ifft2(-1j * tt["kx"] * psih).real
        wx = t.fft.ifft2(1j * tt["kx"] * wh).real
        wy = t.fft.ifft2(1j * tt["ky"] * wh).real
        adv = t.fft.fft2((u*wx + v*wy).to(self._cdtype)) * tt["dealias"]
        return -adv - self.nu * tt["k2"] * wh + self._Fh_t


class FlowObservable(KalmanObservable):
    VORTICITY = "ω"   # point vorticity probe (p sparse channels of the same kind)


# ======================================================================
# The Kalman model — full vorticity field, sparse point observations
# ======================================================================
class KolmogorovFlowModel(KalmanModel):
    """
    Kolmogorov-flow vorticity field as a Kalman state (n = grid², chaotic).

    - ``f`` — advance the field one assimilation cycle (``cycle`` RK4 steps of
      the pseudo-spectral solver); no dense Jacobian exists, so the model
      allows only the ensemble and particle engines.
    - ``h`` — sample the field at ``n_obs`` fixed probe points (+ iid noise
      ``sig_o``); the probes are drawn once from ``seed``.
    - ``localization()`` — Gaspari–Cohn tapers from periodic torus distances
      between grid cells and probes (radius ``loc_radius`` grid units).
    - ``initial_ensemble(N)`` — random climatology snapshots (large, honest
      initial error), from a cached spin-up run.
    - ``forecast(X)`` — batched member propagation, **the AI plug-in point**:
      assign ``model.surrogate = fn`` (columns → columns) to swap the solver
      for a learned forecast without touching the filter.

    Validated lessons carried over from the prototype OSSE (see notebook 13):
    a 16-member localized+inflated EnKF converges to well below the free-run
    error; a *linear* latent space (POD/DMD) hits a representation floor and
    is not offered here; an under-trained one-step surrogate diverges in
    closed loop — check its open-loop rollout error first
    (``flowsurrogate.rollout_rmse``) and budget separate inflation for it.
    """

    NAME = "Kolmogorov Flow"
    DESCRIPTION = ("2D Navier–Stokes vorticity field (Kolmogorov forcing) with sparse "
                   "point probes — ensemble data-assimilation testbed.")
    VERSION = "1.0"

    DEFAULT_ENGINE = KalmanEngineKind.ENSEMBLE
    ALLOWED_ENGINES = frozenset({KalmanEngineKind.ENSEMBLE, KalmanEngineKind.PARTICLE})

    @property
    def observables(self):
        return FlowObservable

    def __init__(self, sensor_id: str,
                 grid: int = 64,
                 nu: float = 1/150.0,
                 chi: float = 2.0,
                 kf: int = 4,
                 dt: float = 0.005,
                 cycle: int = 30,
                 n_obs: int = 100,
                 obs_idx=None,
                 sig_o: float = 0.30,
                 loc_radius: float = 8.0,
                 q_std: float = 0.0,
                 obs_type: str = "vorticity",
                 seed: int = 7,
                 **kwargs):

        self.grid = int(grid)
        self.cycle = int(cycle)
        self.sig_o = float(sig_o)
        self.loc_radius = float(loc_radius)
        self.solver = NS2D(n=self.grid, nu=nu, kf=kf, chi=chi, dt=dt, seed=seed)
        self.surrogate = None          # optional learned forecast (columns → columns)
        self._clim = None              # cached climatology snapshots (d, K)

        # Observation type: "vorticity" (1 scalar per probe) or "gradient"
        # (the two ∂ω/∂x, ∂ω/∂y components per probe — a *linear* operator on
        # the field, so the EnKF assimilates it with no extra approximation).
        self.obs_type = str(obs_type)
        if self.obs_type not in ("vorticity", "gradient"):
            raise ValueError(f"obs_type must be 'vorticity' or 'gradient', got {obs_type!r}")

        d = self.grid * self.grid
        self.n = d
        rng = np.random.default_rng(seed)
        if obs_idx is None:
            obs_idx = rng.choice(d, size=int(n_obs), replace=False)
        self.obs_idx = np.asarray(obs_idx, dtype=int)
        self.n_probes = len(self.obs_idx)
        self.m = self.n_probes * (2 if self.obs_type == "gradient" else 1)

        kwargs.setdefault("fs", 1.0 / (dt * self.cycle))    # one filter step = one cycle
        kwargs.setdefault("Q", np.zeros((d, d)) if q_std == 0.0
                          else np.diag(np.full(d, float(q_std) ** 2)))
        kwargs.setdefault("R", np.diag(np.full(self.m, self.sig_o ** 2)))
        kwargs.setdefault("x0", np.zeros((d, 1)))
        kwargs.setdefault("P0", np.eye(d))
        super().__init__(sensor_id, **kwargs)

    # ------------------------------------------------------------------
    # Dynamics / observation
    # ------------------------------------------------------------------
    def f(self, x, u=None, w=None, t=None):
        """One assimilation cycle of the pseudo-spectral solver."""
        x = np.asarray(x, dtype=float)
        out = self.solver.advance(x.reshape(-1), self.cycle)
        return out.reshape(x.shape) if x.ndim > 1 else out

    def forecast(self, X, t=None):
        """Batched member propagation (d, N) → (d, N); surrogate if assigned."""
        X = np.asarray(X, dtype=float)
        if self.surrogate is not None:
            return np.asarray(self.surrogate(X), dtype=float)
        out = np.empty_like(X)
        for i in range(X.shape[1]):
            out[:, i] = self.solver.advance(X[:, i], self.cycle)
        return out

    def h(self, x, v=None, t=None):
        x = np.asarray(x, dtype=float)
        if self.obs_type == "gradient":
            gx, gy = self.vorticity_gradient(x.reshape(-1))
            z = np.concatenate([gx.ravel()[self.obs_idx],
                                gy.ravel()[self.obs_idx]]).reshape(-1, 1)
        else:
            z = x.reshape(-1)[self.obs_idx].reshape(-1, 1)
        return z if v is None else z + np.asarray(v, dtype=float).reshape(-1, 1)

    def observe(self, X, t=None):
        """Batched observation map (d, N) → (m, N).  For ``obs_type='gradient'``
        the first ``p`` rows are ∂ω/∂x at the probes, the next ``p`` are ∂ω/∂y."""
        X = np.asarray(X, dtype=float)
        if self.obs_type == "gradient":
            gx, gy = self.vorticity_gradient(X)           # (d, N) each
            return np.vstack([gx[self.obs_idx], gy[self.obs_idx]])
        return X[self.obs_idx, :]

    def H(self, x=None, t=None):
        """Linear observation operator (m × d).  Point-selection for vorticity
        obs; a spectral-gradient sampling operator for gradient obs (built once
        and cached — only the ensemble/particle engines run here, which use
        ``observe`` instead, so this serves shape-checking / inspection)."""
        if self.obs_type == "gradient":
            if getattr(self, "_H_grad", None) is None:
                gx = np.zeros((self.n_probes, self.n))
                gy = np.zeros((self.n_probes, self.n))
                e = np.zeros(self.n)
                for j, idx in enumerate(self.obs_idx):     # columns of the operator
                    e[:] = 0.0; e[idx] = 1.0
                    ggx, ggy = self.vorticity_gradient(e)
                    gx[j] = ggx.ravel(); gy[j] = ggy.ravel()
                self._H_grad = np.vstack([gx, gy])
            return self._H_grad
        H = np.zeros((self.m, self.n))
        H[np.arange(self.m), self.obs_idx] = 1.0
        return H

    def F(self, x=None, u=None, t=None):
        """Identity placeholder — the chaotic PDE has no usable dense Jacobian.
        Only the ensemble/particle engines are allowed (``ALLOWED_ENGINES``),
        and both treat the transition as ensemble-implicit, matching this."""
        return np.eye(self.n)

    # ------------------------------------------------------------------
    # DA geometry / climatology
    # ------------------------------------------------------------------
    def torus_distance(self, a_idx, b_idx):
        """Pairwise grid distance between flat indices on the periodic torus."""
        n = self.grid
        ay, ax = np.divmod(np.asarray(a_idx), n)
        by, bx = np.divmod(np.asarray(b_idx), n)
        def wrap(p, q):
            dd = np.abs(p[:, None] - q[None, :])
            return np.minimum(dd, n - dd)
        return np.sqrt(wrap(ax, bx)**2 + wrap(ay, by)**2)

    def localization(self):
        """(rho_xy, rho_yy) Gaspari–Cohn tapers for the ensemble engine.

        For ``obs_type='gradient'`` the two channels of each probe sit at the
        same location, so the tapers are block-tiled: ``rho_xy`` duplicated
        across the ∂x/∂y channels and ``rho_yy`` tiled 2×2."""
        all_idx = np.arange(self.n)
        rho_xy = gaspari_cohn(self.torus_distance(all_idx, self.obs_idx), self.loc_radius)
        rho_yy = gaspari_cohn(self.torus_distance(self.obs_idx, self.obs_idx), self.loc_radius)
        if self.obs_type == "gradient":
            rho_xy = np.hstack([rho_xy, rho_xy])                       # (d, 2p)
            rho_yy = np.block([[rho_yy, rho_yy], [rho_yy, rho_yy]])    # (2p, 2p)
        return rho_xy, rho_yy

    # ------------------------------------------------------------------
    # Spatial differential operators (linear functions of the field)
    # ------------------------------------------------------------------
    def vorticity_gradient(self, field):
        """Spectral vorticity gradient ∇ω = (∂ω/∂x, ∂ω/∂y).

        The spectral ``ik`` operator itself lives on the solver (``NS2D``, the
        single source of truth — it is the same one the dynamics use); this
        wraps it for the DA layer: a *linear* operator on the field, so it drops
        straight into the Kalman machinery as an observation operator
        (:meth:`observe`) or an analysis diagnostic (:meth:`gradient_diagnostic`).
        Accepts a flat ``(d,)`` field, a ``(g, g)`` field, or a batched ensemble
        ``(d, N)``; returns ``(∂ω/∂x, ∂ω/∂y)`` in the matching layout."""
        F = np.asarray(field, dtype=float)
        if F.ndim == 2 and F.shape[0] == self.n:               # batched (d, N)
            g = self.grid
            w = F.T.reshape(-1, g, g)
            wh = np.fft.fft2(w, axes=(-2, -1))
            gx = np.real(np.fft.ifft2(1j * self.solver.kx[None] * wh, axes=(-2, -1)))
            gy = np.real(np.fft.ifft2(1j * self.solver.ky[None] * wh, axes=(-2, -1)))
            return gx.reshape(-1, self.n).T, gy.reshape(-1, self.n).T
        return self.solver.gradient(F)                          # single field

    def velocity(self, field):
        """Incompressible velocity ``(u, v) = ∇⊥ψ`` from the vorticity field —
        delegates to the solver's operator (``NS2D.velocity``).  This is the
        deep-layer flow the storm tracker consumes as *steering* (see
        ``kalman.apps.weather.storms.field_steering``)."""
        return self.solver.velocity(field)

    def velocity_sampler(self, field, region, *, speed=40.0):
        """Return a callable ``(lon, lat) → (v_e, v_n)`` km/h that reads this
        flow field's velocity at the geographic point.

        Maps a geographic ``region = (lon_min, lon_max, lat_min, lat_max)`` onto
        the flow's periodic box (bilinear interpolation, torus-wrapped) and
        scales the non-dimensional velocity to ``speed`` km/h at unit magnitude.
        This is the bridge that lets the field EnKF **supply the deep-layer
        steering flow** to :func:`kalman.apps.weather.storms.field_steering` — the
        environment estimate (with its ensemble uncertainty) driving the
        typhoon tracker.
        """
        u, v = self.velocity(field)
        g = self.grid
        lo0, lo1, la0, la1 = region
        norm = (np.abs(u) + np.abs(v)).mean() + 1e-9

        def sample(lon, lat):
            fx = ((np.asarray(lon) - lo0) / (lo1 - lo0)) % 1.0 * g
            fy = ((np.asarray(lat) - la0) / (la1 - la0)) % 1.0 * g
            i0 = np.floor(fx).astype(int) % g; j0 = np.floor(fy).astype(int) % g
            i1 = (i0 + 1) % g; j1 = (j0 + 1) % g
            ax = fx - np.floor(fx); ay = fy - np.floor(fy)
            def bilinear(A):
                return (A[j0, i0]*(1-ax)*(1-ay) + A[j0, i1]*ax*(1-ay)
                        + A[j1, i0]*(1-ax)*ay + A[j1, i1]*ax*ay)
            ue = bilinear(u) / norm * speed
            vn = bilinear(v) / norm * speed
            return float(ue), float(vn)
        return sample

    def gradient_diagnostic(self, members, k_sigma=1.0):
        """Ensemble estimate of ∇ω **with its uncertainty**.

        ``members`` is the ``(d, N)`` ensemble (e.g. ``engine.members`` after a
        run).  Each member's gradient is a linear image of the member, so the
        ensemble of gradients gives a calibrated mean field and a ±σ band —
        the field-scale analogue of the storm tracker's covariance ellipse.
        Returns a dict of ``(g, g)`` maps: ``gx_mean``/``gy_mean`` (components),
        ``mag_mean`` (|∇ω| mean), ``mag_sigma`` (its ``k_sigma``-σ uncertainty)."""
        gx, gy = self.vorticity_gradient(np.asarray(members, dtype=float))   # (d, N)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        g = self.grid
        return {"gx_mean": gx.mean(1).reshape(g, g),
                "gy_mean": gy.mean(1).reshape(g, g),
                "mag_mean": mag.mean(1).reshape(g, g),
                "mag_sigma": (k_sigma * mag.std(1)).reshape(g, g)}

    def climatology(self, n_snapshots=150, spinup=2500, stride=10):
        """Spin up to the attractor and collect snapshots — cached ``(d, K)``.

        The snapshots seed initial ensembles and set the climatological error
        scale that any assimilation run must beat.  The first call fixes the
        cache — prime it explicitly (e.g. with a shorter spin-up in tests)
        before anything that consumes it.
        """
        if self._clim is None:
            w = self.solver.advance(self.solver.random_ic(), spinup)
            snaps = []
            for _ in range(n_snapshots):
                w = self.solver.advance(w, stride)
                snaps.append(w.ravel().copy())
            self._clim = np.array(snaps).T
        return self._clim

    def initial_ensemble(self, N, rng=None):
        """Random climatology snapshots as columns (d, N) — honest large error."""
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        S = self.climatology()
        return S[:, rng.choice(S.shape[1], size=N, replace=False)].copy()

    def equilibrium_state(self, decorrelate=400):
        """A truth-candidate state on the attractor, decorrelated from the
        climatology snapshots by ``decorrelate`` extra solver steps."""
        S = self.climatology()
        return self.solver.advance(S[:, -1].copy(), decorrelate)

    # ------------------------------------------------------------------
    # Signature figure — vorticity maps instead of 4096 time series
    # ------------------------------------------------------------------
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     figsize=(13, 4), **kwargs):
        """Final analysis field, truth (if ``sample`` given), and their difference."""
        import matplotlib.pyplot as plt
        n = self.grid
        est = np.asarray(record.xk)[-1].reshape(n, n)
        truth = None if sample is None else np.asarray(sample.xk)[-1].reshape(n, n)
        ncols = 1 if truth is None else 3
        fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
        axes = axes[0]
        vmax = np.abs(est if truth is None else truth).max()
        im = axes[0].imshow(est, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[0].set_title("Analysis ω̂ (final cycle)")
        if truth is not None:
            axes[1].imshow(truth, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            axes[1].set_title("Truth ω")
            axes[2].imshow(est - truth, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            axes[2].set_title(f"Error (RMSE {np.sqrt(np.mean((est-truth)**2)):.3f})")
        for ax in axes:
            oy, ox = np.divmod(self.obs_idx, n)
            ax.plot(ox, oy, "k.", ms=2, alpha=0.6)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=list(axes), shrink=0.85, label="vorticity")
        fig.suptitle(f"{self.name()} — probes marked in black", fontweight="bold")
        return fig


# ======================================================================
# OSSE twin-experiment harness
# ======================================================================
class TwinExperiment:
    """
    Cycled identical-twin experiment: truth from the model's real solver,
    shared synthetic observations, one ensemble per registered method::

        osse = TwinExperiment(model, n_cycles=60, n_members=16)
        osse.add_method("enkf", inflation=1.06)
        osse.add_method("surrogate", inflation=1.15, forecast=fn)   # AI forecast
        osse.add_free_run()
        hist = osse.run()

    Every method sees the **same** truth, probes, observation-noise
    realization (one shared ``rng_obs`` stream) and initial ensemble; each
    method's engine draws its perturbed-obs noise from its **own** stream
    (``seed_eps + k``), so the comparison is invariant to which methods are
    registered — the RNG discipline the prototype had to learn the hard way.

    ``run()`` returns ``{name: {"rmse_f", "rmse_a", "spread", "time"}}``
    (forecast/analysis RMSE vs truth, mean ensemble spread, wall time).

    **Asynchronous stations** — real observation networks are neither
    synchronized nor equal-rate.  ``obs_periods=(lo, hi)`` gives every probe
    its own reporting period (uniform in ``[lo, hi]`` cycles) and random
    phase; silent stations contribute NaN, which the engines' masked update
    skips natively.  The schedule is deterministic given ``seed_setup`` and
    drawn from its own stream, so enabling it never changes any other draw.

    **Truth replay** — by default the truth evolves with the model's own solver
    (a *perfect-model* twin).  Pass ``truth_trajectory=`` (one field per cycle,
    ``(T, n)`` or ``(T, grid, grid)``) to replay an **external** truth instead:
    a run of the solver with different physics (imperfect-model experiment) or
    a real reanalysis sequence (see ``kalman.apps.weather.geo`` for the ERA5
    loader chain).  Observations are then synthesized from the replayed fields
    at the model's probes, and every RMSE is measured against them — so the
    same harness, ratios (:meth:`skill`) and RNG discipline cover real data.
    """

    def __init__(self, model: KolmogorovFlowModel, *, n_cycles=60, n_members=16,
                 seed_obs=999, seed_setup=123, seed_eps=200, truth0=None,
                 truth_trajectory=None, obs_periods=None):
        self.model = model
        self.n_cycles = int(n_cycles)
        self.n_members = int(n_members)
        self.rng_obs = np.random.default_rng(seed_obs)
        self._seed_eps = int(seed_eps)
        self._truth0 = truth0
        self._truth_traj = None
        if truth_trajectory is not None:
            traj = np.asarray(truth_trajectory, dtype=float)
            if traj.ndim == 3:
                traj = traj.reshape(traj.shape[0], -1)
            if traj.ndim != 2 or traj.shape[1] != model.n:
                raise ValueError(f"truth_trajectory must be (T, {model.n}) or "
                                 f"(T, {model.grid}, {model.grid}); got {traj.shape}")
            self._truth_traj = traj
            self.n_cycles = min(self.n_cycles, traj.shape[0])
        self._ens0 = model.initial_ensemble(self.n_members,
                                            rng=np.random.default_rng(seed_setup))
        # --- asynchronous observation network (optional) -------------------
        # Real stations are neither synchronized nor equal-rate: each probe
        # gets its own reporting period and phase, and cycles where it stays
        # silent contribute NaN — which the engines' masked update skips
        # natively.  obs_periods=(lo, hi) draws per-station periods uniformly;
        # an array gives them explicitly; None (default) = everyone, every
        # cycle.  The schedule comes from its own seed_setup-derived stream so
        # enabling it never perturbs the ensemble/observation draws.
        self.obs_mask = self.obs_periods = None
        if obs_periods is not None:
            rng_sched = np.random.default_rng(seed_setup + 1)
            p = model.m
            if isinstance(obs_periods, tuple) and len(obs_periods) == 2:
                periods = rng_sched.integers(obs_periods[0], obs_periods[1] + 1, size=p)
            else:
                periods = np.broadcast_to(np.asarray(obs_periods, dtype=int), (p,)).copy()
            phases = rng_sched.integers(0, periods)
            cyc = np.arange(self.n_cycles)[:, None]
            self.obs_mask = ((cyc + phases[None, :]) % periods[None, :]) == 0
            self.obs_periods = periods
        self.methods = {}

    class _MethodView:
        """Per-method model view: optional surrogate forecast and/or additive
        model-error covariance override (``Q = q_std² I``), everything else
        proxied to the shared model."""
        def __init__(self, model, fn=None, q_std=None):
            object.__setattr__(self, "_model", model)
            object.__setattr__(self, "_fn", fn)
            object.__setattr__(self, "_q_std", q_std)
        def forecast(self, X, t=None):
            return self._fn(X) if self._fn is not None else self._model.forecast(X, t=t)
        def Q(self, t=None):
            if self._q_std is None:
                return self._model.Q(t=t)
            return np.diag(np.full(self._model.n, float(self._q_std) ** 2))
        def __getattr__(self, name):
            return getattr(self._model, name)

    def add_method(self, name, engine=None, *, inflation=1.06, forecast=None,
                   q_std=None, assimilate=True, seed=None):
        """Register a method. ``forecast`` (columns → columns) swaps the solver
        for a surrogate; ``q_std`` gives this branch its own additive
        model-error noise (``Q = q_std² I`` drawn each forecast — the standard
        answer when the forecast model is known to be wrong);
        ``assimilate=False`` makes it a free-running control."""
        if engine is None:
            engine = EnsembleKalmanEngine(
                n_members=self.n_members, inflation=inflation,
                seed=self._seed_eps + len(self.methods) if seed is None else seed)
        view = (self.model if forecast is None and q_std is None
                else self._MethodView(self.model, fn=forecast, q_std=q_std))
        self.methods[name] = dict(engine=engine, view=view, assimilate=assimilate)
        return engine

    def add_free_run(self, name="free"):
        """No-assimilation control — the baseline any filter must beat."""
        return self.add_method(name, assimilate=False)

    def run(self, verbose=True, every=15, callback=None, record_fields=False):
        model = self.model
        if self._truth_traj is not None:
            truth = self._truth_traj[0]
        else:
            truth = model.equilibrium_state() if self._truth0 is None else np.asarray(self._truth0).ravel().copy()
        sig = np.sqrt(np.diag(model.R()))

        hist = {}
        for name, m in self.methods.items():
            m["engine"].reset(m["view"])
            m["engine"]._members = self._ens0.copy()   # shared start: fair comparison
            hist[name] = {"rmse_f": [], "rmse_a": [], "spread": [], "time": 0.0}
        self.fields = ({"truth": [], **{n: {"forecast": [], "analysis": []} for n in self.methods}}
                       if record_fields else None)

        for c in range(self.n_cycles):
            truth = model.f(truth) if self._truth_traj is None else self._truth_traj[c]
            y = np.asarray(model.h(truth)).reshape(-1) + self.rng_obs.normal(0.0, sig)
            if self.obs_mask is not None:
                y = np.where(self.obs_mask[c], y, np.nan)   # silent stations this cycle
            if record_fields:
                self.fields["truth"].append(truth.copy())
            for name, m in self.methods.items():
                eng, h = m["engine"], hist[name]
                t0 = time.time()
                mu_f = eng.forecast(m["view"]).ravel()
                h["rmse_f"].append(float(np.sqrt(np.mean((mu_f - truth) ** 2))))
                if m["assimilate"]:
                    eng.analyze(m["view"], y)
                h["time"] += time.time() - t0
                mu_a = eng.mean.ravel()
                h["rmse_a"].append(float(np.sqrt(np.mean((mu_a - truth) ** 2))))
                h["spread"].append(eng.spread())
                if record_fields:
                    self.fields[name]["forecast"].append(mu_f.copy())
                    self.fields[name]["analysis"].append(mu_a.copy())
            if callback is not None:
                callback(c, truth, hist)
            if verbose and (c + 1) % every == 0:
                line = "  ".join(f"{k}={v['rmse_a'][-1]:.3f}" for k, v in hist.items())
                print(f"cycle {c+1:3d}: RMSE_a  {line}")

        self.truth = truth
        self.hist = hist
        return hist

    def climatological_error(self):
        """RMS distance of climatology from the current truth — the 'no skill' scale."""
        S = self.model.climatology()
        return float(np.sqrt(np.mean((S - self.truth[:, None]) ** 2)))

    def skill(self, reference="free", flavor="rmse_a"):
        """Per-cycle RMSE ratio of every method against a ``reference`` method.

        ``skill < 1`` means better than the reference at that cycle.  The
        reference may live in another run's history dict (e.g. the true-solver
        EnKF branch when this experiment holds the surrogate branches) — pass
        that history as ``reference`` directly.  ``flavor`` selects analysis
        (``rmse_a``) or forecast (``rmse_f``) error.
        """
        ref = reference if not isinstance(reference, str) else self.hist[reference]
        ref = np.asarray(ref[flavor] if isinstance(ref, dict) else ref, dtype=float)
        return {name: np.asarray(h[flavor]) / ref
                for name, h in self.hist.items()
                if not (isinstance(reference, str) and name == reference)}

    def animate(self, method=None, *, phases=1, step=1, interval=300,
                figsize=(12.5, 4.2), cmap="RdBu_r"):
        """Interactive replay of the assimilation — truth | estimate | error.

        Default (``phases=1``): **one frame per cycle** — the analysis field
        with a **black** dot at exactly the stations that delivered a
        measurement that cycle; idle stations stay **transparent**.  Nothing
        pulses: the dot pattern changes between cycles only because the
        multirate network (``obs_periods=``) changes who reports.

        ``phases=2`` is the didactic mode: each cycle additionally plays its
        forecast frame with *all* stations idle, making the measurement
        arrival visible as an explicit step (at the cost of an on/off rhythm).

        Requires ``run(record_fields=True)``.  ``step`` subsamples cycles to
        keep the embedded player small; returns the framework's standard
        Jupyter play/scrub player (see ``helpers.animate``).
        """
        import matplotlib.pyplot as plt
        from ...helpers import animate as _animate
        if not getattr(self, "fields", None):
            raise RuntimeError("No recorded fields — call run(record_fields=True) first.")
        if method is None:
            method = next(n for n, m in self.methods.items() if m["assimilate"])

        g = self.model.grid
        oy, ox = np.divmod(self.model.obs_idx, g)
        n_st = len(ox)
        truths = self.fields["truth"]
        F = self.fields[method]
        cycle_phases = (1,) if int(phases) == 1 else (0, 1)
        frames = [(c, phase) for c in range(0, len(truths), int(step))
                  for phase in cycle_phases]
        vmax = float(np.abs(np.asarray(truths)).max())

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        ims, dots = [], []
        for ax, title in zip(axes, (r"truth $\omega$", f"{method} estimate", "error")):
            im = ax.imshow(np.zeros((g, g)), cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
            dot = ax.scatter(ox, oy, s=16, edgecolors="none", zorder=3,
                             facecolors=np.zeros((n_st, 4)))
            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            ims.append(im); dots.append(dot)
        fig.colorbar(ims[0], ax=list(axes), shrink=0.8, label="vorticity")

        def _draw(frame):
            c, phase = frame
            tr = truths[c]
            est = F["forecast" if phase == 0 else "analysis"][c]
            ims[0].set_data(tr.reshape(g, g))
            ims[1].set_data(est.reshape(g, g))
            ims[2].set_data((est - tr).reshape(g, g))
            face = np.zeros((n_st, 4))                       # transparent: idle station
            n_rep = 0
            if phase == 1:
                rep = (self.obs_mask[c] if self.obs_mask is not None
                       else np.ones(n_st, dtype=bool))
                face[rep] = (0.0, 0.0, 0.0, 1.0)             # black: measurement delivered
                n_rep = int(rep.sum())
            for dot in dots:
                dot.set_facecolors(face)
            rmse = np.sqrt(np.mean((est - tr) ** 2))
            fig.suptitle(f"cycle {c+1}/{len(truths)} — "
                         + ("forecast (stations idle)" if phase == 0
                            else f"analysis — {n_rep}/{n_st} stations reported")
                         + f" — RMSE {rmse:.2f}", fontweight="bold")
            return (*ims, *dots)

        return _animate(fig, _draw, frames=frames, interval=interval)

    def plot(self, figsize=(12, 4)):
        """Analysis-RMSE and spread curves for every method."""
        import matplotlib.pyplot as plt
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
        for name, h in self.hist.items():
            cyc = np.arange(1, len(h["rmse_a"]) + 1)
            ax0.semilogy(cyc, h["rmse_a"], label=f"{name} ({h['time']:.1f}s)")
            ax1.plot(cyc, h["spread"], label=name)
        ax0.set_xlabel("cycle"); ax0.set_ylabel("analysis RMSE"); ax0.legend(); ax0.grid(alpha=0.3)
        ax0.set_title("Analysis error vs truth")
        ax1.set_xlabel("cycle"); ax1.set_ylabel("mean ensemble spread"); ax1.legend(); ax1.grid(alpha=0.3)
        ax1.set_title("Ensemble spread (should track RMSE)")
        fig.tight_layout()
        return fig

"""Interacting Multiple Model (IMM) filter.

Pluggable routing — swap the router without touching the filter::

    ikf = InteractiveKalmanFilter(models, Pi)                           # Bayesian default
    ikf = InteractiveKalmanFilter(models, Pi, router=ThresholdRouter())
    ikf = InteractiveKalmanFilter(models, Pi,
                                  router=NeuralRouter(model=MLP(hidden=32)))

Training (for learning routers)::

    ikf.fit(training_zk_list, labels_list)   # collects NIS, trains router
    record = ikf.forward(test_zk)

See ``kalman.framework.routers`` for all built-in routers and
``kalman.framework.routers.nn`` for pluggable neural architectures.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import multivariate_normal

from ..framework.filter import KalmanFilter
from ..framework.io.record import KalmanRecord
from ..framework.routers.base     import KalmanRouter
from ..framework.routers.bayesian import BayesianRouter


class InteractiveKalmanFilter:
    """Multiple models interact to handle regime switching (IMM filter).

    Runs M ``KalmanFilter`` instances in parallel — one per operating regime.
    At every step:

    1. **Interaction / mixing** — states mixed by conditional probs μ_{i|j}.
    2. **Predict + update** — each model runs its own Kalman step.
    3. **NIS** — NIS_j = ν_j^T S_j^{-1} ν_j logged per model.
    4. **Routing** — ``router.route(nis_arr, c, likelihoods)`` → updated μ.
    5. **Combination** — x̂ = Σ μ_j x̂_j.

    Parameters
    ----------
    models : list of KalmanModel
    transition : ndarray (M, M)
        Markov transition matrix.  Rows sum to 1.
    initial_probs : ndarray (M,), optional
        Prior mode probabilities.  Default: uniform.
    router : KalmanRouter or callable, optional
        Any callable ``(nis_arr, c, likelihoods) → mu`` or ``KalmanRouter``
        instance.  Default: ``BayesianRouter()`` (classical IMM).
    """

    NAME        = "Interactive Kalman Filter"
    DESCRIPTION = "IMM — interacting multiple model filter for regime switching."

    def __init__(self, models, transition, initial_probs=None, *, router=None):
        self.M_count = len(models)
        self.models  = models
        self.filters = [KalmanFilter(m) for m in models]
        self.Pi      = np.asarray(transition, dtype=float)
        self.mu      = (np.ones(self.M_count) / self.M_count
                        if initial_probs is None
                        else np.asarray(initial_probs, dtype=float))
        self.router: KalmanRouter = (router if router is not None
                                     else BayesianRouter())
        self.mu_log:             list = []
        self.combined_posterior: list = []
        self.nis_log:            list = []

    @property
    def model(self):
        return self.models[0]

    def sample_rate(self):
        return self.models[0].fs

    def reset(self):
        for f in self.filters:
            f.reset()
        self.mu = np.ones(self.M_count) / self.M_count
        self.mu_log.clear()
        self.combined_posterior.clear()
        self.nis_log.clear()

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, training_zk_list, true_regime_list,
            epochs: int = 500, lr: float = 2e-3, seed=None,
            **kwargs) -> "InteractiveKalmanFilter":
        """Train the router using NIS histories from labelled data.

        Temporarily switches to ``BayesianRouter`` to collect neutral NIS
        histories, then calls ``router.fit(nis_arrays, labels, **kwargs)``.

        For stateless routers (``BayesianRouter``, ``ThresholdRouter``) this
        is a no-op — the call completes without training.

        Parameters
        ----------
        training_zk_list : list of ndarray (T_i, m)
        true_regime_list : list of int ndarray (T_i,)
        epochs, lr, seed
            Forwarded to ``router.fit``.
        """
        saved, self.router = self.router, BayesianRouter()
        nis_arrays, label_arrays = [], []
        for zk, labels in zip(training_zk_list, true_regime_list):
            self.forward(zk, record=False)
            nis_arrays.append(np.array(self.nis_log))
            label_arrays.append(np.asarray(labels, dtype=int))

        self.router = saved
        self.router.fit(nis_arrays, label_arrays,
                        epochs=epochs, lr=lr, seed=seed, **kwargs)
        return self

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, gk, uk=None, *, record=True, **kwargs):
        indexing = kwargs.pop("indexing", "xy")
        self.reset()

        gk_arr = gk.zk if isinstance(gk, KalmanRecord) else gk
        gk_arr = KalmanRecord._normalize_in(gk_arr, indexing=indexing,
                                            dtype=self.models[0].dtype)

        uk_fn = uk if callable(uk) else None
        t     = float(self.models[0].t0) if self.models[0].t0 is not None else 0.0
        dt    = 1.0 / self.sample_rate() if self.sample_rate() else 1.0
        nu_c  = np.zeros((self.models[0].m, 1))

        for z in gk_arr:
            # 1. Markov prediction: c_j = Σ_i Π[i,j] μ_i
            c = np.maximum(self.Pi.T @ self.mu, 1e-12)

            # 2. State mixing
            mu_ij = (self.Pi * self.mu[:, None]) / c   # (M, M)
            for j in range(self.M_count):
                x_mix = sum(mu_ij[i, j] * self.filters[i].x
                            for i in range(self.M_count))
                P_mix = sum(mu_ij[i, j] * (
                    self.filters[i].P
                    + (self.filters[i].x - x_mix) @ (self.filters[i].x - x_mix).T)
                    for i in range(self.M_count))
                self.filters[j].x = x_mix
                self.filters[j].P = P_mix

            # 3. Per-model predict + update; compute NIS and Gaussian likelihood
            if uk_fn is not None:
                x_c = sum(self.mu[j] * self.filters[j].x
                          for j in range(self.M_count))
                P_c = sum(self.mu[j] * (
                    self.filters[j].P
                    + (self.filters[j].x - x_c) @ (self.filters[j].x - x_c).T)
                    for j in range(self.M_count))
                u = np.asarray(uk_fn(x_c, P_c, t, nu=nu_c),
                               dtype=float).reshape(-1, 1)
            else:
                u = None

            nis_step    = np.zeros(self.M_count)
            likelihoods = np.zeros(self.M_count)

            for j in range(self.M_count):
                fj = self.filters[j]
                mj = self.models[j]
                uj = u if u is not None else np.zeros((max(mj.l, 1), 1))

                fj.x, fj.P, F = fj.engine._predict(mj, fj.x, fj.P, u=uj, t=t)
                fj.prior.append((fj.x, fj.P))
                fj.transition.append(F)

                zk, H_j = fj.engine._measure(mj, fj.x, t)
                nu_j = (np.asarray(z, dtype=float).reshape(-1, 1)
                        - np.asarray(zk, dtype=float).reshape(-1, 1))
                S_j  = H_j @ fj.P @ H_j.T + mj.R(t=t)
                fj.innovation.append(nu_j)

                try:
                    nis_step[j]    = float(
                        (nu_j.T @ np.linalg.solve(S_j, nu_j)).item())
                    likelihoods[j] = multivariate_normal.pdf(
                        nu_j.ravel(),
                        mean=np.zeros(nu_j.shape[0]),
                        cov=S_j)
                except np.linalg.LinAlgError:
                    nis_step[j]    = float(nu_j.shape[0])
                    likelihoods[j] = 1e-12

                fj.x, fj.P = fj.engine._update(mj, fj.x, fj.P, z, t=t)
                fj.posterior.append((fj.x, fj.P))
                fj.measurement.append(z)

            self.nis_log.append(nis_step)

            # 4. Route
            nis_arr  = np.array(self.nis_log)
            self.mu  = np.asarray(self.router(nis_arr, c, likelihoods),
                                  dtype=float)
            self.mu  = np.maximum(self.mu, 1e-12)
            self.mu /= self.mu.sum()
            self.mu_log.append(self.mu.copy())

            # 5. Combined estimate
            x_c = sum(self.mu[j] * self.filters[j].x
                      for j in range(self.M_count))
            P_c = sum(self.mu[j] * (
                self.filters[j].P
                + (self.filters[j].x - x_c) @ (self.filters[j].x - x_c).T)
                for j in range(self.M_count))
            self.combined_posterior.append((x_c, P_c))

            h_c  = self.models[0].h(x_c, t=t)
            nu_c = (np.asarray(z, dtype=float).reshape(-1, 1)
                    - np.asarray(h_c, dtype=float).reshape(-1, 1))
            t += dt

        return self._build_record(gk_arr, indexing=indexing) if record else self

    def _build_record(self, zk, indexing="xy"):
        f0 = self.filters[0]
        f0.posterior   = self.combined_posterior
        f0.measurement = f0.measurement if f0.measurement else list(zk)
        return f0.get_record(indexing=indexing)

    def filter(self, z, u=None, **kwargs):
        if u is None:
            u = kwargs.pop("uk", None)
        return self.forward(z, u, **kwargs)

    @property
    def mode_probabilities(self) -> np.ndarray:
        """Mode-probability history, shape (T, M)."""
        return (np.array(self.mu_log) if self.mu_log
                else np.empty((0, self.M_count)))

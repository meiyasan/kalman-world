# kalman/framework/sampler.py
from __future__ import annotations
import numpy as np

from .io.record import KalmanRecord
from ..helpers import format_shape
from .constants import KalmanStateFlavor


class KalmanSampler:
    """
    Generate synthetic trajectories and measurements from a **model**:
      - propagate the state with the model's dynamics (via an engine's predict step),
      - draw process/measurement noise from model.Q / model.R,
      - emit measurements through model.h.

    Sampling is a model capability (it needs only the system + an engine's predict);
    by default it uses ``model.default_engine()``. A filter may pass its own engine
    (e.g. the UKF engine) for consistency.
    """

    def __init__(self, model, engine=None):
        self.model = model
        self._engine = engine if engine is not None else model.default_engine()
        self.reset()   # initialise the running cursor (time + state)

    def reset(self):
        """
        Rewind the sampler stream to the model's initial time/state.

        A ``KalmanSampler`` is *stateful*: successive ``measurement()`` calls on the
        same instance **resume** from the last sample (continuous time and state), so
        you can build a long trajectory in chunks. Call ``reset()`` (alias ``clear()``)
        to start a fresh, independent stream from ``t0`` / ``x0`` / ``P0``.
        (``model.sample()`` always uses a fresh sampler, so it is unaffected.)
        """
        self._t = float(self.model.t0) if self.model.t0 is not None else 0.0
        x0 = self.model.get_initial_state_vector()
        P0 = self.model.get_initial_state_covariance()
        self._x = None if x0 is None else np.array(x0, dtype=float, copy=True)
        self._P = None if P0 is None else np.array(P0, dtype=float, copy=True)
        return self

    clear = reset

    @property
    def time(self) -> float:
        """Current cursor time (where the next ``measurement()`` will resume)."""
        return self._t

    @property
    def engine(self):
        return self._engine
    @property
    def fs(self) -> float:
        return self.model.fs

    @property
    def n(self):
        return self.model._infer_n()
    @property
    def m(self):
        return self.model._infer_m()
    @property
    def l(self):
        return self.model._infer_l()

    def measurement(self, duration: float, uk=None, **kwargs):
        '''
        Generate synthetic measurements for a given duration.
        Returns a KalmanRecord with the sampled truth (xk) and measurements (zk).

        ``covariance`` (default ``False``) selects how the truth is propagated:

          - ``False`` (default) — **fast truth path**: propagate the mean with
            ``model.f`` directly (one dynamics call/step) and inject process noise
            additively, so the returned ``xk`` is a genuine stochastic realisation of
            the system (it actually drifts under ``Q``). The Jacobian and covariance
            recursion are skipped, so ``Pk`` is returned as zeros (not computed). This
            is what you want for filtering/statistics — the truth is a real sample path.
          - ``True``  — route each step through ``engine._predict`` and record the
            predicted covariance ``Pk`` (``F P Fᵀ + Q``); the truth *mean* is the
            deterministic (noise-free) propagation, with ``Q`` reflected solely in
            ``Pk``. On an EKF/UKF engine this also evaluates a per-step numerical
            Jacobian ``model.F``, costing ~``n+2`` dynamics calls per sample. Use it
            when you specifically need the analytic prior covariance band.

        Note: process noise enters the truth *mean* only on the (default)
        ``covariance=False`` path; ``covariance=True`` reflects ``Q`` solely in ``Pk``.
        '''

        inprocess = kwargs.pop("inprocess", True)
        postprocess = kwargs.pop("postprocess", True)
        indexing = kwargs.pop("indexing", "xy")

        covariance = kwargs.pop("covariance", False)
        measurement_noise = kwargs.pop("measurement_noise", True)
        process_noise = kwargs.pop("process_noise", True)

        n_samples = int(duration * self.fs)
        zk = np.zeros((n_samples, self.m, 1))
        xk = np.zeros((n_samples, self.n, 1))
        Pk = np.zeros((n_samples, self.n, self.n))

        # Resume from the running cursor unless an explicit x0/P0 is supplied.
        x = kwargs.pop("x0", None); x = self._x if x is None else x
        P = kwargs.pop("P0", None); P = self._P if P is None else P
        if x is None or x.shape != (self.n, 1):
            raise ValueError(f"Initial state vector has incorrect dimensions. Expected {self.n}x1, got {format_shape(x)}.")
        if P is None or P.shape != (self.n, self.n):
            raise ValueError(f"Initial state covariance has incorrect dimensions. Expected {self.n}x{self.n}, got {format_shape(P)}.")

        uk = uk if uk is not None else [self.model.u0] * n_samples
        wk = self.process_noise(duration, **kwargs) if process_noise else [self.model.w0] * n_samples
        vk = self.measurement_noise(duration, **kwargs) if measurement_noise else [self.model.v0] * n_samples
        t_start = self._t                              # resume time from the previous call
        for i in range(n_samples):
            xk[i] = x
            t_i = t_start + i / self.fs                 # propagate time (for time-varying f/h)
            if covariance:
                Pk[i] = P
                x, P, _ = self.engine._predict(self.model, x, P, u=uk[i], w=wk[i], t=t_i, **kwargs)
            else:
                # Fast truth path: advance the mean with the model's own dynamics
                # (one f-eval) and add process noise additively — no per-step Jacobian
                # or covariance recursion. Pk is left at its pre-allocated zeros.
                u_for = np.asarray(uk[i]).reshape(-1, 1) if self.model.l > 0 else None
                x = self.model.f(x, u=u_for, t=t_i) + wk[i]

        # advance the cursor so the next measurement() resumes at t+1 with the latest state
        self._t = t_start + n_samples / self.fs
        self._x, self._P = x, P

        if inprocess:
            xk = self.model.inprocess(xk, **kwargs)
        # Squeeze trailing unit dim: (N, n, 1) → (N, n) so record.xk indexing works correctly
        xk = KalmanRecord._normalize_out(xk)

        zk = KalmanRecord._normalize_out([self.model.h(xx.reshape(-1, 1), v, t_start + k / self.fs) for xx, v, k in zip(xk, vk, range(n_samples))], indexing=indexing, dtype=self.model.dtype)
        if postprocess:
            zk = self.model.postprocess(zk, **kwargs)

        times = t_start + np.arange(n_samples) / self.fs
        return KalmanRecord(times, zk=zk, xk=xk, Pk=Pk, flavor=KalmanStateFlavor.SAMPLED, model=self.model)

    def process_noise(self, duration: float, **_):
        """
        Generate process noise for the given duration.
        If Q is a block diagonal covariance matrix, ensure proper sampling.
        """

        n_samples = int(duration * self.fs)
        if self.model.Q() is None: process_noise = np.zeros((n_samples, self.n))
        else: # Check if Q is block diagonal
            if np.allclose(self.model.Q(), np.diag(np.diag(self.model.Q()))):  # Diagonal matrix
                process_noise = np.random.normal(0, np.sqrt(np.diag(self.model.Q())), size=(n_samples, self.n))
            else:
                process_noise = np.random.multivariate_normal(np.zeros(self.n), self.model.Q(), size=n_samples)

        process_noise = process_noise.reshape((n_samples, self.n, 1))
        return process_noise

    def measurement_noise(self, duration: float, **_):
        """
        Generate measurement noise for the given duration.
        If R is a block diagonal covariance matrix, ensure proper sampling.
        """

        n_samples = int(duration * self.fs)
        if self.model.R() is None: measurement_noise = np.zeros((n_samples, self.m))
        else: # Check if R is block diagonal
            if np.allclose(self.model.R(), np.diag(np.diag(self.model.R()))):  # Diagonal matrix
                measurement_noise = np.random.normal(0, np.sqrt(np.diag(self.model.R())), size=(n_samples, self.m))
            else:
                measurement_noise = np.random.multivariate_normal(np.zeros(self.m), self.model.R(), size=n_samples)

        measurement_noise = measurement_noise.reshape((n_samples, self.m, 1))
        return measurement_noise

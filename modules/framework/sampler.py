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

        Each step routes through ``engine._predict`` (so ``Pk`` carries the
        analytic prior covariance band, ``F P Fᵀ + Q``) and then injects an
        explicit process-noise draw into the mean, so the returned ``xk`` is a
        genuine stochastic realisation of the system (it actually drifts under
        ``Q``) while ``Pk`` still reflects the model's analytic uncertainty growth.
        '''

        # seed=: the same run every time. It was accepted and ignored before.
        seed = kwargs.pop("seed", None)
        if seed is not None:
            np.random.seed(int(seed))

        inprocess = kwargs.pop("inprocess", True)
        postprocess = kwargs.pop("postprocess", True)
        indexing = kwargs.pop("indexing", "xy")

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
        # Step first, then record: sample k is the state *after* the k-th step
        # (driven by u[k] and w[k]) and z[k] is a measurement of that state.
        # This is the order the filter assumes (predict with u[k], then update
        # with z[k], starting from x0) and the order Kalman::Sampler uses.
        # Recording before the step made every measurement, and every control,
        # one step late for the filter.
        chain = self._native_chain(n_samples, x, P, t_start, uk, wk, kwargs)
        if chain is not None:                           # the loop ran in C++
            xk[:, :, 0], Pk[:] = chain
            x, P = xk[-1].copy(), Pk[-1].copy()
        else:
            for i in range(n_samples):
                t_i = t_start + i / self.fs             # propagate time (for time-varying f/h)
                u_i = np.asarray(uk[i]).reshape(-1, 1) if self.model.l > 0 else None

                x, P, _ = self.engine._predict(self.model, x, P, u=u_i, t=t_i, **kwargs)
                x = x + wk[i]
                xk[i] = x
                Pk[i] = P

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

    def _native_chain(self, n_samples, x, P, t_start, uk, wk, kwargs):
        """The state chain computed by the C++ loop (kalman_propagate), with the
        noise drawn here by numpy so a seeded run does not change. ``None`` when
        the Python loop must run: another engine than the extended one (or the
        linear one on a model of constant matrices), extra arguments for the
        prediction, a model that is not double precision, or no library."""
        from .models import native
        from .engines.extended import ExtendedKalmanEngine
        if kwargs or not native.native_loop_enabled():
            return None
        model = self.model
        matrices = self._matrices_handle()
        if matrices is None and type(self.engine) is not ExtendedKalmanEngine:
            return None
        if n_samples <= 0 or not native.callback_eligible(model):
            return None
        from .native import core
        n, l = int(model.n), int(model.l)
        try:
            draws = np.asarray(wk, dtype=np.float64).reshape(n_samples, n)
            controls = np.asarray(uk, dtype=np.float64).reshape(n_samples, l) if l > 0 else None
        except (TypeError, ValueError):
            return None
        if matrices is not None:
            # A model of constant matrices on the linear engine: the chain runs
            # on those matrices in C++ (a Kalman::Models::Tabulated holding them).
            return core.propagate(matrices, core.create_engine("linear"), n, l, n_samples, x, P,
                                  t_start, self.fs, uk=controls, wk=draws)
        twin = getattr(model, "native_handle", None)
        handle, keep = (twin() if twin is not None else None), None
        if handle is None:
            handle, keep = native.callback_handle(model, "extended", live_noise=True)
        try:
            return core.propagate(handle, core.create_engine("extended"), n, l, n_samples, x, P,
                                  t_start, self.fs, uk=controls, wk=draws)
        except RuntimeError:
            if keep and keep[-1]:
                raise keep[-1][0]                       # the Python exception, not its C++ shadow
            raise

    def _matrices_handle(self):
        """A C++ model of ``self.model``'s matrices when the linear engine samples a
        Python model of constant matrices (``structure`` MATRICES), else ``None``."""
        from .engines.linear import LinearKalmanEngine
        from .constants import KalmanStructureKind
        from .models.native import NativeModel
        model = self.model
        if type(self.engine) is not LinearKalmanEngine or isinstance(model, NativeModel):
            return None
        if getattr(model, "structure", None) is not KalmanStructureKind.MATRICES:
            return None
        from ..native import _stand_in
        try:
            return _stand_in(model, KalmanStructureKind.MATRICES, False)
        except (TypeError, ValueError, AttributeError):
            return None                                 # matrices the C++ model does not take

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

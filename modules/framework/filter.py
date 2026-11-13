# kalman/framework/filter.py
from __future__ import annotations
import numpy as np

from .io.record import KalmanRecord
from .io.collection import KalmanRecordCollection
from .constants import KalmanStateFlavor
from ..helpers import format_shape
from .abstract.metadata import KalmanFilterMetadata
from .abstract.mixins import KalmanFilterMixin


class KalmanFilter(KalmanFilterMetadata, KalmanFilterMixin):
    """
    The Kalman **filter** — workflow (WHEN). Composes a ``model`` (WHAT) and an
    ``engine`` (HOW), owns the run state (``x/P`` + prior/posterior/transition/smooth
    logs) and drives ``forward`` / ``backward``. It implements no covariance math —
    that lives in the engine.

    Engine resolution is declarative: if no engine is passed, one is built from
    ``self.KIND`` (set by public leaves like :class:`LinearKalmanFilter`) or, failing
    that, the model's ``DEFAULT_ENGINE``; the chosen engine's ``KIND`` is validated
    against ``model.ALLOWED_ENGINES``.
    """

    NAME = "Kalman Filter"
    DESCRIPTION = "Forward/backward workflow over a composed model + engine."
    VERSION = "1.0"
    KIND = None   # public leaves (LinearKalmanFilter, ...) set this

    def __init__(self, model=None, engine=None, *, epsilon: float = 1e-5,
                 n=None, m=None, F=None, H=None, Q=None, R=None,
                 fs=1.0, x0=None, P0=None, name=None, **model_kwargs):
        from .engines import resolve_engine

        # Matrix / dimension construction path: when no model is supplied, build a
        # generic linear KalmanModel from the state dimension n, measurement dimension
        # m, and the F/H/Q/R coefficients — so a filter can be created and tuned purely
        # by its matrices, with no domain-specific Model subclass required.
        if model is None:
            model = self._linear_model(n=n, m=m, F=F, H=H, Q=Q, R=R,
                                       fs=fs, x0=x0, P0=P0,
                                       name=name or self.NAME, **model_kwargs)

        self._model = model
        self._engine = engine if engine is not None else resolve_engine(self.KIND or model.DEFAULT_ENGINE)

        allowed = getattr(model, "ALLOWED_ENGINES", None)
        if allowed is not None and self._engine.KIND not in allowed:
            raise TypeError(
                f"{type(model).__name__} disallows engine kind {self._engine.KIND} "
                f"(allowed: {sorted(k.name for k in allowed)})."
            )

        self.epsilon = float(epsilon)
        self.reset()

    @staticmethod
    def _linear_model(*, n=None, m=None, F=None, H=None, Q=None, R=None,
                      fs=1.0, x0=None, P0=None, name="filter", **kw):
        """Build a generic linear :class:`KalmanModel` from dimensions + coefficients.

        ``n`` = state dimension, ``m`` = measurement dimension. Any matrix left as
        ``None`` gets an editable placeholder: ``F`` = identity, ``H`` = observe the
        first ``m`` states, ``Q`` = 1e-3·identity, ``R`` = identity. ``n``/``m`` are
        inferred from ``F``/``H`` when those are given.
        """
        from .model import KalmanModel
        F = None if F is None else np.asarray(F, dtype=float)
        H = None if H is None else np.asarray(H, dtype=float)
        if n is None and F is not None: n = F.shape[0]
        if m is None and H is not None: m = H.shape[0]
        if n is None:
            raise ValueError("Provide a state dimension n (or an F matrix) to size the filter.")
        if m is None: m = n
        if F is None: F = np.eye(n)
        if H is None:
            H = np.zeros((m, n))
            for i in range(min(m, n)): H[i, i] = 1.0
        Q = np.eye(n) * 1e-3 if Q is None else np.asarray(Q, dtype=float)
        R = np.eye(m) * 1.0   if R is None else np.asarray(R, dtype=float)
        return KalmanModel(name, F=F, H=H, Q=Q, R=R, fs=fs, x0=x0, P0=P0, **kw)

    # --- matrix-coefficient accessors: read or edit the underlying linear model ---
    # (so you can do ``kf.F = [[1, dt], [0, 1]]`` after construction).
    @property
    def F(self): return self._model.F()
    @F.setter
    def F(self, M): self._model._F = np.asarray(M, dtype=self._model.dtype)
    @property
    def H(self): return self._model.H()
    @H.setter
    def H(self, M): self._model._H = np.asarray(M, dtype=self._model.dtype)
    @property
    def Q(self): return self._model.Q()
    @Q.setter
    def Q(self, M): self._model._Q = np.asarray(M, dtype=self._model.dtype)
    @property
    def R(self): return self._model.R()
    @R.setter
    def R(self, M): self._model._R = np.asarray(M, dtype=self._model.dtype)

    # initial guess — settable after construction too (``kf.x0 = [...]``, ``kf.P0 = [...]``);
    # routes through the model's initial_state() so each is normalised/validated and the
    # other is preserved. Read back through the same model attributes.
    @property
    def x0(self): return self._model.x0
    @x0.setter
    def x0(self, v): self._model.initial_state(x0=v)
    @property
    def P0(self): return self._model.P0
    @P0.setter
    def P0(self, M): self._model.initial_state(P0=M)

    # ------------------------------------------------------------------
    # Forward filtering
    # ------------------------------------------------------------------
    def forward(self, gk, uk=None, *, record: bool = True, **kwargs):

        # A whole-pass engine (e.g. the combinatorial track finder) is handed the data
        # and returns finished results; there is no per-step loop to run, and none of
        # the time-series normalisation below applies. See KalmanEngine.OWNS_PASS.
        if self.engine.OWNS_PASS:
            self.pass_result = self.engine.forward(self.model, gk, **kwargs)
            self.pass_records = self._pass_records(KalmanStateFlavor.POSTERIOR)
            return self.pass_records if record else self

        indexing    = kwargs.pop("indexing", "xy")
        preprocess  = kwargs.pop("preprocess", True)
        inprocess   = kwargs.pop("inprocess", True)
        postprocess = kwargs.pop("postprocess", True)

        reset  = kwargs.pop("reset", True)
        buffer = kwargs.pop("buffer", 60)  # seconds

        # An explicit reset wins: start a clean run from the model's x0/P0. Only when
        # *appending* (reset=False, streaming) do we roll the window forward once it
        # grows past `buffer` seconds — that path calls reset_to_latest(), which
        # rebases the model's x0/P0 onto the latest estimate, so it must NOT fire on a
        # plain reset=True re-run (doing so silently corrupts x0/P0 across calls).
        if reset:
            self.reset()
        elif len(self.posterior) > buffer * self.sample_rate():
            self.reset_to_latest()

        # --- detect callable controller (closed-loop) before normalisation ---
        uk_fn = uk if callable(uk) else None

        # --- normalise measurements / control ---
        gk = gk.zk if isinstance(gk, KalmanRecord) else gk
        gk = KalmanRecord._normalize_in(gk, indexing=indexing, dtype=self.model.dtype)
        assert gk.shape[1] >= 1, f"gk must have at least 1 column, got {format_shape(gk)}."

        if uk_fn is None:
            uk = uk.zk if isinstance(uk, KalmanRecord) else uk
            if uk is None or len(uk) < 1:
                uk = np.zeros((len(gk), self.l))
            uk = KalmanRecord._normalize_in(uk, indexing=indexing, dtype=self.model.dtype)
            if self.l > 0:
                assert uk is not None and uk.shape[1] == self.l, \
                    f"Control input uk has wrong number of columns. Expected ({self.l},1), got {format_shape(uk)}."

        # --- preprocess (may alter shapes) ---
        if uk_fn is not None:
            zk, _ = self.model.preprocess(gk, None) if preprocess else (gk, None)
        else:
            zk, uk = self.model.preprocess(gk, uk) if preprocess else (gk, uk)
        assert zk.shape[1] == self.m, \
            f"Preprocessed zk has incorrect number of columns. Expected ({self.m},1), got {format_shape(zk)}."
        if uk_fn is None and self.l > 0:
            assert uk is not None and uk.shape[1] == self.l, \
                f"Preprocessed uk has incorrect number of columns. Expected ({self.l},1), got {format_shape(uk)}."

        # --- main loop (engine does predict + masked update); a constant-matrix
        # linear model with no control callback can run the whole pass fused in C ---
        twin = self._native_twin(uk_fn)
        if twin is not None:
            self._forward_twin(twin, zk, uk, uk_fn)
        elif self._native_forward_eligible(uk_fn):
            self._forward_native(zk, uk)
        elif self._callback_eligible(uk_fn):
            self._forward_callback(zk, uk, uk_fn)
        else:
            t = float(self.model.t0) if self.model.t0 is not None else 0.0
            dt = 1.0 / self.sample_rate() if self.sample_rate() else 1.0
            nu = np.zeros((self.model.m, 1))
            u = np.zeros((max(self.l, 1), 1))

            for idx, z in enumerate(zk):
                if uk_fn is not None:
                    u_raw = uk_fn(self.x, self.P, t, nu=nu)
                    u = np.asarray(u_raw, dtype=float).reshape(-1, 1)
                else:
                    u = uk[idx]

                self.measurement.append(z)
                self.control.append(u.ravel() if np.ndim(u) > 1 else u)

                self.x, self.P, F = self.engine._predict(self.model, self.x, self.P, u=u, t=t)
                self.prior.append((self.x, self.P))
                self.transition.append(F)

                h_pred = self.model.h(self.x, t=t)
                nu = np.asarray(z, dtype=float).reshape(-1, 1) - np.asarray(h_pred, dtype=float).reshape(-1, 1)
                self.innovation.append(nu)

                self.x, self.P = self.engine._update(self.model, self.x, self.P, z, t=t)
                self.posterior.append((self.x, self.P))

                t += dt

        return self.get_record(indexing=indexing, inprocess=inprocess, postprocess=postprocess) if record else self

    # ------------------------------------------------------------------
    # A model with a C++ twin (libKalmanApps): the whole pass — the model's
    # f/h, its Jacobians and the EKF steps — runs in Kalman::Filter, in one
    # call. Only the plain extended engine qualifies: a subclass or another
    # engine kind keeps the Python loop.
    # ------------------------------------------------------------------
    def _native_twin(self, uk_fn):
        # In closed loop the controller may retune the Python model as it runs
        # (adaptive Q and R); a twin would not see it, so the C++ loop calls the
        # Python model instead.
        if uk_fn is not None:
            return None
        handle_of = getattr(self._model, "native_handle", None)
        if handle_of is None:
            return None
        from .engines.extended import ExtendedKalmanEngine
        if type(self.engine) is not ExtendedKalmanEngine:
            return None
        return handle_of()

    def _closed_loop_eligible(self) -> bool:
        """A controller function can drive the C++ loop when the model takes a control."""
        from .models import native
        return native.native_loop_enabled() and self._model.l > 0

    def _forward_twin(self, handle, zk, uk, uk_fn=None):
        from .native import core
        from .native import as_contiguous

        model = self._model
        n, m = model.n, model.m
        lib = core.lib

        x0 = as_contiguous(np.asarray(self.x, dtype=float).reshape(-1))
        P0 = as_contiguous(np.asarray(self.P, dtype=float).reshape(n, n))
        core.check(lib.kalman_model_set_x0(handle, core.cdptr(x0)), "set_x0")
        core.check(lib.kalman_model_set_P0(handle, core.cdptr(P0)), "set_P0")
        core.check(lib.kalman_model_set_t0(handle, float(model.t0) if model.t0 is not None else 0.0),
                   "set_t0")

        if uk_fn is not None:
            out = core.run_filter(handle, core.create_engine("extended"), n, m, model.l, zk, control=uk_fn)
            uk = out["controls"]
        elif model.l > 0:
            # One command per measurement, as the Python loop reads them (a longer
            # program is cut, like the loop that stops at the last measurement).
            out = core.run_filter(handle, core.create_engine("extended"), n, m, model.l, zk, uk[:len(zk)])
        else:
            out = core.run_filter(handle, core.create_engine("extended"), n, m, 0, zk)
        self._store_pass(out, zk, uk)

    # ------------------------------------------------------------------
    # A model written in Python, with no C++ twin: the loop and the EKF
    # arithmetic still run in Kalman::Filter, which calls back into the
    # model's own f/h/F/H. Same results as the Python loop below, which
    # stays for the cases this path does not cover (a controller callback,
    # another engine kind, state-dependent noise).
    # ------------------------------------------------------------------
    def _callback_eligible(self, uk_fn) -> bool:
        if uk_fn is not None and not self._closed_loop_eligible():
            return False
        from .models import native
        if not native.native_loop_enabled():
            return False
        from .engines.extended import ExtendedKalmanEngine
        from .engines.unscented import UnscentedKalmanEngine
        from .engines.particle import ParticleEngine
        from .engines.ensemble import EnsembleKalmanEngine
        if type(self.engine) is EnsembleKalmanEngine:
            # Per-member f and h only. A field model's whole-ensemble hooks
            # (a batched solver, localization, its own initial ensemble) have
            # no C++ counterpart: with any of them the Python engine runs.
            hooks = ("forecast", "observe", "localization", "initial_ensemble")
            if any(callable(getattr(self._model, hook, None)) for hook in hooks):
                return False
            if self.engine._rho_xy is not None or self.engine._rho_yy is not None:
                return False
        elif type(self.engine) not in (ExtendedKalmanEngine, UnscentedKalmanEngine, ParticleEngine):
            # A linear model in closed loop (a controller function) runs on the
            # C++ extended engine, which on a linear model with its own constant
            # F and H is the same arithmetic. A linear model whose matrices or
            # noise are functions stays on the Python loop: such a function may
            # keep a counter (the calorimeter steps through its layers that
            # way), and only the Python loop calls it once per step in order.
            from .engines.linear import LinearKalmanEngine
            from .model import KalmanModel
            model = self._model
            if type(self.engine) is not LinearKalmanEngine or uk_fn is None:
                return False
            if any(callable(getattr(model, attr, None)) for attr in ("_F", "_H", "_B", "_L", "_Q", "_R")):
                return False
            if any(getattr(type(model), name) is not getattr(KalmanModel, name) for name in ("F", "H", "Q", "R", "f", "h")):
                return False
        return native.callback_eligible(self._model)

    def _forward_callback(self, zk, uk, uk_fn=None):
        from .models.native import run_callback_filter
        from .engines.unscented import UnscentedKalmanEngine
        model, engine = self._model, self.engine
        from .engines.particle import ParticleEngine
        if type(engine) is UnscentedKalmanEngine:
            kind = {"engine": "unscented", "alpha": engine.alpha, "beta": engine.beta, "kappa": engine.kappa}
        elif type(engine) is ParticleEngine:
            # The C++ engine has its own random stream; seed it from numpy's, so
            # np.random.seed() still makes a run repeatable.
            kind = {"engine": "particle", "n_particles": engine.n_particles,
                    "threshold": engine.resample_threshold, "seed": int(np.random.randint(0, 2**31 - 1))}
        elif type(engine).__name__ == "EnsembleKalmanEngine":
            kind = {"engine": "ensemble", "n_members": engine.n_members, "inflation": engine.inflation,
                    "seed": int(engine.rng.integers(0, 2**31 - 1))}
        else:
            kind = {"engine": "extended"}
        if uk_fn is not None:
            out = run_callback_filter(model, zk, None, x0=self.x, P0=self.P, control=uk_fn, **kind)
            uk = out["controls"]
        else:
            out = run_callback_filter(model, zk, uk if model.l > 0 else None, x0=self.x, P0=self.P, **kind)
        self._store_pass(out, zk, uk)

    def _store_pass(self, out, zk, uk):
        """Record a whole pass returned by the C++ filter."""
        from .native import as_contiguous
        n, m = self._model.n, self._model.m
        N = len(zk)
        self.measurement.extend(as_contiguous(zk))
        self.control.extend(as_contiguous(uk))
        self.prior.extend((out["x_prior"][k].reshape(n, 1), out["P_prior"][k]) for k in range(N))
        self.posterior.extend((out["x_post"][k].reshape(n, 1), out["P_post"][k]) for k in range(N))
        self.transition.extend(out["transitions"][k] for k in range(N))
        self.innovation.extend(out["innovations"][k].reshape(m, 1) for k in range(N))

        self.x = out["x_post"][-1].reshape(n, 1)
        self.P = out["P_post"][-1]

    # ------------------------------------------------------------------
    # C-accelerated fast path (constant-matrix linear model, no control
    # callback): fuses the whole N-step predict+update loop into one
    # ``kf_forward_linear`` call. Falls back transparently to the Python loop
    # above whenever the model/engine don't fit this shape.
    # ------------------------------------------------------------------
    def _native_forward_eligible(self, uk_fn) -> bool:
        if uk_fn is not None:
            return False
        from .native import HAS_C_ACCEL
        if not HAS_C_ACCEL:
            return False
        from .engines.linear import LinearKalmanEngine
        if not isinstance(self.engine, LinearKalmanEngine):
            return False
        model = self._model
        # A C++ model whose matrices change with time (one read from a file with
        # tables) holds no Python callable to give it away: it says so itself.
        from .constants import KalmanStructureKind
        if getattr(model, "structure", None) is KalmanStructureKind.TIME_VARYING:
            return False
        return not any(callable(getattr(model, attr)) for attr in ("_F", "_H", "_B", "_L", "_Q", "_R"))

    def _forward_native(self, zk, uk):
        from .native import lib, ffi, as_contiguous, dptr

        model = self._model
        n, m, l = model.n, model.m, model.l
        N = len(zk)

        F, H, Q, R = (as_contiguous(M()) for M in (model.F, model.H, model.Q, model.R))
        B, L = model.B(), model.L()

        zk_c = as_contiguous(zk)
        uk_c = as_contiguous(uk)
        x0 = as_contiguous(np.asarray(self.x, dtype=float).reshape(-1))
        P0 = as_contiguous(self.P)

        x_prior = np.empty((N, n), dtype=np.float64)
        P_prior = np.empty((N, n, n), dtype=np.float64)
        x_post  = np.empty((N, n), dtype=np.float64)
        P_post  = np.empty((N, n, n), dtype=np.float64)
        innovations = np.empty((N, m), dtype=np.float64)

        lib.kf_forward_linear(
            n, m, l, N,
            dptr(F), dptr(H), dptr(Q), dptr(R),
            dptr(as_contiguous(B)) if B is not None else ffi.NULL,
            dptr(as_contiguous(L)) if L is not None else ffi.NULL,
            dptr(zk_c), dptr(uk_c),
            dptr(x0), dptr(P0),
            dptr(x_prior), dptr(P_prior),
            dptr(x_post), dptr(P_post),
            dptr(innovations),
        )

        self.measurement.extend(zk_c)
        self.control.extend(uk_c)
        self.prior.extend((x_prior[k].reshape(n, 1), P_prior[k]) for k in range(N))
        self.posterior.extend((x_post[k].reshape(n, 1), P_post[k]) for k in range(N))
        self.transition.extend([F] * N)
        self.innovation.extend(innovations[k].reshape(m, 1) for k in range(N))

        self.x = x_post[-1].reshape(n, 1)
        self.P = P_post[-1]

    # ------------------------------------------------------------------
    # Backward smoothing (RTS) — drives engine._smooth
    # ------------------------------------------------------------------
    def backward(self, **kwargs):
        if self.engine.OWNS_PASS:
            if self.pass_result is None:
                raise RuntimeError(
                    f"No results to smooth; run {type(self).__name__}.forward() first.")
            self.pass_result = self.engine.backward(self.model, self.pass_result, **kwargs)
            self.pass_records = self._pass_records(KalmanStateFlavor.SMOOTHED)
            return self.pass_records

        N = len(self.posterior)
        assert N > 0, "No posterior estimates available for smoothing. (run forward() first?)"
        assert len(self.prior) == N, "Prior estimates must match the number of posterior estimates."

        self.smooth = np.zeros((N, 2), dtype=object)
        self.smooth[-1] = self.posterior[-1]

        if N > 1 and self._native_backward_eligible():
            self._backward_native(N)
        else:
            # transition[k + 1] is the one that took posterior[k] to prior[k + 1]:
            # the F of the RTS gain P[k] F^T P_prior[k + 1]^-1.
            for k in reversed(range(N - 1)):
                self.smooth[k] = self.engine._smooth(
                    self.model, self.posterior[k], self.prior[k + 1], self.smooth[k + 1], self.transition[k + 1]
                )

        kwargs.pop("smooth", None)
        return self.get_record(smooth=True, **kwargs)

    # ------------------------------------------------------------------
    # C-accelerated RTS smoother: valid only when the transition matrix is
    # constant across the run (so a single F applies to every step), which
    # is the one assumption ``kf_backward_rts`` makes. That is checked on the
    # transitions the forward pass recorded, not on how the model declares F:
    # a nonlinear model has no stored F at all, and its Jacobian changes at
    # every step.
    # ------------------------------------------------------------------
    def _native_backward_eligible(self) -> bool:
        from .native import HAS_C_ACCEL
        if not HAS_C_ACCEL or not self.transition:
            return False
        first = self.transition[0]
        return all(F is first for F in self.transition)

    def _backward_native(self, N):
        from .native import lib, as_contiguous, dptr

        n = self.model.n
        F = as_contiguous(self.transition[0])
        x_post  = as_contiguous(np.array([x.reshape(-1) for x, _ in self.posterior]))
        P_post  = as_contiguous(np.array([P for _, P in self.posterior]))
        x_prior = as_contiguous(np.array([x.reshape(-1) for x, _ in self.prior]))
        P_prior = as_contiguous(np.array([P for _, P in self.prior]))

        x_smooth = np.empty((N, n), dtype=np.float64)
        P_smooth = np.empty((N, n, n), dtype=np.float64)

        lib.kf_backward_rts(
            n, N,
            dptr(x_post), dptr(P_post),
            dptr(x_prior), dptr(P_prior),
            dptr(F),
            dptr(x_smooth), dptr(P_smooth),
        )

        for k in range(N):
            self.smooth[k] = (x_smooth[k].reshape(n, 1), P_smooth[k])

    # ------------------------------------------------------------------
    # Whole-pass engines: turn their results into records
    # ------------------------------------------------------------------
    def _pass_records(self, flavor) -> KalmanRecordCollection:
        """Wrap an ``OWNS_PASS`` engine's output as a collection of records.

        The only thing the filter needs to know about those results is that each one
        can describe itself — hence the ``to_record(model, flavor)`` contract, which
        keeps the result type (a track, a cluster, …) private to its engine.
        """
        return KalmanRecordCollection(r.to_record(self.model, flavor) for r in self.pass_result)

    # ------------------------------------------------------------------
    # Convenience aliases (mirror the model-level API: filter()/smooth())
    # ------------------------------------------------------------------
    def filter(self, z, u=None, **kwargs):
        """Alias for ``forward`` (``smooth=True`` runs forward then backward)."""
        if u is None:
            u = kwargs.pop("uk", None)
        if kwargs.pop("smooth", False):
            self.forward(z, u, record=False, **kwargs)
            return self.backward(**kwargs)
        return self.forward(z, u, **kwargs)

    def smooth(self, **kwargs):
        """Alias for ``backward`` (RTS smoother over the stored forward pass)."""
        return self.backward(**kwargs)

    # ------------------------------------------------------------------
    # Sampling convenience (uses this filter's engine for the dynamics)
    # ------------------------------------------------------------------
    def sample(self, duration, **kwargs):
        # A whole-pass engine consumes a whole event, not a time series of single
        # measurements, so there is nothing for the time-series sampler to produce.
        if self.engine.OWNS_PASS:
            raise NotImplementedError(
                f"{type(self.engine).__name__} consumes a whole pass, not a time series; "
                f"generate its input from the model instead "
                f"(e.g. {type(self.model).__name__}.generate_event()).")
        from .sampler import KalmanSampler
        return KalmanSampler(self.model, engine=self.engine).measurement(duration, **kwargs)

    # ==================================================================
    # Consistency diagnostics — delegated to KalmanDiagnostics
    # (diagnostics.py), all computed *post-hoc* from the stored forward
    # pass (prior/posterior/measurement) + the model's H/R/h, so no engine
    # changes are needed. Run forward()/filter() first.
    # ==================================================================
    def _diagnostics(self):
        from .diagnostics import KalmanDiagnostics
        return KalmanDiagnostics(self)

    def innovations(self):
        """Per-step innovation ν and its covariance S — see
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.innovations`."""
        return self._diagnostics().innovations()

    def gains(self):
        """Per-step Kalman gain K(t) — see
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.gains`."""
        return self._diagnostics().gains()

    def nis(self):
        """Normalized Innovation Squared (≈ χ²(m), no truth needed) — see
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.nis`."""
        return self._diagnostics().nis()

    def nees(self, truth):
        """Normalized Estimation Error Squared (≈ χ²(n), needs truth) — see
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.nees`."""
        return self._diagnostics().nees(truth)

    def diagnostics(self, sample=None, **kwargs):
        """Draw the filter-consistency dashboard (innovation, gain, NIS, and NEES if a
        ground-truth ``sample`` is given). Delegates to
        :meth:`~kalman.framework.diagnostics.KalmanDiagnostics.plot`; returns the figure."""
        return self._diagnostics().plot(sample=sample, **kwargs)

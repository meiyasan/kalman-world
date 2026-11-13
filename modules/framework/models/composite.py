import numpy as np
from scipy.linalg import block_diag
from typing import List, Optional
from ..abstract.metadata import KalmanModelMetadata
from ..constants import KalmanObservableReadout, KalmanEngineKind, ALL_ENGINES
from ..model import KalmanModel


class CompositeKalmanModel(KalmanModel):

    NAME = "Composite Kalman Model"
    DESCRIPTION = "Composite Kalman model combining multiple submodels (block-stacked F/H/Q/R)."
    VERSION = "1.0"

    # DEFAULT_ENGINE: the extended engine (general), or the linear one for an
    # instance made of matrices -- see _EngineOfParts below.
    ALLOWED_ENGINES = ALL_ENGINES

    """
    Composite Kalman model combining multiple submodels.

    Principles
    -----------
    - F, Q are always block-diagonal (independent state evolution)
    - Observation composition governed by KalmanObservableReadout:
        * INDEPENDENT → z = [h1(x1); h2(x2); ...]
        * FUSED       → z = sum_i h_i(x_i)
    - No custom overrides; complex coupling should be encoded
      in the submodels themselves.
    """

    def __init__(
        self,
        *models: List[KalmanModelMetadata],
        obs_mode: KalmanObservableReadout = KalmanObservableReadout.INDEPENDENT,
        **kwargs,
    ):
        if not models:
            raise ValueError("At least one submodel must be provided.")
        self.models = list(models)

        # Consistency check
        self.fs = models[0].fs
        if any(m.fs != self.fs for m in models):
            raise ValueError("All submodels must share the same sampling rate.")

        # Global dimensions
        self.n = sum(m.n for m in models)
        # FUSED → all submodels share ONE measurement channel (summed); INDEPENDENT → stacked
        self.m = (models[0].m if obs_mode == KalmanObservableReadout.FUSED
                  else sum(m.m for m in models))
        self.l = sum(m.l for m in models)

        # Aggregate state and covariance
        x0 = np.vstack([m.x0 for m in models])
        P0 = block_diag(*[m.P0 for m in models])

        # === Bind unified operators ===
        self.f = self._bind_f()
        self.F = self._bind_F()

        self.h = self._bind_h(obs_mode)
        self.H = self._bind_H(obs_mode)

        # Structural matrices (block-diagonal)
        self.B = self._bind_matrix("B")
        self.L = self._bind_matrix("L")
        # FUSED: one shared measurement channel → additive noise (no block-diagonal M map)
        self.M = ((lambda t=None: None) if obs_mode == KalmanObservableReadout.FUSED
                  else self._bind_matrix("M"))

        # Covariances: Q always block-diagonal (independent state evolution).
        self.Q = lambda t=None: block_diag(*[(m.Q(t) if callable(m.Q) else m.Q) for m in self.models])
        if obs_mode == KalmanObservableReadout.FUSED:
            # Single shared measurement channel → one block's R (submodels share the channel).
            self.R = lambda t=None: (self.models[0].R(t) if callable(self.models[0].R) else self.models[0].R)
        else:
            self.R = lambda t=None: block_diag(*[(m.R(t) if callable(m.R) else m.R) for m in self.models if m.m > 0])

        # Processing hooks
        self.preprocess  = self._bind_preprocess()
        self.inprocess   = self._bind_inprocess()
        self.postprocess = self._bind_postprocess()

        super().__init__("Composite", x0=x0, P0=P0, fs=self.fs, **kwargs)
        self._sensor.clear()
        for m in self.models:
            self._sensor.extend(m._sensor)

        # What was bound here: a part of the contract replaced afterwards (an
        # h of one's own, say) makes the composite code, whatever its parts are.
        self._bound = {name: self.__dict__[name] for name in ("f", "h", "F", "H", "B", "L", "Q", "R")}
        # A stacked matrix that changes with time is a function of time, as on
        # any model built from matrices: the filter and the exporter tabulate
        # what is stored as a function (``_F``, ``_H``, ``_Q``, ``_R``).
        for stored, bound in (("_F", "F"), ("_H", "H"), ("_Q", "Q"), ("_R", "R")):
            if any(callable(getattr(m, stored, None)) for m in self.models):
                setattr(self, stored, self._bound[bound])

    # ==========================================================
    #  What the composite is made of, and the engine that follows
    # ==========================================================
    @property
    def structure(self):
        """The parts' structure: matrices when every part is constant matrices,
        time-varying when every part is matrices and one of them changes with
        time, code (``MAPS``) as soon as one part is code or one of the stacked
        maps was replaced after construction. Both readouts (stacked or summed)
        are linear in the parts' states, so they do not change it."""
        from ..constants import KalmanStructureKind as S
        if self.STRUCTURE is not None:
            return self.STRUCTURE
        bound = self.__dict__.get("_bound")
        if bound is None or any(self.__dict__.get(name) is not fn for name, fn in bound.items()):
            return S.MAPS
        kinds = [m.structure for m in self.models]
        if any(k is S.MAPS for k in kinds):
            return S.MAPS
        return S.TIME_VARYING if any(k is S.TIME_VARYING for k in kinds) else S.MATRICES

    class _EngineOfParts:
        """``DEFAULT_ENGINE`` of a composite. Read on the class it is the class's
        declaration (the extended engine, which serves any part); read on an
        instance made of matrices it is the linear engine, so the filter takes
        the C++ linear pass (``kf_forward_linear``, or the tables of a
        time-varying model) instead of differencing stacked matrices step by
        step. On a model of matrices the two engines are the same arithmetic."""

        def __init__(self, declared):
            self.declared = declared

        def __get__(self, obj, cls=None):
            if obj is None or "_bound" not in obj.__dict__:
                return self.declared
            from ..constants import KalmanStructureKind as S
            return KalmanEngineKind.LINEAR if obj.structure is not S.MAPS else self.declared

    DEFAULT_ENGINE = _EngineOfParts(KalmanEngineKind.EXTENDED)

    # ==========================================================
    #  Core binding methods
    # ==========================================================
    def _bind_f(self):
        """Global state transition: block-diagonal composition."""
        def split_state(x):
            parts, start = [], 0
            for m in self.models:
                parts.append(x[start:start + m.n])
                start += m.n
            return parts

        def split_input(u):
            if u is None or self.l == 0:
                return [None] * len(self.models)
            cuts = np.cumsum([m.l for m in self.models[:-1]])
            return np.array_split(u, cuts)

        def split_noise(w):
            if w is None:
                return [None] * len(self.models)
            cuts = np.cumsum([m.n for m in self.models[:-1]])
            return np.array_split(w, cuts)

        def f(x, u=None, w=None, t=None):
            parts_x = split_state(x)
            parts_u = split_input(u)
            parts_w = split_noise(w)
            out = [m.f(xi, ui, wi, t=t) for m, xi, ui, wi in zip(self.models, parts_x, parts_u, parts_w)]
            return np.vstack(out)

        return f

    def _bind_F(self):
        """Global state Jacobian: block-diagonal composition."""
        def F(x=None, u=None, t=None):
            # Each part is linearised at its own slice of the state (and of the
            # control): a part that is code reads its x, not the stacked one.
            xs, us = self._slices(x, "n"), self._slices(u, "l")
            blocks = [m.F(xi, ui, t) for m, xi, ui in zip(self.models, xs, us)]
            return block_diag(*blocks)
        return F

    def _slices(self, v, dim):
        """``v`` cut into the parts' pieces along their ``n`` (or ``l``); a list
        of ``None`` when ``v`` is ``None`` or has no such pieces."""
        if v is None or (dim == "l" and self.l == 0):
            return [None] * len(self.models)
        v = np.asarray(v)
        cuts = np.cumsum([getattr(m, dim) for m in self.models[:-1]])
        return np.split(v, cuts, axis=0)

    # ---------- Observation ----------
    def _bind_h(self, mode):
        """Global observation mapping according to observation mode."""
        if mode == KalmanObservableReadout.INDEPENDENT:
            def _h(x, v=None, t=None):
                """
                x may be a single state vector (n_total, 1)
                or a batch over time (T, n_total, 1)
                """
                if x is None or x.size == 0:
                    return np.empty((0, 0, 1))

                # Split state by dimension axis=1 (state dimension)
                axis = 1 if x.ndim == 3 else 0
                n_splits = np.cumsum([m.n for m in self.models[:-1]])
                parts_x = np.split(x, n_splits, axis=axis)

                # Split measurement noise v similarly if present
                if v is not None:
                    n_splits_v = np.cumsum([m.m for m in self.models[:-1]])
                    parts_v = np.split(v, n_splits_v, axis=axis)
                else:
                    parts_v = [None] * len(self.models)

                # Apply each submodel's observation mapping
                zs = [m.h(xi, vi, t=t) for m, xi, vi in zip(self.models, parts_x, parts_v)]
                return np.concatenate(zs, axis=axis)
            return _h

        elif mode == KalmanObservableReadout.FUSED:
            def _h(x, v=None, t=None):
                if x is None or x.size == 0:
                    return np.empty((0, 0, 1))

                axis = 1 if x.ndim == 3 else 0
                n_splits = np.cumsum([m.n for m in self.models[:-1]])
                parts_x = np.split(x, n_splits, axis=axis)

                # Combine linearly (sum, not stack). The parts share one channel,
                # so its noise v (one m-vector, R of that channel) is added once to
                # the sum: split by part it left every part after the first with
                # an empty piece, and sampling two in-band modes failed.
                z_sum = None
                for m, xi in zip(self.models, parts_x):
                    zi = m.h(xi, None, t=t)
                    z_sum = zi if z_sum is None else z_sum + zi
                if v is not None:
                    z_sum = z_sum + np.asarray(v).reshape(np.shape(z_sum))
                return z_sum
            return _h

        else:
            raise ValueError(f"Unknown observation mode: {mode}")
    def _bind_H(self, mode):
        """Global observation Jacobian."""
        if mode == KalmanObservableReadout.INDEPENDENT:
            def H(x=None, t=None):
                blocks = [m.H(xi, t) for m, xi in zip(self.models, self._slices(x, "n"))]
                return block_diag(*blocks)
            return H

        elif mode == KalmanObservableReadout.FUSED:
            def H(x=None, t=None):
                """Horizontal stack over the block-stacked state: z = Σ_i H_i x_i."""
                blocks = [m.H(xi, t) for m, xi in zip(self.models, self._slices(x, "n"))]
                return np.hstack(blocks)
            return H

        else:
            raise ValueError(f"Unsupported observation mode: {mode}")

    # ---------- Structural matrices ----------
    def _bind_matrix(self, attr):
        """Bind block-diagonal structural matrices (B, L, or M).

        A submodel returning ``None`` means "unspecified", and the right
        default differs by matrix: for ``B``/``M`` (no control input / no
        measurement-noise shaping) zero is correct — multiplying by zero
        contributes nothing, which is exactly the "not present" case. For
        ``L`` (the process-noise map, ``Q_eff = L @ Q @ L.T`` in
        ``KalmanEngine._predict``), ``None`` means "``Q`` is already in the
        right state-space units, use it unshaped" — the *identity*, not
        zero. Defaulting L to zero silently cancels every leaf's process
        noise once composed, making ``process_uncertainty`` inert for any
        composite built from leaves that don't set an explicit L (e.g. every
        ``AnalyticOscillatorModel``-based window: ``OscillatorModel``,
        ``OscillatorBankModel``, and anything built on them).
        """
        def func(t=None):
            blocks = []
            for m in self.models:
                rows = m.m if attr == "M" else m.n
                mat = getattr(m, attr)(t) if hasattr(m, attr) else None
                if mat is None:
                    mat = np.eye(rows) if attr == "L" else np.zeros((rows, rows))
                blocks.append(mat)
            return block_diag(*blocks)
        return func

    # ==========================================================
    #  Pre / In / Post processing (mode-aware)
    # ==========================================================
    def _bind_preprocess(self):
        """Bind preprocess — always independent of observation mode."""
        def preprocess(zk, uk: Optional[np.ndarray] = None, **kwargs):
            """
            Apply each submodel’s preprocess independently.
            zk: (T, M, 1)
            uk: (T, L, 1)
            """
            if zk is None:
                return np.empty((0, 0, 1)), np.empty((0, 0, 1))

            # Split along channel axis, not time
            zk_parts = np.split(zk, np.cumsum([m.m for m in self.models[:-1]]), axis=1)
            uk_parts = (
                np.split(uk, np.cumsum([m.l for m in self.models[:-1]]), axis=1)
                if uk is not None and self.l > 0
                else [np.empty((zk.shape[0], 0, 1))] * len(self.models)
            )

            out_zk, out_uk = [], []
            for m, zk_i, uk_i in zip(self.models, zk_parts, uk_parts):
                zk_p, uk_p = m.preprocess(zk_i, uk_i, **kwargs)
                out_zk.append(zk_p if zk_p is not None else np.empty((zk.shape[0], 0, 1)))
                out_uk.append(uk_p if uk_p is not None else np.empty((zk.shape[0], 0, 1)))

            return np.concatenate(out_zk, axis=1), np.concatenate(out_uk, axis=1)
        return preprocess


    def _bind_inprocess(self):
        """Bind inprocess — independent across observation channels."""
        def inprocess(zk_hat, **kwargs):
            """
            Transform predicted measurements per submodel.
            zk_hat: (T, M, 1)
            """
            if zk_hat is None or zk_hat.size == 0:
                return np.empty((0, 0, 1))

            parts = np.split(zk_hat, np.cumsum([m.m for m in self.models[:-1]]), axis=1)
            out = [m.inprocess(z_i, **kwargs) for m, z_i in zip(self.models, parts)]
            return np.concatenate(out, axis=1)
        return inprocess


    def _bind_postprocess(self):
        """Bind postprocess — independent of observation mode."""
        def postprocess(zk_hat, **kwargs):
            """
            Apply each submodel’s postprocessing to predicted measurements.
            zk_hat: (T, M, 1)
            """
            if zk_hat is None or zk_hat.size == 0:
                return np.empty((0, 0, 1))

            parts = np.split(zk_hat, np.cumsum([m.m for m in self.models[:-1]]), axis=1)
            out = [m.postprocess(z_i, **kwargs) for m, z_i in zip(self.models, parts)]
            return np.concatenate(out, axis=1)
        return postprocess

    # -----------------------------
    # Abstract methods are dynamically bound in __init__
    # -----------------------------

    def preprocess(self, zk, uk = None, **kwargs):
        pass
    def inprocess(self, zk_hat, **kwargs):
        pass
    def postprocess(self, zk_hat, **kwargs):
        pass

    def f(self, x: np.ndarray, u: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """Global state transition: block-diagonal composition."""
        pass
    def h(self, x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """Global observation mapping according to observation mode."""
        pass
    def F(self, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """Global state Jacobian: block-diagonal composition."""
        pass
    def H(self, x: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """Global observation Jacobian."""
        pass
    def B(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Global control-input Jacobian: block-diagonal composition."""
        pass
    def L(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Global process-noise Jacobian: block-diagonal composition."""
        pass
    def M(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Global measurement-noise Jacobian: block-diagonal composition."""
        pass

    def Q(self, t: Optional[float] = None) -> np.ndarray:
        """Global process noise covariance: block-diagonal composition."""
        pass
    def R(self, t: Optional[float] = None) -> np.ndarray:
        """Global measurement noise covariance: block-diagonal composition."""
        pass

    # -------------------------------------
    # ----- State / Observables enums -----
    # -------------------------------------
    @property
    def states(self):
        """
        Infer the common state across all submodels.
        - If all submodels have identical states → return that state.
        - If submodels have compatible subclasses → return the nearest common base.
        - Otherwise, raise an error.
        """
        if not self.models:
            raise AttributeError("Composite model has no submodels to infer state from.")

        states = []
        for m in self.models:
            if m.states is None:
                raise AttributeError(f"Submodel {m} does not define a 'states' property.")
            states.append(m.states)

        if all(t == states[0] for t in states):
            return states[0]

        common_bases = set.intersection(*(set(t.mro()) for t in states))
        common_bases = [b for b in common_bases if b.__name__ not in ("object", "ABC")]

        if not common_bases:
            raise TypeError(
                "Incompatible submodel states in CompositeKalmanModel: "
                + ", ".join(t.__name__ for t in states)
            )

        base_reference_mro = states[0].mro()
        for cls in base_reference_mro:
            if cls in common_bases:
                return cls

        raise TypeError(f"Failed to resolve common state for submodels: {states}")

    @property
    def observables(self):
        """
        Infer the common observable across all submodels (see ``states``).
        """
        if not self.models:
            raise AttributeError("Composite model has no submodels to infer observable from.")

        observables = []
        for m in self.models:
            if m.observables is None:
                raise AttributeError(f"Submodel {m} does not define a 'observables' property.")
            observables.append(m.observables)

        if all(t == observables[0] for t in observables):
            return observables[0]

        common_bases = set.intersection(*(set(t.mro()) for t in observables))
        common_bases = [b for b in common_bases if b.__name__ not in ("object", "ABC")]

        if not common_bases:
            raise TypeError(
                "Incompatible submodel observables in CompositeKalmanModel: "
                + ", ".join(t.__name__ for t in observables)
            )

        base_reference_mro = observables[0].mro()
        for cls in base_reference_mro:
            if cls in common_bases:
                return cls

        raise TypeError(f"Failed to resolve common observable for submodels: {observables}")

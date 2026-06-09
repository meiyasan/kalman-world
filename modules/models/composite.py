import numpy as np
from scipy.linalg import block_diag
from typing import List, Optional
from ..framework.abstract.metadata import KalmanModelMetadata
from ..framework.constants import KalmanObservableReadout
from ..framework.model import KalmanModel


class KalmanCompositeModel(KalmanModel):

    NAME = "Composite Kalman Model"
    DESCRIPTION = "Composite Kalman model combining multiple submodels."
    VERSION = "1.0"
    
    """
    Composite Kalman model combining multiple submodels.

    Principles
    -----------
    - F, Q are always block-diagonal (independent state evolution)
    - Observation composition governed by KalmanObservableReadout:
        * INDEPENDENT → z = [h1(x1); h2(x2); ...]   (stacked, split-able per submodel)
        * FUSED       → z = sum_i h_i(x_i)           (summed, cannot be split back)
    - No custom overrides; complex coupling should be encoded
      in the submodels themselves.

    Mode-awareness
    --------------
    Several operators depend on ``obs_mode`` (KalmanObservableReadout) because the
    mathematical structure of the measurement vector z changes with the mode:

    - INDEPENDENT: z is a block-concatenation of each submodel's measurement.
      Operators can split z along the channel axis (axis=1) by [m.m for m in models]
      and delegate to each submodel independently.

    - FUSED: z is the element-wise sum of all submodel measurements.
      The individual contributions are irrecoverable from z alone, so operators
      that split z per-submodel (h, H) must be implemented for each mode.

    Operators marked "mode-aware" implement this dispatch: they behave differently
    depending on whether the model was constructed with INDEPENDENT or FUSED.

    Processing hooks
    ----------------
    Only preprocess is mode-aware; _bind_h handles everything after that:
      - preprocess(zk, uk)  → INDEPENDENT: delegates per submodel (split by m.m/m.l)
                               FUSED: pass-through (fused z cannot be demixed)
      - inprocess(xk_hat)   → pass-through; _bind_h composes submodel h's correctly
      - postprocess(zk_hat) → pass-through; _bind_h already ran
    Override any hook in a subclass for domain-specific transforms
    (e.g. thermal-noise filtering in preprocess).
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
        # FUSED: all submodels contribute to the same measurement space → m = m of one submodel
        # INDEPENDENT: each submodel has its own channel → m = sum
        if obs_mode == KalmanObservableReadout.FUSED:
            self.m = models[0].m
        else:
            self.m = sum(m.m for m in models)
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
        if obs_mode == KalmanObservableReadout.FUSED:
            # FUSED: measurement space is shared (dim = models[0].m).
            # M = ∂h/∂v in the fused measurement space = I_m (identity).
            # The block-diagonal _bind_matrix("M") would give (sum_m × sum_m),
            # which would fail the shape check for the (m × m) noise covariance.
            _fused_m = models[0].m
            self.M = lambda t=None: np.eye(_fused_m)
        else:
            self.M = self._bind_matrix("M")

        # Covariances (block-diagonal for Q always; R depends on obs_mode)
        self.Q = lambda t=None: block_diag(*[(m.Q(t) if callable(m.Q) else m.Q) for m in self.models])
        if obs_mode == KalmanObservableReadout.FUSED:
            # All submodels share the same measurement channel → R is single-channel noise.
            # Use the first submodel's R (the noise on the fused measurement).
            _r0 = models[0]
            self.R = lambda t=None: (_r0.R(t) if callable(_r0.R) else _r0.R)
        else:
            self.R = lambda t=None: block_diag(*[(m.R(t) if callable(m.R) else m.R) for m in self.models if m.m > 0])

        # Processing hooks — only preprocess is mode-aware; _bind_h handles the rest
        self.preprocess  = self._bind_preprocess(obs_mode)
        self.inprocess   = self._bind_inprocess()
        self.postprocess = self._bind_postprocess()

        super().__init__("Composite", x0=x0, P0=P0, fs=self.fs, **kwargs)
        self._sensor.clear()
        for m in self.models:
            self._sensor.extend(m._sensor)

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
            blocks = [m.F(x, u, t) for m in self.models]
            return block_diag(*blocks)
        return F

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

                if v is not None:
                    n_splits_v = np.cumsum([m.m for m in self.models[:-1]])
                    parts_v = np.split(v, n_splits_v, axis=axis)
                else:
                    parts_v = [None] * len(self.models)

                # Combine linearly (sum, not stack)
                z_sum = None
                for m, xi, vi in zip(self.models, parts_x, parts_v):
                    zi = m.h(xi, vi, t=t)
                    z_sum = zi if z_sum is None else z_sum + zi
                return z_sum
            return _h

        else:
            raise ValueError(f"Unknown observation mode: {mode}")
    def _bind_H(self, mode):
        """Global observation Jacobian."""
        if mode == KalmanObservableReadout.INDEPENDENT:
            def H(x=None, t=None):
                blocks = [m.H(x, t) for m in self.models]
                return block_diag(*blocks)
            return H

        elif mode == KalmanObservableReadout.FUSED:
            def H(x=None, t=None):
                """Jacobian of fused h w.r.t. full state: horizontal concatenation [H₁|H₂|...]."""
                blocks = [m.H(x, t) for m in self.models]
                return np.concatenate(blocks, axis=1)
            return H

        else:
            raise ValueError(f"Unsupported observation mode: {mode}")

    # ---------- Structural matrices ----------
    def _bind_matrix(self, attr):
        """Bind block-diagonal structural matrices (B, L, or M)."""
        def func(t=None):
            blocks = []
            for m in self.models:
                rows = m.m if attr == "M" else m.n
                mat = getattr(m, attr)(t) if hasattr(m, attr) else None
                if mat is None:
                    mat = np.zeros((rows, rows))
                blocks.append(mat)
            return block_diag(*blocks)
        return func

    # ==========================================================
    #  Pre / In / Post processing
    # ==========================================================
    def _bind_preprocess(self, mode: KalmanObservableReadout):
        """Preprocess binding.

        INDEPENDENT: delegates per submodel by splitting zk/uk on the channel axis (m.m / m.l).
        FUSED: pass-through — the fused measurement cannot be demixed; override in subclass
               if global preprocessing (e.g. normalisation) is needed.
        """
        if mode == KalmanObservableReadout.FUSED:
            def preprocess(zk, uk: Optional[np.ndarray] = None):
                return zk, uk
            return preprocess

        def preprocess(zk, uk: Optional[np.ndarray] = None, **kwargs):
            """zk: (T, M, 1), uk: (T, L, 1)"""
            if zk is None:
                return np.empty((0, 0, 1)), np.empty((0, 0, 1))

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
        """Inprocess binding — identity.  Override in subclass for state-space transforms."""
        return lambda xk_hat, **_: xk_hat

    def _bind_postprocess(self):
        """Postprocess binding — identity.  Override in subclass for output transforms."""
        return lambda zk_hat, **_: zk_hat

    # -----------------------------
    # Abstract methods are dynamically bounded in __init__
    # -----------------------------

    def preprocess(self, zk, uk=None, **kwargs): ...
    def inprocess(self, xk_hat, **kwargs): ...
    def postprocess(self, zk_hat, **kwargs): ...

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
        Infer the common state state across all submodels.
        - If all submodels have identical states → return that state.
        - If submodels have compatible subclasses → return the nearest common base.
        - Otherwise, raise an error.
        """
        if not self.models:
            raise AttributeError("Composite model has no submodels to infer state from.")

        # --- 1. Collect all model states ---
        states = []
        for m in self.models:
            if m.states is None:
                raise AttributeError(f"Submodel {m} does not define a 'states' property.")
            states.append(m.states)

        # --- 2. If all states identical, easy case ---
        if all(t == states[0] for t in states):
            return states[0]

        # --- 3. Find nearest common base class ---
        common_bases = set.intersection(*(set(t.mro()) for t in states))
        # Drop trivial classes like object or generic bases
        common_bases = [b for b in common_bases if b.__name__ not in ("object", "ABC")]

        if not common_bases:
            raise TypeError(
                f"Incompatible submodel states in KalmanCompositeModel: "
                + ", ".join(t.__name__ for t in states)
            )

        # --- 4. Pick the most specific shared base ---
        # The MRO order is important: first common base is the most specific
        base_reference_mro = states[0].mro()
        for cls in base_reference_mro:
            if cls in common_bases:
                return cls

        # --- 5. Fallback (shouldn't happen) ---
        raise TypeError(f"Failed to resolve common state for submodels: {states}")

    @property
    def observables(self):
        """
        Infer the common state observable across all submodels.
        - If all submodels have identical observables → return that observable.
        - If submodels have compatible subclasses → return the nearest common base.
        - Otherwise, raise an error.
        """
        if not self.models:
            raise AttributeError("Composite model has no submodels to infer observable from.")

        # --- 1. Collect all model observables ---
        observables = []
        for m in self.models:
            if m.observables is None:
                raise AttributeError(f"Submodel {m} does not define a 'observables' property.")
            observables.append(m.observables)

        # --- 2. If all observables identical, easy case ---
        if all(t == observables[0] for t in observables):
            return observables[0]

        # --- 3. Find nearest common base class ---
        common_bases = set.intersection(*(set(t.mro()) for t in observables))
        # Drop trivial classes like object or generic bases
        common_bases = [b for b in common_bases if b.__name__ not in ("object", "ABC")]

        if not common_bases:
            raise TypeError(
                f"Incompatible submodel observables in KalmanCompositeModel: "
                + ", ".join(t.__name__ for t in observables)
            )

        # --- 4. Pick the most specific shared base ---
        # The MRO order is important: first common base is the most specific
        base_reference_mro = observables[0].mro()
        for cls in base_reference_mro:
            if cls in common_bases:
                return cls

        # --- 5. Fallback (shouldn't happen) ---
        raise TypeError(f"Failed to resolve common observable for submodels: {observables}")

import numpy as np
from scipy.linalg import block_diag
from typing import List, Optional
from ..framework.abstract.metadata import KalmanModelMetadata
from ..framework.constants import KalmanObservationMode
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
    - Observation composition governed by KalmanObservationMode:
        * INDEPENDENT → z = [h1(x1); h2(x2); ...]
        * FUSED       → z = sum_i h_i(x_i)
    - No custom overrides; complex coupling should be encoded
      in the submodels themselves.
    """

    @property
    def type(self):
        """
        Infer the common state type across all submodels.
        - If all submodels have identical types → return that type.
        - If submodels have compatible subclasses → return the nearest common base.
        - Otherwise, raise an error.
        """
        if not self.models:
            raise AttributeError("Composite model has no submodels to infer type from.")

        # --- 1. Collect all model types ---
        types = []
        for m in self.models:
            m_type = getattr(m, "type", None)
            if m_type is None:
                raise AttributeError(f"Submodel {m} does not define a 'type' property.")
            types.append(m_type)

        # --- 2. If all types identical, easy case ---
        if all(t == types[0] for t in types):
            return types[0]

        # --- 3. Find nearest common base class ---
        common_bases = set.intersection(*(set(t.mro()) for t in types))
        # Drop trivial classes like object or generic bases
        common_bases = [b for b in common_bases if b.__name__ not in ("object", "ABC")]

        if not common_bases:
            raise TypeError(
                f"Incompatible submodel types in KalmanCompositeModel: "
                + ", ".join(t.__name__ for t in types)
            )

        # --- 4. Pick the most specific shared base ---
        # The MRO order is important: first common base is the most specific
        base_reference_mro = types[0].mro()
        for cls in base_reference_mro:
            if cls in common_bases:
                return cls

        # --- 5. Fallback (shouldn't happen) ---
        raise TypeError(f"Failed to resolve common type for submodels: {types}")

    def __init__(
        self,
        *models: List[KalmanModelMetadata],
        obs_mode: KalmanObservationMode = KalmanObservationMode.INDEPENDENT,
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
        self.M = self._bind_matrix("M")

        # Covariances (block-diagonal)
        self.Q = lambda t=None: block_diag(*[(m.Q(t) if callable(m.Q) else m.Q) for m in self.models])
        self.R = lambda t=None: block_diag(*[(m.R(t) if callable(m.R) else m.R) for m in self.models if m.m > 0])

        # Processing hooks
        self.preprocess  = self._bind_preprocess()
        self.inprocess   = self._bind_inprocess()
        self.postprocess = self._bind_postprocess()

        super().__init__("Composite", x0=x0, P0=P0, fs=self.fs, **kwargs)
        self.axis.clear()
        for m in self.models:
            self.axis.extend(m.axis)

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
        if mode == KalmanObservationMode.INDEPENDENT:
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

        elif mode == KalmanObservationMode.FUSED:
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
        if mode == KalmanObservationMode.INDEPENDENT:
            def H(x=None, t=None):
                blocks = [m.H(x, t) for m in self.models]
                return block_diag(*blocks)
            return H

        elif mode == KalmanObservationMode.FUSED:
            def H(x=None, t=None):
                """Sum of local observation Jacobians."""
                blocks = [m.H(x, t) for m in self.models]
                return sum(blocks)
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
import numpy as np
from dataclasses import dataclass, field

from ...helpers import *
from ...runtime.debug import get_debug

from ..constants import KalmanState, KalmanStateFlavor, KalmanObservableFlavor
from enum import Enum as _PyEnum

# =========================
# Record & Collection
# =========================
@dataclass(frozen=True)
class KalmanRecord:
    """
    Posterior-centric record:
      - time: (N,)
      - zk:   (N, m)  measurement vector supplied/used by the filter
      - xk:   (N, n)  state estimate sequence (posterior/prior/smoothed/etc.)
      - Pk:   (N, n, n) state covariance sequence aligned with xk
    """
    time: np.ndarray = field(default_factory=lambda: np.array([])) # time vector
    xk:   np.ndarray = field(default_factory=lambda: np.array([])) # state vector
    Pk:   np.ndarray = field(default_factory=lambda: np.array([])) # state covariance
    zk:   np.ndarray = field(default_factory=lambda: np.array([])) # measurement vector

    # Model reference (a record describes model output: dims, states/observables, mappings)
    model: object = None

    # Optional flavor
    flavor: KalmanStateFlavor = None

    # Properties for dimensions
    @property
    def n(self):
        return self.xk.shape[1] if self.xk.ndim > 1 else 0
    @property
    def m(self):
        return self.zk.shape[1] if self.zk.ndim > 1 else 0
    @property
    def dim(self):
        return len(self.zk[0]) if self.zk is not None and len(self.zk) > 0 else 0

    @property
    def _spc(self):
        """
        States per measurement channel = n // dim.
        Used as the stride in the state-index formula  idx = _spc * axis + comp_id,
        which is correct for both single models and composite models:
          - BallisticModel (n=9, m=3):  _spc = 3  (pos/vel/acc per axis)
          - OscillatorComposite (n=2N, m=N): _spc = 2  (pos+vel per oscillator)
          - AnalyticOscillator (n=2, m=2):   _spc = 1  (one state per obs channel)
        """
        return self.n // max(1, self.dim)
    @property
    def duration(self):
        return 0 if self.time is None else len(self.time) * (self.time[1] - self.time[0])
    @property
    def sample_rate(self):
        return 1 / (self.time[1] - self.time[0]) if self.time is not None and len(self.time) > 1 else 0

    def __len__(self):
        return len(self.xk)
    def __getitem__(self, idx):
        return self.xk[idx]
    def __iter__(self):
        return iter(zip(self.time, self.xk, self.Pk, self.zk))
    
    @staticmethod
    def __freeze_array(a):
        a = np.array(a, copy=True)
        a.setflags(write=False)
        return a

    def __post_init__(self):
        from ..abstract.contracts import KalmanModelABC
        object.__setattr__(self, "time", self.__freeze_array(self.time))
        object.__setattr__(self, "zk",   self.__freeze_array(self.zk))
        object.__setattr__(self, "xk",   self.__freeze_array(self.xk))
        object.__setattr__(self, "Pk",   self.__freeze_array(self.Pk))
        object.__setattr__(self, "model", self.model)
        object.__setattr__(self, "flavor", self.flavor)

        # -----------------------------
        # Type and structural checks
        # -----------------------------
        if not isinstance(self.time, (np.ndarray, list, tuple)):
            raise ValueError(f"'time' must be a numpy array or list, got {type(self.time)}")

        assert self.model is None or any(
            c.__name__ == "KalmanModelABC" for c in type(self.model).__mro__
        ), f"'model' must be a KalmanModel instance, got {type(self.model)}"
        assert self.flavor is None or isinstance(self.flavor, KalmanStateFlavor), \
            f"'flavor' must be of type KalmanStateFlavor, got {type(self.flavor)}"

        # -----------------------------
        # Shape validation if model provided
        # -----------------------------
        if self.model is not None:
            n, m = self.model.n, self.model.m
            N = len(self.xk)

            if self.xk.size > 0:
                assert self.xk.shape[1] == n, \
                    f"xk has wrong state dimension: expected ({N},{n}), got {self.xk.shape}"

            if self.zk.size > 0:
                assert self.zk.shape[1] in (1, m), \
                    f"zk has wrong measurement dimension: expected ({N},{m}) or ({N},1) after postprocess, got {self.zk.shape}"

            if self.Pk.size > 0:
                assert self.Pk.shape[1:] == (n, n), \
                    f"Pk has wrong covariance shape: expected ({N},{n},{n}), got {self.Pk.shape}"

            if len(self.time) not in (0, N):
                raise ValueError(f"time length ({len(self.time)}) does not match state sequence ({N})")

            # Sanity check for NaNs or inf
            if np.isnan(self.xk).any() or np.isinf(self.xk).any():
                raise ValueError("xk contains NaN or Inf values.")
            if np.isnan(self.Pk).any() or np.isinf(self.Pk).any():
                raise ValueError("Pk contains NaN or Inf values.")
            if np.isnan(self.zk).any() or np.isinf(self.zk).any():
                raise ValueError("zk contains NaN or Inf values.")
    @staticmethod
    def _normalize_index(idx, size=None):
        """Convert int, Enum, or tuple/list/array to np.array of ints."""
        if isinstance(idx, (tuple, list, np.ndarray)):
            return np.array([i if hasattr(i, "value") else int(i) for i in idx])
        else:
            i = idx if hasattr(idx, "value") else int(idx)
            if size is not None:
                return np.array([i] * size)
            return np.array([i])

    def _resolve_component_symbol(self, val):
        """
        Accepts an Enum member (KalmanState/KalmanObservable), name (str), or int.
        - If Enum: return as-is
        - If str : try model.states[NAME] then model.observables[NAME]
        - Else   : return as-is (assumed int index)
        """
        from enum import Enum as _PyEnum

        # already an enum (any Enum subtype) -> ok
        if isinstance(val, _PyEnum):
            return val

        # name -> try state then observables
        if isinstance(val, str) and self.model is not None:
            name = val.upper()
            try:
                return self.model.states[name]
            except Exception:
                pass
            try:
                return self.model.observables[name]
            except Exception:
                pass

        # int or anything else -> pass through
        return val

    def _enum_classes(self):
        """
        Returns (StateEnumClass or None, ObsEnumClass or None).
        We expect model.states / model.observables to return Enum classes.
        """
        S = None; O = None
        if self.model is not None:
            try:
                S = self.model.states
            except Exception:
                S = None
            try:
                O = self.model.observables
            except Exception:
                O = None
        return S, O

    def _resolve_name_to_enum(self, name: str, raise_ambiguous: bool = False):
        """
        Map a string name to the unique matching Enum member.
        - If it exists only in states -> return that member
        - If it exists only in observables -> return that member
        - If it exists in BOTH -> raise ValueError (ambiguity)
        - If it exists in NONE -> raise KeyError
        """
        S, O = self._enum_classes()
        key = name.upper()

        s_hit = o_hit = None
        if S is not None and hasattr(S, "__members__") and key in S.__members__:
            s_hit = S.__members__[key]
        if O is not None and hasattr(O, "__members__") and key in O.__members__:
            o_hit = O.__members__[key]
        
        if s_hit and o_hit:
            if raise_ambiguous:
                raise ValueError(f"Ambiguous component '{name}': found in both KalmanState and KalmanObservable enums.\n"
                                 f"Please specify {S.__name__}.{name.upper()} or {O.__name__}.{name.upper()}")
            return s_hit  # Prefer state if ambiguous and not raising
        
        if s_hit:
            return s_hit
        if o_hit:
            return o_hit

        raise KeyError(f"Unknown component name '{name}' in {S.__name__}/{O.__name__} enums.")

    def _resolve_index_to_enum(self, idx: int):
        """
        Map an integer index to a unique Enum member by position.
        - Try states[idx] (iter order) and observables[idx]
        - If both exist and point to different members -> ambiguity error
        - If only one exists -> return it
        - If none exist -> raise IndexError (unless no enums are available, then return idx)
        """
        S, O = self._enum_classes()

        s_hit = o_hit = None

        try:
            if S is not None:
                s_list = list(S)
                if 0 <= idx < len(s_list):
                    s_hit = s_list[idx]
        except Exception:
            pass

        try:
            if O is not None:
                o_list = list(O)
                if 0 <= idx < len(o_list):
                    o_hit = o_list[idx]
        except Exception:
            pass

        if s_hit and o_hit and s_hit is not o_hit:
            raise ValueError(f"Ambiguous index {idx}: matches both State and Observation enums.")
        if s_hit:
            return s_hit
        if o_hit:
            return o_hit

        # If we have neither enum, keep int (backward compatible).
        if S is None and O is None:
            return idx

        raise IndexError(f"Index {idx} does not match any State/Observation member.")
    
    def _normalize_component(self, component, *, preserve_shape: bool = False, count: int | None = None):
        """
        Normalize `component` with ambiguity-safe enum resolution.

        Returns:
        - preserve_shape=True:
            * scalar -> Enum (KalmanState or KalmanObservable)
            * flat list/tuple -> list[Enum]
            * nested list/tuple -> list[tuple[Enum, ...]]  (one nesting level preserved)
        - preserve_shape=False (default / plotting path):
            * returns a tuple[Enum, ...] of length `count` (pad by repeating last;
              truncate if longer). `count` defaults to `self.dim`, but plotters that
              need a fixed number of components regardless of the sensor count pass it
              explicitly — e.g. a 2-D phase plot passes ``count=2`` so a single-sensor
              model (``self.dim == 1``) keeps both phase components instead of
              truncating to one.
        """
        need = max(1, count if count is not None else (self.dim or 1))
        # --- local resolver that yields an Enum or raises ---
        def _norm_one(v):
            if isinstance(v, _PyEnum):
                return v
            if isinstance(v, str):
                return self._resolve_name_to_enum(v)
            if isinstance(v, (int, np.integer)):
                return self._resolve_index_to_enum(int(v))
            # anything else is unsupported for components
            raise TypeError(f"Unsupported component type: {type(v).__name__}")

        # --- scalar path ------------------------------------------------------------
        if not isinstance(component, (list, tuple, np.ndarray)):
            member = _norm_one(component)
            if preserve_shape:
                return member
            # flattening mode: repeat to the requested count (defaults to self.dim)
            return tuple([member] * need)

        # --- sequence path: detect one-level nesting --------------------------------
        seq = list(component)

        def _is_seq(x):
            return isinstance(x, (list, tuple, np.ndarray))

        has_nested = any(_is_seq(x) for x in seq)

        if preserve_shape:
            if not has_nested:
                # flat -> list[Enum]
                return [ _norm_one(x) for x in seq ]
            # nested -> list[tuple[Enum,...]]
            out = []
            for grp in seq:
                if not _is_seq(grp):
                    raise TypeError("Mixed nested/flat component structure is not supported.")
                out.append(tuple(_norm_one(x) for x in grp))
            return out

        # flattening mode for plotting: produce tuple of length self.dim
        flat = []
        if not has_nested:
            flat = [ _norm_one(x) for x in seq ]
        else:
            # flatten one level (left-to-right) then use pad/truncate
            for grp in seq:
                if not _is_seq(grp):
                    raise TypeError("Mixed nested/flat component structure is not supported.")
                for x in grp:
                    flat.append(_norm_one(x))

        if len(flat) == 0:
            # fallback: one arbitrary member replicated (but this should rarely happen)
            return tuple([_norm_one(0)] * max(1, self.dim or 1))

        if len(flat) < need:
            flat += [flat[-1]] * (need - len(flat))
        elif len(flat) > need:
            flat = flat[:need]
        return tuple(flat)

    @staticmethod
    def _normalize_in(x, indexing = None, dtype=None):
        """
        Normalize the input zk to ensure it has the correct shape.
        Converts vectors or lists into a column-shaped numpy array.
        Supports indexing modes "ij" (default) or "xy" for convenience.
        """
        if x is None:
            return None

        indexing = indexing or "xy"
        if indexing not in ["ij", "xy"]:
            raise ValueError(f"Invalid indexing mode '{indexing}'. Use 'ij' or 'xy'.")

        x = np.asarray(x, dtype)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)

        if indexing == "ij":
            x = np.transpose(x, axes=(1, 0))

        return x
    
    @staticmethod
    def _normalize_out(x, indexing = None, dtype=None):
        """
        Normalize the output zk_hat to ensure it has the correct shape.
        Converts vectors or lists into a column-shaped numpy array.
        Supports indexing modes "ij" (default) or "xy" for convenience.
        """        
        if x is None:
            return None

        indexing = indexing or "xy"
        if indexing not in ["ij", "xy"]:
            raise ValueError(f"Invalid indexing mode '{indexing}'. Use 'ij' or 'xy'.")
        
        x = np.asarray(x, dtype)
        # Squeeze trailing unit dim for 3D+ arrays: (N, m, 1) → (N, m).
        # Do NOT squeeze 2D (N, 1) because downstream code relies on shape[1] == m.
        while x.ndim >= 3 and x.shape[-1] == 1:
            x = np.squeeze(x, axis=-1)

        if indexing == "ij":
            x = np.transpose(x, axes=(1, 0))

        return x
        
    def _normalize_idx(self, idx: KalmanState, axis = 0):
        """
        Returns the state index for (component, sensor-axis) using the stride _spc.
        """
        idx = idx if hasattr(idx, 'value') else int(idx)
        idx = self._spc * axis + idx

        assert 0 <= idx < self.n, f"State index out of range. Expected 0 <= idx < {self.n}, got {idx}."
        return idx

    def print(self, **kwargs):
        print(self.__str__(**kwargs))

    def __repr__(self):
        flav = (str(self.flavor) if self.flavor is not None else "Unknown")
        model = (self.model.NAME if getattr(self.model, "NAME", None) else "Model?")
        return f"KalmanRecord(N={len(self)}, n={self.n}, m={self.m}, flavor={flav}, model={model})"

    def __str__(self):
        import textwrap, numpy as np
        
        # ---- local helpers -------------------------------------------------------
        def _shape(x):
            if x is None:
                return "N/A"
            try:
                s = "x".join(map(str, np.shape(x)))
            except Exception:
                s = f"Unsupported({type(x).__name__})"
            return s
        
        def _dims_line(label, arr, extra="", ndots=0):
            if arr is None:
                return f"{label}: N/A{(' ' + extra) if extra else ''}"

            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                shape_str = f"{arr.shape[0]}x(" + "x".join(str(d) for d in arr.shape[1:]) + ")"
            else:
                shape_str = _shape(arr)

            dots = "." * ndots
            return f"({label}) {dots}: {extra if extra else shape_str}"

        def _count_bad(a):
            if a is None:
                return (0,0)
            a = np.asarray(a)
            return (int(np.isnan(a).sum()), int(np.isinf(a).sum()))

        # Compact preview helpers for debug lines
        def _preview_row(row, k):
            """Return a short '[a, b, c, d ...]' style preview of a 1D row."""
            row = np.asarray(row).ravel()
            m = row.shape[0]
            k = max(1, min(k, m))
            head = ", ".join(f"{float(v):.4g}" for v in row[:k])
            if m > k:
                return f"[{head}, …]"
            return f"[{head}]"
        
        def _one_line_matrix_stats(name, A, k):

            """
            One-liner: '<name>: N=<N>, [<first k> .. <last k>], mean=<>, std=<>'
            Shows the first k and last k elements/rows for 1D/2D arrays.
            """
            if A is None:
                return f"{name}: None"
            
            A = np.asarray(A)
            if A.ndim == 0:
                return f"{name}: scalar={float(A):.6g}"
            
            if A.ndim == 1:
                N = A.shape[0]
                first = _preview_row(A[:k], k=k)
                last  = _preview_row(A[-k:], k=k) if N > 1 else first
                return (f"{name}: N={N}, [{first} .. {last}], "
                    f"μ={np.nanmean(A):.6g}, σ={np.nanstd(A):.6g}")
            
            # 2D or higher: use first and last k rows
            N = A.shape[0]
            first_rows = [_preview_row(A[i], k=k) for i in range(min(k, N))]
            last_rows  = [_preview_row(A[i], k=k) for i in range(max(0, N - k), N)]
            first_str = "[" + ", ".join(first_rows) + "]"
            last_str  = "[" + ", ".join(last_rows) + "]"
            return (f"{name}: N={N}, [{first_str} .. {last_str}], "
                f"μ={np.nanmean(A):.6g}, σ={np.nanstd(A):.6g}")

        # ---- snapshot numpy options ---------------------------------------------
        opt = np.get_printoptions()
        np.set_printoptions(linewidth=np.inf, threshold=200, edgeitems=3)

        # ---- gather basics -------------------------------------------------------
        N = len(self)
        n = self.n
        m = self.m
        flav = (str(self.flavor) if self.flavor is not None else "Unknown")
        duration = float(self.duration) if N > 0 else 0.0
        t0 = self.time[0] if self.time is not None and len(self.time) > 0 else 0.0

        # bad counts
        nan_x, inf_x = _count_bad(self.xk)
        nan_p, inf_p = _count_bad(self.Pk)
        nan_z, inf_z = _count_bad(self.zk)

        # covariance quick checks
        diag_neg = None
        if isinstance(self.Pk, np.ndarray) and self.Pk.ndim == 3 and self.Pk.shape[1] == self.Pk.shape[2] and N > 0:
            try:
                d = np.diagonal(self.Pk, axis1=1, axis2=2)
                diag_neg = int((d < 0).sum())
            except Exception:
                diag_neg = None

        # model info
        model_name = getattr(self.model, "NAME", None) or "Model ?"
        model_ver  = getattr(self.model, "VERSION", None) or "?"
        
        model_type = getattr(self.model, "dtype", None) or "N/A"
        if isinstance(model_type, type):
            model_type = model_type.__name__

        # optional long description if the model has one
        description = getattr(self.model, "DESCRIPTION", None) or ""
        if description:
            description = textwrap.fill(description, width=60, subsequent_indent=" " * 8)

        spacer = " " * 4
        s = []

        # ---- header --------------------------------------------------------------
        s.append(f"{self.__class__.__name__}:")
        # s.append("")

        # ---- high-level summary --------------------------------------------------
        s.append(f"{spacer}Summary:")
        s.append(f"{2*spacer}Attached Model....................................: {model_name} (v{model_ver})")
        s.append(f"{2*spacer}Description.......................................: {model_type}")
        s.append(f"{2*spacer}Samples...........................................: {N} data points")
        s.append(f"{2*spacer}Sample Rate.......................................: {self.sample_rate:.6g} Hz")
        s.append(f"{2*spacer}State Dimension (n)...............................: {n}D")
        s.append(f"{2*spacer}Observation Dimension (m).........................: {m}D")
        s.append(f"{2*spacer}Flavor............................................: {flav}")
        s.append(f"{2*spacer}Start Time........................................: {t0:.6g}s")
        s.append(f"{2*spacer}Duration..........................................: {duration:.6g}s")
        # s.append("")
        # ---- dimensions & data quality ------------------------------------------
        s.append(f"{spacer}Dimensions & Data Quality:")

        nan_time, inf_time = _count_bad(self.time)
        nan_z, inf_z = _count_bad(self.zk)
        nan_x, inf_x = _count_bad(self.xk)
        nan_p, inf_p = _count_bad(self.Pk)

        def _bad_str(nan, inf):
            parts = []
            if nan: parts.append(f"#NaN={nan}")
            if inf: parts.append(f"#Inf={inf}")
            return (" [" + ", ".join(parts) + "]") if parts else ""

        dt = 1./self.model.sample_rate()
        extra = f"[{self.time[0]:.6g}s .. {self.time[-1]+dt:.6g}s["
        s.append(f"{2*spacer}Time {_dims_line('s', self.time, extra=extra, ndots=41)}{_bad_str(nan_time, inf_time)}")
        s.append(f"{2*spacer}Measurements {_dims_line('zk', self.zk, ndots=32)}{_bad_str(nan_z, inf_z)}")
        s.append(f"{2*spacer}State Vector {_dims_line('xk', self.xk, ndots=32)}{_bad_str(nan_x, inf_x)}")
        covar_bad = _bad_str(nan_p, inf_p)
        diag_str = (f", Negative Diagonal Entries={diag_neg}" if diag_neg is not None and diag_neg > 0 else "")
        s.append(f"{2*spacer}State Covariance {_dims_line('Pk', self.Pk, ndots=28)}{covar_bad}{diag_str}")
        s.append("")

        # ---- averages per dimension (original behavior kept) ----------------------
        if isinstance(self.xk, np.ndarray) and self.xk.ndim == 2 and self.xk.shape[1] > 0:
            s.append(f"{spacer}State Vector Averages (per dimension):")
            avg_xk = np.nanmean(self.xk, axis=0)
            for i in range(self.xk.shape[1]):
                s.append(f"{2*spacer}xk[{i}] avg = {avg_xk[i]:.6g}")
            s.append("")
        if isinstance(self.zk, np.ndarray) and self.zk.ndim == 2 and self.zk.shape[1] > 0:
            s.append(f"{spacer}Measurement Vector Averages (per dimension):")
            avg_zk = np.nanmean(self.zk, axis=0)
            for j in range(self.zk.shape[1]):
                s.append(f"{2*spacer}zk[{j}] avg = {avg_zk[j]:.6g}")
            s.append("")
        
        # ---- concise one-line debug summaries ------------------------------------
        if get_debug():
            s.append(f"{spacer}Debug (compact one-liners):")
            # time line
            if self.time is not None and len(self.time) > 0:
                s.append(f"{2*spacer}time: N={len(self.time)}, {[float(v) for v in self.time[:2]] if len(self.time)>=1 else []}"
                        f" .. {[float(v) for v in self.time[-2:]] if len(self.time)>=2 else []}, "
                        f"μ = {np.nanmean(self.time):.6g}, σ={np.nanstd(self.time):.6g}")
                
            # xk & zk one-liners
            s.append(f"{2*spacer}{_one_line_matrix_stats('xk  ', self.xk, k=2)}")
            s.append(f"{2*spacer}{_one_line_matrix_stats('zk  ', self.zk, k=2)}")

        np.set_printoptions(**opt)
        return "\n".join(s)

    # Properties for model-dependent information
    def _plot_time(self, **kwargs):
        return "Time (s)" # always seconds, but could be overridden later..

    def _plot_labels(self, component : KalmanState, axis = None, units = False, **kwargs):

        label = (str(component).capitalize()+" " if isinstance(component, Enum) else f"Component #{int(axis)} ") if isinstance(axis, int) else ""
        model = kwargs.get("model", None) or self.model

        # Mix in axis label if axis is provided
        appended_noun = False
        if axis is not None:
            if model is not None and hasattr(model, "sensor") and hasattr(model, "_sensor") and len(model._sensor) > 0:
                safe_axis = min(axis, len(model._sensor) - 1)
                axis_label = model.sensor(safe_axis)
                label = f"{label}{axis_label}"
                appended_noun = bool(axis_label)

        # A bare TRUE flavor reads better as the noun "Truth" (parallel to the
        # "Ideal" curve); when a sensor noun follows it, keep the adjective form
        # ("True position").
        if isinstance(component, Enum) and component == KalmanObservableFlavor.TRUE and not appended_noun:
            label = "Truth "

        # Mix in units if requested
        if isinstance(units, str) and units:
            label = f"{label} ({units})"
        elif units:
            units_val = component.value
            label = f"{label} ({units_val})" if units_val else label

        return label

    def _plot_kwargs(self, **kwargs):

        _kwargs = {}
        flavor = kwargs.pop("flavor", None)
        axis = kwargs.pop("axis", None)
        kind = kwargs.pop("kind", None)
        model = kwargs.pop("model", None) or self.model # to be used to compute labels
        units = kwargs.pop("units", False)
        common = kwargs.pop("common", False)

        # Label handling
        _kwargs["label"] = kwargs.pop("label", f"{self._plot_labels(flavor, axis=axis, units=units, model=model)}")

        # Marker handling
        marker = kwargs.pop("marker", None)
        marker = marker or ("o" if kind == "scatter" else None)
        if marker: _kwargs["marker"] = marker

        # Colors handling
        colors = kwargs.get("colors", None) or "tab10"
        if isinstance(colors, str):
            try:
                cmap = plt.get_cmap(colors)
                colors = [cmap(i) for i in range(cmap.N)]
            except Exception:
                colors = None

        _kwargs["alpha"] = kwargs.pop("alpha", 1)
        if flavor == KalmanObservableFlavor.MEASURED:
            _kwargs["alpha"] = 2./3 * _kwargs["alpha"]

        color = kwargs.pop("color", None)
        if isinstance(color, int) and color > 0 and flavor == KalmanObservableFlavor.IDEAL and common:
            color = "black"
        else:
            if not isinstance(color, str):
                color = colors[color % len(colors)] if colors and isinstance(color, int) and len(colors) > 0 else "gray"
            if   flavor == KalmanObservableFlavor.MEASURED:
                color = lighten(fade(color))
            elif flavor in (KalmanObservableFlavor.TRUE, KalmanObservableFlavor.IDEAL):
                color = lighten(vivid(color))
            elif flavor == KalmanStateFlavor.PRIOR:
                color = darken(color, 0.25)
            elif flavor in (KalmanStateFlavor.POSTERIOR, KalmanStateFlavor.SMOOTHED):
                color = darken(color)

        _kwargs["color"] = color
    
        for key, val in kwargs.items():
            if key not in _kwargs and val is not None:
                _kwargs[key] = val

        if kind != "scatter":
            _kwargs["linestyle"] = kwargs.get("linestyle", None)
            if flavor == KalmanObservableFlavor.TRUE:
                _kwargs["linewidth"] = kwargs.get("linewidth", 1.5)
                _kwargs["linestyle"] = _kwargs["linestyle"] or kwargs.get("linestyle", "--")
            elif flavor == KalmanObservableFlavor.IDEAL:
                _kwargs["linewidth"] = kwargs.get("linewidth", 1.5)
                _kwargs["linestyle"] = _kwargs["linestyle"] or kwargs.get("linestyle", ":")

            _kwargs["linestyle"] = _kwargs["linestyle"] or kwargs.get("linestyle", "-")

        return _kwargs

    # ---------- Arrow helpers ----------
    def _arrow_indices(self, N: int, arrows: int | None = None) -> list[int]:
        """
        Choose ~uniform indices along a curve to place arrows.
        If `arrows` is None:
          - use 5 for longer series (N >= 100), else 4.
        Positions at fractions i/(arrows+1), i = 1..arrows  (≈ N/5 or N/4).
        """
        if N < 2:
            return []
        if arrows is None:
            arrows = 5 if N >= 100 else 4
        arrows = max(1, int(arrows))
        idxs = []
        for i in range(1, arrows + 1):
            j = int(round(i * (N - 1) / (arrows + 1)))
            if 0 <= j < N - 1:
                idxs.append(j)
        return sorted(set(idxs))

    def _smooth_dir_pts(self, arrays, j, w):
        """
        Return (tail_coords, head_coords) for an arrow centered at index j.
        Each coordinate is the mean over a window of width w on each side,
        which suppresses noise-driven direction flips.

        tail = mean(a[max(0, j-w) : j+1])
        head = mean(a[j : min(N-1, j+w)+1])
        """
        N = len(arrays[0])
        j0 = max(0, j - w)
        j1 = min(N - 1, j + w)
        tail = [float(np.mean(a[j0 : j + 1])) for a in arrays]
        head = [float(np.mean(a[j  : j1 + 1])) for a in arrays]
        return tail, head

    def _draw_arrows_1d(self, ax, t, y, color, arrows=None, lw=1, size=15):
        t, y = np.asarray(t), np.asarray(y)
        N = len(t)
        idxs = self._arrow_indices(N, arrows=arrows)
        w = max(1, min(N // 15, 50))
        for j in idxs:
            (t0, y0), (t1, y1) = self._smooth_dir_pts((t, y), j, w)
            if np.isclose(t0, t1) and np.isclose(y0, y1):
                continue
            ax.annotate(
                "", xy=(t1, y1), xytext=(t0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=size)
            )

    def _draw_arrows_2d(self, ax, x, y, color, arrows=None, lw=1, size=15):
        x, y = np.asarray(x), np.asarray(y)
        N = len(x)
        idxs = self._arrow_indices(N, arrows=arrows)
        w = max(1, min(N // 15, 50))
        for j in idxs:
            (x0, y0), (x1, y1) = self._smooth_dir_pts((x, y), j, w)
            if np.isclose(x0, x1) and np.isclose(y0, y1):
                continue
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=size)
            )

    def _draw_arrows_3d(self, ax, x, y, z, color='r', arrows=None, lw=1, size=15):
        """
        Draw '−|>' arrowheads along a 3D curve using projected FancyArrowPatch.
        No tails — just visible heads aligned to curve direction.
        """
        import numpy as np
        from mpl_toolkits.mplot3d import proj3d
        from matplotlib.patches import FancyArrowPatch

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                super().__init__((0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def do_3d_projection(self, renderer=None):
                xs3d, ys3d, zs3d = map(np.ravel, self._verts3d)
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                return np.min(zs)

        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        N = len(x)
        idxs = self._arrow_indices(N, arrows=arrows)
        if not idxs:
            return

        w = max(1, min(N // 15, 50))
        for j in idxs:
            if j + 1 >= N:
                continue
            (x0, y0, z0), (x1, y1, z1) = self._smooth_dir_pts((x, y, z), j, w)
            if np.allclose([x0, y0, z0], [x1, y1, z1]):
                continue
            arrow = Arrow3D([x0, x1], [y0, y1], [z0, z1],
                            mutation_scale=size,
                            lw=lw,
                            arrowstyle='-|>',
                            color=color,
                            shrinkA=0, shrinkB=0)
            ax.add_artist(arrow)
            
    def plot(self, **kwargs):
        """
        Unified plotting entrypoint using kwargs only.

        kwargs:
        - view:        "auto" | "1d" | "2d" | "3d"   (default: "auto")
        - component:   int | Enum | sequence of them (optional)
        - dim:         int  | sequence of ints        (optional)
        - ...          any other kwargs forwarded to _plot1D/_plot2D/_plot3D
        """
        if self.time is None or len(self.time) < 1:
            return None, None

        # ----------------------- pull kwargs with defaults ------------------------
        component  = kwargs.pop("component", None)
        dim       = kwargs.pop("dim", None)

        view       = kwargs.pop("view", "auto")
        slider     = kwargs.pop("slider", False) or kwargs.pop("interactive", False) or kwargs.pop("replay", False)
        error      = kwargs.pop("error", False)

        dim_explicit  = dim is not None
        is_multi_comp = isinstance(component, (list, tuple)) and len(component) > 1

        def _is_single(x):
            if x is None: return False
            return not isinstance(x, (list, tuple)) or (isinstance(x, (list, tuple)) and len(x) == 1)

        if view == "auto" and _is_single(component) and _is_single(dim):
            view = "1d"
        # A multi-component request without an explicit axis is a per-component
        # *dashboard* (one panel per component vs time), not a cross-axis phase
        # plot: resolve it to 1d so each component indexes its own state slot
        # (idx = _spc*axis + comp_id). Otherwise auto-filling dim to the available
        # sensor axes (e.g. [0, 1]) forces a 2d phase plot whose second axis
        # overflows on models that carry one component per axis (AnalyticOscillator
        # re/im, _spc=1 → iy = _spc*1 + 1 = 2 on a 2-wide state vector).
        elif view == "auto" and is_multi_comp and not dim_explicit:
            view = "1d"

        # ----------------------- small local helpers ------------------------------
        def _need_by_view(v: str) -> int:
            if v == "1d": return 1
            if v == "2d": return 2
            if v == "3d": return 3
            if v == "auto": return None
            raise ValueError(f"view must be 'auto'|'1d'|'2d'|'3d', got {v}")

        def _decide_view(v, dim_len):
            if v in ("1d","2d","3d"): return v
            # auto
            return "3d" if dim_len >= 3 else ("2d" if dim_len == 2 else "1d")

        def _norm_dim(d, dim_available):
            if d is None:
                # default to XYZ (trim to available)
                base = [0, 1, 2]
                k = 3 if (dim_available or 0) >= 3 else (2 if (dim_available or 0) == 2 else 1)
                return base[:k]
            if not isinstance(d, (list, tuple)): d = [int(d)]
            out = [int(x) for x in d]
            for x in out:
                if not (0 <= x < max(1, dim_available or 1)):
                    raise ValueError(f"Axis {x} out of range [0,{max(1, dim_available or 1)-1}]")
            return out

        # ----------------------- normalize dim/component/view -------------------
        dim_available = self.dim or 1
        dim  = _norm_dim(dim, dim_available)
        view = _decide_view(view, len(dim))
        need = _need_by_view(view) or len(dim)

        # if dim shorter than need, extend with 0,1,2…; if longer, trim.
        # Clamp extra axes to the available spatial axes so a 2D *phase* plot on a
        # single-sensor model (e.g. position-vs-speed) reuses axis 0 for both
        # components instead of indexing a non-existent axis.
        base_axes = [0, 1, 2]
        if len(dim) < need:
            extra = [min(a, dim_available - 1) for a in base_axes[len(dim):need]]
            dim = list(dim) + extra
        elif len(dim) > need:
            dim = list(dim[:need])

        # An explicit multi-component 2d/3d plot with no axis given is a *phase /
        # state-space* portrait: the distinct components share one sensor axis
        # (idx = _spc*axis + comp_id), so collapse the auto-filled axis list onto
        # dim[0] rather than spreading components across separate sensor axes
        # (which overflows on one-component-per-axis models, e.g. _spc=1).
        if is_multi_comp and not dim_explicit and view in ("2d", "3d"):
            dim = [dim[0]] * need

        # ----------------------- interactive (time-slider) variant ----------------
        if slider:
            sigma = kwargs.pop("sigma", None)   # ±sigma·√Pkk band on the 1d panels
            # multi-panel dashboard: a list of components and/or a measurement-error pad
            is_list = isinstance(component, (list, tuple)) and len(component) > 1
            if view != "2d" and (is_list or error):
                return self._replay(component=list(component) if is_list else [component],
                                   dim=dim[0], view="1d", sample=kwargs.get("sample"), error=error, sigma=sigma,
                                   jshtml=kwargs.pop("jshtml", False),
                                   frames=kwargs.pop("frames", 80), interval=kwargs.pop("interval", 90))
            if view == "2d" or len(dim) >= 2:
                cc = list(component) if isinstance(component, (list, tuple)) else [component, component]
                return self._replay(component=cc[:2], dim=dim[:2], view="2d",
                                   sample=kwargs.get("sample"), jshtml=kwargs.pop("jshtml", False),
                                   frames=kwargs.pop("frames", 150), interval=kwargs.pop("interval", 80))
            c = component[0] if isinstance(component, (list, tuple)) else (component if component is not None else 0)
            return self._replay(component=c, dim=dim[0], view="1d", sigma=sigma,
                               sample=kwargs.get("sample"), jshtml=kwargs.pop("jshtml", False),
                               frames=kwargs.pop("frames", 150), interval=kwargs.pop("interval", 80))

        # `need` is the view's component count (1d→1, 2d→2, 3d→3); pass it so phase
        # plots keep all components even when the model has fewer sensor axes.
        component = self._normalize_component(component, count=need)

        # ----------------------- dispatch -----------------------------------------
        if view == "3d":
            fig, ax = self._plot3D(component, dim=dim, **kwargs)
        elif view == "2d":
            fig, ax = self._plot2D(component, dim=dim, **kwargs)
        else:
            fig, ax = self._plot1D(component, dim=dim, **kwargs)

        if ax.get_title() == "":
            ax.set_title(kwargs.pop("title", self.model.name() if self.model else "Generic Kalman Model"))

        return fig, ax

    def replay(self, *args, **kwargs):
        """Deprecated: use ``model.plot(record, sample=sample, interactive=True)``
        (or ``record.plot(replay=True, ...)``). Kept as a thin alias."""
        import warnings
        warnings.warn(
            "record.replay(...) is deprecated; use model.plot(record, sample=sample, "
            "interactive=True) to animate the model's signature figure.",
            DeprecationWarning, stacklevel=2,
        )
        return self._replay(*args, **kwargs)

    def _replay(self, component=0, dim=0, sample=None, view="1d", frames=80,
               interval=90, jshtml=False, figsize=None, dpi=80, title=None, error=False, sigma=None):
        """
        Interactive **time-cursor replay** of the reconstruction — play the filtered
        estimate as a function of time, optionally overlaying the **truth** and the noisy
        **measurements** (pass the sampled record as ``sample=``) so you can watch the
        filter track the data.

        Works both in a plain **Python window** (a GUI backend → ``plt.show()`` plays it)
        and in **Jupyter** (``jshtml=True`` returns an embedded play/scrub-slider player).
        It is also reachable as ``record.plot(slider=True, ...)`` with the same
        ``component`` / ``dim`` / ``sample`` arguments as :meth:`plot`.

            rec = model.filter(z)
            rec.replay(jshtml=True)                      # estimate of component 0 vs time
            rec.replay(component="speed", sample=truth, jshtml=True)
            rec.replay(component=["position","speed"], view="2d", jshtml=True)  # phase plane

        Parameters
        ----------
        component : int | Enum | str | pair
            State component(s) (same meaning as in :meth:`plot`); a pair is needed for ``view="2d"``.
        dim : int | pair
            Spatial/sensor axis (or pair of axes for 2D).
        sample : KalmanRecord | None
            A sampled record (truth ``xk`` + measurements ``zk``) to overlay.
        view : "1d" | "2d"
            Component-vs-time (default) or a two-component phase plot.
        jshtml : bool
            If True, return an ``IPython.display.HTML`` slider player; else a ``FuncAnimation``.
        """
        import matplotlib.pyplot as plt

        t = np.asarray(self.time)
        if t is None or len(t) < 2:
            return None
        N = len(t)

        # Multi-panel dashboard: a list of components and/or a measurement-error
        # pad, all driven by ONE joint time slider (uses the standard palette).
        comps = list(component) if isinstance(component, (list, tuple, np.ndarray)) else [component]
        if view != "2d" and (len(comps) > 1 or error):
            return self._replay_dashboard(
                comps, dim=dim, sample=sample, frames=frames, interval=interval,
                jshtml=jshtml, figsize=figsize, dpi=dpi, title=title, error=error, sigma=sigma)

        fr = list(range(0, N, max(1, N // max(1, frames)))) + [N - 1]
        base_title = title or (self.model.name() if self.model is not None else "Kalman record")

        def _resolve(c):
            """Return (component_index, label_object) without the ambiguous-index error."""
            if isinstance(c, (int, np.integer)):
                cid = int(c)
                try:
                    states = list(self.model.states)
                    lbl = states[cid] if 0 <= cid < len(states) else cid
                except Exception:
                    lbl = cid
                return cid, lbl
            e = self._normalize_component(c)[0]
            return int(e), e

        if view == "2d":
            cl = list(component) if isinstance(component, (list, tuple, np.ndarray)) else [component, component]
            dl = list(dim) if isinstance(dim, (list, tuple)) else [int(dim), int(dim)]
            ax_x, ax_y = int(dl[0]), int(dl[1])
            (c0id, c0lbl), (c1id, c1lbl) = _resolve(cl[0]), _resolve(cl[1])
            ix = self._spc * ax_x + c0id
            iy = self._spc * ax_y + c1id
            X, Y = self.xk[:, ix], self.xk[:, iy]
            fig, ax = plt.subplots(figsize=figsize or (6.0, 6.0), dpi=dpi)
            ax.plot(X, Y, color="lightgray", lw=1.0, zorder=1)
            if sample is not None:
                ax.plot(sample.xk[:, ix], sample.xk[:, iy], color="crimson", lw=1.0, alpha=0.55,
                        zorder=2, label="truth")
            ln, = ax.plot([], [], color="navy", lw=2.0, zorder=3, label="estimate")
            hd, = ax.plot([], [], "o", color="navy", ms=6, zorder=4)
            meas = None
            if sample is not None:
                try:
                    if self.model.exclusive_mapping(ix, ax_x) and self.model.exclusive_mapping(iy, ax_y):
                        meas = np.asarray(sample.zk)[:, [ax_x, ax_y]]
                except Exception:
                    meas = None
            msc = ax.scatter([], [], s=5, color="lightgray", zorder=0, label="measured") if meas is not None else None
            ax.set_xlabel(self._plot_labels(c0lbl, axis=ax_x))
            ax.set_ylabel(self._plot_labels(c1lbl, axis=ax_y))
            ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")

            def _draw(k):
                ln.set_data(X[:k + 1], Y[:k + 1]); hd.set_data([X[k]], [Y[k]])
                if msc is not None: msc.set_offsets(meas[:k + 1])
                ax.set_title(f"{base_title}    t = {t[k]:.3g} s")
                return (ln, hd) + ((msc,) if msc is not None else ())
        else:
            c = component[0] if isinstance(component, (list, tuple, np.ndarray)) else component
            comp_id, comp = _resolve(c)
            ax_a = int(dim[0]) if isinstance(dim, (list, tuple)) else int(dim)
            idx = self._spc * ax_a + comp_id
            y = self.xk[:, idx]
            fig, ax = plt.subplots(figsize=figsize or (8.5, 4.0), dpi=dpi)
            ax.plot(t, y, color="lightgray", lw=1.0, zorder=1)
            # ±sigma uncertainty band from Pk (full-time, drawn once like the guide)
            if sigma and isinstance(self.Pk, np.ndarray) and self.Pk.ndim == 3 and self.Pk.shape[1] > idx:
                sd = np.sqrt(np.clip(self.Pk[:, idx, idx], 0.0, None))
                ax.fill_between(t, y - sigma * sd, y + sigma * sd, color="navy",
                                alpha=0.15, linewidth=0, zorder=-1, label=f"±{sigma:g}σ")  # behind scatter (zorder=0)
            if sample is not None and sample.xk.shape[1] > idx:
                ax.plot(t, sample.xk[:, idx], color="crimson", lw=1.2, alpha=0.6, zorder=2, label="truth")
            ln, = ax.plot([], [], color="navy", lw=2.0, zorder=3, label="estimate")
            hd, = ax.plot([], [], "o", color="navy", ms=6, zorder=4)
            cur = ax.axvline(t[0], color="gray", ls="--", lw=1, zorder=2)
            meas = None
            if sample is not None:
                try:
                    if self.model.exclusive_mapping(idx, ax_a):
                        meas = np.asarray(sample.zk)[:, ax_a]
                except Exception:
                    meas = None
            msc = ax.scatter([], [], s=7, color="lightgray", zorder=0, label="measured") if meas is not None else None
            ax.set_xlim(t[0], t[-1])
            ax.set_xlabel(self._plot_time()); ax.set_ylabel(self._plot_labels(comp, axis=ax_a))
            ax.grid(alpha=0.3)
            if sample is not None: ax.legend(fontsize=8, loc="best")

            def _draw(k):
                ln.set_data(t[:k + 1], y[:k + 1]); hd.set_data([t[k]], [y[k]]); cur.set_xdata([t[k]])
                if msc is not None: msc.set_offsets(np.column_stack([t[:k + 1], meas[:k + 1]]))
                ax.set_title(f"{base_title}    t = {t[k]:.3g} s")
                return (ln, hd, cur) + ((msc,) if msc is not None else ())

        return animate(fig, _draw, frames=fr, interval=interval, jshtml=jshtml)

    def _replay_dashboard(self, components, dim=0, sample=None, frames=80, interval=90,
                          jshtml=False, figsize=None, dpi=80, title=None, error=False, sigma=None):
        """
        Multi-panel time-cursor replay: one panel per state component, plus an
        optional measurement-error histogram, all driven by a single joint time
        slider. Colors follow the standard per-flavor palette (Measured / Ideal /
        True / Smoothed) via :meth:`_plot_kwargs`, exactly like :meth:`plot`.

        Reached via ``replay(component=[...], error=True)`` or, equivalently,
        ``plot(replay=True, component=[...], error=True)``.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.asarray(self.time); N = len(t)
        fr = list(range(0, N, max(1, N // max(1, frames)))) + [N - 1]
        base_title = title or (self.model.name() if self.model is not None else "Kalman record")
        ax_a = int(dim[0]) if isinstance(dim, (list, tuple, np.ndarray)) else int(dim)

        def _resolve(c):
            if isinstance(c, (int, np.integer)):
                cid = int(c)
                try:
                    states = list(self.model.states)
                    lbl = states[cid] if 0 <= cid < len(states) else cid
                except Exception:
                    lbl = cid
                return cid, lbl
            e = self._normalize_component(c)[0]
            return int(e), e

        npan = len(components) + (1 if error else 0)
        fig, axes = plt.subplots(1, npan, figsize=figsize or (4.8 * npan, 4.2), dpi=dpi)
        axes = np.atleast_1d(axes).ravel()

        truth = KalmanRecord._normalize_out(sample.xk) if sample is not None else None
        ideal = self.model.ideal(duration=self.duration) if (sample is not None and self.model is not None) else None
        ideal = KalmanRecord._normalize_out(ideal) if ideal is not None else None

        panels = []
        for j, comp in enumerate(components):
            ax = axes[j]
            comp_id, comp_lbl = _resolve(comp)
            idx = self._spc * ax_a + comp_id
            y = self.xk[:, idx]

            kw_main  = self._plot_kwargs(flavor=self.flavor, axis=ax_a, color=0)
            kw_true  = self._plot_kwargs(flavor=KalmanObservableFlavor.TRUE,  axis=ax_a, color=0)
            kw_ideal = self._plot_kwargs(flavor=KalmanObservableFlavor.IDEAL, axis=ax_a, color=0)
            kw_meas  = self._plot_kwargs(flavor=KalmanObservableFlavor.MEASURED, axis=ax_a, color=0, kind="scatter", s=7)
            col = kw_main["color"]

            ax.plot(t, y, color="lightgray", lw=1.0, zorder=1)               # faint full guide
            # ±sigma uncertainty band from Pk (drawn once over all time, like the guide)
            if sigma and isinstance(self.Pk, np.ndarray) and self.Pk.ndim == 3 and self.Pk.shape[1] > idx:
                sd = np.sqrt(np.clip(self.Pk[:, idx, idx], 0.0, None))
                ax.fill_between(t, y - sigma * sd, y + sigma * sd, color=col,
                                alpha=0.15, linewidth=0, zorder=-1, label=f"±{sigma:g}σ")  # behind scatter (zorder=0)
            if ideal is not None and ideal.shape[1] > idx:
                ax.plot(t, ideal[:, idx], zorder=2, **kw_ideal)
            if truth is not None and truth.shape[1] > idx:
                ax.plot(t, truth[:, idx], zorder=2, **kw_true)

            meas = None
            if sample is not None and self.model is not None and self.model.exclusive_mapping(idx, ax_a):
                meas = np.asarray(sample.zk)[:, ax_a]
            msc = ax.scatter([], [], zorder=0, color=kw_meas["color"], s=kw_meas.get("s", 7),
                             alpha=kw_meas.get("alpha", 0.6), label=kw_meas["label"]) if meas is not None else None
            (ln,) = ax.plot([], [], color=col, lw=kw_main.get("linewidth", 2.0), zorder=3, label=kw_main["label"])
            (hd,) = ax.plot([], [], "o", color=col, ms=6, zorder=4)
            cur = ax.axvline(t[0], color="gray", ls="--", lw=1, zorder=2)

            ax.set_xlim(t[0], t[-1]); ax.set_xlabel(self._plot_time())
            ax.set_ylabel(self._plot_labels(comp_lbl, axis=ax_a, units=True))
            ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best"); ax.set_title(base_title)
            panels.append((y, ln, hd, cur, msc, meas))

        # Optional measurement-error histogram pad (accumulates toward its Gaussian).
        err_ax = err = bins = err_col = None
        if error and sample is not None:
            err_ax = axes[len(components)]
            zk  = np.asarray(sample.zk)[:, ax_a]
            ref = np.asarray(sample.xk)[:, self._spc * ax_a + 0]
            err = zk - ref
            bins = np.linspace(float(np.nanmin(err)), float(np.nanmax(err)), 40)
            err_col = self._plot_kwargs(flavor=self.flavor, axis=ax_a, color=0)["color"]

        def _draw(k):
            for (y, ln, hd, cur, msc, meas) in panels:
                ln.set_data(t[:k + 1], y[:k + 1]); hd.set_data([t[k]], [y[k]]); cur.set_xdata([t[k]])
                if msc is not None:
                    msc.set_offsets(np.column_stack([t[:k + 1], meas[:k + 1]]))
            if err_ax is not None:
                err_ax.cla()
                e = err[:k + 1]
                err_ax.hist(e, bins=bins, color="lightgray", label="Measured Z")
                if k > 5 and np.std(e) > 0:
                    from scipy.stats import norm
                    mu, sg = float(np.mean(e)), float(np.std(e))
                    xs = np.linspace(bins[0], bins[-1], 200); bw = bins[1] - bins[0]
                    err_ax.plot(xs, len(e) * bw * norm.pdf(xs, mu, sg), color=err_col, lw=1.6,
                                label=f"μ={mu:.2g}, σ={sg:.1f}")
                    err_ax.legend(fontsize=8)
                err_ax.set_xlabel("Measurement Error"); err_ax.set_ylabel("Counts"); err_ax.set_title(base_title)
            fig.suptitle(f"t = {t[k]:.3g} s", fontsize=12)
            return ()

        return animate(fig, _draw, frames=fr, interval=interval, jshtml=jshtml)

    def _plot1D(self, component, **kwargs):
        """
        Plot 1D time series for one component across one or more spatial axes.

        Args (kwargs):
        dim: None | int | [int, ...]
            - None -> plot all axes [0..self.dim-1]
            - int  -> plot that axis
            - list -> plot the listed axes (e.g., [0,2])
        sample, ideal, truth, measurement: arrays or will be derived as in _plot1D original
        legend: bool (default True)
        sharey: bool (default False)  # if False and a single axis, set ylabel
        units: bool (default True)    # draw reference lines once when no fig/canvas provided
        grid: bool (default True)
        s: scatter size for measurements (default 1)
        color: base color id or explicit color (forwarded)
        legend_loc: str (default "best")
        arrows: Optional[int] -> if None, auto (4 or 5 per curve)
        Returns:
        (figure, axis)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.time is None or len(self.time) < 1:
            return None, None

        # ----------------- read/normalize inputs -----------------
        dim_arg  = kwargs.pop("dim", None)
        legend   = kwargs.pop("legend", True)
        sharey   = kwargs.pop("sharey", False)
        grid     = kwargs.pop("grid", True)
        s_size   = kwargs.get("s", 1)

        legend_loc = kwargs.get("legend_loc", "best")
        arrows_ct = kwargs.get("arrows", None)

        sample      = kwargs.pop("sample", None)
        ideal_in    = kwargs.pop("ideal", None)
        truth_in    = kwargs.pop("truth", None)
        meas_in     = kwargs.pop("measurement", None)
        commons     = kwargs.pop("commons", [])
        sigma       = kwargs.pop("sigma", None)   # if set (e.g. 2 or 3), shade ±sigma·√Pkk band

        # Derive defaults; ideal=False disables the ideal line explicitly
        if ideal_in is None:
            ideal = self.model.ideal(duration=self.duration) if self.model else None
        elif ideal_in is False:
            ideal = None
        else:
            ideal = ideal_in
        ideal = KalmanRecord._normalize_out(ideal) if (sample is not None and ideal is not None) else None

        truth = truth_in or (sample.xk if sample is not None else None)
        truth = KalmanRecord._normalize_out(truth) if (sample is not None and truth is not None) else None

        measurement = meas_in or (sample.zk if sample is not None else None)
        measurement = KalmanRecord._normalize_out(measurement) if (sample is not None and measurement is not None) else None

        # Canvas
        c = kwargs.get("canvas", None) or get_canvas(**kwargs)
        plt.sca(c)

        # dims: list[int]
        if dim_arg is None:
            axes = list(range(self.dim))
        elif isinstance(dim_arg, (list, tuple, np.ndarray)):
            axes = list(map(int, dim_arg))
        else:
            axes = [int(dim_arg)]

        for a in axes:
            if not (0 <= a < self.dim):
                raise ValueError(f"Invalid axis {a} considering {self.dim}D model")

        # component (use first if a sequence)
        component = component[0] if isinstance(component, (list, tuple, np.ndarray)) else component
        component = self._normalize_component(component)[0]
        comp_id   = int(component)

        # Base color id; allow passing a concrete color via kwargs['color']
        base_color_id = len(get_lines(**kwargs))

        # ----------------- plotting loop -----------------
        for i, a in enumerate(axes):
            idx = self._spc * a + comp_id

            # measurement
            if measurement is not None and self.model is not None and self.model.exclusive_mapping(idx, a):
                h = c.scatter(self.time, measurement[:, a],
                            **self._plot_kwargs(flavor=KalmanObservableFlavor.MEASURED,
                                                axis=a, kind="scatter", zorder=-1,
                                                s=s_size, common=self.flavor in commons,
                                                color=kwargs.get("color", base_color_id + i)))

            # ideal
            if ideal is not None and ideal.shape[1] > idx:
                h, = c.plot(self.time, ideal[:, idx],
                            **self._plot_kwargs(flavor=KalmanObservableFlavor.IDEAL,
                                                axis=a, common=KalmanObservableFlavor.IDEAL in commons,
                                                color=kwargs.get("color", base_color_id + i)))
                self._draw_arrows_1d(c, self.time, ideal[:, idx], color=h.get_color(), arrows=arrows_ct)

            # truth
            if truth is not None and truth.shape[1] > idx:
                h, = c.plot(self.time, truth[:, idx],
                            **self._plot_kwargs(flavor=KalmanObservableFlavor.TRUE,
                                                axis=a, common=KalmanObservableFlavor.TRUE in commons,
                                                color=kwargs.get("color", base_color_id + i)))
                self._draw_arrows_1d(c, self.time, truth[:, idx], color=h.get_color(), arrows=arrows_ct)

            # filtered/state (main)
            h, = c.plot(self.time, self.xk[:, idx],
                        **self._plot_kwargs(flavor=self.flavor, axis=a, common=self.flavor in commons,
                                            color=kwargs.get("color", base_color_id + i)))
            # label once under flavor name
            self._draw_arrows_1d(c, self.time, self.xk[:, idx], color=h.get_color(), arrows=arrows_ct)

            # ±sigma uncertainty band from the state covariance Pk (diagonal element idx)
            if sigma and isinstance(self.Pk, np.ndarray) and self.Pk.ndim == 3 and self.Pk.shape[1] > idx:
                sd = np.sqrt(np.clip(self.Pk[:, idx, idx], 0.0, None))
                c.fill_between(self.time,
                               self.xk[:, idx] - sigma * sd,
                               self.xk[:, idx] + sigma * sd,
                               color=h.get_color(), alpha=0.18, linewidth=0,
                               zorder=-2,   # backdrop: stays behind the measurement scatter (zorder=-1)
                               label=(f"±{sigma:g}σ" if i == 0 else None))

        # ----------------- labels / legend / grid -----------------
        # X label: time formatter like your original (allow override)
        c.set_xlabel(kwargs.pop("xlabel", self._plot_time()))

        # Y label:
        user_ylabel = kwargs.pop("ylabel", None)
        if user_ylabel is not None:
            c.set_ylabel(user_ylabel)
        elif not sharey and len(axes) == 1:
            c.set_ylabel(self._plot_labels(component, axis=axes[0]))

        c.grid(grid)

        if legend:
            c.legend(fontsize=10, handlelength=4, handleheight=0.01, loc=legend_loc)

        return get_figure(c), get_axis(c)

    def _plot2D(self, component, **kwargs):
        """
        2D phase plot:
        X = component[0] projected on axis dim[0]
        Y = component[1] projected on axis dim[1]

        Usage:
        _plot2D([compX, compY], dim=[0,1])
        _plot2D([compX, compY], dim=[1,0])
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.time is None or len(self.time) < 1:
            return None, None

        # --- args / inputs ---
        dim_arg    = kwargs.pop("dim", None)          # e.g., [0,1] or [1,0]
        legend     = kwargs.pop("legend", True)
        grid       = kwargs.pop("grid", True)
        s_size     = kwargs.get("s", 1)
        legend_loc = kwargs.get("legend_loc", "best")
        arrows_ct  = kwargs.get("arrows", None)
        commons    = kwargs.get("commons", [])

        # components: need two
        if not isinstance(component, (list, tuple, np.ndarray)) or len(component) < 2:
            raise ValueError("plot2D requires component[0] and component[1].")

        # normalize both components (phase plot always needs exactly 2)
        comp_norm = self._normalize_component(component, count=2)
        comp0_id  = int(comp_norm[0])
        comp1_id  = int(comp_norm[1])

        # dims: two axes (x-axis index, y-axis index)
        if dim_arg is None:
            if self.dim < 2:
                raise ValueError("plot2D needs at least 2 spatial axes or an explicit dim=[x,y].")
            ax_x, ax_y = 0, 1
        elif isinstance(dim_arg, (list, tuple, np.ndarray)) and len(dim_arg) == 2:
            ax_x, ax_y = int(dim_arg[0]), int(dim_arg[1])
        else:
            raise ValueError("`dim` must be a 2-element list/tuple like [0,1] or [1,0].")

        for a in (ax_x, ax_y):
            if not (0 <= a < self.dim):
                raise ValueError(f"Invalid axis {a} considering {self.dim}D model")

        # indices into state arrays: idx = _spc * axis + component_id
        idx_x = self._spc * ax_x + comp0_id
        idx_y = self._spc * ax_y + comp1_id

        # --- inputs like _plot1D ---
        sample       = kwargs.pop("sample", None)
        ideal_in     = kwargs.pop("ideal", None)
        truth_in     = kwargs.pop("truth", None)
        meas_in      = kwargs.pop("measurement", None)

        if ideal_in is None:
            ideal = self.model.ideal(duration=self.duration) if self.model else None
        elif ideal_in is False:
            ideal = None
        else:
            ideal = ideal_in
        ideal = KalmanRecord._normalize_out(ideal) if (sample is not None and ideal is not None) else None

        truth = truth_in or (sample.xk if sample is not None else None)
        truth = KalmanRecord._normalize_out(truth) if (sample is not None and truth is not None) else None

        measurement = meas_in or (sample.zk if sample is not None else None)
        measurement = KalmanRecord._normalize_out(measurement) if (sample is not None and measurement is not None) else None

        # --- canvas ---
        c = kwargs.get("canvas", None) or get_canvas(**kwargs)
        plt.sca(c)
        colorid = len(get_lines(**kwargs))
        base_color = kwargs.get("color", colorid)

        # --- styles ---
        kw_main  = self._plot_kwargs(flavor=self.flavor, axis=ax_x, color=base_color, common=self.flavor in commons)
        kw_true  = self._plot_kwargs(flavor=KalmanObservableFlavor.TRUE,  axis=ax_x, color=base_color, common=KalmanObservableFlavor.TRUE in kwargs.get("commons", []))
        kw_ideal = self._plot_kwargs(flavor=KalmanObservableFlavor.IDEAL, axis=ax_x, color=base_color, common=KalmanObservableFlavor.IDEAL in kwargs.get("commons", []))
        kw_meas  = self._plot_kwargs(flavor=KalmanObservableFlavor.MEASURED, axis=ax_x, zorder=-1,
                                     kind="scatter", s=s_size, color=base_color, alpha=kwargs.get("alpha", 0.7),
                                     common=KalmanObservableFlavor.MEASURED in kwargs.get("commons", []))

        # --- measurement scatter with exclusive mapping on BOTH axes ---
        if measurement is not None and self.model is not None:
            mx_ok = self.model.exclusive_mapping(idx_x, ax_x)
            my_ok = self.model.exclusive_mapping(idx_y, ax_y)
            if mx_ok and my_ok and measurement.shape[1] > max(ax_x, ax_y):
                h = c.scatter(measurement[:, ax_x], measurement[:, ax_y], **kw_meas)

        # --- ideal curve ---
        if ideal is not None and ideal.shape[1] > max(idx_x, idx_y):
            IX, IY = ideal[:, idx_x], ideal[:, idx_y]
            h, = c.plot(IX, IY, **kw_ideal)
            self._draw_arrows_2d(c, IX, IY, color=h.get_color(), arrows=arrows_ct)

        # --- truth curve ---
        if truth is not None and truth.shape[1] > max(idx_x, idx_y):
            TX, TY = truth[:, idx_x], truth[:, idx_y]
            h, = c.plot(TX, TY, **kw_true)
            self._draw_arrows_2d(c, TX, TY, color=h.get_color(), arrows=arrows_ct)

        # --- filtered/state curve ---
        X = self.xk[:, idx_x]; Y = self.xk[:, idx_y]
        h, = c.plot(X, Y, **kw_main)
        self._draw_arrows_2d(c, X, Y, color=h.get_color(), arrows=arrows_ct)

        # --- labels / legend / grid ---
        c.set_xlabel(kwargs.pop("xlabel", self._plot_labels(comp_norm[0], axis=ax_x)))
        c.set_ylabel(kwargs.pop("ylabel", self._plot_labels(comp_norm[1], axis=ax_y)))
        c.grid(grid)

        if legend:
            c.legend(fontsize=10, handlelength=4, handleheight=0.01, loc=legend_loc)

        return get_figure(c), get_axis(c)
    
    def _plot3D(self, component, **kwargs):
        """
        3D state trajectory with optional projections ("projectors") and start/end
        connectors to the projection planes. Arrows (≈N/4 or N/5) show direction.

        X = component[0] on axis dim[0]
        Y = component[1] on axis dim[1]
        Z = component[2] on axis dim[2]
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        t = self.time
        if t is None or len(t) == 0:
            return None, None

        # --- normalize components (need 3) ---------------------------------------
        comps = self._normalize_component(component, count=3)
        if len(comps) < 3:
            raise ValueError("plot3D requires component[0], component[1], component[2].")

        c0, c1, c2 = int(comps[0]), int(comps[1]), int(comps[2])

        # --- dimensions -----------------------------------------------------------
        dim_arg = kwargs.pop("dim", None)
        if dim_arg is None:
            if self.dim < 3:
                raise ValueError("plot3D needs dim>=3 or an explicit dim=[dx,dy,dz].")
            dx, dy, dz = 0, 1, 2
        else:
            if not (isinstance(dim_arg, (list, tuple, np.ndarray)) and len(dim_arg) == 3):
                raise ValueError("`dim` must be a 3-element list/tuple like [0,1,2].")
            dx, dy, dz = map(int, dim_arg)

        for a in (dx, dy, dz):
            if not (0 <= a < self.dim):
                raise ValueError(f"Invalid axis {a} considering {self.dim}D model")

        # indices in flattened state: idx = _spc * axis + comp_id
        idx_x = self._spc * dx + c0
        idx_y = self._spc * dy + c1
        idx_z = self._spc * dz + c2

        # --- derive series like in _plot1D/_plot2D --------------------------------
        sample      = kwargs.pop("sample", None)
        ideal_in    = kwargs.pop("ideal", None)
        truth_in    = kwargs.pop("truth", None)
        meas_in     = kwargs.pop("measurement", None)

        if ideal_in is None:
            ideal = self.model.ideal(duration=self.duration) if self.model else None
        elif ideal_in is False:
            ideal = None
        else:
            ideal = ideal_in
        ideal = KalmanRecord._normalize_out(ideal) if (sample is not None and ideal is not None) else None

        truth = truth_in or (sample.xk if sample is not None else None)
        truth = KalmanRecord._normalize_out(truth) if (sample is not None and truth is not None) else None

        measurement = meas_in or (sample.zk if sample is not None else None)
        measurement = KalmanRecord._normalize_out(measurement) if (sample is not None and measurement is not None) else None

        # main state
        X = self.xk[:, idx_x]; Y = self.xk[:, idx_y]; Z = self.xk[:, idx_z]
        X0 = self._normalize_out(self.xk[0])[idx_x]
        Y0 = self._normalize_out(self.xk[0])[idx_y]
        Z0 = self._normalize_out(self.xk[0])[idx_z]

        # optional series
        TX = TY = TZ = None
        if truth is not None and truth.shape[1] > max(idx_x, idx_y, idx_z):
            TX, TY, TZ = truth[:, idx_x], truth[:, idx_y], truth[:, idx_z]

        IX = IY = IZ = None
        if ideal is not None and ideal.shape[1] > max(idx_x, idx_y, idx_z):
            IX, IY, IZ = ideal[:, idx_x], ideal[:, idx_y], ideal[:, idx_z]

        # --- figure / axis --------------------------------------------------------
        fig = kwargs.get("fig") or plt.figure(figsize=kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
        plt.sca(ax)

        # styling helpers
        colorid   = len(get_lines(**kwargs))
        base_color = kwargs.get("color", colorid)
        alpha     = kwargs.get("alpha", 0.75)
        legend    = kwargs.pop("legend", True)
        legend_loc = kwargs.pop("legend_loc", "best")
        grid      = kwargs.pop("grid", True)
        projectors = kwargs.pop("projectors", True)
        connectors = kwargs.pop("connectors", True)
        arrows_ct  = kwargs.get("arrows", None)
        s_size     = kwargs.get("s", 1)
        commons    = kwargs.get("commons", [])

        # --- limits ---------------------------------------------------------------
        def _finite_minmax(*arrays):
            vals = []
            for a in arrays:
                if a is not None:
                    aa = np.asarray(a).ravel()
                    aa = aa[np.isfinite(aa)]
                    if aa.size:
                        vals.append((aa.min(), aa.max()))
            if not vals:
                return (0.0, 1.0)
            lo = min(v[0] for v in vals)
            hi = max(v[1] for v in vals)
            return (lo, hi if hi != lo else hi + 1.0)

        xlim = kwargs.pop("xlim", _finite_minmax(X, TX, IX))
        ylim = kwargs.pop("ylim", _finite_minmax(Y, TY, IY))
        zlim = kwargs.pop("zlim", _finite_minmax(Z, TZ, IZ))

        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

        # clip initial point to box
        x0 = float(np.clip(X0, *xlim))
        y0 = float(np.clip(Y0, *ylim))
        z0 = float(np.clip(Z0, *zlim))
        
        # Primary styles derived from base_color via _plot_kwargs
        kw_main  = self._plot_kwargs(flavor=self.flavor,                     common=self.flavor in commons, axis=dx, color=base_color)
        kw_true  = self._plot_kwargs(flavor=KalmanObservableFlavor.TRUE,     common=KalmanObservableFlavor.TRUE in commons,  axis=dx, color=base_color)
        kw_ideal = self._plot_kwargs(flavor=KalmanObservableFlavor.IDEAL,    common=KalmanObservableFlavor.IDEAL in commons, axis=dx, color=base_color)
        kw_meas  = self._plot_kwargs(flavor=KalmanObservableFlavor.MEASURED, common=KalmanObservableFlavor.MEASURED in commons,
                                    axis=dx, zorder=-1,color=base_color, kind="scatter", s=s_size, alpha=alpha)

        # Aux (projections/connectors) – route through _plot_kwargs but we’ll keep legend proxies in base color
        def kw_proj_state(axis):
            kw = self._plot_kwargs(flavor=self.flavor, axis=axis, color=axis+1, alpha=0.65)
            axis_str = self.model.sensor(axis) if self.model else "#" + str(axis+1)
            kw["label"] = f"Projection {axis_str}"
            return kw
        def kw_proj_true(axis):
            kw = self._plot_kwargs(flavor=KalmanObservableFlavor.TRUE, common=KalmanObservableFlavor.TRUE in commons, axis=axis, color=axis+1, alpha=0.65)
            axis_str = self.model.sensor(axis) if self.model else "#" + str(axis+1)
            kw["label"] = f"Projection {axis_str}"
            return kw
        def kw_proj_ideal(axis):
            kw = self._plot_kwargs(flavor=KalmanObservableFlavor.IDEAL, common=KalmanObservableFlavor.IDEAL in commons, axis=axis, color=axis+1, alpha=0.65)
            axis_str = self.model.sensor(axis) if self.model else "#" + str(axis+1)
            kw["label"] = f"Projection {axis_str}"
            return kw
        def kw_meas_proj(axis):
            kw = self._plot_kwargs(flavor=KalmanObservableFlavor.MEASURED, common=KalmanObservableFlavor.MEASURED in commons, axis=axis, color=axis+1, kind="scatter", zorder=-1, s=s_size, alpha=alpha)
            axis_str = self.model.sensor(axis) if self.model else "#" + str(axis+1)
            kw["label"] = f"Projection {axis_str}"
            return kw
        def kw_conn(axis):
            return self._plot_kwargs(flavor=self.flavor, axis=axis, color=axis+1, linewidth=0.5, alpha=0.25, linestyle="-.")

        # --- measurement scatter --------------------------------------------------
        if measurement is not None and self.model is not None:
            mx_ok = self.model.exclusive_mapping(idx_x, dx)
            my_ok = self.model.exclusive_mapping(idx_y, dy)
            mz_ok = self.model.exclusive_mapping(idx_z, dz)
            if mx_ok and my_ok:
                ax.scatter(measurement[:, dx], measurement[:, dy], np.full_like(measurement[:, dz], min(zlim)),
                        depthshade=False, **kw_meas_proj(dz))
            if mx_ok and mz_ok:
                ax.scatter(measurement[:, dx], np.full_like(measurement[:, dy], max(ylim)), measurement[:, dz],
                        depthshade=False, **kw_meas_proj(dy))
            if my_ok and mz_ok:
                ax.scatter(np.full_like(measurement[:, dx], min(xlim)), measurement[:, dy], measurement[:, dz],
                        depthshade=False, **kw_meas_proj(dx))
            if mx_ok and my_ok and mz_ok and measurement.shape[1] > max(dx, dy, dz):
                ax.scatter(measurement[:, dx], measurement[:, dy], measurement[:, dz],
                        depthshade=False, **kw_meas)
            
        # --- ancillary series -----------------------------------------------------
        if IX is not None:
            h_ideal, = ax.plot(IX, IY, IZ, **kw_ideal)
            self._draw_arrows_3d(ax, IX, IY, IZ, color=h_ideal.get_color(), arrows=arrows_ct)
        if TX is not None:
            h_truth, = ax.plot(TX, TY, TZ, **kw_true)
            self._draw_arrows_3d(ax, TX, TY, TZ, color=h_truth.get_color(), arrows=arrows_ct)

        # --- subtle reference box lines (neutral) --------------------------------
        ref_style = dict(color="black", linestyle=":", linewidth=0.75, alpha=0.8, zorder=-10)
        if xlim[0] < x0 < xlim[1]:
            ax.plot([xlim[0], xlim[1]], [y0, y0], [z0, z0], **ref_style)
        if ylim[0] < y0 < ylim[1]:
            ax.plot([x0, x0], [ylim[0], ylim[1]], [z0, z0], **ref_style)
        if zlim[0] < z0 < zlim[1]:
            ax.plot([x0, x0], [y0, y0], [zlim[0], zlim[1]], **ref_style)

        # --- projections (drawn; legend will show proxies in base color) ----------
        if projectors:
            if IX is not None:
                ax.plot(IX, IY, np.full_like(IZ, min(zlim)), **kw_proj_ideal(dz))
                ax.plot(IX, np.full_like(IX, max(ylim)), IZ, **kw_proj_ideal(dy))
                ax.plot(np.full_like(IY, min(xlim)), IY, IZ, **kw_proj_ideal(dx))
            if TX is not None:
                ax.plot(TX, TY, np.full_like(TZ, min(zlim)), **kw_proj_true(dz))
                ax.plot(TX, np.full_like(TX, max(ylim)), TZ, **kw_proj_true(dy))
                ax.plot(np.full_like(TY, min(xlim)), TY, TZ, **kw_proj_true(dx))
            if X is not None and Y is not None and Z is not None:
                ax.plot(X, Y, np.full_like(Z, min(zlim)), **kw_proj_state(dz))
                ax.plot(X, np.full_like(X, max(ylim)), Z, **kw_proj_state(dy))
                ax.plot(np.full_like(Y, min(xlim)), Y, Z, **kw_proj_state(dx))

        # --- connectors -----------------------------------------------------------
        if connectors and X is not None and len(X) > 0:
            for j in (0, len(X) - 1):
                xi, yi, zi = float(X[j]), float(Y[j]), float(Z[j])
                ax.plot([xi, xi], [yi, yi], [zlim[0], zi], **kw_conn(dz))
                ax.plot([xi, xi], [ylim[1], yi], [zi, zi], **kw_conn(dy))
                ax.plot([xlim[0], xi], [yi, yi], [zi, zi], **kw_conn(dx))

        # --- main trajectory + arrows --------------------------------------------
        h_main, = ax.plot(X, Y, Z, **kw_main)
        self._draw_arrows_3d(ax, X, Y, Z, color=h_main.get_color(), arrows=arrows_ct)

        # --- labels / cosmetics ---------------------------------------------------
        ax.set_xlabel(kwargs.pop("xlabel", self._plot_labels(comps[0], axis=dx)))
        ax.set_ylabel(kwargs.pop("ylabel", self._plot_labels(comps[1], axis=dy)))
        ax.set_zlabel(kwargs.pop("zlabel", self._plot_labels(comps[2], axis=dz)))
        ax.grid(grid)
        ax.set_box_aspect([1, 1, 1])

        # --- CLEAN LEGEND: only base color entries (trajectory + three projections)
        if legend:

            base_col = h_main.get_color()

            # pull applied styles for consistency
            main_ls  = kw_main.get("linestyle", "-")
            main_lw  = kw_main.get("linewidth", 1.5)

            kw_meas_projx = kw_proj_state(dx)
            kw_meas_projy = kw_proj_state(dy)
            kw_meas_projz = kw_proj_state(dz)

            # proxies use base color (not the axis-colored projection lines)
            proxies = [
                Line2D([0], [0], color=base_col, linestyle=main_ls, linewidth=main_lw, label=str(self.flavor).capitalize() + " 3D"),
                Line2D([0], [0], **kw_meas_projx),
                Line2D([0], [0], **kw_meas_projy),
                Line2D([0], [0], **kw_meas_projz),
            ]
            ax.legend(proxies, [h.get_label() for h in proxies],
                    fontsize=10, handlelength=4, handleheight=0.01, loc=legend_loc)

        return get_figure(ax), get_axis(ax)

    def plot_error(self, flavor: KalmanStateFlavor | KalmanObservableFlavor = KalmanObservableFlavor.MEASURED, component=None, dim=None, **kwargs):
        """
        Error plotting dispatcher with sample-driven data:
        - If 'sample' is provided:
            truth       := sample.xk
            ideal       := model.ideal(...) if available
            measurement := sample.zk  (used as zk_hat)
        - If 'sample' absent:
            use provided truth/ideal/measurement; if measurement is missing,

        Modes:
        - Observation flavor and no component  -> measurement error (zk - zk_hat)
        - component provided                   -> state component error (xk - reference)
        - else                                 -> L2 state error ||xk - reference||_2
        """
        from enum import Enum

        if self.time is None or len(self.time) == 0:
            return None, None

        # ---------------- inputs / references (sample-first logic) ----------------
        sample = kwargs.pop("sample", None)

        # Start from incoming kwargs (may be None); will override if sample given
        ideal       = kwargs.pop("ideal", None)
        truth       = kwargs.pop("truth", None)
        measurement = kwargs.pop("measurement", None)

        if sample is not None:
            # Use user's exact scheme
            ideal = ideal or (self.model.ideal(duration=self.duration) if self.model else None)
            ideal = KalmanRecord._normalize_out(ideal) if ideal is not None else None

            truth = truth or getattr(sample, "xk", None)
            truth = KalmanRecord._normalize_out(truth) if truth is not None else None

            measurement = measurement or getattr(sample, "zk", None)
            measurement = KalmanRecord._normalize_out(measurement) if measurement is not None else None
        else:
            # No sample: normalize whatever was provided
            if ideal is not None:       ideal = KalmanRecord._normalize_out(ideal)
            if truth is not None:       truth = KalmanRecord._normalize_out(truth)
            if measurement is not None: measurement = KalmanRecord._normalize_out(measurement)

        # pick reference for state error (prefer truth)
        reference = truth if truth is not None else ideal

        # figure/axis + cosmetics
        c       = kwargs.pop("canvas", None) or get_canvas(**kwargs)
        plt.sca(c)

        legend  = kwargs.pop("legend", True)

        # default flavor for labels/colors
        flavor = flavor or self.flavor
        colorid = len(get_lines(**kwargs))

        # helpers
        def _axes_from_dim(d):
            if d is None:
                return list(range(self.dim))
            if isinstance(d, (list, tuple, np.ndarray)):
                return [int(x) for x in d]
            return [int(d)]

        def _as_enum(v, enum_cls):
            if isinstance(v, enum_cls): return v
            if isinstance(v, Enum):     return v
            if isinstance(v, str):      return enum_cls[v.upper()]
            return enum_cls(int(v))

        axes = _axes_from_dim(dim)

        # ---------------- dispatch ----------------
        if isinstance(flavor, KalmanObservableFlavor) and component is None:
            # Measurement error: uses 'measurement' as zk_hat when available
            fig, ax = self._plot_error_measurement(
                flavor=flavor, axes=axes, zk0=measurement,
                canvas=c, legend=legend, color=kwargs.pop("color", colorid), **kwargs)

            if ax.get_title() == "":
                ax.set_title(kwargs.pop("title", self.model.name() if self.model else "Generic Kalman Model"))
            return fig, ax

        if component is not None:
            
            component = self._normalize_component(component)

            # State component error (requires reference)
            if reference is None:
                raise ValueError("Provide 'sample' or ('truth' or 'ideal') to compute state error.")

            fig, ax = self._plot_error_state(
                component=component, axes=axes, xk0=reference,
                canvas=c, legend=legend, **kwargs
            )

            if ax.get_title() == "":
                ax.set_title(kwargs.pop("title", self.model.name() if self.model else "Generic Kalman Model"))
            return fig, ax

        # L2 over full state (requires reference)
        if reference is None:
            raise ValueError("Provide 'sample' or ('truth' or 'ideal') to compute L2 state error.")
        fig, ax = self._plot_error_l2(
            xk0=reference, canvas=c, legend=legend, **kwargs
        )

        if ax.get_title() == "":
            ax.set_title(kwargs.pop("title", self.model.name() if self.model else "Generic Kalman Model"))
        return fig, ax


    def _plot_error_measurement(self, *, flavor, axes, zk0, **kwargs):
        """
        Plot zk0 error per axis: zk - zk_hat.
        Here zk_hat is taken from 'zk0' (preferred: sample.zk).
        If 'zk0' is None, we error.
        """

        kind = kwargs.pop("kind", "hist")
        bins = kwargs.pop("bins", 50)
        legend = kwargs.pop("legend", True)
        
        c = kwargs.pop("canvas", None) or get_canvas(**kwargs)
        plt.sca(c)

        if self.zk is None:
            raise ValueError("Cannot plot measurement error: 'zk' is None.")
        if zk0 is None:
            raise ValueError("Cannot plot measurement error: no predicted measurement available (pass 'sample' or 'measurement').")

        ymin, ymax = np.inf, -np.inf
        # Pop the user's colour once (not per axis): a per-axis base index keeps
        # X/Y/Z visually distinct and pairs each histogram with its own Gaussian
        # fit in the same hue (the faded bars + a saturated fit line).
        base_color = kwargs.pop("color", None)
        for i, a in enumerate(axes):
            if a >= self.zk.shape[1] or a >= zk0.shape[1]:
                continue
            err = self.zk[:, a] - zk0[:, a]
            ymin = min(ymin, np.nanmin(err))
            ymax = max(ymax, np.nanmax(err))

            # Per-axis colour: honour an explicit string colour, otherwise step
            # through the palette so each axis gets its own hue.
            axis_color = base_color if isinstance(base_color, str) else \
                (base_color if isinstance(base_color, int) else 0) + i

            if kind == "hist":
                # draw histogram and capture counts/edges (counts needed for scaling)
                counts, edges, _ = c.hist(err, bins=bins, **self._plot_kwargs(flavor=flavor, axis=a, color=axis_color))

                # fit Gaussian (MLE via mean/std with ddof=0)
                mu = float(np.mean(err))
                sigma = float(np.std(err, ddof=0)) or 1e-12  # avoid zero

                # overlay Gaussian scaled to counts
                x = np.linspace(edges[0], edges[-1], 512)
                inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
                pdf = inv * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

                n = np.sum(counts)
                bw = float(np.mean(np.diff(edges)))  # assumes ~uniform bins
                y = n * pdf * bw

                # solid, saturated line in the same hue as this axis' bars
                fit_kw = self._plot_kwargs(color=axis_color, axis=a, linewidth=2)

                if legend:
                    fit_kw["label"] = f"Gaussian Fit\n(μ={mu:.3g}, σ={sigma:.3g})"
                c.plot(x, y, **fit_kw)

            else:
                c.plot(self.time, err, **self._plot_kwargs(flavor=flavor, axis=a, color=axis_color))
        
        if kind == "hist":
            c.set_xlabel(kwargs.pop("xlabel", "Measurement Error"))
            c.set_ylabel(kwargs.pop("ylabel", "Counts"))
        else:
            c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
            c.set_ylabel(kwargs.pop("ylabel", "Absolute Measurement Error"))
        c.grid(kwargs.pop("grid", True))
        if legend: c.legend()

        return get_figure(c), get_axis(c)

    def _plot_error_state(self, *, component, axes, xk0, canvas, legend, **kwargs):
        """
        Plot per-axis state component error: xk[:, idx] - xk0[:, idx]
        """
        c = canvas
        plt.sca(c)

        ymin, ymax = np.inf, -np.inf

        def _state_index(component, axis):
            k = component if hasattr(component, "value") else int(component)
            return self._spc * int(axis) + int(k)

        colorid = len(get_lines(canvas=c))
        comp_one = component[0] if isinstance(component, (tuple, list)) else component

        for i, a in enumerate(axes):
            idx = _state_index(comp_one, a)
            if idx >= self.n or idx >= xk0.shape[1]:
                continue
            err = self.xk[:, idx] - xk0[:, idx]
            ymin = min(ymin, np.nanmin(err))
            ymax = max(ymax, np.nanmax(err))
            kw = self._plot_kwargs(flavor=self.flavor, axis=a, color=kwargs.get("color", colorid + i))
            kw["label"] = f"{self._plot_labels(comp_one, axis=a)} error"
            c.plot(self.time, err, **kw)

        
        c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
        c.set_ylabel(kwargs.pop("ylabel", f"Absolute {self._plot_labels(component)} Error"))
        c.grid(kwargs.pop("grid", True))
        
        if legend: c.legend()
        return get_figure(c), get_axis(c)

    def _plot_error_l2(self, *, xk0, canvas, legend, **kwargs):
        """
        Plot total L2 state error per time: ||xk - xk0||_2
        """
        c = canvas
        plt.sca(c)

        n_cols = min(self.xk.shape[1], xk0.shape[1])
        e  = self.xk[:, :n_cols] - xk0[:, :n_cols]
        l2 = np.sqrt(np.sum(e*e, axis=1))

        kw = self._plot_kwargs(flavor=self.flavor, axis=0, color=kwargs.get("color", 0))
        kw["label"] = f"{str(self.flavor).capitalize()} L2 error"
        c.plot(self.time, l2, **kw)


        c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
        c.set_ylabel(kwargs.pop("ylabel", "Least Square State Errors (L2)"))
        c.grid(kwargs.pop("grid", True))
        if legend: c.legend()
        return get_figure(c), get_axis(c)

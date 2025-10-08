import numpy as np
from dataclasses import dataclass, field

from enum import Enum
from ..helpers import *

from ..debug import get_debug

# =========================
# Types / Utilities
# =========================

class KalmanStateType(Enum):
    def __int__(self): return self.value
    def __str__(self): return self.name.lower().capitalize()

class KalmanStateFlavor(Enum):
    SAMPLED   = 0
    PRIOR     = 1
    POSTERIOR = 2
    SMOOTHED  = 3
    def __int__(self): return self.value
    def __str__(self): return self.name.lower().capitalize()

class KalmanObservationFlavor(Enum):
    IDEAL     = 0
    TRUE      = 1
    MEASURED  = 2
    def __int__(self): return self.value
    def __str__(self): return self.name.lower().capitalize()

# =========================
# Record & Collection
# =========================
@dataclass(frozen=True)
class KalmanStateRecord:
    """
    Posterior-centric record:
      - time: (N,)
      - zk:   (N, m)  measurement vector supplied/used by the filter
      - xk:   (N, n)  state estimate sequence (posterior/prior/smoothed/etc.)
      - Pk:   (N, n, n) state covariance sequence aligned with xk
    """
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    xk:   np.ndarray = field(default_factory=lambda: np.array([])) # state vector
    Pk:   np.ndarray = field(default_factory=lambda: np.array([])) # state covariance
    zk:   np.ndarray = field(default_factory=lambda: np.array([])) # measurement vector

    model: object = None # KalmanStateModel
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
    def duration(self):
        return 0 if self.time is None else len(self.time) * (self.time[1] - self.time[0])
    @property
    def sample_rate(self):
        return 1 / (self.time[1] - self.time[0]) if self.time is not None and len(self.time) > 1 else 0

    def __len__(self):
        return len(self.time)

    @staticmethod
    def __freeze_array(a):
        a = np.array(a, copy=True)
        a.setflags(write=False)
        return a

    def __post_init__(self):
        object.__setattr__(self, "time", self.__freeze_array(self.time))
        object.__setattr__(self, "zk",   self.__freeze_array(self.zk))
        object.__setattr__(self, "xk",   self.__freeze_array(self.xk))
        object.__setattr__(self, "Pk",   self.__freeze_array(self.Pk))

        object.__setattr__(self, "model", self.model)
        assert self.model is None or hasattr(self.model, "NAME"), f"'model' must be a KalmanStateModel instance, got {type(self.model)}"
        object.__setattr__(self, "flavor", self.flavor)
        assert self.flavor is None or isinstance(self.flavor, KalmanStateFlavor), f"'flavor' must be of type KalmanStateFlavor, got {type(self.flavor)}"

    @staticmethod
    def _normalize_index(idx, size=None):
        """Convert int, Enum, or tuple/list/array to np.array of ints."""
        if isinstance(idx, (tuple, list, np.ndarray)):
            return np.array([i.value if hasattr(i, "value") else int(i) for i in idx])
        else:
            i = idx.value if hasattr(idx, "value") else int(idx)
            if size is not None:
                return np.array([i] * size)
            return np.array([i])

    def _normalize_component(self, component, type: KalmanStateType = None):
        """
        Normalize component to a tuple of length `dim`.

        Rules:
        - If a single value is passed, repeat it `dim` times.
        - If a list/tuple/array of length < dim, repeat the last element to reach dim.
        - If length > dim, truncate to dim.
        - Always returns a tuple of length dim.
        """
        if type is not None and not issubclass(type, Enum):
            raise ValueError(f"'type' must be a subclass of Enum, got {type}.")

        if not isinstance(component, (list, tuple, np.ndarray)):
            components = [component]
        else:
            components = list(component)

        if not components:
            components = [0]

        # Extend or truncate to match dim
        if len(components) < self.dim:
            components += [components[-1]] * (self.dim - len(components))
        elif len(components) > self.dim:
            components = components[:self.dim]

        # Convert all to enum
        if type is not None:
            components = [str2enum(c, type) for c in components]

        return tuple(components)

    @staticmethod
    def _normalize_in(x, indexing = None):
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

        x = np.asarray(x)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)

        if indexing == "ij":
            x = np.transpose(x, axes=(1, 0))

        return x
    
    @staticmethod
    def _normalize_out(x, indexing = None):
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
        
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[-1] == 1:
            x = np.squeeze(x, axis=-1)

        if indexing == "ij":
            x = np.transpose(x, axes=(1, 0))

        return x
        
    def _normalize_idx(self, idx: KalmanStateType, axis = 0):
        """
        Returns the list of state indices that contribute to measurement[meas_idx].
        """
        idx = idx.value if hasattr(idx, 'value') else int(idx)
        idx = self.dim * axis + idx

        assert 0 <= idx < self.dim * self.m, f"Measurement index out of range. Expected 0 <= idx < {self.dim * self.m}, got {idx}."
        return idx

    def print(self, **kwargs):
        print(self.__str__(**kwargs))

    def __repr__(self):
        flav = (str(self.flavor) if self.flavor is not None else "Unknown")
        model = (self.model.NAME if getattr(self.model, "NAME", None) else "Model?")
        return f"KalmanStateRecord(N={len(self)}, n={self.n}, m={self.m}, flavor={flav}, model={model})"
        
    def __str__(self):
        import textwrap, numpy as np
        
        # ---- local helpers -------------------------------------------------------
        def _shape(x):
            if x is None:
                return "None"
            try:
                s = "x".join(map(str, np.shape(x)))
            except Exception:
                s = f"Unsupported({type(x).__name__})"
            return s
        
        def _dims_line(label, arr, extra="", ndots=0):
            if arr is None:
                return f"{label}: None{(' ' + extra) if extra else ''}"

            # Display as X0 x (X1 x X2 x ...) for any ndim >= 2
            dt = f" dtype={arr.dtype}" if isinstance(arr, np.ndarray) else ""

            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                shape_str = f"{arr.shape[0]}x(" + "x".join(str(d) for d in arr.shape[1:]) + ")"
            else:
                shape_str = _shape(arr)

            dots = "." * ndots
            return f"({label}) {dots}: {shape_str} {dt}{(' ' + extra) if extra else ''}"

        def _stats1d(a):
            if a is None or len(a) == 0:
                return "empty"
            a = np.asarray(a)
            finite = np.isfinite(a)
            if not finite.any():
                return "all non-finite"
            return f"min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}, mean={np.nanmean(a):.6g}"

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
        t_stats = _stats1d(self.time)

        # time metrics
        approx_fs = None
        if N >= 2 and self.time is not None:
            dt = np.diff(self.time)
            finite_dt = dt[np.isfinite(dt)]
            if finite_dt.size:
                mean_dt = float(np.mean(finite_dt))
                approx_fs = (1.0/mean_dt) if mean_dt > 0 else None
        
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
        fs_model   = getattr(self.model, "fs", None)

        # optional long description if the model has one
        description = getattr(self.model, "DESCRIPTION", None) or ""
        if description:
            description = textwrap.fill(description, width=60, subsequent_indent=" " * 8)

        spacer = " " * 4
        s = []

        # ---- header --------------------------------------------------------------
        s.append(f"{self.__class__.__name__}:")
        s.append("")
        s.append(f"{spacer}{model_name} (v.{model_ver})")
        if description:
            s.append(f"{2*spacer}{description}")
        s.append("")

        # ---- high-level summary --------------------------------------------------
        s.append(f"{spacer}Summary:")
        s.append(f"{2*spacer}Samples (N).......................................: {N}")
        s.append(f"{2*spacer}State Dimension (n)...............................: {n}")
        s.append(f"{2*spacer}Observation Dimension (m).........................: {m}")
        s.append(f"{2*spacer}Flavor............................................: {flav}")
        s.append(f"{2*spacer}Duration..........................................: {duration:.6g} s")
        if approx_fs is not None:
            s.append(f"{2*spacer}Sampling Rate.....................................: {approx_fs:.6g} Hz")
        if fs_model is not None:
            s.append(f"{2*spacer}Model Sampling Rate...............................: {fs_model} Hz")
        s.append(f"{2*spacer}Time Stats........................................: {t_stats}")
        s.append("")
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

        s.append(f"{2*spacer}Time {_dims_line('t', self.time, ndots=41)}{_bad_str(nan_time, inf_time)}")
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
        return "Time (s)" # always seconds, but could be overridden later

    def _plot_labels(self, component : KalmanStateType, axis = None, units = False, **kwargs):

        labels = kwargs.get("labels", None) or []
        label = labels[int(component)] if component in labels else None
        label = label or (str(component)+" " if isinstance(component, Enum) else f"Component #{int(axis)} ") if isinstance(axis, int) else ""

        model = kwargs.get("model", None) or self.model

        # Mix in axis label if axis is provided
        if axis is not None:
            if model is not None and hasattr(model, "axis"):
                axis_label = model.axis[axis % len(model.axis)]
                label = f"{label}{axis_label}"

        # Mix in units if requested
        if isinstance(units, str) and units:
            label = f"{label} ({units})"
        elif units:
            units_val = self._plot_units(component, model=model)
            label = f"{label} ({units_val})" if units_val else label

        return label

    def _plot_units(self, component : KalmanStateType, default = "∅", **kwargs):
        model = kwargs.get("model", None) or self.model
        units = kwargs.get("units", None) or model.units if model else {}
        return units[str(component)] if str(component) in units else default

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
        if str(flavor) == str(KalmanObservationFlavor.MEASURED):
            _kwargs["alpha"] = 2./3 * _kwargs["alpha"]


        color = kwargs.pop("color", None)
        if isinstance(color, int) and color > 0 and str(flavor) == str(KalmanObservationFlavor.IDEAL) and common:
            color = "black"
        else:
            if not isinstance(color, str):
                color = colors[color % len(colors)] if colors and isinstance(color, int) and len(colors) > 0 else "gray"
            if str(flavor) == str(KalmanObservationFlavor.MEASURED):
                color = lighten(fade(color))
            elif str(flavor) in (str(KalmanObservationFlavor.TRUE), str(KalmanObservationFlavor.IDEAL)):
                color = lighten(vivid(color))
            elif str(flavor) == str(KalmanStateFlavor.PRIOR):
                color = darken(color, 0.25)
            elif str(flavor) in (str(KalmanStateFlavor.POSTERIOR), str(KalmanStateFlavor.SMOOTHED)):
                color = darken(color)

        _kwargs["color"] = color
    
        for key, val in kwargs.items():
            if key not in _kwargs and val is not None:
                _kwargs[key] = val

        if kind != "scatter":
            _kwargs["linestyle"] = kwargs.get("linestyle", None)
            if str(flavor) == str(KalmanObservationFlavor.TRUE):
                _kwargs["linewidth"] = kwargs.get("linewidth", 1.5)
                _kwargs["linestyle"] = _kwargs["linestyle"] or kwargs.get("linestyle", "--")
            elif str(flavor) == str(KalmanObservationFlavor.IDEAL):
                _kwargs["linewidth"] = kwargs.get("linewidth", 1.5)
                _kwargs["linestyle"] = _kwargs["linestyle"] or kwargs.get("linestyle", ":")

            _kwargs["linestyle"] = _kwargs["linestyle"] or kwargs.get("linestyle", "-")

        return _kwargs

    # ---------- Arrow helpers (N/4 or N/5 placement) ----------
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

    def _draw_arrows_1d(self, ax, t, y, color, arrows=None, lw=1):
        idxs = self._arrow_indices(len(t), arrows=arrows)
        for j in idxs:
            ax.annotate(
                "", xy=(t[j+1], y[j+1]), xytext=(t[j], y[j]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0)
            )

    def _draw_arrows_2d(self, ax, x, y, color, arrows=None, lw=1):
        idxs = self._arrow_indices(len(x), arrows=arrows)
        for j in idxs:
            ax.annotate(
                "", xy=(x[j+1], y[j+1]), xytext=(x[j], y[j]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0)
            )

    def _draw_arrows_3d(self, ax, x, y, z, color, arrows=None, lw=1):
        """
        Use 3D quivers between j -> j+1 for selected indices.
        """
        import numpy as np
        idxs = self._arrow_indices(len(x), arrows=arrows)
        if not idxs:
            return
        xs = np.array([x[j] for j in idxs])
        ys = np.array([y[j] for j in idxs])
        zs = np.array([z[j] for j in idxs])
        us = np.array([x[j+1] - x[j] for j in idxs])
        vs = np.array([y[j+1] - y[j] for j in idxs])
        ws = np.array([z[j+1] - z[j] for j in idxs])
        ax.quiver(xs, ys, zs, us, vs, ws, color=color, linewidth=lw,
                  arrow_length_ratio=0.25, length=1.0, normalize=False)

    def plot(self, **kwargs):
        """
        Unified plotting entrypoint using kwargs only.

        kwargs:
        - view:        "auto" | "1d" | "2d" | "3d"   (default: "auto")
        - component:   int | Enum | sequence of them (optional)
        - dim:         int  | sequence of ints        (optional)
        - type:        Enum subclass for component normalization (optional)
        - ...          any other kwargs forwarded to _plot1D/_plot2D/_plot3D
        """
        if self.time is None or len(self.time) < 1:
            return None, None

        # ----------------------- pull kwargs with defaults ------------------------
        component  = kwargs.pop("component", None)
        dim       = kwargs.pop("dim", None)
        type  = kwargs.pop("type", getattr(self.model, "type", None))

        view       = kwargs.pop("view", "auto")
        def _is_single(x):
            if x is None: return False
            return not isinstance(x, (list, tuple)) or (isinstance(x, (list, tuple)) and len(x) == 1)

        if view == "auto" and _is_single(component) and _is_single(dim):
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

        def _norm_component(comp, need, enum_t):
            # makes a tuple of length 'need'; accepts ints/enums/strings
            if comp is None:
                lst = [0] * need
            elif isinstance(comp, (list, tuple)):
                lst = list(comp)
            else:
                lst = [comp]

            # pad or truncate
            if len(lst) < need:
                lst += [lst[-1]] * (need - len(lst))
            elif len(lst) > need:
                lst = lst[:need]

            # convert to enum if type given
            if enum_t is not None:
                if not issubclass(enum_t, Enum):
                    raise ValueError("'type' must be an Enum subclass")
                try:
                    lst = [lst[i] if isinstance(lst[i], enum_t) else str2enum(lst[i], enum_t) for i in range(len(lst))]
                except Exception:
                    # fallback: try enum_t(value)
                    for i, v in enumerate(lst):
                        if not isinstance(v, enum_t):
                            try:
                                lst[i] = enum_t(v)
                            except Exception:
                                # last resort: by name
                                if isinstance(v, str):
                                    lst[i] = enum_t[v.upper()]
                                else:
                                    raise
            return tuple(lst)

        # ----------------------- normalize dim/component/view -------------------
        dim_available = self.dim or 1
        dim = _norm_dim(dim, dim_available)
        view = _decide_view(view, len(dim))
        need = _need_by_view(view) or len(dim)

        # if dim shorter than need, extend with 0,1,2…; if longer, trim
        base_axes = [0, 1, 2]
        if len(dim) < need:
            dim = list(dim) + base_axes[len(dim):need]
        elif len(dim) > need:
            dim = list(dim[:need])

        component = _norm_component(component, need, type)

        # final sanity
        if len(component) != need or len(dim) != need:
            raise RuntimeError("Normalization failed to align component and dim.")

        # ----------------------- dispatch -----------------------------------------
        if view == "3d":
            fig, ax = self._plot3D(component, dim=dim, type=type, **kwargs)
        elif view == "2d":
            fig, ax = self._plot2D(component, dim=dim, type=type, **kwargs)
        else:
            fig, ax = self._plot1D(component, dim=dim, type=type, **kwargs)

        if ax.get_title() == "":
            ax.set_title(kwargs.pop("title", self.model.name() if self.model else "Generic Kalman Model"))

        return fig, ax

    def _plot1D(self, component, **kwargs):
        """
        Plot 1D time series for one component across one or more spatial axes.

        Args (kwargs):
        dim: None | int | [int, ...]
            - None -> plot all axes [0..self.dim-1]
            - int  -> plot that axis
            - list -> plot the listed axes (e.g., [0,2])
        type: Optional component interpretation passed to _normalize_component
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
        kind     = kwargs.pop("type", None)
        legend   = kwargs.pop("legend", True)
        sharey   = kwargs.pop("sharey", False)
        units    = kwargs.pop("units", True)
        grid     = kwargs.pop("grid", True)
        s_size   = kwargs.get("s", 1)

        legend_loc = kwargs.get("legend_loc", "best")
        arrows_ct = kwargs.get("arrows", None)

        sample      = kwargs.pop("sample", None)
        ideal_in    = kwargs.pop("ideal", None)
        truth_in    = kwargs.pop("truth", None)
        meas_in     = kwargs.pop("measurement", None)
        commons     = kwargs.pop("commons", [])

        # Derive defaults like your original
        ideal = ideal_in or (self.model.ideal(duration=self.duration) if self.model else None)
        ideal = KalmanStateRecord._normalize_out(ideal) if (sample is not None and ideal is not None) else None

        truth = truth_in or (sample.xk if sample is not None else None)
        truth = KalmanStateRecord._normalize_out(truth) if (sample is not None and truth is not None) else None

        measurement = meas_in or (sample.zk if sample is not None else None)
        measurement = KalmanStateRecord._normalize_out(measurement) if (sample is not None and measurement is not None) else None

        # Canvas
        c = kwargs.get("canvas", None) or get_canvas(**kwargs)
        plt.sca(c)

        # dims → list[int]
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
        component = self._normalize_component(component, type=kind)[0]
        comp_id   = int(component.value)

        # Base color id; allow passing a concrete color via kwargs['color']
        base_color_id = len(get_lines(**kwargs))

        # ----------------- plotting loop -----------------
        handles, labels = [], []
        used_label = set()

        for i, a in enumerate(axes):
            idx = self.dim * a + comp_id

            # measurement
            if measurement is not None and self.model is not None and self.model.exclusive_mapping(idx, a):
                h = c.scatter(self.time, measurement[:, a],
                            **self._plot_kwargs(flavor=KalmanObservationFlavor.MEASURED,
                                                axis=a, kind="scatter",
                                                s=s_size, common=self.flavor in commons,
                                                color=kwargs.get("color", base_color_id + i)))

            # ideal
            if ideal is not None and ideal.shape[1] > idx:
                h, = c.plot(self.time, ideal[:, idx],
                            **self._plot_kwargs(flavor=KalmanObservationFlavor.IDEAL,
                                                axis=a, common=KalmanObservationFlavor.IDEAL in commons,
                                                color=kwargs.get("color", base_color_id + i)))
                self._draw_arrows_1d(c, self.time, ideal[:, idx], color=h.get_color(), arrows=arrows_ct)

            # truth
            if truth is not None and truth.shape[1] > idx:
                h, = c.plot(self.time, truth[:, idx],
                            **self._plot_kwargs(flavor=KalmanObservationFlavor.TRUE,
                                                axis=a, common=KalmanObservationFlavor.TRUE in commons,
                                                color=kwargs.get("color", base_color_id + i)))
                self._draw_arrows_1d(c, self.time, truth[:, idx], color=h.get_color(), arrows=arrows_ct)

            # filtered/state (main)
            h, = c.plot(self.time, self.xk[:, idx],
                        **self._plot_kwargs(flavor=self.flavor, axis=a, common=self.flavor in commons,
                                            color=kwargs.get("color", base_color_id + i)))
            # label once under flavor name
            self._draw_arrows_1d(c, self.time, self.xk[:, idx], color=h.get_color(), arrows=arrows_ct)

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
        kind       = kwargs.pop("type", None)
        legend     = kwargs.pop("legend", True)
        units      = kwargs.pop("units", True)
        grid       = kwargs.pop("grid", True)
        s_size     = kwargs.get("s", 1)
        legend_loc = kwargs.get("legend_loc", "best")
        arrows_ct  = kwargs.get("arrows", None)
        commons    = kwargs.get("commons", [])

        # components: need two
        if not isinstance(component, (list, tuple, np.ndarray)) or len(component) < 2:
            raise ValueError("plot2D requires component[0] and component[1].")

        # normalize both components
        comp_norm = self._normalize_component(component, type=kind)
        comp0_id  = int(comp_norm[0].value)
        comp1_id  = int(comp_norm[1].value)

        # dims → two axes (x-axis index, y-axis index)
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

        # indices into state arrays following your convention: idx = self.dim * axis + component_id
        idx_x = self.dim * ax_x + comp0_id
        idx_y = self.dim * ax_y + comp1_id

        # --- inputs like _plot1D ---
        sample       = kwargs.pop("sample", None)
        ideal_in     = kwargs.pop("ideal", None)
        truth_in     = kwargs.pop("truth", None)
        meas_in      = kwargs.pop("measurement", None)

        ideal = ideal_in or (self.model.ideal(duration=self.duration) if self.model else None)
        ideal = KalmanStateRecord._normalize_out(ideal) if (sample is not None and ideal is not None) else None

        truth = truth_in or (sample.xk if sample is not None else None)
        truth = KalmanStateRecord._normalize_out(truth) if (sample is not None and truth is not None) else None

        measurement = meas_in or (sample.zk if sample is not None else None)
        measurement = KalmanStateRecord._normalize_out(measurement) if (sample is not None and measurement is not None) else None

        # --- canvas ---
        c = kwargs.get("canvas", None) or get_canvas(**kwargs)
        plt.sca(c)
        colorid = len(get_lines(**kwargs))
        base_color = kwargs.get("color", colorid)

        # --- styles ---
        kw_main  = self._plot_kwargs(flavor=self.flavor, axis=ax_x, color=base_color, common=self.flavor in commons)
        kw_true  = self._plot_kwargs(flavor=KalmanObservationFlavor.TRUE,  axis=ax_x, color=base_color, common=KalmanObservationFlavor.TRUE in kwargs.get("commons", []))
        kw_ideal = self._plot_kwargs(flavor=KalmanObservationFlavor.IDEAL, axis=ax_x, color=base_color, common=KalmanObservationFlavor.IDEAL in kwargs.get("commons", []))
        kw_meas  = self._plot_kwargs(flavor=KalmanObservationFlavor.MEASURED, axis=ax_x,
                                     kind="scatter", s=s_size, color=base_color, alpha=kwargs.get("alpha", 0.7),
                                     common=KalmanObservationFlavor.MEASURED in kwargs.get("commons", []))

        handles, labels = [], []

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

        if legend and handles:
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
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        t = self.time
        if t is None or len(t) == 0:
            return None, None

        # --- normalize components (need 3) ---------------------------------------
        type_ = kwargs.pop("type", None)
        comps = self._normalize_component(component, type=type_)
        if len(comps) < 3:
            raise ValueError("plot3D requires component[0], component[1], component[2].")

        c0, c1, c2 = int(comps[0].value), int(comps[1].value), int(comps[2].value)

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

        # indices in flattened state (idx = self.dim*axis + comp_id)
        idx_x = self.dim * dx + c0
        idx_y = self.dim * dy + c1
        idx_z = self.dim * dz + c2

        # --- derive series like in _plot1D/_plot2D --------------------------------
        sample      = kwargs.pop("sample", None)
        ideal_in    = kwargs.pop("ideal", None)
        truth_in    = kwargs.pop("truth", None)
        meas_in     = kwargs.pop("measurement", None)

        ideal = ideal_in or (self.model.ideal(duration=self.duration) if self.model else None)
        ideal = KalmanStateRecord._normalize_out(ideal) if (sample is not None and ideal is not None) else None

        truth = truth_in or (sample.xk if sample is not None else None)
        truth = KalmanStateRecord._normalize_out(truth) if (sample is not None and truth is not None) else None

        measurement = meas_in or (sample.zk if sample is not None else None)
        measurement = KalmanStateRecord._normalize_out(measurement) if (sample is not None and measurement is not None) else None

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
        ax = fig.add_subplot(projection="3d")
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
        kw_main  = self._plot_kwargs(flavor=self.flavor,                      common=self.flavor in commons, axis=dx, color=base_color)
        kw_true  = self._plot_kwargs(flavor=KalmanObservationFlavor.TRUE,     common=KalmanObservationFlavor.TRUE in commons,  axis=dx, color=base_color)
        kw_ideal = self._plot_kwargs(flavor=KalmanObservationFlavor.IDEAL,    common=KalmanObservationFlavor.IDEAL in commons, axis=dx, color=base_color)
        kw_meas  = self._plot_kwargs(flavor=KalmanObservationFlavor.MEASURED, common=KalmanObservationFlavor.MEASURED in commons,
                                    axis=dx, color=base_color, kind="scatter", s=s_size, alpha=alpha)

        # Aux (projections/connectors) – route through _plot_kwargs but we’ll keep legend proxies in base color
        def kw_proj_state(axis):
            kw = self._plot_kwargs(flavor=self.flavor, axis=axis, color=axis+1, alpha=0.65)
            axis_str = self.model.axis[axis] if self.model else "#" + str(axis+1)
            kw["label"] = f"Projection {axis_str}"
            return kw
        def kw_proj_true(axis):
            kw = self._plot_kwargs(flavor=KalmanObservationFlavor.TRUE, common=KalmanObservationFlavor.TRUE in commons, axis=axis, color=axis+1, alpha=0.65)
            axis_str = self.model.axis[axis] if self.model else "#" + str(axis+1)
            kw["label"] = f"Projection {axis_str}"
            return kw
        def kw_proj_ideal(axis):
            kw = self._plot_kwargs(flavor=KalmanObservationFlavor.IDEAL, common=KalmanObservationFlavor.IDEAL in commons, axis=axis, color=axis+1, alpha=0.65)
            axis_str = self.model.axis[axis] if self.model else "#" + str(axis+1)
            kw["label"] = f"Projection {axis_str}"
            return kw
        def kw_meas_proj(axis):
            kw = self._plot_kwargs(flavor=KalmanObservationFlavor.MEASURED, common=KalmanObservationFlavor.MEASURED in commons, axis=axis, color=axis+1, kind="scatter", s=s_size, alpha=alpha)
            axis_str = self.model.axis[axis] if self.model else "#" + str(axis+1)
            kw["label"] = f"Projection {axis_str}"
            return kw
        def kw_conn(axis):
            return self._plot_kwargs(flavor=self.flavor, axis=axis, color=axis+1, linewidth=0.6, alpha=0.6)

        # --- measurement scatter --------------------------------------------------
        if measurement is not None and self.model is not None:
            mx_ok = self.model.exclusive_mapping(idx_x, dx)
            my_ok = self.model.exclusive_mapping(idx_y, dy)
            mz_ok = self.model.exclusive_mapping(idx_z, dz)
            if mx_ok and my_ok and mz_ok and measurement.shape[1] > max(dx, dy, dz):
                ax.scatter(measurement[:, dx], measurement[:, dy], measurement[:, dz],
                        depthshade=False, **kw_meas)
            if mx_ok and my_ok:
                ax.scatter(measurement[:, dx], measurement[:, dy], np.full_like(measurement[:, dz], z0),
                        depthshade=False, **kw_meas_proj(dz))
            if mx_ok and mz_ok:
                ax.scatter(measurement[:, dx], np.full_like(measurement[:, dy], ylim[1]), measurement[:, dz],
                        depthshade=False, **kw_meas_proj(dy))
            if my_ok and mz_ok:
                ax.scatter(np.full_like(measurement[:, dx], xlim[0]), measurement[:, dy], measurement[:, dz],
                        depthshade=False, **kw_meas_proj(dx))

        # --- ancillary series -----------------------------------------------------
        if IX is not None:
            h_ideal, = ax.plot(IX, IY, IZ, **kw_ideal)
            self._draw_arrows_3d(ax, IX, IY, IZ, color=h_ideal.get_color(), arrows=arrows_ct)
        if TX is not None:
            h_truth, = ax.plot(TX, TY, TZ, **kw_true)
            self._draw_arrows_3d(ax, TX, TY, TZ, color=h_truth.get_color(), arrows=arrows_ct)

        # --- subtle reference box lines (neutral) --------------------------------
        ref_style = dict(color="black", linestyle=":", linewidth=0.75, alpha=0.8)
        if xlim[0] < x0 < xlim[1]:
            ax.plot([xlim[0], xlim[1]], [y0, y0], [z0, z0], **ref_style)
        if ylim[0] < y0 < ylim[1]:
            ax.plot([x0, x0], [ylim[0], ylim[1]], [z0, z0], **ref_style)
        if zlim[0] < z0 < zlim[1]:
            ax.plot([x0, x0], [y0, y0], [zlim[0], zlim[1]], **ref_style)

        # --- projections (drawn; legend will show proxies in base color) ----------
        if projectors:
            if X is not None and Y is not None and Z is not None:
                ax.plot(X, Y, np.full_like(Z, z0), **kw_proj_state(dz))
                ax.plot(X, np.full_like(X, ylim[1]), Z, **kw_proj_state(dy))
                ax.plot(np.full_like(Y, xlim[0]), Y, Z, **kw_proj_state(dx))
            if IX is not None:
                ax.plot(IX, IY, np.full_like(IZ, z0), **kw_proj_ideal(dz))
                ax.plot(IX, np.full_like(IX, ylim[1]), IZ, **kw_proj_ideal(dy))
                ax.plot(np.full_like(IY, xlim[0]), IY, IZ, **kw_proj_ideal(dx))
            if TX is not None:
                ax.plot(TX, TY, np.full_like(TZ, z0), **kw_proj_true(dz))
                ax.plot(TX, np.full_like(TX, ylim[1]), TZ, **kw_proj_true(dy))
                ax.plot(np.full_like(TY, xlim[0]), TY, TZ, **kw_proj_true(dx))

        # --- connectors -----------------------------------------------------------
        if connectors and X is not None and len(X) > 0:
            for j in (0, len(X) - 1):
                xi, yi, zi = float(X[j]), float(Y[j]), float(Z[j])
                ax.plot([xi, xi], [yi, yi], [z0, zi], **kw_conn(dz))
                ax.plot([xi, xi], [ylim[0], yi], [zi, zi], **kw_conn(dy))
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
                Line2D([0], [0], color=base_col, linestyle=main_ls, linewidth=main_lw, label=str(self.flavor) + " 3D"),
                Line2D([0], [0], **kw_meas_projx),
                Line2D([0], [0], **kw_meas_projy),
                Line2D([0], [0], **kw_meas_projz),
            ]
            ax.legend(proxies, [h.get_label() for h in proxies],
                    fontsize=10, handlelength=4, handleheight=0.01, loc=legend_loc)

        return get_figure(ax), get_axis(ax)

    def plot_error(self, flavor: KalmanStateFlavor | KalmanObservationFlavor = KalmanObservationFlavor.MEASURED, component=None, dim=None, **kwargs):
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
            ideal = KalmanStateRecord._normalize_out(ideal) if ideal is not None else None

            truth = truth or getattr(sample, "xk", None)
            truth = KalmanStateRecord._normalize_out(truth) if truth is not None else None

            measurement = measurement or getattr(sample, "zk", None)
            measurement = KalmanStateRecord._normalize_out(measurement) if measurement is not None else None
        else:
            # No sample: normalize whatever was provided
            if ideal is not None:       ideal = KalmanStateRecord._normalize_out(ideal)
            if truth is not None:       truth = KalmanStateRecord._normalize_out(truth)
            if measurement is not None: measurement = KalmanStateRecord._normalize_out(measurement)

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
        if isinstance(flavor, KalmanObservationFlavor) and component is None:
            # Measurement error: uses 'measurement' as zk_hat when available
            fig, ax = self._plot_error_measurement(
                flavor=flavor, axes=axes, zk0=measurement,
                canvas=c, legend=legend, color=kwargs.pop("color", colorid), **kwargs)

            if ax.get_title() == "":
                ax.set_title(kwargs.pop("title", self.model.name() if self.model else "Generic Kalman Model"))
            return fig, ax

        if component is not None:
            # State component error (requires reference)
            enum_t = kwargs.pop("type", getattr(self.model, "flavor", None))
            if enum_t is not None and isinstance(enum_t, flavor) and issubclass(enum_t, Enum):
                try:
                    component = _as_enum(component, enum_t)
                except Exception:
                    component = component  # fallback

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
        for a in axes:
            if a >= self.zk.shape[1] or a >= zk0.shape[1]:
                continue
            err = self.zk[:, a] - zk0[:, a]
            ymin = min(ymin, np.nanmin(err))
            ymax = max(ymax, np.nanmax(err))

            if kind == "hist":
                # draw histogram and capture counts/edges (counts needed for scaling)
                counts, edges, _ = c.hist(err, bins=bins, **self._plot_kwargs(flavor=flavor, axis=a))

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

                fit_kw = self._plot_kwargs(color=kwargs.pop("color", "red"), axis=a)

                if legend:
                    fit_kw["label"] = f"Gaussian Fit\n(μ={mu:.3g}, σ={sigma:.3g})"
                c.plot(x, y, **fit_kw)

            else:
                c.plot(self.time, err, **self._plot_kwargs(flavor=flavor, axis=a, color=kwargs.pop("color", None)))
        
        c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
        if kind == "hist":
            c.set_title(kwargs.pop("ylabel", "Absolute Measurement Error"))
            c.set_ylabel(kwargs.pop("xlabel", "Counts"))
        else :
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
            k = component.value if hasattr(component, "value") else int(component)
            return self.dim * int(axis) + int(k)

        for a in axes:
            idx = _state_index(component, a)
            if idx >= self.n or idx >= xk0.shape[1]:
                continue
            err = self.xk[:, idx] - xk0[:, idx]
            ymin = min(ymin, np.nanmin(err))
            ymax = max(ymax, np.nanmax(err))
            c.plot(self.time, err,
                color=self._colors(a, self.flavor, "black"),
                linestyle=self._linestyles(a, self.flavor, "-"),
                label=f"{self._axis(a, component)} error")

        
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

        c.plot(self.time, l2,
            color=self._colors(0, self.flavor, "black"),
            linestyle=self._linestyles(0, self.flavor, "-"),
            label=f"{str(self.flavor)} L2 error")


        c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
        c.set_ylabel(kwargs.pop("ylabel", "Least Square State Errors (L2)"))
        c.grid(kwargs.pop("grid", True))
        if legend: c.legend()
        return get_figure(c), get_axis(c)
    
class KalmanStateCollection:
    
    def __init__(self, records=()):
        # use a list so add/remove/clear work
        self._records = list(records)

    def add(self, record: KalmanStateRecord):
        self._records.append(record)

    def remove(self, record: KalmanStateRecord):
        self._records.remove(record)

    def clear(self):
        self._records.clear()

    def list(self):
        return list(range(len(self._records)))

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index: int) -> KalmanStateRecord:
        if 0 <= index < len(self._records):
            return self._records[index]
        return None

    def __iter__(self):
        return iter(self._records)

    def __repr__(self):
        return f"KalmanStateCollection(N={len(self)})"

    def __str__(self, **kwargs):

        max = kwargs.pop("max", None if get_debug() else 8)
        extra = f"(only display {max})" if max is not None and len(self) > max else ""
        lines = [f"KalmanStateCollection with {len(self)} records: {extra}"]

        n = len(self)
        if max is not None and n > max:
            show = max // 2
            for i, rec in enumerate(self._records[:show]):
                lines.append(f"  {i}: {rec!r}")
            lines.append(f"  ... ({n - max} more records) ...")
            for i, rec in enumerate(self._records[-show:], start=n-show):
                lines.append(f"  {i}: {rec!r}")
        else:
            for i, rec in enumerate(self._records):
                lines.append(f"  {i}: {rec!r}")
        return "\n".join(lines)

    def print(self, **kwargs):
        print(self.__str__(**kwargs))

    def plot(self, **kwargs):
        """
        Overlay plot for all records with one combined legend.
        kwargs forwarded to each record.plot().
        """
        from ..helpers import get_canvas, get_axis, get_figure, get_ylim
        from matplotlib.lines import Line2D

        c = kwargs.pop("canvas", None) or get_canvas(**kwargs)
        legend = kwargs.pop("legend", len(self._records) <= 4)
        handles, labels = [], []

        samples = kwargs.pop("samples", None)

        for i, rec in enumerate(self._records):

            sample = samples[i] if isinstance(samples, (list, tuple, KalmanStateCollection)) and i < len(samples) else samples
            if sample is None: kwargs["sample"] = kwargs.get("sample", None)

            rec.plot(sample=sample, commons=[KalmanObservationFlavor.IDEAL], canvas=c, legend=False, color=i, **kwargs)
            if legend and len(self._records) < kwargs.get("legend_max", 8):
                plt_kwargs = { k: v for k, v in rec._plot_kwargs(color=i, **kwargs).items() if k in ("color", "linewidth", "linestyle") }
                plt_kwargs["color"] = darken(plt_kwargs["color"])
                handles.append(Line2D([0], [0], **plt_kwargs))
                labels.append(f"Record #{i+1}")

        if samples is not None and rec is not None:
            plt_kwargs = { k: v for k, v in rec._plot_kwargs(flavor=KalmanObservationFlavor.IDEAL, common=True, color=1, **kwargs).items() if k in ("color", "linewidth", "linestyle") }
            handles.insert(0, Line2D([0], [0], **plt_kwargs))
            labels.insert(0, str(KalmanObservationFlavor.IDEAL))
        
        if kwargs.get("legend", True) and handles:
            c.legend(handles, labels, fontsize=10, handlelength=4, handleheight=0.01)

        return get_figure(c), get_axis(c)

    def plot_error(self, **kwargs):
        """
        Overlay error plots for all records on a single axes, with one combined legend.

        Parameters (kwargs):
        samples      : None | KalmanStateCollection | list[KalmanStateRecord] | single sample
                        - If a collection/list is provided, samples[i] is passed to record i.
                        - Else the same 'sample' (if provided) is used for all.
        legend       : bool (default: len(self._records) <= 4)
        legend_max   : int  (default: 10)  # max number of legend entries
        legend_loc   : str  (default: "best")
        alpha_decay  : float in (0,1] (default: 0.9)
                        - Multiplies alpha for successive records to reduce clutter.
        color        : optional; if provided, all records share this base color
                        (still routed through _plot_kwargs). Otherwise uses color=i.
        ...          : forwarded to each record.plot_error()
                        (e.g., flavor, component, dim, kind, bins, grid, title, etc.)

        Returns:
        (figure, axis)
        """
        from ..helpers import get_canvas, get_axis, get_figure
        from matplotlib.lines import Line2D
        import copy

        if len(self._records) == 0:
            # Nothing to plot — return an empty canvas
            c = kwargs.pop("canvas", None) or get_canvas(**kwargs)
            return get_figure(c), get_axis(c)

        c          = kwargs.pop("canvas", None) or get_canvas(**kwargs)
        legend     = kwargs.pop("legend", len(self._records) <= 4)
        legend_max = kwargs.pop("legend_max", 10)
        legend_loc = kwargs.pop("legend_loc", "best")
        alpha_decay = float(kwargs.pop("alpha_decay", 0.9))
        alpha_decay = alpha_decay if 0 < alpha_decay <= 1 else 0.9

        # Allow per-record samples (mirrors .plot())
        samples = kwargs.pop("samples", None)

        handles, labels = [], []

        # We’ll not mutate the caller’s kwargs; copy for safety per-record
        for i, rec in enumerate(self._records):
            # Build per-record kwargs
            local = copy.deepcopy(kwargs)
            local["legend"] = False  # consolidate legend here

            # Ensure color coherence:
            # - if user passed an explicit color, keep it
            # - otherwise use color=i
            base_color = kwargs.get("color", i)
            local["color"] = base_color

            # Apply alpha decay to help readability on heavy overlays
            if "alpha" in local:
                try:
                    local["alpha"] = float(local["alpha"]) * (alpha_decay ** i)
                except Exception:
                    pass
            else:
                # Set a gentle default alpha if not specified
                local["alpha"] = 0.95 * (alpha_decay ** i)

            # Per-record sample selection
            if isinstance(samples, (list, tuple, KalmanStateCollection)):
                sample_i = samples[i] if i < len(samples) else None
                if sample_i is not None:
                    local["sample"] = sample_i
            elif samples is not None:
                local["sample"] = samples
            else:
                # Respect a pre-existing 'sample' already in kwargs if present
                if "sample" in kwargs:
                    local["sample"] = kwargs["sample"]

            # Draw the record’s error onto the shared canvas
            rec.plot_error(canvas=c, **local)

            # Construct a legend handle that matches the style used for this record’s error
            # Try to use the same 'flavor' as the one used in plot_error() if provided.
            flavor_for_handle = local.get("flavor", kwargs.get("flavor", rec.flavor))
            plt_kwargs = {
                k: v for k, v in rec._plot_kwargs(
                    flavor=flavor_for_handle,
                    color=base_color,
                    alpha=local.get("alpha", 1.0),
                    **local
                ).items()
                if k in ("color", "linewidth", "linestyle", "alpha")
            }

            handles.append(Line2D([0], [0], **plt_kwargs))
            labels.append(f"Record #{i+1}")

            # Cap legend size if requested
            if legend and len(handles) >= legend_max:
                break

        # Final legend
        if legend and handles:
            c.legend(handles, labels, fontsize=10, handlelength=4, handleheight=0.01, loc=legend_loc)

        return get_figure(c), get_axis(c)
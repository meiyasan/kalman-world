import numpy as np
from dataclasses import dataclass, field

from enum import Enum
from ..helpers import *

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



    # inside KalmanStateRecord
    def print(self):
        print(self.__str__())

    def __repr__(self):
        # short, one-liner representation
        N = len(self)
        n = self.n
        m = self.m
        flav = (str(self.flavor) if self.flavor is not None else "Unknown")
        model = (self.model.NAME if getattr(self.model, "NAME", None) else "Model?")
        return f"KalmanStateRecord(N={N}, n={n}, m={m}, flavor={flav}, model={model})"

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

        def _print_matrix(name, mat, prefix=""):
            if mat is None:
                return ""
            text = str(mat).replace("\n", f"\n{prefix}")
            return f"{prefix}{name} = {text}\n"

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
        monotonic  = "N/A"
        if N >= 2 and self.time is not None:
            dt = np.diff(self.time)
            finite_dt = dt[np.isfinite(dt)]
            if finite_dt.size:
                mean_dt = float(np.mean(finite_dt))
                approx_fs = (1.0/mean_dt) if mean_dt > 0 else None
            monotonic = "Monotonic" if np.all(dt >= 0) else "Not Monotonic"

        # bad counts
        nan_x, inf_x = _count_bad(self.xk)
        nan_p, inf_p = _count_bad(self.Pk)
        nan_z, inf_z = _count_bad(self.zk)

        # covariance quick checks
        P_shape = np.shape(self.Pk)
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
        # Custom model name formatting
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
        s.append(f"{2*spacer}Time Monotonicity.................................: {monotonic}")
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

        # ---- debug dump (optional) ----------------------------------------------
        try:
            from ..helpers import get_debug
        except Exception:
            def get_debug(): return False

        if get_debug():
            s.append(f"{spacer}Debug Dump (first/last rows and small slices):")
            # time
            if self.time is not None and len(self.time) > 0:
                s.append(_print_matrix("time[:10]", self.time[:10], prefix=2*spacer))
                if len(self.time) > 10:
                    s.append(_print_matrix("time[-10:]", self.time[-10:], prefix=2*spacer))
            # xk
            if isinstance(self.xk, np.ndarray) and self.xk.size:
                rows = min(3, self.xk.shape[0])
                s.append(_print_matrix("xk[:3,:]", self.xk[:rows, :], prefix=2*spacer))
                if self.xk.shape[0] > rows:
                    s.append(_print_matrix("xk[-3:,:]", self.xk[-rows:, :], prefix=2*spacer))
            # Pk
            if isinstance(self.Pk, np.ndarray) and self.Pk.size:
                rows = min(2, self.Pk.shape[0])
                s.append(_print_matrix("Pk[:2,:,:]", self.Pk[:rows, :, :], prefix=2*spacer))
            # zk
            if isinstance(self.zk, np.ndarray) and self.zk.size:
                rows = min(3, self.zk.shape[0])
                s.append(_print_matrix("zk[:3,:]", self.zk[:rows, :], prefix=2*spacer))
                if self.zk.shape[0] > rows:
                    s.append(_print_matrix("zk[-3:,:]", self.zk[-rows:, :], prefix=2*spacer))

        np.set_printoptions(**opt)
        return "\n".join(s)

    # Properties for model-dependent information
    def _labels(self, type : KalmanStateType, default = None, **kwargs):
        labels = kwargs.get("labels", None) or []
        label = labels[int(type)] if type in labels else None
        default = default or (f"{str(type)} Component #{int(type)}")
        return label or (str(type) if isinstance(type, Enum) else default)

    def _units(self, type : KalmanStateType, default = "∅", **kwargs):
        model = kwargs.get("model", None) or self.model
        units = kwargs.get("units", None) or model.units if model else {}
        return units[str(type)] if str(type) in units else default

    def _coords(self, axis, type=None, default = None, **kwargs):
        model = kwargs.get("model", None) or self.model
        coords = kwargs.get("coords", None) or (model.coords if model and hasattr(model, "coords") else [])
        coord = coords[int(axis) % len(coords)] if coords else None
        default = (f"{str(type)} Dimension #{int(axis)}") if type else (f"Dimension #{int(axis)}")
        return coord if coord else default

    def _colors(self, axis, type : KalmanStateType, default = "gray", **kwargs):
        model = kwargs.get("model", None) or self.model
        colors = kwargs.get("colors", None) or model.colors if model else {}
        return colors[str(type)].get(axis, None) if str(type) in colors else default

    def _linestyles(self, axis, type : KalmanStateType, default = "-", **kwargs):
        model = kwargs.get("model", None) or self.model
        linestyles = kwargs.get("linestyles", None) or model.linestyles if model else {}
        return linestyles[str(type)].get(axis, None) if str(type) in linestyles else default

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

    def __len__(self):
        return len(self.time)

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

        ax.set_title(kwargs.pop("title", self.model.name() if self.model else "Generic Kalman Model"))
        return fig, ax
    
    def _plot1D(self, component, **kwargs):

        if self.time is None or len(self.time) < 1:
            return None, None

        dim = kwargs.pop("dim", None)
        type = kwargs.pop("type", None)

        # Determine axes parameters
        component = component[0] if isinstance(component, (list, tuple, np.ndarray)) else component
        dim = dim[0] if isinstance(dim, (list, tuple, np.ndarray)) else dim

        sample = kwargs.pop("sample", None)
        ideal = kwargs.pop("ideal", None) or self.model.ideal(duration=self.duration) if self.model else None
        ideal = KalmanStateRecord._normalize_out(ideal) if sample is not None and ideal is not None else None
        truth = kwargs.pop("truth", None) or sample.xk if sample is not None else None
        truth = KalmanStateRecord._normalize_out(truth) if sample is not None and truth is not None else None
        measurement = kwargs.pop("measurement", None) or sample.zk if sample is not None else None
        measurement = KalmanStateRecord._normalize_out(measurement) if sample is not None and measurement is not None else None

        # Determine limits with padding, comparing with current plot limits
        padding = kwargs.pop("padding", 0.05)
        legend = kwargs.pop("legend", True)

        # --- 2D branch ---
        c = kwargs.pop("canvas", None) or get_canvas(**kwargs)

        if dim is None: axes = list(range(self.dim))
        elif isinstance(dim, (list, tuple)): axes = dim
        else: axes = [dim]

        for a in axes:
            if not (0 <= a < self.dim):
                raise ValueError(f"Invalid axis {a} considering {self.dim}D model")

        component = self._normalize_component(component, type=type)[0]  # Use only the first component for 1D plot

        # Initialize y_min and y_max
        current_ylim = c.get_ylim() # Check all ylim from all canvas 
        y_min, y_max = current_ylim if current_ylim else (np.inf, -np.inf)
        for a in axes:

            idx = self.dim * a + int(component.value)
            if measurement is not None and self.model is not None and self.model.exclusive_mapping(idx, a):
                y_min = min(y_min, measurement[:, a].min())
                y_max = max(y_max, measurement[:, a].max())
            if ideal is not None:
                y_min = min(y_min, ideal[:, idx].min())
                y_max = max(y_max, ideal[:, idx].max())
            if truth is not None and truth.shape[1] > idx:
                y_min = min(y_min, truth[:, idx].min())
                y_max = max(y_max, truth[:, idx].max())

            y_min = min(y_min, self.xk[:, idx].min())
            y_max = max(y_max, self.xk[:, idx].max())
            
            y_range = y_max - y_min
            y_min -= padding * y_range
            y_max += padding * y_range

            if "fig" not in kwargs and kwargs.pop("units", True):
                X0 = self._normalize_out(self.xk[0])[idx]
                c.axhline(X0, color='black', alpha=1, linestyle=':', linewidth=0.75)  # x-axis reference
                c.axvline(0, color='black', alpha=1, linestyle=':', linewidth=0.75)  # y-axis reference

            if measurement is not None and self.model is not None and self.model.exclusive_mapping(idx, a):
                color = self._colors(a, KalmanObservationFlavor.MEASURED) or "gray"
                c.scatter(self.time, measurement[:, a], s=1, label=f"{str(KalmanObservationFlavor.MEASURED)} {self._coords(a, KalmanObservationFlavor.MEASURED)}", color=color)

            if ideal is not None:
                c.plot(self.time, ideal[:, idx], linestyle=self._linestyles(a, KalmanObservationFlavor.IDEAL) or "-" , label=f"{str(KalmanObservationFlavor.IDEAL)} {self._coords(a, KalmanObservationFlavor.IDEAL)}", color=self._colors(a, KalmanObservationFlavor.IDEAL, "gray"))
            if truth is not None and truth.shape[1] > idx:
                c.plot(self.time, truth[:, idx], linestyle=self._linestyles(a, KalmanObservationFlavor.TRUE) or "--", label=f"{str(KalmanObservationFlavor.TRUE)} {self._coords(a, KalmanObservationFlavor.TRUE)}", color=self._colors(a, KalmanObservationFlavor.TRUE, "gray"))

            c.plot(self.time, self.xk[:, idx], label=f"{str(self.flavor)} {self._coords(a, self.flavor)}", color=self._colors(a, self.flavor, "gray"))

        c.set_xlim(kwargs.pop("xlim", None))
        c.set_ylim(kwargs.pop("ylim", (y_min, y_max)))

        c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
        if not kwargs.pop("sharey", False):
            if legend:
                c.set_ylabel(kwargs.pop("label", f"{self._labels(component)} ({self._units(component)})"))
            else:
                c.set_ylabel(kwargs.pop("label", f"{self._labels(component)} ({self._units(component)}) along {self._coords(dim or 0)}"))

        c.grid(kwargs.pop("grid", True))
        if legend:
            c.legend(fontsize=10, handlelength=4, handleheight=0.01)

        return get_figure(c), get_axis(c)

    def _plot2D(self, component, **kwargs):
        """
        Plot the ballistic trajectory in 2D (x vs y). Supports multiple components.
        `component` can be a single value, tuple/list, or array of length 2.
        """
        t = self.time
        if t is None or len(t) == 0:
            return None, None

        # Normalize component to a tuple of length 2
        component = self._normalize_component(component, type=kwargs.pop("type", None))
        sample = kwargs.pop("sample", None)
        ideal = kwargs.pop("ideal", None) or self.model.ideal(duration=self.duration) if self.model else None
        ideal = KalmanStateRecord._normalize_out(ideal) if sample is not None and ideal is not None else None
        truth = kwargs.pop("truth", None) or (sample.xk if sample is not None else None)
        truth = KalmanStateRecord._normalize_out(truth) if sample is not None and truth is not None else None
        measurement = kwargs.pop("measurement", None) or (sample.zk if sample is not None else None)
        measurement = KalmanStateRecord._normalize_out(measurement) if sample is not None and measurement is not None else None

        # Common kwargs
        padding = kwargs.pop("padding", 0.05)
        projectors = kwargs.pop("projectors", True)
        connectors = kwargs.pop("connectors", True)

        # Canvas / figure
        fig = kwargs.get("fig") or plt.figure(figsize=kwargs.pop("figsize",(8,8)))
        ax = fig.add_subplot(1,1,1)

        # Map components to state indices
        i0, i1 = self._normalize_idx(component[0], axis=0), self._normalize_idx(component[1], axis=1)
        X,  Y  = self.xk[:, i0], self.xk[:, i1]
        X0, Y0 = self._normalize_out(self.xk[0])[i0], self._normalize_out(self.xk[0])[i1]
        TX, TY = (truth[:, i0], truth[:, i1]) if truth is not None else (None, None)
        IX, IY = (ideal[:, i0], ideal[:, i1]) if ideal is not None else (None, None)

        # Measurements
        if self.model is not None and self.model.exclusive_mapping((i0,i1), component):
            if measurement is not None and measurement.shape[1] >= 2:
                ax.scatter(measurement[:,0], measurement[:,1], s=1, color='black', alpha=0.75, label="Measurements")

        # Determine plot limits
        xmin = min(X.min(), TX.min() if TX is not None else X.min(), IX.min() if IX is not None else X.min())
        xmax = max(X.max(), TX.max() if TX is not None else X.max(), IX.max() if IX is not None else X.max())
        ymin = min(Y.min(), TY.min() if TY is not None else Y.min(), IY.min() if IY is not None else Y.min())
        ymax = max(Y.max(), TY.max() if TY is not None else Y.max(), IY.max() if IY is not None else Y.max())

        # Apply padding
        xpad = padding * (xmax - xmin)
        ypad = padding * (ymax - ymin)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

        # Plot ideal, truth, estimate
        if IX is not None:
            ax.plot(IX, IY, label='Ideal', color=self._colors(0, KalmanObservationFlavor.IDEAL),
                    linestyle=self._linestyles(0, KalmanObservationFlavor.IDEAL))
        if TX is not None:
            ax.plot(TX, TY, label='Truth', color=self._linestyles(0, KalmanObservationFlavor.TRUE, "--"))

        ax.plot(X, Y, label=f"{str(self.flavor)} {self._coords(0, self.flavor)}", color=self._colors(0, self.flavor, "black"), linewidth=1.5)

        # Projectors / connectors (optional)
        if projectors:
            ax.plot(X, [Y0]*len(X), color=self._colors(1, self.flavor, "gray"),
                    linestyle=self._linestyles(1, self.flavor, "-"), alpha=0.5)
            ax.plot([X0]*len(Y), Y, color=self._colors(0, self.flavor, "gray"),
                    linestyle=self._linestyles(0, self.flavor, "-"), alpha=0.5)
            if connectors:
                for j in (0, len(X)-1):
                    ax.plot([X0, X[j]], [Y0, Y[j]], color='gray', linestyle='-.', alpha=0.5, linewidth=0.5)

        # Labels and grid
        ax.set_xlabel(f"{self._coords(0)} {self._labels(component[0])} ({self._units(component[0])})")
        ax.set_ylabel(f"{self._coords(1)} {self._labels(component[1])} ({self._units(component[1])})")
        ax.grid(kwargs.pop("grid", True))

        if kwargs.pop("legend", True): ax.legend(fontsize=10)
        return fig, ax

    def _plot3D(self, component, **kwargs):
        """
        Plot the ballistic trajectory:
          - 3D view (with projections & connectors) if dim>2 and axis=None.
          - Otherwise: 2D plot of Position / Speed / Acceleration vs time.
        """
        t = self.time
        if t is None or len(t) == 0:
            return None, None
        
        # Normalize component to a list of length 3
        component = self._normalize_component(component, type=kwargs.pop("type", None))

        # Determine axes parameter
        sample = kwargs.pop("sample", None)
        ideal = kwargs.pop("ideal", None) or self.model.ideal(duration=self.duration) if self.model else None
        ideal = KalmanStateRecord._normalize_out(ideal) if sample is not None and ideal is not None else None
        truth = kwargs.pop("truth", None) or sample.xk if sample is not None else None
        truth = KalmanStateRecord._normalize_out(truth) if sample is not None and truth is not None else None
        measurement = kwargs.pop("measurement", None) or sample.zk if sample is not None else None
        measurement = KalmanStateRecord._normalize_out(measurement) if sample is not None and measurement is not None else None

        # Common kwargs for plot styling
        plot_kwargs = kwargs.pop("plot", {"color": "black", "linestyle": "-.", "linewidth": 0.5, "alpha": 0.5})

        # Padding factor
        padding = kwargs.pop("padding", 0)

        fig = kwargs.get("fig") or plt.figure(figsize=kwargs.pop("figsize",(10,10)))
        ax = fig.add_subplot(projection='3d')

        i0, i1, i2 = self._normalize_idx(component[0], axis=0), self._normalize_idx(component[1], axis=1), self._normalize_idx(component[2], axis=2)
        X,  Y,  Z  = self.xk[:, i0], self.xk[:, i1], self.xk[:, i2]
        X0, Y0, Z0 = self._normalize_out(self.xk[0])[i0], self._normalize_out(self.xk[0])[i1], self._normalize_out(self.xk[0])[i2]
        TX, TY, TZ = None, None, None
        if truth is not None:
            TX, TY, TZ = truth[:, i0], truth[:, i1], truth[:, i2]
        IX, IY, IZ = None, None, None
        if ideal is not None:
            IX, IY, IZ = ideal[:, i0], ideal[:, i1], ideal[:, i2]

        projectors = kwargs.pop("projectors", True)
        connectors = kwargs.pop("connectors", True)

        if self.model is not None and self.model.exclusive_mapping([0,1,2], component):
            if measurement is not None and measurement.shape[1] >= 3:
                ax.scatter(measurement[:,0], measurement[:,1], measurement[:,2], label='Measurements', color='black', s=1, alpha=0.75, depthshade=False)

        # Ensure X0, Y0, Z0 are within the limit range
        x0, y0, z0 = np.clip(X0, X.min(), X.max()), np.clip(Y0, Y.min(), Y.max()), np.clip(Z0, Z.min(), Z.max())
        
        # Apply padding
        if isinstance(padding, (tuple, list)) and len(padding) == 3:
            xpad, ypad, zpad = padding
        else:
            xpad = ypad = zpad = padding

        xlim = (min(X.min(), TX.min() if TX is not None else X.min(), IX.min() if IX is not None else X.min()) - xpad*(X.max()-X.min()),
                max(X.max(), TX.max() if TX is not None else X.max(), IX.max() if IX is not None else X.max()) + xpad*(X.max()-X.min()))
        ylim = (min(Y.min(), TY.min() if TY is not None else Y.min(), IY.min() if IY is not None else Y.min()) - ypad*(Y.max()-Y.min()),
                max(Y.max(), TY.max() if TY is not None else Y.max(), IY.max() if IY is not None else Y.max()) + ypad*(Y.max()-Y.min()))
        zlim = (min(Z.min(), TZ.min() if TZ is not None else Z.min(), IZ.min() if IZ is not None else Z.min()) - zpad*(Z.max()-Z.min()),
                max(Z.max(), TZ.max() if TZ is not None else Z.max(), IZ.max() if IZ is not None else Z.max()) + zpad*(Z.max()-Z.min()))

        xlim = kwargs.pop("xlim", xlim)
        if xlim[0] != xlim[1]: ax.set_xlim(xlim)
        ylim = kwargs.pop("ylim", ylim)
        if ylim[0] != ylim[1]: ax.set_ylim(ylim)
        zlim = kwargs.pop("zlim", zlim)
        if zlim[0] != zlim[1]: ax.set_zlim(zlim)

        # Lines along XYZ-axis
        if x0 != xlim[0] and x0 != xlim[1]:
            ax.plot([xlim[0], xlim[1]], [y0, y0], [z0, z0], **plot_kwargs)
        if y0 != ylim[0] and y0 != ylim[1]:
            ax.plot([x0, x0], [ylim[0], ylim[1]], [z0, z0], **plot_kwargs)
        if z0 != zlim[0] and z0 != zlim[1]:
            if x0 != xlim[0]:
                ax.plot([xlim[0], xlim[1]], [y0, y0], [zlim[0], zlim[0]], **plot_kwargs)
            if y0 != ylim[0]:
                ax.plot([x0, x0], [ylim[0], ylim[1]], [zlim[0], zlim[0]], **plot_kwargs)

        # Outline XY plane
        if z0 != zlim[0] and z0 != zlim[1]:
            ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[0]], [z0, z0], **plot_kwargs)
            ax.plot([xlim[1], xlim[1]], [ylim[0], ylim[1]], [z0, z0], **plot_kwargs)

        # Projecting lines from Z0 to Zmin
        if z0 != zlim[0]:        
            ax.plot([xlim[0], xlim[0]], [ylim[0], ylim[0]], [zlim[0], z0], **plot_kwargs)
            ax.plot([xlim[0], xlim[0]], [ylim[1], ylim[1]], [zlim[0], z0], **plot_kwargs)
            ax.plot([xlim[1], xlim[1]], [ylim[0], ylim[0]], [zlim[0], z0], **plot_kwargs)
            ax.plot([xlim[1], xlim[1]], [ylim[1], ylim[1]], [zlim[0], z0], **plot_kwargs)
        
            if x0 != xlim[0] and x0 != xlim[1]:
                ax.plot([x0, x0], [ylim[0], ylim[0]], [zlim[0], z0], **plot_kwargs)
                ax.plot([x0, x0], [ylim[1], ylim[1]], [zlim[0], z0], **plot_kwargs)
            if x0 != xlim[0] and x0 != xlim[1] and y0 != ylim[0] and y0 != ylim[1]:
                ax.plot([x0, x0], [y0, y0], [zlim[0], z0], **plot_kwargs)
            if y0 != ylim[0] and y0 != ylim[1]:
                ax.plot([xlim[1], xlim[1]], [y0, y0], [zlim[0], z0], **plot_kwargs)
                ax.plot([xlim[0], xlim[0]], [y0, y0], [zlim[0], z0], **plot_kwargs)

        if ideal is not None:
            ax.plot(IX, IY, IZ, label='Ideal', color=self._colors(0, KalmanObservationFlavor.IDEAL, "black"), linestyle=self._linestyles(0, KalmanObservationFlavor.IDEAL))
        if truth is not None:
            ax.plot(TX, TY, TZ, label='Truth', color=self._colors(0, KalmanObservationFlavor.TRUE, "black"), linestyle=self._linestyles(0, KalmanObservationFlavor.TRUE))

        if projectors:

            z0, y0, x0 = float(Z0), float(ylim[1]), float(xlim[0])

            if X is not None and Y is not None and Z is not None:
                ax.plot(X,           Y,          [Z0]*len(Z), color=self._colors(0, KalmanStateFlavor.POSTERIOR, "gray"), alpha=0.75, linestyle=self._linestyles(0, KalmanStateFlavor.POSTERIOR, "-"))
                ax.plot(X,          [y0]*len(X), Z,           color=self._colors(1, KalmanStateFlavor.POSTERIOR, "gray"), alpha=0.75, linestyle=self._linestyles(1, KalmanStateFlavor.POSTERIOR, "-"))
                ax.plot([x0]*len(Y), Y,          Z,           color=self._colors(2, KalmanStateFlavor.POSTERIOR, "gray"), alpha=0.75, linestyle=self._linestyles(2, KalmanStateFlavor.POSTERIOR, "-"))

            if IX is not None and IY is not None and IZ is not None:
                ax.plot(IX,           IY,          [Z0]*len(IZ), color=self._colors(0, KalmanObservationFlavor.IDEAL, "gray"), alpha=0.75, linestyle=self._linestyles(0, KalmanObservationFlavor.IDEAL, "-"))
                ax.plot(IX,          [y0]*len(IX), IZ,           color=self._colors(1, KalmanObservationFlavor.IDEAL, "gray"), alpha=0.75, linestyle=self._linestyles(1, KalmanObservationFlavor.IDEAL, "-"))
                ax.plot([x0]*len(IY), IY,          IZ,           color=self._colors(2, KalmanObservationFlavor.IDEAL, "gray"), alpha=0.75, linestyle=self._linestyles(2, KalmanObservationFlavor.IDEAL, "-"))

            if TX is not None and TY is not None and TZ is not None:
                ax.plot(TX,           TY,          [Z0]*len(TZ), color=self._colors(0, KalmanObservationFlavor.TRUE, "gray"), alpha=0.75, linestyle=self._linestyles(0, KalmanObservationFlavor.TRUE, "-"))
                ax.plot(TX,          [y0]*len(TX), TZ,           color=self._colors(1, KalmanObservationFlavor.TRUE, "gray"), alpha=0.75, linestyle=self._linestyles(1, KalmanObservationFlavor.TRUE, "-"))
                ax.plot([x0]*len(TY), TY,          TZ,           color=self._colors(2, KalmanObservationFlavor.TRUE, "gray"), alpha=0.75, linestyle=self._linestyles(2, KalmanObservationFlavor.TRUE, "-"))
            
            if connectors:
                for j in (0, len(X)-1):
                    xi, yi, zi = float(X[j]), float(Y[j]), float(Z[j])
                    ax.plot([xi,xi],[yi,yi],[z0,zi], color=self._colors(0, KalmanStateFlavor.POSTERIOR, "gray"), linestyle='-.', linewidth=0.5, alpha=0.75)
                    ax.plot([xi,xi],[y0,yi],[zi,zi], color=self._colors(1, KalmanStateFlavor.POSTERIOR, "gray"), linestyle='-.', linewidth=0.5, alpha=0.75)
                    ax.plot([x0,xi],[yi,yi],[zi,zi], color=self._colors(2, KalmanStateFlavor.POSTERIOR, "gray"), linestyle='-.', linewidth=0.5, alpha=0.75)

        if X is not None and Y is not None and Z is not None:
            ax.plot(X, Y, Z, label=f"{str(self.flavor)}", color=self._colors(0, self.flavor, "black"), linewidth=1.5)
            ax.set_xlabel(kwargs.pop("xlabel", self._coords(0) + " " + self._labels(component[0]) + " (" + self._units(component[0]) + ")"))
            ax.set_ylabel(kwargs.pop("ylabel", self._coords(1) + " " + self._labels(component[1]) + " (" + self._units(component[1]) + ")"))
            ax.set_zlabel(kwargs.pop("zlabel", self._coords(2) + " " + self._labels(component[2]) + " (" + self._units(component[2]) + ")"))
        
        ax.grid(kwargs.pop("grid", True))
        ax.set_box_aspect([1,1,1])
        if kwargs.pop("legend", True):
            ax.legend(fontsize=10, handlelength=4, handleheight=0.01)

        return fig, ax

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
        legend  = kwargs.pop("legend", True)
        padding = kwargs.pop("padding", 0.05)

        # default flavor for labels/colors
        flavor = flavor or self.flavor

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
                canvas=c, legend=legend, padding=padding, **kwargs
            )
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
                canvas=c, legend=legend, padding=padding, **kwargs
            )
            return fig, ax

        # L2 over full state (requires reference)
        if reference is None:
            raise ValueError("Provide 'sample' or ('truth' or 'ideal') to compute L2 state error.")
        fig, ax = self._plot_error_l2(
            xk0=reference, canvas=c, legend=legend, padding=padding, **kwargs
        )
        return fig, ax


    def _plot_error_measurement(self, *, flavor, axes, zk0, canvas, legend, padding, **kwargs):
        """
        Plot zk0 error per axis: zk - zk_hat.
        Here zk_hat is taken from 'zk0' (preferred: sample.zk).
        If 'zk0' is None, we error.
        """
        c = canvas
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
            c.plot(self.time, err,
                color=self._colors(a, flavor, "gray"),
                label=f"{str(flavor)} error @ {self._coords(a, flavor)}")

        if np.isfinite(ymin) and np.isfinite(ymax):
            yr = max(1e-12, ymax - ymin)
            c.set_ylim(ymin - padding*yr, ymax + padding*yr)
        
        c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
        c.set_ylabel(kwargs.pop("ylabel", "Absolute Measurement Error"))
        c.grid(kwargs.pop("grid", True))
        if legend: c.legend()
        return get_figure(c), get_axis(c)

    def _plot_error_state(self, *, component, axes, xk0, canvas, legend, padding, **kwargs):
        """
        Plot per-axis state component error: xk[:, idx] - xk0[:, idx]
        """
        c = canvas
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
                label=f"{self._coords(a, component)} error")

        # cosmetics
        if np.isfinite(ymin) and np.isfinite(ymax):
            yr = max(1e-12, ymax - ymin)
            c.set_ylim(ymin - padding*yr, ymax + padding*yr)
        
        c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
        c.set_ylabel(kwargs.pop("ylabel", f"Absolute {self._labels(component)} Error"))
        c.grid(kwargs.pop("grid", True))
        
        if legend: c.legend()
        return get_figure(c), get_axis(c)

    def _plot_error_l2(self, *, xk0, canvas, legend, padding, **kwargs):
        """
        Plot total L2 state error per time: ||xk - xk0||_2
        """
        c = canvas
        n_cols = min(self.xk.shape[1], xk0.shape[1])
        e  = self.xk[:, :n_cols] - xk0[:, :n_cols]
        l2 = np.sqrt(np.sum(e*e, axis=1))

        c.plot(self.time, l2,
            color=self._colors(0, self.flavor, "black"),
            linestyle=self._linestyles(0, self.flavor, "-"),
            label=f"{str(self.flavor)} L2 error")

        if np.isfinite(l2).any():
            ymin, ymax = np.nanmin(l2), np.nanmax(l2)
            yr = max(1e-12, ymax - ymin)
            c.set_ylim(ymin - padding*yr, ymax + padding*yr)

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
        return f"KalmanStateCollection with {len(self)} records"

    def __str__(self):
        return self.__repr__()
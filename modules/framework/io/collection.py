from ...helpers import *
from ..constants import  KalmanObservationFlavor
from .record import KalmanStateRecord
from ...runtime.debug import get_debug

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
        from ...helpers import get_canvas, get_axis, get_figure, get_ylim
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
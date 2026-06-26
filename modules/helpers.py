from scipy.signal import resample_poly, butter, filtfilt
from fractions import Fraction

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.colors as mcolors
import colorsys
from collections.abc import Iterable, Mapping

from enum import Enum
import random
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

try:
    from colorspacious import cspace_convert
    _HAS_COLORSPACIOUS = True
except ImportError:
    _HAS_COLORSPACIOUS = False

def barycentric_interpolation(*colors, space="rgb"):
    """
    Compute the barycentric (average) color of a collection of colors.

    Parameters
    ----------
    *colors : str | tuple | list | dict
        Color specifications (names, hex codes, RGB tuples, or nested containers).
        Dicts are flattened using their values.
    space : {"rgb","lab"}, default="rgb"
        Color space in which to average:
        - "rgb": arithmetic mean in RGB (numerical average).
        - "lab": perceptual mean in Lab (requires colorspacious).

    Returns
    -------
    tuple of floats
        (R, G, B) each in [0,1].
    """
    def _flatten(items):
        for i in items:
            if isinstance(i, str):
                yield i
            elif isinstance(i, Mapping):
                yield from _flatten(i.values())
            elif isinstance(i, Iterable):
                yield from _flatten(i)
            else:
                yield i

    flat_colors = list(_flatten(colors))
    if not flat_colors:
        raise ValueError("No colors provided")

    # Convert to RGB triples
    rgbs = np.array([mcolors.to_rgb(c) for c in flat_colors])

    if space == "lab":
        if not _HAS_COLORSPACIOUS:
            raise RuntimeError("Lab averaging requires `colorspacious` installed.")
        labs = cspace_convert(rgbs, "sRGB1", "CIELab")
        mean_lab = labs.mean(axis=0)
        mean_rgb = cspace_convert(mean_lab[np.newaxis, :], "CIELab", "sRGB1")[0]
    else:
        mean_rgb = rgbs.mean(axis=0)

    return tuple(np.clip(mean_rgb, 0, 1))

def rainbow(N, l=0.9, s=1):
    """Generate N distinct colors in a rainbow spectrum."""
    return [colorsys.hls_to_rgb(i / N, l, s) for i in range(N)]

def darken(color, percent=0.5):
    """Darken by moving lightness toward 0 (black)."""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = l * (1 - percent)   # interpolate toward 0
    return colorsys.hls_to_rgb(h, l, s)

def lighten(color, percent=0.5):
    """Lighten by moving lightness toward 1 (white)."""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = l + (1 - l) * percent   # interpolate toward 1
    return colorsys.hls_to_rgb(h, l, s)

def fade(color, percent=0.5):
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s = (1 - percent) * s   # interpolate to 0
    return colorsys.hls_to_rgb(h, l, s)

def vivid(color, percent=0.5):
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s = s + (1 - s) * percent   # interpolate to 1
    return colorsys.hls_to_rgb(h, l, s)

def rotate(color, degrees = 30, units="degrees"):
    """Return a color with the hue rotated by the given degrees."""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)

    if units == "radians":
        degrees = degrees * 180 / np.pi

    h = (h + degrees / 360.0) % 1.0
    return colorsys.hls_to_rgb(h, l, s)

def format_shape(x):
    try:
        return "x".join(map(str, np.asarray(x).shape))
    except Exception:
        return str(type(x))

def show_canvas(canvas = None, **kwargs):
    from IPython.display import display
    display(canvas)

def get_canvas(ncols: int = 1, nrows: int = 1, canvas = None, squeeze = True, **kwargs):

    # Normalize graphics input
    canvas = kwargs.pop("canvas", None)
    fig = kwargs.pop("fig", None)
    if fig:
        if canvas: raise ValueError(f"Ambiguous plot provided: you provided both canvas and figures.")
        canvas = fig

    ax = kwargs.pop("ax", None)
    if ax:
        if fig:    raise ValueError(f"Ambiguous plot provided: you provided both figures and axes.")
        if canvas: raise ValueError(f"Ambiguous plot provided: you provided both canvas and axes.")
        canvas = ax

    # Check if specific canvas is passed (Fig, Axis, Tuple, List or Array)
    if isinstance(canvas, Axes):
        canvas = np.array([[canvas]] if not squeeze else [canvas])
        return canvas if nrows * ncols > 1 else canvas[0]

    if hasattr(canvas, "__iter__") and all(isinstance(a, Axes) for a in canvas): # if list/tuple or array containing Axes
        ax = list(canvas)
        canvas = np.array(ax).reshape(nrows, ncols) if not squeeze else np.array(ax).reshape(nrows * ncols)
        return canvas if nrows * ncols > 1 else canvas[0]
    
    if isinstance(canvas, Figure): # if passing a Figure
        
        fig = canvas
        ax = fig.axes

        if len(ax) != nrows * ncols:
            fig.clf()
            ax = [fig.add_subplot(nrows, ncols, idx+1) for idx in range(nrows * ncols)]

        canvas = np.array(ax).reshape(nrows, ncols) if not squeeze else ax
        return canvas if nrows * ncols > 1 else canvas[0]

    fig_kwargs = {}
    for k in ["sharex", "sharey", "squeeze", "width_ratios", "height_ratios", "subplot_kw", "gridspec_kw"]:
        if k in kwargs:
            fig_kwargs[k] = kwargs.pop(k)

    if "fig_kwargs" in kwargs:
        fig_kwargs.update(kwargs.pop("fig_kwargs"))

    figsize = fig_kwargs.pop("figsize", None) or kwargs.pop("figsize", (5,5))
    
    fig_kwargs.setdefault("figsize", (5,5))
    fig_kwargs["figsize"] = (ncols * figsize[0], nrows * figsize[1])

    fig, ax = plt.subplots(nrows, ncols, **fig_kwargs)

    canvas = np.array(ax).reshape(nrows, ncols) if not squeeze else np.array(ax).reshape(nrows * ncols) # canvas output 
    return canvas if nrows * ncols > 1 else canvas[0]

def get_lines(**kwargs):
    fig = kwargs.get("fig", None)
    ax = kwargs.get("ax", None) or kwargs.get("canvas", None) or plt.gca()
    return ax.lines

def get_ylim(**kwargs):
    fig = kwargs.get("fig", None)
    ax = kwargs.get("ax", None) or kwargs.get("canvas", None) or plt.gca()
    return ax.get_ylim() if len(ax.lines) > 0 else None

def get_figure(canvas = None, **kwargs):
    
    canvas = plt.gcf() if canvas is None else canvas
    canvas = get_canvas(squeeze=False, **kwargs) if canvas is None else canvas
    if isinstance(canvas, Figure): # if passing a Figure
        return canvas

    return canvas if isinstance(canvas, Axes) else canvas[0].figure
    
def get_axis(canvas = None, **kwargs):
    fig = get_figure(canvas, **kwargs)
    return fig.axes

def set_canvas_size(canvas = None, size = (8, 6)):
    get_figure(plt.gcf() if canvas is None else canvas).set_size_inches(size)  # Set to specified size

def get_canvas3d(*args, **kwargs):
    kwargs["projection"] = "3d"
    return get_canvas(*args, **kwargs)

def animate(fig, draw, frames, *, interval=90, jshtml=None, blit=False):
    """Build a matplotlib ``FuncAnimation`` and return it ready for the current environment.

    One animation primitive serves both worlds, which is why every ``_plot_interactive`` /
    replay routine funnels through ``FuncAnimation``:

      - inside a **Jupyter** kernel it returns a self-contained ``IPython.display.HTML``
        play/scrub player (via ``anim.to_jshtml()``) that survives notebook serialization
        with no live kernel;
      - in a plain **Python** session it returns the raw ``FuncAnimation`` so a GUI
        backend's ``plt.show()`` plays it live.

    Parameters
    ----------
    fig : matplotlib Figure        the figure to animate (closed here so the static frame
                                   isn't also rendered alongside the player).
    draw : callable                per-frame update ``draw(frame)`` (the ``_draw`` closure).
    frames : iterable | int        frame indices forwarded to ``FuncAnimation``.
    interval, blit                 standard ``FuncAnimation`` knobs.
    jshtml : bool | None           force the HTML player (True) or the raw anim (False);
                                   ``None`` auto-detects whether an IPython kernel is active.
    """
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, draw, frames=frames, interval=interval, blit=blit)
    plt.close(fig)
    if jshtml is None:
        try:
            from IPython import get_ipython
            jshtml = getattr(get_ipython(), "kernel", None) is not None
        except Exception:
            jshtml = False
    if jshtml:
        try:
            from IPython.display import HTML
            import matplotlib as mpl
            # to_jshtml() silently DROPS frames once the embedded size passes
            # animation.embed_limit (20 MB default) — the usual "animation doesn't finish"
            # cause. Lift it for the render so long replays play to the end, then restore.
            prev = mpl.rcParams.get("animation.embed_limit", 20.0)
            mpl.rcParams["animation.embed_limit"] = max(prev, 100.0)   # MB
            try:
                html = anim.to_jshtml()
            finally:
                mpl.rcParams["animation.embed_limit"] = prev
            return HTML(html)
        except Exception:
            return anim
    return anim

def str2enum(e, _type: Enum):
    if isinstance(e, str):
        try:
            e = _type[e.upper()]
        except KeyError:
            raise ValueError(f"Invalid component '{e}'. Must be one of {list(_type)}")

    if not isinstance(e, _type):
        raise ValueError(f"Component must be one of {list(_type)}; got {e.__repr__()}")

    return e

def apply_crop(data, crop):

    if crop is None:
        return data
    if isinstance(crop, tuple):
        start, end = crop
        start = start or 0
        end = end or len(data)
    else:
        start, end = crop, len(data) - crop
    return data[start:end]

def as_col(x, name = "array"):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"{name} must be (d,1) or (d,), got {format_shape(x)}")
    return x

def numerical_jacobian(func, x: np.ndarray, *args, eps: float = 1e-6) -> np.ndarray:
    x = as_col(x)
    y0 = as_col(func(x, *args))
    m, n = y0.shape[0], x.shape[0]
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        dx = np.zeros_like(x)
        step = eps * max(1.0, abs(x[j, 0]))
        dx[j, 0] = step
        y1 = as_col(func(x + dx, *args))
        J[:, j] = ((y1 - y0) / step).ravel()
    return J

def format_shape(arr):
    """
    Returns a human-readable string for array shape:
    - (n,) -> "N"
    - (n,1) -> "Nx1"
    - (n,m) -> "NxM"
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.ndim == 0:
        return "0"
    if arr.ndim == 1:
        return f"{arr.shape[0]}"
    elif arr.ndim == 2:
        return f"{arr.shape[0]}x{arr.shape[1]}"
    else:
        return "x".join(map(str, arr.shape))

def bandpass_filter(signal, fs, fc, bandwidth, order=4):
    low = (fc - bandwidth/2) / (fs / 2)
    high = (fc + bandwidth/2) / (fs / 2)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def resample_poly_float(y, fs_old, fs_new, max_denominator=1e12):
    """
    Resample a 1-D signal y from sample rate fs_old to fs_new
    using polyphase filtering with integer up/down factors.

    Parameters
    ----------
    y : array_like
        Input signal.
    fs_old : float
        Original sample rate.
    fs_new : float
        Desired sample rate.
    max_denominator : int
        Maximum denominator when rationally approximating fs_new/fs_old.

    Returns
    -------
    y_resampled : ndarray
        The resampled signal.
    """
    # 1) Find a simple rational approximation to the rate change
    frac = Fraction(fs_new, fs_old).limit_denominator(max_denominator)
    up, down = frac.numerator, frac.denominator

    # 2) Do the polyphase resampling
    y_resampled = resample_poly(y, up, down)

    return y_resampled

def make_multirate_input(sensor_data, base_fs):
    """
    Interleave multiple sensors running at different rates into a single
    NaN-padded measurement matrix at `base_fs`.

    Parameters
    ----------
    sensor_data : list of (zk, fs_sensor) tuples
        Each tuple is an (N_i, m_i) measurement array and its sample rate.
        The duration is inferred from each array's length and its fs_sensor.
    base_fs : float
        The base filter sample rate.

    Returns
    -------
    zk_out : ndarray, shape (N_base, m_total)
        NaN where the sensor did not fire at that base-rate step.
    """
    if not sensor_data:
        raise ValueError("sensor_data must contain at least one (zk, fs) pair.")

    # Determine total duration from the longest sensor
    durations = [len(zk) / fs for zk, fs in sensor_data]
    duration = max(durations)
    N_base = int(round(duration * base_fs))

    # Total observable dimension
    m_total = sum(zk.shape[1] if zk.ndim == 2 else 1 for zk, _ in sensor_data)

    zk_out = np.full((N_base, m_total), np.nan)

    col = 0
    for zk, fs_sensor in sensor_data:
        zk = np.atleast_2d(zk) if zk.ndim == 1 else zk
        m_i = zk.shape[1]
        n_sensor = len(zk)
        # Map each sensor sample index to the nearest base-rate index
        for k in range(n_sensor):
            base_idx = int(round(k * base_fs / fs_sensor))
            if base_idx < N_base:
                zk_out[base_idx, col:col + m_i] = zk[k]
        col += m_i

    return zk_out

def vectorize(z):
		return np.stack((z.real, z.imag), axis=-1)

def complexify(v):
		z_real, z_imag = v[:, 0], v[:, 1]
		return z_real + 1j * z_imag


def kplot(signals, fs, labels=None, xlim=None):
    """
    Plots multiple time-domain signals stacked vertically with legends on the right.

    Parameters:
    - signals : list of np.ndarray
        List of signals to plot.
    - fs : float
        Sampling frequency [Hz].
    - labels : list of str, optional
        Legends for each signal.
    - xlim : tuple (xmin, xmax), optional
        X-axis limits in seconds.
    """

    num_signals = len(signals)
    if labels is None:
        labels = [f"Signal {i+1}" for i in range(num_signals)]

    t = np.arange(len(signals[0])) / fs
    width, height = 10, 2
    fig, axes = plt.subplots(nrows=num_signals, sharex=True, figsize=(width, height * num_signals))

    if num_signals == 1:
        axes = [axes]

    for ax, signal, label in zip(axes, signals, labels):
        ax.plot(t, signal, color='#141414', linewidth=1.1, label=label)
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=':', color='gray', linewidth=1)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

        if xlim is not None:
            ax.set_xlim(xlim)

    axes[-1].set_xlabel("Temps [s]")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, right=0.85)

from scipy.signal import resample_poly, butter, filtfilt
from fractions import Fraction

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from enum import Enum
import random
import numpy as np
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

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

    # Rearrange size (if new subplot set)
    kwargs.setdefault("figsize", (5,5)) # Rearrange the total canvas size based on figsize and the number of rows and cols
    kwargs["figsize"] = (ncols * kwargs["figsize"][0], nrows * kwargs["figsize"][1])

    fig, ax = plt.subplots(nrows, ncols, **kwargs)

    canvas = np.array(ax).reshape(nrows, ncols) if not squeeze else np.array(ax).reshape(nrows * ncols) # canvas output 
    return canvas if nrows * ncols > 1 else canvas[0]

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

def format_shape(arr):
    """
    Returns a human-readable string for array shape:
    - (n,) -> "N"
    - (n,1) -> "Nx1"
    - (n,m) -> "NxM"
    """
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

"""
Computer-vision object tracking with linear Kalman filters — the input is always a
per-frame object **detector**'s output; this module owns the *filtering*, so a notebook
never has to assemble ``F``/``H`` or a tracking loop by hand. Three levels, sharing one
idea (detections in → smoothed, occlusion-robust state out):

* :class:`DetectionModel` — **single object, centroid** ``x = [x, y, vx, vy]`` observing
  ``(x, y)``. A constant-velocity filter that smooths a detector's centroid jitter and
  **coasts through occlusions** (a ``NaN`` measurement row → predict-only, covariance grows
  until re-acquisition).
* :class:`BoxTrackModel` — **single object, bounding box** in the SORT parameterization
  ``x = [x, y, s, r, vx, vy, vs]`` observing ``[x, y, s, r]`` (centre, area ``s = w·h``,
  aspect ``r = w/h``). Now the *box size* is filtered/predicted too, so it coasts at its
  last size rather than being faked from the covariance.
* :class:`SortTracker` — **multiple objects** (SORT): per-frame detector **boxes** are
  matched to existing tracks by IoU + Hungarian assignment, with track **birth/death** and
  ``max_age`` occlusion coasting. Each track is one :class:`KalmanTrack` over a shared
  :class:`BoxTrackModel`. :func:`synthetic_multi` generates a reproducible crossing-objects
  scene (with occlusion) to drive it.

The detector is swappable (a Haar cascade, an Otsu blob, a neural net…); this module only
needs centroids (``DetectionModel``) or boxes (``BoxTrackModel``/``SortTracker``). See the
``10_ComputerVision`` notebook. Distinct from ``apps.tracking`` — the combinatorial
drift-chamber finder (HEP) — which shares the *combinatorial* idea but a different domain.
"""
from __future__ import annotations

import os
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from ..helpers import set_random_seed
from ..framework.constants import KalmanObservable, KalmanState, KalmanStateFlavor
from ..framework.io.record import KalmanRecord
from ..framework.models.linear import LinearKalmanModel


# ===== State / observable enums (distinct values → 4 / 2 real members) =====
class DetectionState(KalmanState):
    X  = "x [px]"
    Y  = "y [px]"
    VX = "vx [px/s]"
    VY = "vy [px/s]"


class DetectionObservable(KalmanObservable):
    X = "x [px]"
    Y = "y [px]"


class DetectionModel(LinearKalmanModel):
    """Constant-velocity 2-D centroid tracker ``x = [x, y, vx, vy]`` (position observed).

    Parameters
    ----------
    sensor_id : str
        Camera / track name.
    measurement_uncertainty : float
        Detector centroid noise std [px] (so ``R = σ²·I₂``).
    process_uncertainty : float
        White-noise-acceleration std [px/s²] driving ``Q`` (how much manoeuvring to allow).
    fs : float
        Frame rate [fps].
    frame : (int, int)
        Image size ``(W, H)`` in pixels — used by the synthetic generator.
    """

    NAME = "Linear Object Detection Tracker"
    DESCRIPTION = ("Constant-velocity 2-D centroid tracker [x, y, vx, vy]; observes (x, y) "
                   "and coasts (predict-only) through occlusions.")
    VERSION = "1.0"

    @property
    def states(self):
        return DetectionState

    @property
    def observables(self):
        return DetectionObservable

    def __init__(self, sensor_id: str | int,
                 measurement_uncertainty: float = 8.0,
                 process_uncertainty: float = 40.0,
                 fs: float = 30.0,
                 frame: tuple[int, int] = (640, 480),
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)
        self._meas_unc = float(measurement_uncertainty)
        self.frame = tuple(frame)
        dt = self.dtype(1.0 / fs)

        # Constant-velocity, state laid out [x, y, vx, vy].
        F = np.array([[1.0, 0.0, dt,  0.0],
                      [0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=self.dtype)
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=self.dtype)

        # White-noise-acceleration Q (per axis), interleaved for [x, y, vx, vy].
        qa = self.dtype(process_uncertainty) ** 2
        d3, d2 = dt ** 3 / 3.0, dt ** 2 / 2.0
        Q = qa * np.array([[d3, 0.0, d2, 0.0],
                           [0.0, d3, 0.0, d2],
                           [d2, 0.0, dt, 0.0],
                           [0.0, d2, 0.0, dt]], dtype=self.dtype)

        R = (self._meas_unc ** 2) * np.eye(2, dtype=self.dtype)

        if kwargs.get("P0") is None:
            p, v = self._meas_unc ** 2, (10.0 * self._meas_unc) ** 2
            kwargs["P0"] = np.diag([p, p, v, v]).astype(self.dtype)

        super().__init__(sensor_id, F=F, H=H, Q=Q, R=R, fs=fs, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs): return zk, uk
    def inprocess(self, state_vector, **kwargs): return state_vector
    def postprocess(self, xk_hat, **kwargs):     return xk_hat

    # ------------------------------------------------------------------
    # Synthetic detector (reproducible stand-in for a real object detector)
    # ------------------------------------------------------------------
    def synthetic_track(self, duration: float, motion: str = "parabola", seed=None) -> KalmanRecord:
        """Ground-truth object trajectory as a SAMPLED record ``[x, y, vx, vy]``.

        ``motion`` ∈ {``"linear"``, ``"parabola"``, ``"bounce"``}.
        """
        if seed is not None:
            set_random_seed(seed)
        fs = float(self.fs)
        N = int(round(duration * fs))
        t = np.arange(N) / fs
        dt = 1.0 / fs
        W, H = self.frame
        u = t / t[-1] if N > 1 else np.zeros(N)

        x = 0.08 * W + 0.84 * W * u
        if motion == "linear":
            y = 0.20 * H + 0.60 * H * u
        elif motion == "parabola":
            y = 0.85 * H - 0.70 * H * np.sin(np.pi * u)            # arc (image-y points down)
        elif motion == "bounce":
            y = 0.85 * H - 0.70 * H * np.abs(np.sin(2.0 * np.pi * u))
        else:
            raise ValueError(f"Unknown motion {motion!r}; use 'linear', 'parabola' or 'bounce'.")

        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        xk = np.column_stack([x, y, vx, vy]).astype(self.dtype)
        zk = xk[:, :2].copy()
        Pk = np.zeros((N, 4, 4), dtype=self.dtype)
        return KalmanRecord(t, xk=xk, Pk=Pk, zk=zk, model=self, flavor=KalmanStateFlavor.SAMPLED)

    def detect(self, truth: KalmanRecord, occlusion=None, dropout: float = 0.0, seed=None) -> np.ndarray:
        """Noisy per-frame centroid detections ``(N, 2)`` from a truth track.

        ``occlusion=(t0, t1)`` blanks detections in that time window (``NaN`` → predict-only);
        ``dropout`` randomly drops a fraction of the remaining frames.
        """
        if seed is not None:
            set_random_seed(seed)
        pos = np.asarray(truth.xk)[:, :2].astype(float)
        t = np.asarray(truth.time)
        z = pos + self._meas_unc * np.random.randn(*pos.shape)
        if occlusion is not None:
            t0, t1 = occlusion
            z[(t >= t0) & (t <= t1)] = np.nan
        if dropout > 0.0:
            z[np.random.rand(len(z)) < dropout] = np.nan
        return z

    # ==================================================================
    # Signature figure — model.plot(record, sample=truth, detections=z)
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     detections=None, sigma=2.0, figsize=(15, 6), **kwargs):
        """Left: the image plane — truth path, detector scatter (gap at the occlusion),
        Kalman track and a few ±σ covariance ellipses. Right: x(t)/y(t) with ±σ bands,
        showing the uncertainty inflate across the occlusion and snap back on re-detection."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        t = np.asarray(record.time)
        xk = np.asarray(record.xk).reshape(len(record), -1)
        Pk = np.asarray(record.Pk)
        ex = np.sqrt(np.maximum(Pk[:, 0, 0], 0.0))
        ey = np.sqrt(np.maximum(Pk[:, 1, 1], 0.0))
        xt = np.asarray(sample.xk).reshape(len(sample), -1) if sample is not None else None
        z = np.asarray(detections, dtype=float) if detections is not None else None

        fig, (axI, axT) = plt.subplots(1, 2, figsize=figsize,
                                       gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.25))

        # --- Left: image plane ---
        if xt is not None:
            axI.plot(xt[:, 0], xt[:, 1], color="gray", lw=1.6, ls="--", alpha=0.9, label="truth")
        if z is not None:
            m = ~np.isnan(z[:, 0])
            axI.scatter(z[m, 0], z[m, 1], s=14, color="black", alpha=0.4, label="detections", zorder=2)
        axI.plot(xk[:, 0], xk[:, 1], color="crimson", lw=2.0, label="Kalman track", zorder=3)
        for k in np.linspace(0, len(t) - 1, 7).astype(int):       # ±σ ellipses along the track
            axI.add_patch(Ellipse((xk[k, 0], xk[k, 1]),
                                  2 * sigma * ex[k], 2 * sigma * ey[k],
                                  fill=False, edgecolor="crimson", alpha=0.4, lw=1.0))
        axI.set_xlim(0, self.frame[0]); axI.set_ylim(self.frame[1], 0)   # image coords (y down)
        axI.set_xlabel("x [px]"); axI.set_ylabel("y [px]")
        axI.legend(fontsize=8, loc="best"); axI.grid(True, alpha=0.3)
        axI.set_title(f"{self.name()} — image plane")

        # --- Right: x(t) / y(t) with ±σ bands ---
        for ax, comp, e, lbl, col in ((axT, 0, ex, "x", "steelblue"), (axT, 1, ey, "y", "darkorange")):
            ax.fill_between(t, xk[:, comp] - sigma * e, xk[:, comp] + sigma * e,
                            color=col, alpha=0.2)
            ax.plot(t, xk[:, comp], color=col, lw=1.8, label=f"{lbl} Kalman")
            if xt is not None:
                ax.plot(t, xt[:, comp], color=col, lw=1.0, ls="--", alpha=0.7, label=f"{lbl} truth")
            if z is not None:
                mm = ~np.isnan(z[:, comp])
                ax.scatter(t[mm], z[mm, comp], s=8, color=col, alpha=0.3)
        # shade the occlusion span (where the detector returned no measurement)
        if z is not None:
            gap = np.isnan(z[:, 0])
            if gap.any():
                axT.axvspan(t[gap].min(), t[gap].max(), color="gray", alpha=0.12, label="occlusion")
        axT.set_xlabel("Time (s)"); axT.set_ylabel("position [px]")
        axT.legend(fontsize=8, ncol=2, loc="best"); axT.grid(True, alpha=0.3)
        axT.set_title(r"Coordinates with $\pm\sigma$ (uncertainty grows through occlusion)")

        return fig

    # ==================================================================
    # Interactive player — model.plot(record, interactive=True, frames=..., detections=...)
    # ==================================================================
    def _plot_interactive(self, record, sample=None, *, frames=None, detections=None,
                          component=None, error=True, sigma=3.0, interval=90,
                          figsize=(7, 4), dpi=66, max_frames=64, trail=True, **kwargs):
        """Play/scrub-slider playback of the tracker.

        Pass ``frames`` (a list of ``(H,W,3)`` RGB images, e.g. decoded video frames) to
        animate the **footage** with the detection (green) and the Kalman box (red, sized
        by ``±sigma·σ``) overlaid — the box keeps moving and grows while coasting through a
        detector dropout. Without ``frames`` it falls back to the generic joint time-slider
        over the state components. Returns a Jupyter play/scrub player (via
        :func:`kalman.helpers.animate`).

        ``trail`` draws the cumulative image-plane path (cyan); set ``False`` for handheld /
        moving-camera footage, where pixel-position history mixes camera motion with the
        target's and just looks like a tangle.

        The player is subsampled to ``max_frames`` and rendered at ``dpi`` to keep the
        embedded HTML under Matplotlib's animation size limit.
        """
        if frames is None:
            return super()._plot_interactive(record, sample=sample, component=component,
                                             error=error, **kwargs)

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from ..helpers import animate

        frames = list(frames)
        N = min(len(frames), len(record))
        sel = np.unique(np.linspace(0, N - 1, min(N, int(max_frames))).astype(int))
        xk = np.asarray(record.xk)
        Pk = np.asarray(record.Pk)
        det = np.asarray(detections, dtype=float) if detections is not None else None
        H, W = frames[0].shape[:2]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(frames[sel[0]])
        trail_line, = ax.plot([], [], color="cyan", lw=1.5, alpha=0.75, zorder=4, visible=trail)
        pos_dot = ax.scatter([], [], s=55, c="red", zorder=6)
        box = Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", lw=2.0)
        ax.add_patch(box)
        det_dot = ax.scatter([], [], s=45, c="lime", edgecolors="black", zorder=5)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        title = ax.set_title("")

        def _draw(j):
            k = int(sel[j])
            im.set_data(frames[k])
            px, py = xk[k, 0], xk[k, 1]
            sx = np.sqrt(max(Pk[k, 0, 0], 0.0)); sy = np.sqrt(max(Pk[k, 1, 1], 0.0))
            bw, bh = max(20.0, sigma * sx), max(16.0, sigma * sy)
            box.set_bounds(px - bw, py - bh, 2 * bw, 2 * bh)
            if trail:
                trail_line.set_data(xk[:k + 1, 0], xk[:k + 1, 1])
            pos_dot.set_offsets([[px, py]])
            occluded = det is not None and bool(np.isnan(det[k, 0]))
            if det is not None and not occluded:
                det_dot.set_offsets([[det[k, 0], det[k, 1]]]); det_dot.set_visible(True)
            else:
                det_dot.set_visible(False)
            title.set_text(f"frame {k}   t = {k / self.fs:.2f} s"
                           + ("   — coasting (predict-only)" if occluded else ""))
            return im, trail_line, pos_dot, box, det_dot, title

        return animate(fig, _draw, frames=range(len(sel)), interval=interval,
                       jshtml=kwargs.pop("jshtml", None))


# ======================================================================
#  SORT — bounding-box state, IoU/Hungarian association, multi-object
# ======================================================================
# The classic "Simple Online and Realtime Tracking" recipe: filter the box in the
# scale/aspect parameterization [x, y, s, r] (s = area, r = aspect) rather than just the
# centroid, and associate per-frame detector boxes to existing tracks by IoU. This is the
# proper, named home for what a notebook would otherwise hand-assemble inline.

class BoxTrackState(KalmanState):
    X  = "x [px]"
    Y  = "y [px]"
    S  = "s [px^2]"
    R  = "r [w/h]"
    VX = "vx [px/s]"
    VY = "vy [px/s]"
    VS = "vs [px^2/s]"


class BoxTrackObservable(KalmanObservable):
    X = "x [px]"
    Y = "y [px]"
    S = "s [px^2]"
    R = "r [w/h]"


# ----- box <-> [x, y, s, r] conversions (SORT parameterization) -----
def box_to_xysr(bbox) -> np.ndarray:
    """``[x1, y1, x2, y2]`` → column ``[x, y, s, r]`` (centre, area s=w·h, aspect r=w/h)."""
    b = np.asarray(bbox, dtype=float).ravel()
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2.0, b[1] + h / 2.0, w * h, w / h if h else 0.0]).reshape(4, 1)


def xysr_to_box(state) -> np.ndarray:
    """``[x, y, s, r, ...]`` → ``[x1, y1, x2, y2]`` (inverse of :func:`box_to_xysr`)."""
    b = np.asarray(state, dtype=float).ravel()
    s, r = max(float(b[2]), 1e-6), max(float(b[3]), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([b[0] - w / 2.0, b[1] - h / 2.0, b[0] + w / 2.0, b[1] + h / 2.0])


def iou_batch(det, trk) -> np.ndarray:
    """Pairwise IoU between detection boxes ``det (N,4)`` and track boxes ``trk (M,4)`` → ``(N, M)``."""
    det = np.asarray(det, dtype=float).reshape(-1, 1, 4)
    trk = np.asarray(trk, dtype=float).reshape(1, -1, 4)
    xx1 = np.maximum(det[..., 0], trk[..., 0]); yy1 = np.maximum(det[..., 1], trk[..., 1])
    xx2 = np.minimum(det[..., 2], trk[..., 2]); yy2 = np.minimum(det[..., 3], trk[..., 3])
    w = np.clip(xx2 - xx1, 0.0, None); h = np.clip(yy2 - yy1, 0.0, None)
    inter = w * h
    area_d = (det[..., 2] - det[..., 0]) * (det[..., 3] - det[..., 1])
    area_t = (trk[..., 2] - trk[..., 0]) * (trk[..., 3] - trk[..., 1])
    return inter / np.clip(area_d + area_t - inter, 1e-9, None)


def _greedy_match(iou: np.ndarray):
    """Fallback assignment (no SciPy): repeatedly take the highest remaining IoU pair."""
    iou = iou.copy()
    pairs = []
    while iou.size and iou.max() > 0:
        d, t = np.unravel_index(np.argmax(iou), iou.shape)
        pairs.append((int(d), int(t)))
        iou[d, :] = -1.0
        iou[:, t] = -1.0
    return pairs


def associate_detections(det, trk, iou_threshold: float = 0.3):
    """Match detections to tracks by IoU (Hungarian if SciPy is present, else greedy).

    Returns ``(matches, unmatched_detections, unmatched_tracks)`` where ``matches`` is a
    list of ``(det_index, track_index)`` pairs with IoU ≥ ``iou_threshold``.
    """
    det = np.asarray(det, dtype=float).reshape(-1, 4)
    trk = np.asarray(trk, dtype=float).reshape(-1, 4)
    if len(trk) == 0:
        return [], list(range(len(det))), []
    if len(det) == 0:
        return [], [], list(range(len(trk)))

    iou = iou_batch(det, trk)
    try:
        from scipy.optimize import linear_sum_assignment
        di, ti = linear_sum_assignment(-iou)
        pairs = list(zip(di.tolist(), ti.tolist()))
    except Exception:                                  # SciPy missing → greedy fallback
        pairs = _greedy_match(iou)

    matched_d = {d for d, _ in pairs}
    matched_t = {t for _, t in pairs}
    un_det = [d for d in range(len(det)) if d not in matched_d]
    un_trk = [t for t in range(len(trk)) if t not in matched_t]
    matches = []
    for d, t in pairs:
        if iou[d, t] < iou_threshold:                  # too weak → leave both unmatched
            un_det.append(d); un_trk.append(t)
        else:
            matches.append((d, t))
    return matches, un_det, un_trk


class BoxTrackModel(LinearKalmanModel):
    """Constant-velocity **bounding-box** tracker — SORT state ``[x, y, s, r, vx, vy, vs]``.

    Observes ``[x, y, s, r]`` (centre, area, aspect); the aspect ratio ``r`` is modelled as
    constant (no rate). ``Q``/``R``/``P0`` follow the canonical SORT recipe (relative
    weights), scaled by ``process_uncertainty`` / ``measurement_uncertainty`` (default 1.0).

    Single-object use: inject a sequence of detector boxes and filter::

        model = BoxTrackModel("ball", fs=25)
        record = model.filter_boxes(boxes)        # boxes: [x1,y1,x2,y2] or None per frame
        smoothed = model.record_to_boxes(record)  # back to [x1,y1,x2,y2] (coasts on misses)
    """

    NAME = "SORT Box Tracker"
    DESCRIPTION = ("Constant-velocity bounding-box tracker [x, y, s, r, vx, vy, vs]; observes "
                   "[x, y, s, r] and coasts (predict-only) through detector misses.")
    VERSION = "1.0"

    @property
    def states(self):
        return BoxTrackState

    @property
    def observables(self):
        return BoxTrackObservable

    def __init__(self, sensor_id: str | int,
                 measurement_uncertainty: float = 1.0,
                 process_uncertainty: float = 1.0,
                 fs: float = 30.0,
                 frame: tuple[int, int] = (640, 480),
                 **kwargs):
        self.dtype = kwargs.pop("dtype", np.float64)
        self.frame = tuple(frame)
        dt = self.dtype(1.0 / fs)

        F = np.eye(7, dtype=self.dtype)
        F[0, 4] = dt; F[1, 5] = dt; F[2, 6] = dt           # x,y,s advance by their velocities
        H = np.zeros((4, 7), dtype=self.dtype)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0        # observe [x, y, s, r]

        # Canonical SORT covariances (relative), scaled by the *_uncertainty knobs.
        R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(self.dtype) * float(measurement_uncertainty)
        Q = np.eye(7, dtype=self.dtype)
        Q[-1, -1] *= 0.01                                  # low process noise on area-rate
        Q[4:, 4:] *= 0.01                                  # and on the velocities
        Q *= float(process_uncertainty)

        if kwargs.get("P0") is None:
            P0 = np.eye(7, dtype=self.dtype) * 10.0
            P0[4:, 4:] *= 1000.0                            # large prior on unobserved velocities
            kwargs["P0"] = P0

        super().__init__(sensor_id, F=F, H=H, Q=Q, R=R, fs=fs, **kwargs)

    # ---- explicit input injection: detector boxes -> measurement rows ----
    @staticmethod
    def boxes_to_measurements(boxes) -> np.ndarray:
        """Sequence of boxes (``[x1,y1,x2,y2]`` or ``None``/``NaN`` for a miss) → ``(N, 4)``
        measurement rows ``[x, y, s, r]`` with ``NaN`` rows where the detector missed."""
        rows = []
        for b in boxes:
            a = None if b is None else np.asarray(b, dtype=float).ravel()
            if a is None or a.size < 4 or np.any(np.isnan(a)):
                rows.append([np.nan] * 4)
            else:
                rows.append(box_to_xysr(a).ravel())
        return np.asarray(rows, dtype=float)

    @staticmethod
    def record_to_boxes(record) -> np.ndarray:
        """Filtered record (state rows ``[x,y,s,r,...]``) → ``(N, 4)`` boxes ``[x1,y1,x2,y2]``."""
        return np.array([xysr_to_box(x) for x in np.asarray(record.xk)])

    def initial_from_box(self, box):
        """Seed ``x0`` from a single detector box (velocities zero)."""
        x0 = np.zeros((7, 1), dtype=self.dtype)
        x0[:4, 0] = box_to_xysr(box).ravel()
        self.initial_state(x0=x0)
        return self

    def filter_boxes(self, boxes, **kwargs) -> KalmanRecord:
        """Inject a per-frame box sequence and run the filter (seeds ``x0`` from the first
        valid box if not already set). The explicit, named alternative to assembling the
        measurement stream and calling :meth:`filter` by hand."""
        z = self.boxes_to_measurements(boxes)
        valid = np.where(~np.isnan(z[:, 0]))[0]
        if len(valid):
            x0 = np.zeros((7, 1), dtype=self.dtype)
            x0[:4, 0] = z[valid[0]]
            self.initial_state(x0=x0)
        return self.filter(z, **kwargs)


# Per-track public view returned by SortTracker.step (one row per confirmed track).
TrackedBox = namedtuple("TrackedBox", ["id", "box", "coasting", "hits", "age"])


class KalmanTrack:
    """One object's online box filter — a thin wrapper holding its own ``(x, P)`` and
    stepping the shared :class:`BoxTrackModel` through the linear engine."""

    def __init__(self, model: BoxTrackModel, engine, box, track_id: int):
        self.model = model
        self.engine = engine
        self.id = int(track_id)
        self.x = np.zeros((7, 1), dtype=float)
        self.x[:4, 0] = box_to_xysr(box).ravel()
        self.P = np.array(model.P0, dtype=float).copy()
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def predict(self, t=None) -> np.ndarray:
        """Advance one frame (predict-only); returns the predicted box."""
        if self.x[6, 0] + self.x[2, 0] <= 0:               # keep area from going negative
            self.x[6, 0] = 0.0
        self.x, self.P, _ = self.engine._predict(self.model, self.x, self.P, t=t)
        self.age += 1
        self.time_since_update += 1
        return xysr_to_box(self.x)

    def update(self, box, t=None) -> None:
        """Correct with a matched detection box."""
        self.time_since_update = 0
        self.hits += 1
        self.x, self.P = self.engine._update(self.model, self.x, self.P, box_to_xysr(box), t=t)

    @property
    def box(self) -> np.ndarray:
        return xysr_to_box(self.x)


class SortTracker:
    """Multi-object SORT tracker: per-frame detector **boxes** in, stable per-ID tracks out.

    Each :meth:`step` predicts every track, matches detections by IoU + Hungarian
    assignment (:func:`associate_detections`), updates matches, **spawns** a track per
    unmatched detection, and **retires** tracks unseen for more than ``max_age`` frames. A
    track is reported once it has ``min_hits`` hits; while a confirmed track is between
    detections it is still reported with ``coasting=True`` (predict-only through occlusion).
    """

    NAME = "SortTracker"          # single source of truth for the method name (eval, plots, tables)

    def __init__(self, max_age: int = 10, min_hits: int = 3, iou_threshold: float = 0.3,
                 measurement_uncertainty: float = 1.0, process_uncertainty: float = 1.0,
                 fs: float = 30.0, frame: tuple[int, int] = (640, 480), **model_kwargs):
        self.model = BoxTrackModel("sort", measurement_uncertainty=measurement_uncertainty,
                                   process_uncertainty=process_uncertainty, fs=fs,
                                   frame=frame, **model_kwargs)
        self.engine = self.model.default_engine()
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.tracks: list[KalmanTrack] = []
        self.frame_count = 0
        self._next_id = 0

    def step(self, boxes) -> list[TrackedBox]:
        """Process one frame's detections (``(M, 4)`` boxes, possibly empty)."""
        self.frame_count += 1
        det = (np.asarray(boxes, dtype=float).reshape(-1, 4)
               if boxes is not None and len(boxes) else np.empty((0, 4)))

        # 1) advance every existing track
        trk_boxes = np.zeros((len(self.tracks), 4))
        for i, trk in enumerate(self.tracks):
            trk_boxes[i] = trk.predict()

        # 2) IoU/Hungarian association
        matches, un_det, un_trk = associate_detections(det, trk_boxes, self.iou_threshold)

        # 3) update matched, 4) birth unmatched detections
        for d, t in matches:
            self.tracks[t].update(det[d])
        for d in un_det:
            self.tracks.append(KalmanTrack(self.model, self.engine, det[d], self._next_id))
            self._next_id += 1

        # 5) death: drop tracks unseen for too long
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6) report confirmed tracks (coasting flag for predict-only frames)
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                out.append(TrackedBox(id=t.id, box=t.box, coasting=t.time_since_update > 0,
                                      hits=t.hits, age=t.age))
        return out

    def run(self, detections) -> list[list[TrackedBox]]:
        """Run over a whole sequence — an iterable of per-frame ``(M, 4)`` detection arrays —
        returning the per-frame list of :class:`TrackedBox`."""
        return [self.step(d) for d in detections]


def synthetic_multi(n_frames: int = 120, frame: tuple[int, int] = (640, 480),
                    n_objects: int = 3, box_wh: tuple[int, int] = (50, 100),
                    noise: float = 3.0, seed=None) -> list[np.ndarray]:
    """Reproducible crossing-objects scene for :class:`SortTracker` testing.

    Returns a list of ``n_frames`` per-frame ``(M, 4)`` ``[x1, y1, x2, y2]`` detection arrays.
    Three objects move on crossing linear paths; object 1 is occluded (no detection) for
    ~15 frames in the middle to exercise coasting. Object 2 drifts diagonally.
    """
    if seed is not None:
        set_random_seed(seed)
    W, H = frame
    bw, bh = box_wh
    t = np.linspace(0.0, 1.0, n_frames)

    cx = np.column_stack([
        0.05 * W + 0.90 * W * t,          # 0: left → right
        0.95 * W - 0.90 * W * t,          # 1: right → left (crosses 0 near centre)
        0.20 * W + 0.60 * W * t,          # 2: diagonal
    ])
    cy = np.column_stack([
        np.full(n_frames, 0.30 * H),      # 0: horizontal band
        np.full(n_frames, 0.30 * H),      # 1: same band → crossing with 0
        0.10 * H + 0.80 * H * t,         # 2: top → bottom
    ])

    occ_start = int(0.38 * n_frames)
    occ_end   = int(0.52 * n_frames)

    out = []
    for k in range(n_frames):
        row = []
        for i in range(n_objects):
            if i == 1 and occ_start <= k < occ_end:
                continue
            x1 = cx[k, i] - bw / 2.0 + noise * np.random.randn()
            y1 = cy[k, i] - bh / 2.0 + noise * np.random.randn()
            row.append([x1, y1, x1 + bw, y1 + bh])
        out.append(np.asarray(row, float) if row else np.empty((0, 4)))
    return out


# ======================================================================
#  MOTChallenge evaluation (CLEAR-MOT) — real benchmark, optional `motmetrics`
# ======================================================================
# Score the SAME SortTracker on real MOTChallenge sequences (MOT15/MOT16) with the standard
# CLEAR-MOT metrics, by injecting the benchmark's own public detections — i.e. the real
# Kalman input, not ones we invented. Keeps the "detections in → tracks out" contract.

def mot_detections(det_path, conf_thresh: float = 0.0) -> dict:
    """MOTChallenge ``det.txt`` → ``{frame: (M,4) [x1,y1,x2,y2]}`` keeping rows with conf ≥ thresh."""
    a = np.atleast_2d(np.loadtxt(det_path, delimiter=",", usecols=(0, 2, 3, 4, 5, 6)))
    by = {}
    for fr, x, y, w, h, c in a:
        if c < conf_thresh:
            continue
        by.setdefault(int(fr), []).append([x, y, x + w, y + h])
    return {f: np.asarray(v, float) for f, v in by.items()}


def sort_mot_hypothesis(det_by_frame, **tracker_kwargs) -> np.ndarray:
    """Run :class:`SortTracker` over ``{frame:(M,4)}`` → MOTChallenge rows
    ``(N,10): frame,id,x,y,w,h,1,-1,-1,-1`` (detection frames only, not coasted predictions)."""
    tracker_kwargs.setdefault("fs", 1.0)               # dt = 1 frame (canonical SORT)
    tr = SortTracker(**tracker_kwargs)
    fmax = max(det_by_frame) if det_by_frame else 0
    rows = []
    for f in range(1, fmax + 1):
        for tb in tr.step(det_by_frame.get(f, np.empty((0, 4)))):
            if tb.coasting:                            # report detections, not pure predictions
                continue
            x1, y1, x2, y2 = tb.box
            rows.append([f, tb.id + 1, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1])
    return np.asarray(rows, float).reshape(-1, 10)


def _mot_active_gt_rows(gt_path) -> np.ndarray:
    """MOT gt.txt → mot15 rows for evaluable targets. Auto-detects MOT16 (flag/class columns →
    keep active pedestrians: flag==1 & class==1) vs MOT15 (no class → keep all)."""
    a = np.atleast_2d(np.loadtxt(gt_path, delimiter=","))
    col7 = a[:, 7] if a.shape[1] > 7 else np.full(len(a), -1.0)
    is_mot16 = (a.shape[1] > 7
                and np.all(np.isin(a[:, 6], (0.0, 1.0)))
                and np.all((col7 >= 1) & (col7 <= 12)))
    rows = []
    for r in a:
        if is_mot16 and not (r[6] == 1 and r[7] == 1):
            continue
        rows.append([int(r[0]), int(r[1]), r[2], r[3], r[4], r[5], 1, -1, -1, -1])
    return np.asarray(rows, float)


def mot_sequences_from_root(root) -> dict:
    """Discover ``{name: (det_path, gt_path)}`` under a MOTChallenge ``train`` root (gt present)."""
    import glob
    seqs = {}
    for det_path in sorted(glob.glob(os.path.join(root, "**", "det", "det.txt"), recursive=True)):
        seqdir = os.path.dirname(os.path.dirname(det_path))
        gt_path = os.path.join(seqdir, "gt", "gt.txt")
        if os.path.exists(gt_path):
            seqs[os.path.basename(seqdir)] = (det_path, gt_path)
    return seqs


def evaluate_mot(sequences, *, conf_thresh: float = 0.0, method_name: str | None = None,
                 metrics=None, **tracker_kwargs):
    """CLEAR-MOT score :class:`SortTracker` on MOTChallenge sequences (needs ``motmetrics``).

    ``sequences`` : ``{name: (det_path, gt_path)}`` (MOT15 or MOT16 format, auto-detected) — e.g.
    from :func:`mot_sequences_from_root`. Returns the motmetrics summary DataFrame (per sequence
    + ``OVERALL``); note motmetrics' ``motp`` is a distance (``1 − IoU``), so overlap-% =
    ``100·(1 − motp)``. The method is labelled ``method_name`` (defaults to ``SortTracker.NAME``).
    """
    try:
        import motmetrics as mm
    except ImportError as exc:                         # keep motmetrics an optional dependency
        raise ImportError("evaluate_mot needs `motmetrics` (pip install motmetrics).") from exc
    if not hasattr(np, "asfarray"):                    # motmetrics ≤1.4 on NumPy ≥2.0
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
    import tempfile

    method_name = method_name or SortTracker.NAME
    metrics = metrics or ['num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost',
                          'num_false_positives', 'num_misses', 'num_switches', 'num_objects']
    accs, names = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (det_path, gt_path) in sequences.items():
            hyp = sort_mot_hypothesis(mot_detections(det_path, conf_thresh), **tracker_kwargs)
            hp = os.path.join(tmp, "hyp.txt"); np.savetxt(hp, hyp, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d")
            gp = os.path.join(tmp, "gt.txt");  np.savetxt(gp, _mot_active_gt_rows(gt_path), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d")
            gt = mm.io.loadtxt(gp, fmt="mot15-2D")
            hd = mm.io.loadtxt(hp, fmt="mot15-2D")
            accs.append(mm.utils.compare_to_groundtruth(gt, hd, "iou", distth=0.5))
            names.append(name)
    summary = mm.metrics.create().compute_many(accs, metrics=metrics, names=names,
                                               generate_overall=len(names) > 1)
    summary.attrs["method_name"] = method_name
    return summary

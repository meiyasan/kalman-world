"""Object-detection filtering — the DETECTION variant."""
from __future__ import annotations
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

import os
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from ...helpers import set_random_seed
from ...framework.constants import KalmanObservable, KalmanState, KalmanStateFlavor
from ...framework.io.record import KalmanRecord
from ...framework.models.linear import LinearKalmanModel



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

        # Continuous-time generator, state laid out [x, y, vx, vy]: ẋ=vx, ẏ=vy,
        # v̇x=0, v̇y=0 (constant velocity, axes decoupled). Passed as A= (not
        # F=): the base KalmanModel discretizes exactly via expm(A/fs).
        A = np.array([[0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=self.dtype)
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=self.dtype)

        # Continuous white-noise-acceleration disturbance driving v̇x, v̇y
        # independently (Lc's two columns); passed as Qc=/Lc= so the base
        # model Van Loan-discretizes it exactly alongside A=.
        Lc = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=self.dtype)
        Qc = self.dtype(process_uncertainty) ** 2 * np.eye(2, dtype=self.dtype)

        R = (self._meas_unc ** 2) * np.eye(2, dtype=self.dtype)

        if kwargs.get("P0") is None:
            p, v = self._meas_unc ** 2, (10.0 * self._meas_unc) ** 2
            kwargs["P0"] = np.diag([p, p, v, v]).astype(self.dtype)

        super().__init__(sensor_id, A=A, H=H, Qc=Qc, Lc=Lc, R=R, fs=fs, **kwargs)

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
        from ...helpers import animate

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



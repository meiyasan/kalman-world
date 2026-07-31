"""Bounding-box tracking + SORT association — the TRACK variant."""
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



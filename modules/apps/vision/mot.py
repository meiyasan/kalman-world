"""MOT-challenge dataset loading and evaluation helpers."""
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
from .boxtrack import SortTracker
from ...framework.constants import KalmanObservable, KalmanState, KalmanStateFlavor
from ...framework.io.record import KalmanRecord
from ...framework.models.linear import LinearKalmanModel


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

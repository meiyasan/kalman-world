"""
kalman.apps.vision — computer-vision tracking domain.

- detection — :class:`DetectionModel` (per-detection smoothing);
- boxtrack  — :class:`BoxTrackModel`, :class:`SortTracker`, association;
- mot       — MOT dataset loaders and evaluation.
"""
from .detection import (DetectionModel, DetectionState, DetectionObservable)
from .boxtrack import (BoxTrackModel, BoxTrackState, BoxTrackObservable, SortTracker, KalmanTrack, TrackedBox, box_to_xysr, xysr_to_box, iou_batch, associate_detections, synthetic_multi)
from .mot import (mot_detections, sort_mot_hypothesis, mot_sequences_from_root, evaluate_mot)

VARIANTS = {
    "detection": DetectionModel,
    "boxtrack": BoxTrackModel,
}

__all__ = [
    "DetectionModel",
    "DetectionState",
    "DetectionObservable",
    "BoxTrackModel",
    "BoxTrackState",
    "BoxTrackObservable",
    "SortTracker",
    "KalmanTrack",
    "TrackedBox",
    "box_to_xysr",
    "xysr_to_box",
    "iou_batch",
    "associate_detections",
    "mot_detections",
    "sort_mot_hypothesis",
    "mot_sequences_from_root",
    "evaluate_mot",
    "synthetic_multi",
    "VARIANTS",
]

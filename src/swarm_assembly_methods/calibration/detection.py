"""AprilTag detection utilities."""

import cv2
import numpy as np
from pupil_apriltags import Detector

from .board import BoardParams


def create_detector(n_threads: int = 4) -> Detector:
    return Detector(
        families="tag36h11",
        nthreads=n_threads,
        quad_decimate=1.0,
        quad_sigma=0.0,
    )


def to_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    g_min, g_max = int(gray.min()), int(gray.max())
    scale = 255.0 / max(1, g_max - g_min)
    return cv2.convertScaleAbs(gray, alpha=scale, beta=-g_min * scale)


def detect_tags(bgr: np.ndarray, detector: Detector, max_width: int = 1920):
    """Run AprilTag detection; returns list of detections with corners in original coords."""
    gray = to_uint8(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    h, w = gray.shape
    scale_back = 1.0
    if w > max_width:
        f = max_width / w
        gray = cv2.resize(gray, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
        scale_back = 1.0 / f

    dets = detector.detect(gray)
    for d in dets:
        d.corners *= scale_back
    return dets


def dets_to_dict(dets, board: BoardParams) -> dict[int, np.ndarray]:
    """Convert detections to {tag_id: corners (4,2)} filtering by board grid."""
    out = {}
    for d in dets:
        tid = int(d.tag_id)
        if board.id_offset <= tid < board.id_offset + board.rows * board.cols:
            out[tid] = d.corners.astype(np.float32)
    return out

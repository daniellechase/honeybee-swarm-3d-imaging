"""Stereo rectification helpers."""

import cv2
import numpy as np


def compute_rectification(K1, d1, K2, d2, R, T, W, H, alpha=0.0):
    """Compute stereo rectification maps.

    Returns ((R1, R2, P1, P2), (map1x, map1y, map2x, map2y)).
    """
    image_size = (int(W), int(H))
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, d1, K2, d2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, image_size, cv2.CV_32FC1)
    return (R1, R2, P1, P2), (map1x, map1y, map2x, map2y)


def rectify_points(pts_uv, K, dist, R_rect, P_rect):
    """Undistort + rectify 2-D points. Returns (N,2) array."""
    pts = pts_uv.reshape(-1, 1, 2).astype(np.float64)
    rect = cv2.undistortPoints(pts, K, dist, R=R_rect, P=P_rect)
    return rect.reshape(-1, 2)

"""Apply stereo rectification to 2D track point dicts."""

import numpy as np
import cv2


def rectify_track_dict(tracks, K, dist, R_rect, P_rect):
    """
    Rectify all points in a tracks dict.

    Parameters
    ----------
    tracks   : dict  {tid: np.ndarray (N, 3) columns [frame, x, y]}
    K, dist  : camera intrinsics
    R_rect   : rectification rotation (from stereoRectify)
    P_rect   : rectified projection matrix

    Returns
    -------
    dict  {tid: np.ndarray (N, 3) columns [frame, x_rect, y_rect]}
    """
    out = {}
    for tid, pts in tracks.items():
        frames = pts[:, 0:1]                          # (N, 1)
        xy = pts[:, 1:3].reshape(-1, 1, 2).astype(np.float64)
        xy_rect = cv2.undistortPoints(xy, K, dist, R=R_rect, P=P_rect)
        xy_rect = xy_rect.reshape(-1, 2)
        out[tid] = np.hstack([frames, xy_rect])
    return out

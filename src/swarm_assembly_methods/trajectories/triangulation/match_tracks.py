"""
Per-frame Hungarian stereo matching of rectified 2D tracks → 3D trajectories.

Algorithm per frame:
  1. Pool all right-camera detections into a lookup.
  2. For each left track with a detection at this frame, find valid right
     candidates via epipolar (|dy| <= y_thresh_px) and disparity gates
     (derived from zmin/zmax).
  3. Triangulate valid (left, right) pairs.
  4. Build a cost matrix: velocity-prior distance if track has history,
     else |dy| as fallback.
  5. Solve with Hungarian assignment; update EMA velocity state.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _build_right_lookup(track_R):
    """Pool right tracks into {frame: np.ndarray (M, 2)} of unique points."""
    by_frame = defaultdict(list)
    for pts in track_R.values():
        for frame, xR, yR in pts:
            by_frame[int(frame)].append((xR, yR))

    result = {}
    for t, pts in by_frame.items():
        unique = []
        for xR, yR in pts:
            if not any(abs(e[0] - xR) < 1.0 and abs(e[1] - yR) < 1.0 for e in unique):
                unique.append([xR, yR])
        result[t] = np.array(unique, dtype=float)
    return result


def _build_left_lookup(track_L):
    """Index left tracks as {frame: {tid: (xL, yL)}}."""
    by_frame = defaultdict(dict)
    for tid, pts in track_L.items():
        for frame, xL, yL in pts:
            by_frame[int(frame)][tid] = (xL, yL)
    return by_frame


def match_tracks(
    track_L,
    track_R,
    fx_rect,
    fy_rect,
    cx,
    cy,
    baseline_m,
    zmin_m,
    zmax_m,
    y_thresh_px,
    use_velocity_prior=True,
    max_3d_dist=0.15,
    max_pred_error=None,
    vel_ema_alpha=0.3,
    smoothness_lambda=0.0,
):
    """
    Match rectified left and right 2D tracks frame-by-frame and triangulate.

    Parameters
    ----------
    track_L, track_R : dict  {tid: np.ndarray (N, 3) [frame, x, y]}
        Rectified 2D tracks for left and right cameras.
    fx_rect, fy_rect, cx, cy : float
        Rectified camera intrinsics (from P1 after stereoRectify).
    baseline_m : float
        Stereo baseline in metres.
    zmin_m, zmax_m : float
        Depth range for disparity gate.
    y_thresh_px : float
        Max allowed vertical (epipolar) distance between matched points.
    use_velocity_prior : bool
        Gate candidates by distance from velocity-predicted 3D position.
    max_3d_dist : float or None
        Max metres from the *last known position* per frame (absolute displacement gate).
        Candidates with ||pt3d - last_pos|| > max_3d_dist * dt are rejected.
    max_pred_error : float or None
        Max metres from the *velocity-predicted position* (prediction error gate).
        Candidates with ||pt3d - pred_pos|| > max_pred_error * dt are rejected.
        Independent of max_3d_dist; either or both can be set.  None = disabled.
    vel_ema_alpha : float
        EMA smoothing for velocity update (higher = faster response).
    smoothness_lambda : float
        Weight on acceleration penalty added to cost: lambda * |new_vel - current_vel|.
        Set to 0 to disable (pure distance cost). Penalises abrupt velocity changes,
        biasing assignment toward smoother trajectories.

    Returns
    -------
    pd.DataFrame  columns: traj_id, t, X, Y, Z, xL, yL, xR, yR
    """
    dmin = fx_rect * baseline_m / zmax_m
    dmax = fx_rect * baseline_m / zmin_m

    right_by_frame = _build_right_lookup(track_R)
    left_by_frame  = _build_left_lookup(track_L)

    track_state = {}   # {tidL: {pos, vel, last_t, n_obs}}
    accepted    = defaultdict(list)  # {tidL: [(t, X, Y, Z, xL, yL, xR, yR, pred_error)]}

    common_frames = sorted(set(left_by_frame) & set(right_by_frame))

    for frame_t in common_frames:
        left_frame  = left_by_frame[frame_t]
        right_frame = right_by_frame[frame_t]
        if not left_frame or len(right_frame) == 0:
            continue

        tids_L  = list(left_frame.keys())
        n_L     = len(tids_L)

        xL_arr = np.array([left_frame[tid][0] for tid in tids_L])
        yL_arr = np.array([left_frame[tid][1] for tid in tids_L])
        xR_arr = right_frame[:, 0]
        yR_arr = right_frame[:, 1]

        # ---- epipolar + disparity gate ----
        dy   = np.abs(yL_arr[:, None] - yR_arr[None, :])   # (n_L, n_R)
        disp = xL_arr[:, None] - xR_arr[None, :]
        valid = (dy <= y_thresh_px) & (disp >= dmin) & (disp <= dmax)
        vi, vj = np.where(valid)

        if len(vi) == 0:
            continue

        # ---- triangulate valid pairs ----
        d_v = disp[vi, vj]
        Z_v = fx_rect * baseline_m / d_v
        X_v = (xL_arr[vi] - cx) * Z_v / fx_rect
        Y_v = (yL_arr[vi] - cy) * Z_v / fy_rect
        pts3d_v = np.stack([X_v, Y_v, Z_v], axis=1)

        # ---- velocity state for this frame ----
        pred_pos  = np.full((n_L, 3), np.nan)
        last_pos  = np.full((n_L, 3), np.nan)
        dt_arr    = np.ones(n_L, dtype=float)
        has_prior = np.zeros(n_L, dtype=bool)

        for i, tidL in enumerate(tids_L):
            if tidL in track_state:
                st = track_state[tidL]
                dt = frame_t - st["last_t"]
                dt_arr[i]   = max(float(dt), 1e-6)
                last_pos[i] = st["pos"]
                pred_pos[i] = st["pos"] + st["vel"] * dt_arr[i]
                has_prior[i] = True

        # ---- cost matrix ----
        cost_mat = np.full((n_L, len(xR_arr)), 1e9)

        for k in range(len(vi)):
            i, j = vi[k], vj[k]
            pt3d = pts3d_v[k]

            if use_velocity_prior and has_prior[i]:
                dist_pred = float(np.linalg.norm(pt3d - pred_pos[i]))
                dist_last = float(np.linalg.norm(pt3d - last_pos[i]))
                if max_3d_dist is not None and dist_last > max_3d_dist * dt_arr[i]:
                    continue
                if max_pred_error is not None and dist_pred / dt_arr[i] > max_pred_error:
                    continue
                cost = dist_pred
                if smoothness_lambda > 0.0:
                    new_vel = (pt3d - last_pos[i]) / dt_arr[i]
                    accel = float(np.linalg.norm(
                        new_vel - track_state[tids_L[i]]["vel"]
                    ))
                    cost += smoothness_lambda * accel
                cost_mat[i, j] = cost
            else:
                # no velocity prior yet — use epipolar distance as cost, no distance gate
                cost_mat[i, j] = float(abs(yL_arr[i] - yR_arr[j]))

        # ---- Hungarian assignment ----
        pts3d_lookup = {(vi[k], vj[k]): pts3d_v[k] for k in range(len(vi))}
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        for i, j in zip(row_ind, col_ind):
            if cost_mat[i, j] >= 1e9:
                continue
            pt3d = pts3d_lookup.get((i, j))
            if pt3d is None:
                continue
            tidL = tids_L[i]
            if has_prior[i]:
                err = (pt3d - pred_pos[i]) / dt_arr[i]
                pe   = float(np.linalg.norm(err))
                pe_x, pe_y, pe_z = float(abs(err[0])), float(abs(err[1])), float(abs(err[2]))
            else:
                pe = pe_x = pe_y = pe_z = np.nan
            accepted[tidL].append((
                frame_t, pt3d[0], pt3d[1], pt3d[2],
                xL_arr[i], yL_arr[i], xR_arr[j], yR_arr[j],
                pe, pe_x, pe_y, pe_z,
            ))

            # ---- update EMA velocity state ----
            if tidL in track_state:
                st = track_state[tidL]
                dt = frame_t - st["last_t"]
                if dt > 0:
                    raw_vel = (pt3d - st["pos"]) / dt
                    st["vel"] = ((1.0 - vel_ema_alpha) * st["vel"]
                                 + vel_ema_alpha * raw_vel)
                st["pos"]    = pt3d
                st["last_t"] = frame_t
                st["n_obs"] += 1
            else:
                track_state[tidL] = {
                    "pos": pt3d, "vel": np.zeros(3),
                    "last_t": frame_t, "n_obs": 1,
                }

    # ---- assemble DataFrame ----
    if not accepted:
        return pd.DataFrame(columns=["traj_id", "t", "X", "Y", "Z",
                                      "xL", "yL", "xR", "yR",
                                      "pred_error", "pred_error_x", "pred_error_y", "pred_error_z"])

    rows = []
    for tidL, pts in accepted.items():
        for t, X, Y, Z, xL, yL, xR, yR, pe, pe_x, pe_y, pe_z in pts:
            rows.append({"traj_id": tidL, "t": t,
                          "X": X, "Y": Y, "Z": Z,
                          "xL": xL, "yL": yL, "xR": xR, "yR": yR,
                          "pred_error": pe,
                          "pred_error_x": pe_x,
                          "pred_error_y": pe_y,
                          "pred_error_z": pe_z})
    df = pd.DataFrame(rows).sort_values(["traj_id", "t"]).reset_index(drop=True)
    return df

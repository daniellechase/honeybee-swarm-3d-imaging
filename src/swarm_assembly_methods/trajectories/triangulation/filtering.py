"""Filter 3D trajectories by length and continuity."""

import numpy as np
import pandas as pd


def _accel_mask(grp, max_accel):
    """
    Return a boolean array (len = len(grp)) that is True for rows where the
    3D acceleration magnitude exceeds max_accel (m/s²).

    Acceleration at index k: ||X[k+1] - 2*X[k] + X[k-1]|| * fps²
    We use fps=1 (frame units) here since max_accel should be given in m/frame²
    when called from filter_tracks_3d which does not have fps.  The caller is
    responsible for unit consistency.
    """
    xyz = grp[["X", "Y", "Z"]].to_numpy()
    bad = np.zeros(len(grp), dtype=bool)
    for k in range(1, len(grp) - 1):
        d2 = xyz[k+1] - 2*xyz[k] + xyz[k-1]
        if np.linalg.norm(d2) > max_accel:
            bad[k] = True
    return bad


def _speed_mask(grp, max_speed):
    """
    Return a boolean array (len = len(grp)) that is True for rows where the
    frame-to-frame 3D speed exceeds max_speed (m/frame).

    Speed at index k: ||X[k] - X[k-1]||  (forward difference, frame units).
    First frame is never flagged.
    """
    xyz = grp[["X", "Y", "Z"]].to_numpy()
    bad = np.zeros(len(grp), dtype=bool)
    for k in range(1, len(grp)):
        if np.linalg.norm(xyz[k] - xyz[k-1]) > max_speed:
            bad[k] = True
    return bad


def filter_tracks_3d(df, min_length=10, max_gap=0, max_accel=None, max_speed=None):
    """
    Filter 3D trajectories.

    Parameters
    ----------
    df         : pd.DataFrame  with columns traj_id, t, X, Y, Z, xL
    min_length : int   minimum number of *matched* (non-interpolated) frames
    max_gap    : int   maximum allowed gap (consecutive interpolated frames).
                       Trajectories are split at any gap exceeding this value;
                       resulting segments shorter than min_length are dropped.
                       0 = no gaps allowed (split at every interpolated frame).
    max_accel  : float or None
                       Maximum allowed 3D acceleration magnitude in m/frame²
                       (i.e. ||X(t+1) - 2X(t) + X(t-1)||).  Offending frames
                       are dropped; the resulting gap is then subject to max_gap.
                       None = disabled.
    max_speed  : float or None
                       Maximum allowed frame-to-frame 3D displacement in m/frame
                       (i.e. ||X(t) - X(t-1)||).  Offending frames are dropped;
                       the resulting gap is then subject to max_gap.
                       None = disabled.

    Returns
    -------
    pd.DataFrame — filtered copy with new contiguous traj_id values
    """
    new_rows = []
    next_id = 0

    for tid, grp in df.groupby("traj_id"):
        grp = grp.sort_values("t").reset_index(drop=True)

        # 1. Drop frames that exceed accel or speed thresholds
        if max_accel is not None:
            accel_bad = _accel_mask(grp, max_accel)
        else:
            accel_bad = np.zeros(len(grp), dtype=bool)

        if max_speed is not None:
            speed_bad = _speed_mask(grp, max_speed)
        else:
            speed_bad = np.zeros(len(grp), dtype=bool)

        cut_bad = accel_bad | speed_bad
        if cut_bad.any():
            grp = grp[~cut_bad].reset_index(drop=True)

        if len(grp) == 0:
            continue

        is_interp = grp["xL"].isna().to_numpy()

        # 2. Split on runs of interpolated frames longer than max_gap
        segment_start = 0
        run = 0

        def _emit(seg_df):
            nonlocal next_id
            if seg_df["xL"].notna().sum() >= min_length:
                out = seg_df.copy()
                out["traj_id"] = next_id
                new_rows.append(out)
                next_id += 1

        for k, interp in enumerate(is_interp):
            if interp:
                run += 1
            else:
                if run > max_gap and k > segment_start:
                    _emit(grp.iloc[segment_start : k - run])
                    segment_start = k
                run = 0

        # final segment
        _emit(grp.iloc[segment_start:])

    if not new_rows:
        return df.iloc[0:0].copy()

    return pd.concat(new_rows, ignore_index=True)

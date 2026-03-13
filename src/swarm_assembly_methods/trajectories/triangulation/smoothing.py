"""Savitzky-Golay smoothing of 3D trajectories with uniform-grid resampling."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _sec_to_window(window_sec, fps):
    """Convert a duration in seconds to the nearest valid (odd) SG window in frames."""
    w = max(3, int(round(window_sec * fps)))
    if w % 2 == 0:
        w += 1
    return w


def smooth_tracks_3d(df, window_sec=0.1, fps=60, polyorder=3):
    """
    Apply Savitzky-Golay smoothing to X, Y, Z per trajectory.

    Gaps in matched frames are linearly interpolated onto a uniform frame grid
    before smoothing (SG requires uniform spacing). Only originally-matched rows
    are returned; interpolated fill rows are discarded after smoothing.

    Parameters
    ----------
    df         : pd.DataFrame  columns include traj_id, t, X, Y, Z
    window_sec : float         smoothing window in seconds (0 = no smoothing)
    fps        : float         camera frame rate
    polyorder  : int           SG polynomial order
    """
    if not window_sec:
        print("  Smoothing: disabled (window_sec=0)")
        return df
    window_length = _sec_to_window(window_sec, fps)
    print(f"  Smoothing: {window_sec}s → {window_length} frames at {fps}fps, "
          f"polyorder={polyorder}")

    out = []
    for tid, grp in df.groupby("traj_id"):
        grp = grp.sort_values("t").copy()
        t_matched = grp["t"].to_numpy(int)

        # build uniform frame grid spanning the trajectory
        t_full = np.arange(t_matched[0], t_matched[-1] + 1)

        # shrink window to fit available frames (keep odd, >= polyorder+1)
        w = min(window_length, len(t_full))
        if w % 2 == 0:
            w -= 1
        min_w = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
        w = max(w, min_w)

        if len(t_full) < w or len(t_full) < 3:
            # too short to smooth — return as-is
            out.append(grp)
            continue

        # save raw values before smoothing
        grp["X_raw"] = grp["X"]
        grp["Y_raw"] = grp["Y"]
        grp["Z_raw"] = grp["Z"]

        for col in ("X", "Y", "Z"):
            # interpolate to uniform grid
            col_uniform = np.interp(t_full, t_matched, grp[col].to_numpy())
            # smooth on uniform grid
            col_smooth = savgol_filter(col_uniform, w, polyorder, mode="nearest")
            # sample back at original matched frames only
            idx = t_matched - t_matched[0]   # indices into t_full
            grp[col] = col_smooth[idx]

        out.append(grp)

    return pd.concat(out).reset_index(drop=True)

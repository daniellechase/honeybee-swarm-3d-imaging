"""Statistics for 3D trajectories."""

import numpy as np


def analyze_3d_tracks(df, fps=60, label=""):
    """
    Print statistics for a 3D trajectory DataFrame.

    Parameters
    ----------
    df    : pd.DataFrame  columns: traj_id, t, X, Y, Z, xL (NaN = interpolated)
    fps   : float         used to convert frame counts to seconds
    label : str           prefix for print output
    """
    tag = f"[{label}] " if label else ""

    if df.empty:
        print(f"{tag}No 3D tracks.")
        return

    tids = df["traj_id"].unique()
    n = len(tids)

    # matched (non-interpolated) frames per track
    # xL is NaN for interpolated rows in our pipeline; if it contains coordinate
    # values (never NaN) the distinction doesn't apply — skip in that case.
    if "xL" in df.columns and df["xL"].isna().any():
        matched_len = df[df["xL"].notna()].groupby("traj_id").size().reindex(tids, fill_value=0).values
    else:
        matched_len = None
    # total frames (including interpolated)
    total_len   = df.groupby("traj_id").size().reindex(tids).values
    # frame span (first to last)
    t_col = "t" if "t" in df.columns else df.columns[1]
    spans = (df.groupby("traj_id")[t_col].max() - df.groupby("traj_id")[t_col].min()).reindex(tids).values

    # per-track path length (3D arc length) and mean speed
    path_lengths = []
    mean_speeds  = []
    for tid in tids:
        grp = df[df["traj_id"] == tid].sort_values(t_col)
        xyz = grp[["X", "Y", "Z"]].to_numpy()
        diffs = np.diff(xyz, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        path_lengths.append(seg_lens.sum())
        dt = grp[t_col].diff().dropna().to_numpy() / fps
        speeds = seg_lens / dt[dt > 0][:len(seg_lens)]
        mean_speeds.append(speeds.mean() if len(speeds) else 0.0)

    path_lengths = np.array(path_lengths)
    mean_speeds  = np.array(mean_speeds)

    def _pcts(arr):
        return (np.percentile(arr, 25), np.percentile(arr, 50),
                np.percentile(arr, 75), np.percentile(arr, 90))

    print(f"\n{tag}3D Trajectory Statistics:")
    print(f"  Total trajectories : {n}")
    if matched_len is not None:
        print(f"  Matched frames     : min={matched_len.min()}  max={matched_len.max()}"
              f"  mean={matched_len.mean():.1f}  median={np.median(matched_len):.1f}")
        p = _pcts(matched_len)
        print(f"                       p25={p[0]:.0f}  p50={p[1]:.0f}  p75={p[2]:.0f}  p90={p[3]:.0f}")
    print(f"  Duration (s)       : min={spans.min()/fps:.2f}  max={spans.max()/fps:.2f}"
          f"  mean={spans.mean()/fps:.2f}")
    print(f"  Path length (m)    : min={path_lengths.min():.3f}  max={path_lengths.max():.3f}"
          f"  mean={path_lengths.mean():.3f}  median={np.median(path_lengths):.3f}")
    p = _pcts(path_lengths)
    print(f"                       p25={p[0]:.3f}  p50={p[1]:.3f}  p75={p[2]:.3f}  p90={p[3]:.3f}")
    print(f"  Mean speed (m/s)   : min={mean_speeds.min():.3f}  max={mean_speeds.max():.3f}"
          f"  mean={mean_speeds.mean():.3f}  median={np.median(mean_speeds):.3f}")
    p = _pcts(mean_speeds)
    print(f"                       p25={p[0]:.3f}  p50={p[1]:.3f}  p75={p[2]:.3f}  p90={p[3]:.3f}")

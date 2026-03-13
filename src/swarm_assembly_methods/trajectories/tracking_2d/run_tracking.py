"""2D tracking core: track one .npy detection file, return tracks dict."""

from .io_utils import load_detections_npy, save_tracks
from .tracker import track_2d_points, interpolate_track_gaps, analyze_2d_tracks


def track_npy_file(npy_path, trk_cfg, out_path, label=""):
    """
    Run 2D tracking on a single .npy detection file.

    Saves the result to out_path and returns the tracks dict.

    Parameters
    ----------
    npy_path : Path   — input detection .npy
    trk_cfg  : dict   — tracking_2d config section
    out_path : Path   — where to save _tracks.npy
    label    : str    — used for progress printing

    Returns
    -------
    dict  {tid: [(frame, x, y, interp), ...]}
    """
    df = load_detections_npy(npy_path)
    n_raw = len(df)

    # Drop detections with NaN/inf coordinates — these cause cost matrix errors
    bad = ~(df["x"].between(-1e6, 1e6) & df["y"].between(-1e6, 1e6))
    if bad.any():
        print(f"  [{label}] Warning: dropping {bad.sum()} detections with invalid coords")
        df = df[~bad]

    print(f"  [{label}] {n_raw} detections, "
          f"frames {df['t'].min()}–{df['t'].max()}")

    tracks, pred_residuals = track_2d_points(
        df,
        t_col="t",
        min_len=trk_cfg.get("min_track_len", 5),
        max_dist=trk_cfg.get("max_dist", 40),
        max_frame_skip=trk_cfg.get("max_frame_skip", 1),
        use_velocity=trk_cfg.get("use_velocity", True),
        angle_penalty_weight=trk_cfg.get("angle_penalty_weight", 0.0),
        min_speed_for_angle=trk_cfg.get("min_speed_for_angle", 2.0),
        vel_ema_alpha=trk_cfg.get("vel_ema_alpha", 0.3),
        verbose=True,
        camera_name=label,
    )

    if trk_cfg.get("interpolate_gaps", True):
        tracks = interpolate_track_gaps(tracks)

    analyze_2d_tracks(tracks, camera_name=label)
    save_tracks(tracks, out_path)
    print(f"  [{label}] Saved 2D tracks: {out_path}")
    return tracks, n_raw, pred_residuals

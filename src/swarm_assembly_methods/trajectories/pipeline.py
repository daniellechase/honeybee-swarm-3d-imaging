"""
Unified trajectories pipeline: 2D tracking → stereo triangulation → smoothing → filtering → visualization.

For each matched GH file pair (left/right):
  1. 2D track left detections       → save {prefix}_left_tracks.parquet
  2. 2D track right detections      → save {prefix}_right_tracks.parquet
  3. Rectify 2D tracks
  4. Stereo match + triangulate     → df3d
  5. Filter trajectories
  6. Smooth X/Y/Z (Savitzky-Golay)  → save {prefix}_3d.parquet
  7. Plot XY/XZ/YZ projections      → save {prefix}_3d_projections.png
"""

import re
import csv
from pathlib import Path

import numpy as np

from .config import get_output_dir
from .tracking_2d.run_tracking import track_npy_file
from .tracking_2d.io_utils import load_tracks as load_tracks_npy_dict
from .triangulation.rectify_tracks import rectify_track_dict
from .triangulation.match_tracks import match_tracks
from .triangulation.io_utils import save_tracks_3d
from .triangulation.smoothing import smooth_tracks_3d
from .triangulation.filtering import filter_tracks_3d
from .triangulation.stats_3d import analyze_3d_tracks
from swarm_assembly_methods.figures.figtraj.quiver import compute_velocities_3d

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swarm_assembly_methods.calibration.io_utils import load_intrinsics_json, load_extrinsics_json
from swarm_assembly_methods.calibration.rectification import compute_rectification


def _gh_prefix(path):
    m = re.search(r"(GH\d+)", path.stem, re.IGNORECASE)
    return m.group(1).upper() if m else path.stem


def _filter_by_gh(files, gh_filter):
    if gh_filter is None:
        return files
    stems = [s.upper() for s in gh_filter]
    return [f for f in files if any(s in f.stem.upper() for s in stems)]


def _tracks_dict_to_arrays(tracks):
    """Convert {tid: [(frame,x,y,interp),...]} to {tid: np.ndarray (N,3) [frame,x,y]}."""
    return {
        tid: np.array([(pt[0], pt[1], pt[2]) for pt in pts], dtype=float)
        for tid, pts in tracks.items()
    }


def run_trajectories(cfg):
    det_cfg  = cfg["detections"]
    trk_cfg  = cfg["tracking_2d"]
    tri_cfg  = cfg["tracking_3d"]
    cal_cfg  = cfg["calibration"]
    sm_cfg   = cfg.get("smoothing", {})
    filt_cfg = cfg.get("filtering", {})
    gh_filter = det_cfg.get("gh_files", None)

    out = get_output_dir(cfg)

    # ---- load calibration once ----
    K1, d1, _ = load_intrinsics_json(Path(cal_cfg["intrinsics_left"]))
    K2, d2, _ = load_intrinsics_json(Path(cal_cfg["intrinsics_right"]))
    R, T_mm, dk_global, size_wh = load_extrinsics_json(Path(cal_cfg["extrinsics"]))
    baseline_m = float(np.linalg.norm(T_mm)) / 1000.0

    # ---- load per-video dk from calibration yaml sweep_dk.results ----
    dk_per_video = {}
    cal_yaml_path = cal_cfg.get("calibration_yaml")
    if cal_yaml_path:
        import yaml as _yaml
        with open(cal_yaml_path) as f:
            cal_yaml = _yaml.safe_load(f)
        dk_per_video = cal_yaml.get("sweep_dk", {}).get("results", {}) or {}
        dk_per_video = {k.upper(): int(v) for k, v in dk_per_video.items()}
        print(f"Loaded per-video dk from calibration yaml: {dk_per_video}")
    print(f"Loaded extrinsics: baseline={baseline_m:.4f} m, global dk={dk_global}")

    W, H = int(size_wh[0]), int(size_wh[1])
    alpha = tri_cfg.get("rectification_alpha", 0.0)
    (R1, R2, P1, P2), _ = compute_rectification(K1, d1, K2, d2, R, T_mm, W, H, alpha=alpha)

    fx_rect = float(P1[0, 0])
    fy_rect = float(P1[1, 1])
    cx      = float(P1[0, 2])
    cy      = float(P1[1, 2])

    # ---- pair left/right npy files by order in gh_files list ----
    # gh_files: [left_stem, right_stem] pairs by index (left[0] <-> right[0], etc.)
    # Each side is filtered independently then zipped in order.
    left_files  = _filter_by_gh(sorted(Path(det_cfg["left_npy_dir"]).glob("*.npy")),  gh_filter)
    right_files = _filter_by_gh(sorted(Path(det_cfg["right_npy_dir"]).glob("*.npy")), gh_filter)

    if len(left_files) != len(right_files):
        print(f"Warning: {len(left_files)} left files but {len(right_files)} right files — zipping by order.")

    for left_npy, right_npy in zip(left_files, right_files):
        prefix = _gh_prefix(left_npy)

        # look up per-video dk: sweep_dk keys are like "GH43" (first 4 chars)
        # match against the full prefix (e.g. "GH430142") by checking which key is a prefix of it
        dk = dk_global
        for key, val in dk_per_video.items():
            if prefix.upper().startswith(key.upper()):
                dk = val
                break
        print(f"\n{'='*60}")
        print(f"Processing {prefix}  (dk={dk})")
        print(f"  L: {left_npy.name}")
        print(f"  R: {right_npy.name}")

        # ---- 2D tracking ----
        left_out  = out / f"{prefix}_left_tracks.parquet"
        right_out = out / f"{prefix}_right_tracks.parquet"

        print("\n-- 2D tracking (left) --")
        tracks_L_raw, n_raw_L, resid_L = track_npy_file(left_npy,  trk_cfg, left_out,  label=f"L/{prefix}")

        print("\n-- 2D tracking (right) --")
        tracks_R_raw, n_raw_R, resid_R = track_npy_file(right_npy, trk_cfg, right_out, label=f"R/{prefix}")

        # ---- align right frames to left frame numbers using dk ----
        # sweep_dk convention: right_frame + dk = left_frame
        # so add dk to right frame numbers to put them on left's timeline
        tracks_R_arrays = _tracks_dict_to_arrays(tracks_R_raw)
        if dk != 0:
            tracks_R_arrays = {
                tid: np.column_stack([pts[:, 0] + dk, pts[:, 1:]])
                for tid, pts in tracks_R_arrays.items()
            }

        # ---- rectify track points ----
        tracks_L = rectify_track_dict(_tracks_dict_to_arrays(tracks_L_raw), K1, d1, R1, P1)
        tracks_R = rectify_track_dict(tracks_R_arrays, K2, d2, R2, P2)

        # ---- stereo match + triangulate ----
        print("\n-- Stereo matching + triangulation --")
        df3d = match_tracks(
            track_L=tracks_L,
            track_R=tracks_R,
            fx_rect=fx_rect,
            fy_rect=fy_rect,
            cx=cx,
            cy=cy,
            baseline_m=baseline_m,
            zmin_m=tri_cfg.get("zmin_m", 0.4),
            zmax_m=tri_cfg.get("zmax_m", 2.1),
            y_thresh_px=tri_cfg["y_thresh_px"],
            use_velocity_prior=tri_cfg.get("use_velocity_prior", True),
            max_3d_dist=tri_cfg.get("max_3d_dist", 0.15),
            max_pred_error=tri_cfg.get("max_pred_error", None),
            vel_ema_alpha=tri_cfg.get("vel_ema_alpha", 0.3),
            smoothness_lambda=tri_cfg.get("smoothness_lambda", 0.0),
        )
        n_3d_pts   = len(df3d)
        n_3d_trajs = df3d["traj_id"].nunique()
        print(f"  3D trajectories: {n_3d_trajs}, total points: {n_3d_pts}")

        # ---- filter ----
        if filt_cfg:
            df3d = filter_tracks_3d(
                df3d,
                min_length=filt_cfg.get("min_length", 10),
                max_gap=filt_cfg.get("max_gap", 0),
                max_accel=filt_cfg.get("max_accel", None),
                max_speed=filt_cfg.get("max_speed", None),
            )
            print(f"  After filtering: {df3d['traj_id'].nunique()} trajectories")

        n_3d_filt_pts   = len(df3d)
        n_3d_filt_trajs = df3d["traj_id"].nunique()

        # ---- smooth ----
        if sm_cfg:
            fps = cfg["session"].get("fps", 60)
            df3d = smooth_tracks_3d(
                df3d,
                window_sec=sm_cfg.get("window_sec", 0.1),
                fps=fps,
                polyorder=sm_cfg.get("polyorder", 3),
            )

        fps = cfg["session"].get("fps", 60)
        analyze_3d_tracks(df3d, fps=fps, label=prefix)

        # ---- compact traj_ids to 0-based sequential integers ----
        df3d["traj_id_orig"] = df3d["traj_id"]
        id_map = {old: new for new, old in enumerate(sorted(df3d["traj_id"].unique()))}
        df3d["traj_id"] = df3d["traj_id"].map(id_map)

        # ---- compute velocities (needed for quiver plots + saved to parquet) ----
        df3d = compute_velocities_3d(df3d, fps)

        out_path = save_tracks_3d(df3d, out / f"{prefix}_3d")
        print(f"  Saved 3D tracks: {out_path}")

        # ---- detection funnel table ----
        n_2d_matched_L = sum(
            1 for pts in tracks_L_raw.values() for pt in pts if pt[3] == 0
        )
        n_2d_matched_R = sum(
            1 for pts in tracks_R_raw.values() for pt in pts if pt[3] == 0
        )
        rows = [
            ("Raw detections (left)",          n_raw_L,         "left camera"),
            ("Raw detections (right)",          n_raw_R,         "right camera"),
            ("In 2D tracks, matched (left)",    n_2d_matched_L,  "left camera"),
            ("In 2D tracks, matched (right)",   n_2d_matched_R,  "right camera"),
            ("In 3D trajectories",              n_3d_pts,        f"{n_3d_trajs} trajs"),
            ("After length/gap filter",         n_3d_filt_pts,   f"{n_3d_filt_trajs} trajs"),
        ]
        col_w = max(len(r[0]) for r in rows)
        sep = "-" * (col_w + 36)
        print(f"\n{'Detection funnel':^{col_w + 36}}")
        print(sep)
        print(f"  {'Stage':<{col_w}}  {'Count':>10}  {'Notes':>14}")
        print(sep)
        for stage, count, notes in rows:
            print(f"  {stage:<{col_w}}  {str(count):>10}  {str(notes):>14}")
        print(sep)

        csv_path = out / f"{prefix}_detection_funnel.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "count", "notes"])
            writer.writerows(rows)
        print(f"  Saved detection funnel: {csv_path}")



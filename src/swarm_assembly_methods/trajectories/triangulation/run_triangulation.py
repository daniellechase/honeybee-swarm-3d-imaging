"""Core logic: rectify 2D tracks, match stereo, triangulate to 3D."""

from pathlib import Path

import numpy as np

from .config import get_output_paths
from .io_utils import load_tracks_npy, save_tracks_3d
from .match_tracks import match_tracks
from .rectify_tracks import rectify_track_dict

# Reuse calibration helpers
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from swarm_assembly_methods.calibration.io_utils import load_intrinsics_json, load_extrinsics_json
from swarm_assembly_methods.calibration.rectification import compute_rectification


def run_triangulation(cfg):
    tri_cfg  = cfg["triangulation"]
    cal_cfg  = cfg["calibration"]
    det_cfg  = cfg["detections"]
    out_cfg  = cfg["output"]

    # ---- load calibration ----
    K1, d1, _ = load_intrinsics_json(Path(cal_cfg["intrinsics_left"]))
    K2, d2, _ = load_intrinsics_json(Path(cal_cfg["intrinsics_right"]))
    R, T_mm, _, size_wh = load_extrinsics_json(Path(cal_cfg["extrinsics"]))
    T_m = T_mm / 1000.0
    baseline_m = float(np.linalg.norm(T_m))

    W, H = int(size_wh[0]), int(size_wh[1])
    alpha = tri_cfg.get("rectification_alpha", 0.0)
    (R1, R2, P1, P2), _ = compute_rectification(K1, d1, K2, d2, R, T_mm, W, H, alpha=alpha)

    fx_rect = float(P1[0, 0])
    fy_rect = float(P1[1, 1])
    cx      = float(P1[0, 2])
    cy      = float(P1[1, 2])

    # ---- locate track files ----
    left_dir  = Path(det_cfg["left_tracks_dir"])
    right_dir = Path(det_cfg["right_tracks_dir"])
    gh_filter = det_cfg.get("gh_files", None)

    left_files  = sorted(left_dir.glob("*_tracks.npy"))
    right_files = sorted(right_dir.glob("*_tracks.npy"))

    if gh_filter is not None:
        stems = [s.upper() for s in gh_filter]
        left_files  = [f for f in left_files  if any(s in f.stem.upper() for s in stems)]
        right_files = [f for f in right_files if any(s in f.stem.upper() for s in stems)]

    # pair left/right files by GH prefix
    def _gh_prefix(p):
        import re
        m = re.search(r"(GH\d+)", p.stem, re.IGNORECASE)
        return m.group(1).upper() if m else p.stem

    right_map = {_gh_prefix(f): f for f in right_files}
    out_paths = get_output_paths(cfg)

    for left_file in left_files:
        prefix = _gh_prefix(left_file)
        right_file = right_map.get(prefix)
        if right_file is None:
            print(f"  No matching right file for {left_file.name}, skipping.")
            continue

        print(f"\nProcessing {prefix}")
        print(f"  L: {left_file.name}")
        print(f"  R: {right_file.name}")

        track_L_raw = load_tracks_npy(left_file)
        track_R_raw = load_tracks_npy(right_file)
        print(f"  Loaded {len(track_L_raw)} left tracks, {len(track_R_raw)} right tracks")

        # ---- rectify 2D track points ----
        track_L = rectify_track_dict(track_L_raw, K1, d1, R1, P1)
        track_R = rectify_track_dict(track_R_raw, K2, d2, R2, P2)

        # ---- stereo match + triangulate ----
        df3d = match_tracks(
            track_L=track_L,
            track_R=track_R,
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
            vel_ema_alpha=tri_cfg.get("vel_ema_alpha", 0.3),
        )

        print(f"  3D trajectories: {df3d['traj_id'].nunique()}, "
              f"total points: {len(df3d)}")

        out_file = out_paths["tracks_3d"] / f"{prefix}_3d"
        save_tracks_3d(df3d, out_file)
        print(f"  Saved: {out_file}.parquet")

"""
Visualization pipeline: load 3D parquet tracks → quiver + traj count plots.

Reads all *_3d.parquet files from tracks_dir and produces PDFs in figures_dir.
"""

import sys
from pathlib import Path

from .config import get_tracks_dir, get_figures_dir
from .quiver import (
    plot_combined_quiver_and_projections,
    plot_depth_sliced_quiver,
    plot_traj_count_3proj,
    plot_traj_count_histogram,
    load_localization,
    transform_shed_to_gate,
)
from .boundary import load_boundary
from .trajectory import plot_trajectory_on_video_frames

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swarm_assembly_methods.utils import no_overwrite_path
from swarm_assembly_methods.calibration.io_utils import load_intrinsics_json, load_extrinsics_json
from swarm_assembly_methods.calibration.rectification import compute_rectification
from swarm_assembly_methods.trajectories.tracking_2d.io_utils import load_tracks as load_2d_tracks


def _resolve_path(p, data_root):
    """Resolve p against data_root if it is a relative path."""
    if not p:
        return p
    path = Path(p)
    if data_root and not path.is_absolute():
        return str(Path(data_root) / path)
    return str(path)


def _load_boundary_for_prefix(bnd_cfg, prefix, data_root=None):
    """
    Load boundary for a given parquet prefix.

    boundary config can specify masks in two ways:

    1. Single pair (applied to all parquet files):
         mask_left:  /path/left.npz
         mask_right: /path/right.npz

    2. Per-video pairs list (each entry matched by prefix substring):
         mask_pairs:
           - key: GH440142
             left:  /path/GH440142_left.npz
             right: /path/GH440269_right.npz
           - key: GH430142
             left:  /path/GH430142_left.npz
             right: /path/GH430269_right.npz
    """
    if not bnd_cfg:
        return None

    mask_pairs = bnd_cfg.get("mask_pairs")
    if mask_pairs:
        mask_left = mask_right = None
        for pair in mask_pairs:
            if str(pair["key"]).upper() in prefix.upper():
                mask_left  = pair["left"]
                mask_right = pair["right"]
                break
        if not mask_left or not mask_right:
            print(f"  No boundary mask pair matched prefix '{prefix}' — skipping boundary.")
            return None
    else:
        mask_left  = bnd_cfg.get("mask_left")
        mask_right = bnd_cfg.get("mask_right")
        if not mask_left or not mask_right:
            return None

    return load_boundary(
        mask_left_path   = _resolve_path(mask_left,                data_root),
        mask_right_path  = _resolve_path(mask_right,               data_root),
        intrinsics_left  = _resolve_path(bnd_cfg["intrinsics_left"],  data_root),
        intrinsics_right = _resolve_path(bnd_cfg["intrinsics_right"], data_root),
        extrinsics       = _resolve_path(bnd_cfg["extrinsics"],       data_root),
        flat_z           = bnd_cfg.get("flat_z", True),
        min_width_px     = bnd_cfg.get("min_width_px", 0),
        flat_top         = bnd_cfg.get("flat_top", False),
    )


def run_visualization(cfg, parquet_file=None):
    viz_cfg = cfg["visualization"]
    fig_dir = get_figures_dir(cfg)
    _no_ow = cfg.get("no_overwrite", False)

    def sp(path):
        """Resolve save path, optionally avoiding overwrite."""
        return no_overwrite_path(path) if _no_ow else path

    bnd_cfg = cfg.get("boundary", {})

    # ---- Load board extents (optional) ----
    board_kwargs = {}
    board_cfg = cfg.get("board", {})
    data_root = cfg.get("data_root")
    board_extents_path = board_cfg.get("extents_json")
    if board_extents_path:
        if data_root and not Path(board_extents_path).is_absolute():
            board_extents_path = str(Path(data_root) / board_extents_path)
        import json
        with open(board_extents_path) as _f:
            _extents = json.load(_f)
        # extents_json may have a single key (e.g. "gate") or be keyed by video prefix
        board_key = board_cfg.get("key")
        if board_key and board_key in _extents:
            _e = _extents[board_key]
        else:
            _e = next(iter(_extents.values()))
        board_kwargs = dict(
            x_min_m       = _e["x_min_m"],
            x_max_m       = _e["x_max_m"],
            x_center_m    = _e["x_center_m"],
            z_center_m    = _e["z_center_m"],
            y_center_m    = float(board_cfg.get("y_center_m", 0.1)),
            rect_height_m = float(board_cfg.get("rect_height_m", 0.02)),
            fill_alpha    = float(board_cfg.get("fill_alpha", 1.0)),
            color         = board_cfg.get("color", "black"),
            linewidth     = float(board_cfg.get("linewidth", 1.5)),
            linestyle     = board_cfg.get("linestyle", "--"),
        )
        print(f"Board extents loaded: x=[{board_kwargs['x_min_m']}, {board_kwargs['x_max_m']}] "
              f"z_center={board_kwargs['z_center_m']} y_center={board_kwargs['y_center_m']}")

    _board_kwargs_orig = dict(board_kwargs)  # snapshot before loop (unshifted)

    if parquet_file is not None:
        parquet_files = [Path(parquet_file)]
        tracks_dir = parquet_files[0].parent
    else:
        tracks_dir = get_tracks_dir(cfg)
        parquet_files = sorted(tracks_dir.glob("*_3d.parquet"))
    if not parquet_files:
        print(f"No *_3d.parquet files found in {tracks_dir}")
        return

    quiver_kwargs = dict(
        x_range          = viz_cfg.get("x_range"),
        y_range          = viz_cfg.get("y_range"),
        z_range          = viz_cfg.get("z_range"),
        t_range          = viz_cfg.get("t_range"),
        speed_min        = viz_cfg.get("speed_min"),
        speed_max        = viz_cfg.get("speed_max"),
        bin_size_m       = viz_cfg.get("bin_size_m", 0.1),
        bin_size_m_3d    = viz_cfg.get("bin_size_m_3d", 0.1),
        min_count        = viz_cfg.get("min_count", 10),
        quiver_stride    = viz_cfg.get("quiver_stride", 1),
        arrow_scale      = viz_cfg.get("arrow_scale", 1.5),
        arrow_width      = viz_cfg.get("arrow_width", 0.008),
        arrow_color_2d   = viz_cfg.get("arrow_color_2d", "black"),
        arrow_length     = viz_cfg.get("arrow_length", 0.05),
        arrow_color_3d   = viz_cfg.get("arrow_color_3d", "black"),
        vmin             = viz_cfg.get("vmin"),
        vmax             = viz_cfg.get("vmax"),
        colormap         = viz_cfg.get("colormap", "plasma"),
        normalize_arrows = viz_cfg.get("normalize_arrows", "flux"),
        elev             = viz_cfg.get("elev", 10),
        azim             = viz_cfg.get("azim", -65),
        figsize          = tuple(viz_cfg.get("figsize", [26, 6])),
        dpi              = viz_cfg.get("dpi", 300),
    )

    for parquet_path in parquet_files:
        prefix = parquet_path.stem.replace("_3d", "")
        sub_dir = fig_dir
        sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Visualizing {prefix}  →  {sub_dir}")

        board_kwargs = dict(_board_kwargs_orig)  # reset to unshifted for each parquet

        bnd = _load_boundary_for_prefix(bnd_cfg, prefix, data_root=cfg.get("data_root"))
        boundary_kwargs = dict(
            boundary_outline      = bnd["outline"]       if bnd else None,
            boundary_axis_center  = bnd["axis_center"]   if bnd else None,
            boundary_radius       = bnd["radius"]         if bnd else None,
            boundary_color        = bnd_cfg.get("color", "black"),
            boundary_linewidth    = bnd_cfg.get("linewidth", 2),
        )

        df = pd.read_parquet(parquet_path)
        if df["traj_id"].nunique() == 0:
            print("  No trajectories — skipping.")
            continue
        if "vx" not in df.columns or df["vx"].isna().all():
            print("  No velocity data — run analysis pipeline first.")
            continue

        # ---- optional: coordinate transform (shed → gate) ----
        loc_cfg = cfg.get("localization")
        if loc_cfg and loc_cfg.get("source", "").lower() == "shed":
            loc_path = Path(loc_cfg["localization_json"])
            if not loc_path.exists():
                print(f"  Warning: localization_json not found: {loc_path}")
            else:
                R_loc, T_loc = load_localization(loc_path)
                df = transform_shed_to_gate(df, R_loc, T_loc)
                print(f"  Transformed {prefix} from shed → gate frame.")

        # ---- optional: re-center coordinates ----
        import numpy as _np
        origin = viz_cfg.get("origin_m")
        if origin is not None:
            ox, oz = float(origin[0]), float(origin[2])
        else:
            ox = oz = 0.0

        # auto y origin: use flat_top boundary y per video so swarm top is always at y=0
        if board_cfg.get("y_from_boundary", False) and bnd is not None:
            oy = float(bnd["outline"][0, 1])  # flat_top y in original coords (metres)
            print(f"  Auto y origin from boundary flat_top: oy={oy:.6f} m")
        elif origin is not None:
            oy = float(origin[1])
        else:
            oy = 0.0

        if ox != 0.0 or oy != 0.0 or oz != 0.0:
            df = df.copy()
            df["X"] -= ox;  df["Y"] -= oy;  df["Z"] -= oz
            print(f"  Re-centered: subtracted [{ox:.4f}, {oy:.4f}, {oz:.4f}]")
            # shift boundary
            if bnd is not None:
                bnd = {k: v for k, v in bnd.items()}
                bnd["outline"]     = bnd["outline"]     - _np.array([ox, oy, oz])
                bnd["axis_center"] = bnd["axis_center"] - _np.array([ox, oy, oz])
                boundary_kwargs = dict(
                    boundary_outline      = bnd["outline"],
                    boundary_axis_center  = bnd["axis_center"],
                    boundary_radius       = bnd["radius"],
                    boundary_color        = bnd_cfg.get("color", "black"),
                    boundary_linewidth    = bnd_cfg.get("linewidth", 2),
                )
            # shift board x/z (measured); y_center_m = 0 since flat_top is now at y=0
            if board_kwargs:
                board_kwargs = {**board_kwargs,
                    "x_min_m":    board_kwargs["x_min_m"]    - ox,
                    "x_max_m":    board_kwargs["x_max_m"]    - ox,
                    "x_center_m": board_kwargs["x_center_m"] - ox,
                    "z_center_m": board_kwargs["z_center_m"] - oz,
                    "y_center_m": 0.0,  # flat_top is now at y=0
                }

        _board_extents = board_kwargs or None

        print("-- Quiver plot --")
        plot_combined_quiver_and_projections(
            df, **quiver_kwargs, **boundary_kwargs,
            board_extents=_board_extents,
            save_path=sp(sub_dir / f"{prefix}_quiver.pdf"),
        )

        print("-- Quiver plot (equal-length arrows) --")
        equal_arrow_scale = viz_cfg.get("equal_arrow_scale", quiver_kwargs["arrow_scale"])
        plot_combined_quiver_and_projections(
            df, **{**quiver_kwargs, "normalize_arrows": True, "arrow_scale": equal_arrow_scale},
            **boundary_kwargs,
            board_extents=_board_extents,
            save_path=sp(sub_dir / f"{prefix}_quiver_equal_arrows.pdf"),
        )

        print("-- Trajectory count plots --")
        count_kwargs = dict(
            x_range    = quiver_kwargs["x_range"],
            y_range    = quiver_kwargs["y_range"],
            z_range    = quiver_kwargs["z_range"],
            t_range    = quiver_kwargs["t_range"],
            bin_size_m = quiver_kwargs["bin_size_m"],
            colormap   = viz_cfg.get("count_colormap", "YlGnBu"),
            figsize    = quiver_kwargs["figsize"],
            dpi        = quiver_kwargs["dpi"],
            **boundary_kwargs,
        )
        plot_traj_count_3proj(df, metric="unique",
                              save_path=sp(sub_dir / f"{prefix}_traj_count.pdf"),
                              board_extents=_board_extents,
                              **count_kwargs)
        plot_traj_count_3proj(df, metric="density",
                              save_path=sp(sub_dir / f"{prefix}_traj_length.pdf"),
                              board_extents=_board_extents,
                              **count_kwargs)

        # ---- trajectory video overlay (optional) ----
        _run_traj_for_prefix(cfg, df, tracks_dir, sub_dir, prefix)


def _dk_for_prefix(traj_cfg, prefix):
    """Look up per-video dk from calibration yaml sweep_dk.results, fall back to traj_cfg.dk."""
    import yaml as _yaml
    cal_yaml_path = traj_cfg.get("calibration_yaml")
    if cal_yaml_path:
        with open(cal_yaml_path) as f:
            cal = _yaml.safe_load(f)
        results = (cal.get("sweep_dk") or {}).get("results") or {}
        results = {k.upper(): int(v) for k, v in results.items()}
        for key, val in results.items():
            if prefix.upper().startswith(key.upper()):
                return val
    return traj_cfg.get("dk", 0)


def _run_traj_for_prefix(cfg, df, tracks_dir, fig_dir, prefix):
    import numpy as np

    traj_cfg = cfg.get("trajectory")
    if not traj_cfg:
        return

    if not {"xL", "yL", "xR", "yR"}.issubset(df.columns):
        print(f"  No xL/yL/xR/yR columns in 3d parquet — skipping trajectory overlay.")
        return

    print("-- Trajectory video overlay --")

    # Build track_L from xL/yL already in the 3d parquet (already rectified)
    has_orig = "traj_id_orig" in df.columns
    track_L = {}
    for tid, grp in df.groupby("traj_id"):
        orig_id = int(grp["traj_id_orig"].iloc[0]) if has_orig else int(tid)
        pts = grp[["t", "xL", "yL"]].dropna().sort_values("t")
        if len(pts) > 0:
            track_L[orig_id] = pts[["t", "xL", "yL"]].values.tolist()

    K1, d1, _ = load_intrinsics_json(Path(traj_cfg["intrinsics_left"]))
    K2, d2, _ = load_intrinsics_json(Path(traj_cfg["intrinsics_right"]))
    R, T_mm, _, size_wh = load_extrinsics_json(Path(traj_cfg["extrinsics"]))
    W, H = int(size_wh[0]), int(size_wh[1])
    (R1, R2, P1, P2), _ = compute_rectification(K1, d1, K2, d2, R, T_mm, W, H, alpha=0.0)

    dk = _dk_for_prefix(traj_cfg, prefix)
    print(f"  dk={dk}")

    plot_trajectory_on_video_frames(
        video_path_left  = traj_cfg["video_left"],
        video_path_right = traj_cfg["video_right"],
        K1=K1, d1=d1, R1=R1, P1r=P1,
        K2=K2, d2=d2, R2=R2, P2r=P2,
        df3d             = df,
        track_L          = track_L,
        dk               = dk,
        traj_id          = traj_cfg.get("traj_id"),
        pick             = traj_cfg.get("pick", "longest"),
        n_trajectories   = traj_cfg.get("n_trajectories", 5),
        length_min       = traj_cfg.get("length_min"),
        length_max       = traj_cfg.get("length_max"),
        t_min            = traj_cfg.get("t_min"),
        t_max            = traj_cfg.get("t_max"),
        random_seed      = traj_cfg.get("random_seed"),
        trail_color_start = tuple(traj_cfg.get("trail_color_start", [255, 255, 0])),
        trail_color_end   = tuple(traj_cfg.get("trail_color_end",   [0, 0, 255])),
        point_radius     = traj_cfg.get("point_radius", 5),
        line_thickness   = traj_cfg.get("line_thickness", 2),
        figsize          = tuple(traj_cfg.get("figsize", [20, 7])),
        dpi              = traj_cfg.get("dpi", 150),
        debug_interp     = traj_cfg.get("debug_interp", False),
        save_dir         = fig_dir / "trajectories",
        prefix           = prefix,
    )


def run_trajectory_visualization(cfg):
    """Run only trajectory video overlays (skips quiver/count plots)."""
    tracks_dir = get_tracks_dir(cfg)
    fig_dir    = get_figures_dir(cfg)

    parquet_files = sorted(tracks_dir.glob("*_3d.parquet"))
    if not parquet_files:
        print(f"No *_3d.parquet files found in {tracks_dir}")
        return

    for parquet_path in parquet_files:
        prefix = parquet_path.stem.replace("_3d", "")
        sub_dir = fig_dir
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Trajectory overlay: {prefix}  →  {sub_dir}")
        df = pd.read_parquet(parquet_path)
        if df["traj_id"].nunique() == 0:
            print("  No trajectories — skipping.")
            continue
        _run_traj_for_prefix(cfg, df, tracks_dir, sub_dir, prefix)

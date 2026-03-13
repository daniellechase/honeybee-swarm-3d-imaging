import argparse
from .config import load_config
from .pipeline import run_trajectories
from swarm_assembly_methods.utils import rename_config_with_run_id, resolve_config_path


def main():
    parser = argparse.ArgumentParser(
        description="Run 2D tracking + stereo triangulation pipeline."
    )
    parser.add_argument("--config", required=True, help="Path to trajectories YAML config")
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    rename_config_with_run_id(config_path, cfg)
    run_trajectories(cfg)


def main_stats():
    import pandas as pd
    from pathlib import Path
    from .triangulation.stats_3d import analyze_3d_tracks
    from .triangulation.visualize_3d import plot_3d_projections

    parser = argparse.ArgumentParser(
        description="Print stats and save projection plots for an existing 3D parquet."
    )
    parser.add_argument("--parquet", required=True, help="Path to *_3d.parquet file")
    parser.add_argument("--fps", type=float, default=60.0, help="Camera FPS (default: 60)")
    parser.add_argument("--n-tracks", type=int, default=15, help="Tracks to show in projection plot (default: 15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    df = pd.read_parquet(parquet_path)
    print(f"Columns: {list(df.columns)}")
    prefix = parquet_path.stem.replace("_3d", "")

    required = {"traj_id", "X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        print(f"  Skipping stats — missing columns: {missing}")
    else:
        analyze_3d_tracks(df, fps=args.fps, label=prefix)

    proj_required = {"traj_id", "X", "Y", "Z"}
    if not (proj_required - set(df.columns)):
        out_path = parquet_path.with_name(f"{prefix}_3d_projections.png")
        plot_3d_projections(df, n_tracks=args.n_tracks, seed=args.seed,
                            fps=args.fps, out_path=out_path)
        print(f"  Saved projections: {out_path}")
    else:
        print("  Skipping projections — missing required columns.")

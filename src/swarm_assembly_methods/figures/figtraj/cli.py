import argparse
from .config import load_config
from .quiver_pipeline import run_visualization, run_trajectory_visualization


def main_quiver():
    parser = argparse.ArgumentParser(
        description="Quiver + trajectory count figures from 3D tracks."
    )
    parser.add_argument("--config", required=True, help="Path to figures YAML config")
    parser.add_argument("--tracks-dir", help="Override input.tracks_dir from config")
    parser.add_argument("--parquet", help="Run on a single parquet file instead of scanning tracks_dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.tracks_dir:
        cfg["input"]["tracks_dir"] = args.tracks_dir
    run_visualization(cfg, parquet_file=args.parquet)


def main_trajectory():
    parser = argparse.ArgumentParser(
        description="Trajectory video overlay figures."
    )
    parser.add_argument("--config", required=True, help="Path to figures YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_trajectory_visualization(cfg)

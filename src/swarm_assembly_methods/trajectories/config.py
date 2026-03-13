"""Unified config loading and output path resolution for trajectories pipeline."""

from pathlib import Path
import yaml


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_output_dir(cfg):
    """
    Return the tracks output directory, inserting run_id after the root if present.

    Example:
        output_dir: "data/tracks/S02/0722"
        run_id:     "paper"
        → data/paper/tracks/S02/0722/
    """
    base = Path(cfg["output"]["output_dir"])
    run_id = cfg.get("run_id")
    if run_id:
        parts = base.parts
        out = Path(parts[0]) / str(run_id) / Path(*parts[1:])
    else:
        out = base
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_figures_dir(cfg):
    """Return the figures directory, creating it if needed."""
    out = Path(cfg["output"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out

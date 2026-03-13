"""Configuration loading and output path resolution for triangulation."""

from pathlib import Path
import yaml


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_output_paths(cfg):
    out_dir = Path(cfg["output"]["tracks_3d_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"tracks_3d": out_dir}

"""
Configuration loading and output path resolution for 2D tracking.
"""

from pathlib import Path
import yaml


def load_config(yaml_path):
    """Load a YAML config file and return as a dict."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_output_paths(cfg):
    """
    Derive output directory paths from config.

    Returns a dict with keys:
        "left"  -> Path  (tracks_dir / left_camera)
        "right" -> Path  (tracks_dir / right_camera)

    Both directories are created if they do not exist.
    """
    tracks_dir = Path(cfg["output"]["tracks_dir"])
    paths = {
        "left":  tracks_dir / "left_camera",
        "right": tracks_dir / "right_camera",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

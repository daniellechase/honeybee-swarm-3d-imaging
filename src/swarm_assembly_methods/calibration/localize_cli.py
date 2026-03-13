import argparse
from pathlib import Path

import yaml

from .stereo_localize import run_stereo_localize


def main():
    parser = argparse.ArgumentParser(
        description="Interactively localize two stereo pairs relative to each other."
    )
    parser.add_argument("--config", required=True, help="Path to stereo-localize YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_stereo_localize(cfg)

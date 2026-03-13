import argparse
from pathlib import Path

import yaml

from .board_localize import run_board_localize

_REPO_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(
        description="Interactively localize board X-extents and Z-center from video frames."
    )
    parser.add_argument("--config", required=True,
                        help="Path to board-localize YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg.get("data_root") or str(_REPO_ROOT)
    run_board_localize(cfg, data_root=data_root)

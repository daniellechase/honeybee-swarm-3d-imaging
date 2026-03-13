"""Load and validate morphology YAML config."""

from pathlib import Path
from typing import Any

import yaml


_REQUIRED = ["data", "session"]


def load_config(yaml_path: str | Path) -> dict[str, Any]:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    for key in _REQUIRED:
        if key not in cfg:
            raise KeyError(f"Missing required config section: '{key}'")
    return cfg


def get_output_dir(cfg: dict) -> Path:
    plots_cfg = cfg.get("plots", {})
    d = plots_cfg.get("save_dir", "figures/figmorph")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cache_path(cfg: dict) -> Path | None:
    mc = cfg.get("metrics_cache", {})
    path = mc.get("path")
    return Path(path) if path else None


def use_cache(cfg: dict) -> bool:
    mc = cfg.get("metrics_cache", {})
    return bool(mc.get("use_cache", True))

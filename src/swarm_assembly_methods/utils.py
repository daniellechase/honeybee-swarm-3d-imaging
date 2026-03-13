"""Shared utilities for all pipelines."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def resolve_session_paths(
    raw_root: str,
    processed_root: str,
    figures_root: str,
    subject: str,
    date: str,
    stereo_side: str,
    run_id: str | None = None,
) -> dict:
    """
    Derive standard data paths from session identifiers.

    Returns a dict with keys:
        mask_base_left, mask_base_right,
        video_base_left,
        bee_detect_base_left, bee_detect_base_right,
        weight_csv,       (first *.csv in scale dir, or None if not found)
        metrics_cache,
        save_dir

    Any key can be overridden by setting it explicitly in the config's input section.
    """
    rr  = Path(raw_root)
    pr  = Path(processed_root)
    fr  = Path(figures_root)
    tag = f"gopro_pair_{stereo_side}"
    sd  = f"{subject}_{date}"

    scale_dir = rr / subject / date / "scale"
    weight_csv: str | None = None
    if scale_dir.is_dir():
        csvs = sorted(scale_dir.glob("*.csv"))
        if csvs:
            weight_csv = str(csvs[0])

    return dict(
        mask_base_left        = str(rr / "swarm_detection" / subject / date / tag / "left_camera"),
        mask_base_right       = str(rr / "swarm_detection" / subject / date / tag / "right_camera"),
        video_base_left       = str(rr / subject / date / tag / "left_camera"),
        bee_detect_base_left  = str(rr / "bee_detection"  / subject / date / tag / "left_camera"),
        bee_detect_base_right = str(rr / "bee_detection"  / subject / date / tag / "right_camera"),
        weight_csv            = weight_csv,
        metrics_cache         = str(pr / "morphology" / subject / date / f"{sd}_metrics.parquet"),
        save_dir              = str(fr / "figmorph" / subject / date / run_id) if run_id else str(fr / "figmorph" / subject / date),
    )


def resolve_save_dir(cfg: dict, key: str, default: str) -> Path:
    """
    Resolve the output directory, inserting run_id after the root component.

    Without run_id: returns save_dir as-is.
    With run_id:    inserts run_id as the second path component.

    Example:
        save_dir: "figures/figmorph/S02_0722"
        run_id:   "030526_00"
        → figures/030526_00/morphology/S02_0722/
    """
    base = Path(cfg.get(key, default))
    run_id = cfg.get("run_id")
    if not run_id:
        return base
    parts = base.parts
    return Path(parts[0]) / str(run_id) / Path(*parts[1:])


def resolve_config_path(config_path: str | Path) -> Path:
    """
    Resolve a config path that may have already been renamed with a run_id.

    If the given path doesn't exist, look for a file in the same directory
    whose stem starts with the given stem (e.g. S02_0722_paper.yaml when
    S02_0722.yaml was passed). Raises FileNotFoundError if nothing matches.
    """
    config_path = Path(config_path)
    if config_path.exists():
        return config_path
    # Original file was renamed — look for stem_*.yaml in the same directory
    candidates = sorted(config_path.parent.glob(f"{config_path.stem}_*.yaml"))
    if len(candidates) == 1:
        print(f"  Config path resolved: {config_path.name} → {candidates[0].name}")
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Config '{config_path}' not found and multiple renamed variants exist: "
            + ", ".join(c.name for c in candidates)
            + ". Pass the full path explicitly."
        )
    raise FileNotFoundError(f"Config not found: {config_path}")


def rename_config_with_run_id(config_path: str | Path, cfg: dict) -> Path:
    """
    If cfg contains a run_id, rename the config file on disk to include it.

    Example:
        configs/figures/quiver/S02_0722.yaml  +  run_id: "030526_00"
        → configs/figures/quiver/S02_0722_030526_00.yaml

    Returns the (possibly renamed) config path.
    """
    config_path = Path(config_path)
    run_id = cfg.get("run_id")
    if not run_id:
        return config_path
    run_id_str = str(run_id)
    # Don't rename if already contains the run_id
    if run_id_str in config_path.stem:
        return config_path
    new_path = config_path.with_stem(f"{config_path.stem}_{run_id_str}")
    config_path.rename(new_path)
    print(f"  Config renamed: {config_path.name} → {new_path.name}")
    return new_path


def no_overwrite_path(path: str | Path) -> Path:
    """
    If path already exists, append _1, _2, ... before the suffix until free.

    Example:
        quiver.pdf        → quiver_1.pdf  (if quiver.pdf exists)
        quiver_1.pdf      → quiver_2.pdf  (if quiver_1.pdf also exists)
    """
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def update_yaml_field(config_path: str | Path, keys: list[str], value) -> None:
    """
    Update a nested key in a YAML config file in-place, preserving formatting.

    keys: list of key names forming the path, e.g. ["data", "frames_per_video"]
    """
    from ruamel.yaml import YAML
    ry = YAML()
    ry.preserve_quotes = True
    config_path = Path(config_path)
    with open(config_path) as f:
        data = ry.load(f)
    node = data
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    with open(config_path, "w") as f:
        ry.dump(data, f)
    print(f"  Updated {'.'.join(str(k) for k in keys)} in {config_path.name}")


def save_config_copy(config_path: str | Path, save_dir: str | Path) -> None:
    """
    Copy the config YAML into save_dir with a timestamp so every run is
    independently traceable and reruns with different params don't overwrite.

    Example: S02_0722_20260305_143022.yaml
    """
    config_path = Path(config_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = save_dir / f"{config_path.stem}_{timestamp}.yaml"
    shutil.copy2(config_path, dest)
    print(f"  Config saved: {dest}")

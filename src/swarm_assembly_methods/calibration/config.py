"""Load and validate calibration YAML config; derive output paths."""

from pathlib import Path
from typing import Any

import yaml


_REQUIRED_KEYS = ["session", "cameras", "frame_export", "detection"]


def load_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load a calibration YAML config and validate required keys."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in _REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required config section: '{key}'")

    # Derive data_root and output_root from subject/date if not explicitly set
    sess = cfg["session"]
    raw_root = Path(cfg["raw_root"])
    processed_root = Path(cfg["processed_root"])
    subject = sess["subject"]
    date = str(sess["date"])
    run_id = sess.get("run_id")

    sess["data_root"] = raw_root / subject / date

    out = processed_root / "calibration" / subject / date
    if run_id:
        out = out / str(run_id)
    sess["output_root"] = out

    # Derive full video paths from filename only
    side = sess["stereo_side"]
    cam_cfg = cfg["cameras"]
    cam_cfg["left_video"]  = sess["data_root"] / f"gopro_pair_{side}" / "left_camera"  / cam_cfg["left_video"]
    cam_cfg["right_video"] = sess["data_root"] / f"gopro_pair_{side}" / "right_camera" / cam_cfg["right_video"]

    # Derive npy dirs for sweep_dk
    if "sweep_dk" in cfg:
        npy_base = raw_root / "bee_detection" / subject / date / f"gopro_pair_{side}"
        cfg["sweep_dk"].setdefault("cam1_npy", str(npy_base / "left_camera"))
        cfg["sweep_dk"].setdefault("cam2_npy", str(npy_base / "right_camera"))

    return cfg


def get_output_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Derive all standard output directories/files from the config.

    Any key can be overridden in the config under an ``outputs`` section, e.g.::

        outputs:
          intrinsics_left: "/some/other/path/left_intrinsics.json"
    """
    data_root = cfg["session"]["data_root"]
    out_root = cfg["session"]["output_root"]
    board = cfg["session"]["board_type"]
    side = cfg["session"]["stereo_side"]

    raw_cal_base = data_root / "calibration_data" / f"{board}_calibration_board"
    cal_base = out_root / "calibration_data" / f"{board}_calibration_board"
    frames_dir = raw_cal_base / "frames"
    intrinsics_dir = cal_base / "intrinsics"
    extrinsics_dir = cal_base / "extrinsics"
    rectification_dir = cal_base / "rectification_debug"

    defaults = {
        "cal_base": cal_base,
        "frames_left": frames_dir / f"{side}_left",
        "frames_right": frames_dir / f"{side}_right",
        "intrinsics_left": intrinsics_dir / f"{board}_board_{side}_left_intrinsics.json",
        "intrinsics_right": intrinsics_dir / f"{board}_board_{side}_right_intrinsics.json",
        "extrinsics": extrinsics_dir / f"{board}_board_{side}_left_to_{side}_right_stereo_extrinsics.json",
        "rectification_debug": rectification_dir / side,
    }

    overrides = cfg.get("outputs", {}) or {}
    for key, val in overrides.items():
        if key not in defaults:
            raise KeyError(f"Unknown output key '{key}'. Valid keys: {list(defaults)}")
        defaults[key] = Path(val)

    return defaults

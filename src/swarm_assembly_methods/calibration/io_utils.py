"""I/O helpers: load/save calibration JSON, list images, pair frames."""

import json
import re
import warnings
from pathlib import Path

import numpy as np

_FRAME_RE = re.compile(r"_(\d+)\.", re.IGNORECASE)


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_intrinsics_json(path: Path):
    """Returns (K, dist, image_size_wh)."""
    d = load_json(path)
    K = np.array(d["K"], dtype=np.float64)
    dist_raw = d.get("distCoeffs")
    if dist_raw is None:
        warnings.warn("distCoeffs not found; assuming zeros.")
        dist = np.zeros(5, dtype=np.float64)
    else:
        dist = np.array(dist_raw, dtype=np.float64).reshape(-1, 1)
    img_size = tuple(d.get("image_size", [None, None]))
    return K, dist, img_size


def load_extrinsics_json(path: Path):
    """Returns (R, T_mm, right_offset, image_size_wh)."""
    d = load_json(path)
    R = np.array(d["R_cam2_from_cam1"], dtype=np.float64)
    T = np.array(d["T_cam2_from_cam1_mm"], dtype=np.float64).reshape(3, 1)
    pairing = d.get("pairing", {}) or {}
    right_offset = int(pairing.get("dk", pairing.get("RIGHT_OFFSET", 0)))
    size_wh = d.get("image_size_W_H", None)
    return R, T, right_offset, size_wh


def frame_id(path: Path) -> int:
    m = _FRAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Couldn't parse frame id from: {path.name}")
    return int(m.group(1))


def can_parse_frame_id(path: Path) -> bool:
    return _FRAME_RE.search(path.name) is not None


def list_images(folder: Path) -> list[Path]:
    files = []
    for ext in ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.PNG", "*.png"):
        files.extend(folder.glob(ext))
    files = sorted(files, key=lambda p: frame_id(p) if can_parse_frame_id(p) else p.name)
    return files


def pair_by_offset(files_left, files_right, offset: int):
    """Pair rule: right_frame_id = left_frame_id + offset.
    Returns list of (left_path, right_path, left_fid, right_fid).
    """
    right_map = {frame_id(p): p for p in files_right if can_parse_frame_id(p)}
    pairs = []
    for pL in files_left:
        if not can_parse_frame_id(pL):
            continue
        fL = frame_id(pL)
        fR = fL + offset
        pR = right_map.get(fR)
        if pR is not None:
            pairs.append((pL, pR, fL, fR))
    return pairs

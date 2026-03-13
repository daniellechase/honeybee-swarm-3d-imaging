"""Core logic for step 6: view rectified frames from bee videos."""

import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import get_output_paths
from .io_utils import load_extrinsics_json, load_intrinsics_json
from .rectification import compute_rectification


def _ffmpeg_extract_frame(video_path: Path, frame0: int, fps: float = 60.0) -> np.ndarray:
    tmp = Path(os.getenv("TEMP", "/tmp")) / f"tmp_{os.getpid()}_{video_path.stem}_{int(frame0)}.png"
    timestamp = frame0 / fps
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{timestamp:.6f}",
        "-i", str(video_path),
        "-frames:v", "1", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    img = cv2.imread(str(tmp), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"ffmpeg failed for {video_path} frame {frame0}")
    return img


def run_view_rectified(cfg: dict[str, Any]):
    root = cfg["session"]["data_root"]
    paths = get_output_paths(cfg)
    vr_cfg = cfg.get("view_rectified", {})
    rect_cfg = cfg.get("rectification", {})

    K1, d1, _ = load_intrinsics_json(paths["intrinsics_left"])
    K2, d2, _ = load_intrinsics_json(paths["intrinsics_right"])
    R, Tmm, dk, _ = load_extrinsics_json(paths["extrinsics"])

    sync_frame = vr_cfg.get("sync_frame", 6000)  # 1-based
    alpha = rect_cfg.get("alpha", 0.0)
    fps = cfg["frame_export"].get("fps", 60.0)

    left_video = Path(cfg["cameras"]["left_video"])
    right_video = Path(cfg["cameras"]["right_video"])

    left0 = sync_frame - 1
    right0 = (sync_frame - dk) - 1

    print(f"Extracting left frame {left0}, right frame {right0}...")
    imgL = _ffmpeg_extract_frame(left_video, left0, fps)
    imgR = _ffmpeg_extract_frame(right_video, right0, fps)

    Hvid, Wvid = imgL.shape[:2]
    if imgR.shape[:2] != (Hvid, Wvid):
        raise RuntimeError(f"Frame sizes differ: {imgL.shape[:2]} vs {imgR.shape[:2]}")

    (_, _, _, _), (m1x, m1y, m2x, m2y) = compute_rectification(
        K1, d1, K2, d2, R, Tmm, Wvid, Hvid, alpha=alpha,
    )

    imgL_rect = cv2.remap(imgL, m1x, m1y, interpolation=cv2.INTER_LINEAR)
    imgR_rect = cv2.remap(imgR, m2x, m2y, interpolation=cv2.INTER_LINEAR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2RGB))
    ax1.set_title("Left rectified")
    ax2.set_title("Right rectified")
    ax1.axis("off")
    ax2.axis("off")

    for y in np.linspace(0, Hvid - 1, 10).astype(int):
        ax1.plot([0, Wvid], [y, y], color="lime", alpha=0.15, linewidth=1)
        ax2.plot([0, Wvid], [y, y], color="lime", alpha=0.15, linewidth=1)

    plt.tight_layout()
    out_path = paths["rectification_debug"] / f"rectified_frame_{sync_frame}.png"
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

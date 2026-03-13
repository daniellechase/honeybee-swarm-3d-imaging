"""
ML identification figures.

Three outputs from a single video frame + mask + detections:

1. swarm_with_detections  — full GoPro frame, swarm outline, bee bounding boxes + dots
2. swarm_mask_crop        — tight crop, white background, swarm filled + outlined
3. swarm_crop_on_image    — tight crop of original frame with swarm outline only
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _extract_frame_number(filename: str) -> int:
    nums = re.findall(r"(\d+)", os.path.splitext(filename)[0])
    if not nums:
        raise ValueError(f"No frame number found in filename: {filename}")
    return int(nums[-1])


def read_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def find_mask_for_frame(mask_dir: str | Path, frame_idx: int) -> tuple[str, int]:
    """Return (path, frame_num) of the NPZ mask closest to frame_idx."""
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith(".npz")]
    candidates = []
    for f in files:
        try:
            candidates.append((_extract_frame_number(f), f))
        except ValueError:
            continue
    if not candidates:
        raise RuntimeError(f"No mask NPZ files found in {mask_dir}")
    nums = np.array([c[0] for c in candidates])
    idx = int(np.argmin(np.abs(nums - frame_idx)))
    frame_num, fname = candidates[idx]
    if frame_num != frame_idx:
        print(f"  WARNING: no exact mask for frame {frame_idx}, using closest: {frame_num}")
    return str(Path(mask_dir) / fname), frame_num


def load_detections_at_frame(
    detection_path: str | Path, frame_idx: int
) -> tuple[np.ndarray, int]:
    """
    Load detections array and filter to frame_idx.
    If no detections exist at that frame, snaps to nearest frame with detections.
    Returns (det_rows, actual_frame_idx).
    det_rows columns: [frame, cx, cy, x1, y1, x2, y2]
    """
    detections = np.load(str(detection_path)).astype(np.float32)
    det = detections[detections[:, 0].astype(int) == frame_idx]
    if len(det) == 0:
        frames = np.unique(detections[:, 0].astype(int))
        nearest = int(frames[np.argmin(np.abs(frames - frame_idx))])
        print(f"  No detections at frame {frame_idx}, snapping to nearest: {nearest}")
        frame_idx = nearest
        det = detections[detections[:, 0].astype(int) == frame_idx]
    print(f"  Detections at frame {frame_idx}: {len(det)}")
    return det, frame_idx


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def plot_full_frame_with_detections(
    frame_rgb: np.ndarray,
    contours: list,
    det: np.ndarray,
    save_path: str | Path,
    dpi: int = 400,
    contour_color: tuple = (1.0, 0.0, 0.0),
    contour_lw: float = 1.0,
    bbox_color: tuple = (0.0, 0.0, 1.0),
    bbox_lw: float = 0.5,
    point_color: tuple = (1.0, 0.0, 0.0),
    point_size: float = 2,
) -> None:
    """Full GoPro frame with swarm outline + bee bounding boxes + center dots."""
    H, W = frame_rgb.shape[:2]
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame_rgb)
    ax.axis("off")

    for c in contours:
        c = c.squeeze()
        if c.ndim != 2 or len(c) < 3:
            continue
        ax.add_patch(Polygon(c, closed=True, fill=False,
                             edgecolor=contour_color, linewidth=contour_lw,
                             joinstyle="round", capstyle="round", zorder=3))

    if len(det) > 0:
        cx, cy = det[:, 1], det[:, 2]
        x1, y1, x2, y2 = det[:, 3], det[:, 4], det[:, 5], det[:, 6]
        for i in range(len(det)):
            ax.add_patch(patches.Rectangle(
                (x1[i], y1[i]), x2[i] - x1[i], y2[i] - y1[i],
                linewidth=bbox_lw, edgecolor=bbox_color, facecolor="none", zorder=4,
            ))
        ax.scatter(cx, cy, s=point_size, c=[point_color], zorder=5, linewidths=0)

    _save(fig, save_path, dpi)


def plot_swarm_crop_white_bg(
    binary_mask: np.ndarray,
    contours: list,
    save_path: str | Path | None,
    dpi: int = 400,
    crop_pad: int = 10,
    fill_color: tuple = (1.0, 0.0, 0.0),
    fill_alpha: float = 0.3,
    contour_color: tuple = (1.0, 0.0, 0.0),
    contour_lw: float = 0.8,
) -> tuple[int, int, int, int]:
    """Tight crop around swarm on white background, filled + outlined. Returns crop bbox."""
    H, W = binary_mask.shape[:2]
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Swarm mask is empty.")
    x_min = max(int(xs.min()) - crop_pad, 0)
    x_max = min(int(xs.max()) + crop_pad, W - 1)
    y_min = max(int(ys.min()) - crop_pad, 0)
    y_max = min(int(ys.max()) + crop_pad, H - 1)
    crop_w = x_max - x_min + 1
    crop_h = y_max - y_min + 1

    bg = np.ones((crop_h, crop_w, 3), dtype=np.uint8) * 255
    fig = plt.figure(figsize=(crop_w / dpi, crop_h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(bg)
    ax.axis("off")

    for c in contours:
        c = c.squeeze()
        if c.ndim != 2 or len(c) < 3:
            continue
        c_crop = c.copy()
        c_crop[:, 0] -= x_min
        c_crop[:, 1] -= y_min
        ax.add_patch(Polygon(c_crop, closed=True, fill=True,
                             facecolor=fill_color, alpha=fill_alpha,
                             edgecolor="none", zorder=1))
        ax.add_patch(Polygon(c_crop, closed=True, fill=False,
                             edgecolor=contour_color, linewidth=contour_lw,
                             joinstyle="round", capstyle="round", zorder=2))

    _save(fig, save_path, dpi)
    return x_min, x_max, y_min, y_max


def plot_swarm_crop_on_image(
    frame_rgb: np.ndarray,
    contours: list,
    x_min: int, x_max: int, y_min: int, y_max: int,
    save_path: str | Path,
    dpi: int = 400,
    contour_color: tuple = (1.0, 0.0, 0.0),
    contour_lw: float = 0.8,
) -> None:
    """Tight crop of original frame with swarm outline only."""
    crop_img = frame_rgb[y_min:y_max + 1, x_min:x_max + 1]
    crop_h, crop_w = crop_img.shape[:2]

    fig = plt.figure(figsize=(crop_w / dpi, crop_h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(crop_img)
    ax.axis("off")

    for c in contours:
        c = c.squeeze()
        if c.ndim != 2 or len(c) < 3:
            continue
        c_crop = c.copy()
        c_crop[:, 0] -= x_min
        c_crop[:, 1] -= y_min
        ax.add_patch(Polygon(c_crop, closed=True, fill=False,
                             edgecolor=contour_color, linewidth=contour_lw,
                             joinstyle="round", capstyle="round", zorder=2))

    _save(fig, save_path, dpi)


def plot_detection_crops(
    frame_rgb: np.ndarray,
    det: np.ndarray,
    save_dir: str | Path,
    fmt: str = "pdf",
    dpi: int = 150,
    n_crops: int = 10,
    crop_w: int = 40,
    crop_h: int = 40,
    crop_padding: int = 2,
    auto_crop: bool = False,
    auto_crop_quantile: float = 0.95,
    bbox_color: tuple = (0.0, 0.0, 1.0),
    bbox_lw: float = 2.0,
    point_color: tuple = (1.0, 0.0, 0.0),
) -> None:
    """
    Save fixed-size crops around N selected bee detections.

    Detections are sampled spread across the frame (by distance from center).
    If auto_crop=True, crop size is derived from the detection bbox distribution.
    Saved as save_dir/crop_00.pdf, crop_01.pdf, ...
    """
    if len(det) == 0:
        print("  No detections — skipping crops.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cx, cy = det[:, 1], det[:, 2]
    x1, y1, x2, y2 = det[:, 3], det[:, 4], det[:, 5], det[:, 6]

    # auto crop size from detection bbox distribution
    if auto_crop:
        bw, bh = x2 - x1, y2 - y1
        crop_w = int(np.ceil(np.quantile(bw, auto_crop_quantile))) + 2 * crop_padding
        crop_h = int(np.ceil(np.quantile(bh, auto_crop_quantile))) + 2 * crop_padding
        crop_w += crop_w % 2
        crop_h += crop_h % 2
        print(f"  Auto crop size: {crop_w}×{crop_h} px")

    # select detections spread across the frame
    if len(det) <= n_crops:
        crop_indices = np.arange(len(det))
    else:
        img_cx, img_cy = frame_rgb.shape[1] / 2, frame_rgb.shape[0] / 2
        dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
        sorted_idx = np.argsort(dist)
        crop_indices = sorted_idx[np.linspace(0, len(sorted_idx) - 1, n_crops, dtype=int)]

    # pad frame so crops at edges keep fixed size
    pad_x = crop_w // 2 + 2
    pad_y = crop_h // 2 + 2
    frame_pad = cv2.copyMakeBorder(frame_rgb, pad_y, pad_y, pad_x, pad_x,
                                   cv2.BORDER_REFLECT_101)

    for plot_idx, det_idx in enumerate(crop_indices):
        cxf_p = float(cx[det_idx]) + pad_x
        cyf_p = float(cy[det_idx]) + pad_y
        bx1 = int(np.round(cxf_p - crop_w / 2))
        by1 = int(np.round(cyf_p - crop_h / 2))
        crop = frame_pad[by1:by1 + crop_h, bx1:bx1 + crop_w]

        ox = bx1 - pad_x
        oy = by1 - pad_y

        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(crop)
        ax.add_patch(patches.Rectangle(
            (x1[det_idx] - ox, y1[det_idx] - oy),
            x2[det_idx] - x1[det_idx], y2[det_idx] - y1[det_idx],
            linewidth=bbox_lw, edgecolor=bbox_color, facecolor="none",
        ))
        ax.plot(cx[det_idx] - ox, cy[det_idx] - oy, "o",
                color=point_color, markersize=6, markeredgewidth=0)
        ax.set_xlim(-0.5, crop_w - 0.5)
        ax.set_ylim(crop_h - 0.5, -0.5)
        ax.axis("off")
        _save(fig, save_dir / f"crop_{plot_idx:02d}.{fmt}", dpi)

    print(f"  Saved {len(crop_indices)} crops ({crop_w}×{crop_h} px) to {save_dir}")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _save(fig, path: str | Path | None, dpi: int) -> None:
    if path is None:
        plt.close(fig)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".").lower()
    fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {path}")

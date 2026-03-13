"""
ML identification figures pipeline.

Reads a YAML config and produces three figures for a single frame:
  swarm_with_detections — full frame + outline + bee detections
  swarm_mask_crop       — tight crop on white background
  swarm_crop_on_image   — tight crop of original frame with outline
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from swarm_assembly_methods.utils import resolve_save_dir
from .figures import (
    read_video_frame,
    find_mask_for_frame,
    load_detections_at_frame,
    plot_full_frame_with_detections,
    plot_swarm_crop_white_bg,
    plot_swarm_crop_on_image,
    plot_detection_crops,
)


def run_ml_identification_figures(cfg: dict, config_path=None) -> None:
    inp      = cfg.get("input", {})
    style    = cfg.get("style", {})
    save_dir = resolve_save_dir(cfg, "save_dir", "figures/ml_identification")
    fmt      = cfg.get("format", "pdf")
    dpi      = int(cfg.get("dpi", 400))

    video_path      = inp["video"]
    mask_dir        = inp["mask_dir"]
    detection_path  = inp.get("detection_path")
    frame_idx       = int(inp["frame_idx"])

    # ---- Load frame ----
    print(f"Reading frame {frame_idx} from {Path(video_path).name} ...")
    frame_bgr = read_video_frame(video_path, frame_idx)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]

    # ---- Load mask ----
    mask_path, mask_frame = find_mask_for_frame(mask_dir, frame_idx)
    mask = np.load(mask_path)["mask"]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask > 0).astype(np.uint8)

    # ---- Load detections ----
    det = np.zeros((0, 7), dtype=np.float32)
    actual_frame = frame_idx
    if detection_path:
        det, actual_frame = load_detections_at_frame(detection_path, frame_idx)
        if actual_frame != frame_idx:
            # re-read frame to match detection frame
            frame_bgr = read_video_frame(video_path, actual_frame)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ---- Contours ----
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- Style params ----
    def _color(key, default):
        v = style.get(key, default)
        return tuple(v) if isinstance(v, list) else v

    main_contour_color = _color("main_contour_color", (1.0, 0.0, 0.0))
    main_contour_lw    = float(style.get("main_contour_lw", 1.0))
    bbox_color         = _color("bbox_color", (0.0, 0.0, 1.0))
    bbox_lw            = float(style.get("bbox_lw", 0.5))
    point_color        = _color("point_color", (1.0, 0.0, 0.0))
    point_size         = float(style.get("point_size", 2))
    crop_pad           = int(style.get("crop_pad", 10))
    crop_fill_color    = _color("crop_fill_color", (1.0, 0.0, 0.0))
    crop_fill_alpha    = float(style.get("crop_fill_alpha", 0.3))
    crop_contour_color = _color("crop_contour_color", (1.0, 0.0, 0.0))
    crop_contour_lw    = float(style.get("crop_contour_lw", 0.8))
    n_crops            = int(style.get("n_crops", 10))
    crop_w             = int(style.get("crop_w", 40))
    crop_h             = int(style.get("crop_h", 40))
    crop_padding       = int(style.get("crop_padding", 2))
    auto_crop          = bool(style.get("auto_crop", False))
    auto_crop_quantile = float(style.get("auto_crop_quantile", 0.95))
    crop_bbox_lw       = float(style.get("crop_bbox_lw", 2.0))

    # ---- Generate figures ----
    print("\nGenerating figures...")

    plot_full_frame_with_detections(
        frame_rgb, contours, det,
        save_path=save_dir / f"swarm_with_detections.{fmt}",
        dpi=dpi,
        contour_color=main_contour_color,
        contour_lw=main_contour_lw,
        bbox_color=bbox_color,
        bbox_lw=bbox_lw,
        point_color=point_color,
        point_size=point_size,
    )

    x_min, x_max, y_min, y_max = plot_swarm_crop_white_bg(
        binary_mask, contours,
        save_path=None,
        dpi=dpi,
        crop_pad=crop_pad,
        fill_color=crop_fill_color,
        fill_alpha=crop_fill_alpha,
        contour_color=crop_contour_color,
        contour_lw=crop_contour_lw,
    )

    plot_swarm_crop_on_image(
        frame_rgb, contours,
        x_min, x_max, y_min, y_max,
        save_path=save_dir / f"swarm_crop_on_image.{fmt}",
        dpi=dpi,
        contour_color=crop_contour_color,
        contour_lw=crop_contour_lw,
    )

    if detection_path:
        plot_detection_crops(
            frame_rgb, det,
            save_dir=save_dir / "detection_crops",
            fmt=fmt,
            dpi=dpi,
            n_crops=n_crops,
            crop_w=crop_w,
            crop_h=crop_h,
            crop_padding=crop_padding,
            auto_crop=auto_crop,
            auto_crop_quantile=auto_crop_quantile,
            bbox_color=bbox_color,
            bbox_lw=crop_bbox_lw,
            point_color=point_color,
        )

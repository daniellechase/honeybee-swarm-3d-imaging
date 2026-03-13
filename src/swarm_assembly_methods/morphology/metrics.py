"""
Per-frame swarm morphology metrics from binary masks.

Computes: area_px, width_px, length_px, volume_px3 (pixel solid-of-revolution),
volume_m3 and volume_axisym_m3 (stereo metric, when calibration is available).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# NumPy 2.0 compatibility
_np_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Letterbox parameters for GoPro footage (1280×720 with 160px side padding)
_LB_SIZE_WH   = (1280, 720)
_LB_PAD_LEFT  = 160
_LB_PAD_RIGHT = 160


def _unletterbox(mask_u8: np.ndarray, calib_size_wh: tuple[int, int]) -> np.ndarray:
    """Undo GoPro letterboxing and resize to calibration image size."""
    import cv2
    h, w = mask_u8.shape[:2]
    if (w, h) == calib_size_wh:
        return mask_u8
    if (w, h) == _LB_SIZE_WH:
        crop = mask_u8[:, _LB_PAD_LEFT: _LB_SIZE_WH[0] - _LB_PAD_RIGHT]
        return cv2.resize(crop, calib_size_wh, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(mask_u8, calib_size_wh, interpolation=cv2.INTER_NEAREST)


def compute_axisym_volume_metric(
    maskL_rect: np.ndarray,
    maskR_rect: np.ndarray,
    P1: np.ndarray,
    T21: np.ndarray,
) -> float:
    """
    Axisymmetric solid-of-revolution volume from rectified silhouette masks.

    Uses single median depth per frame for robustness against row-level
    disparity noise at the tapered top/bottom of the swarm.

    Returns float (m³) or np.nan if fewer than 2 valid rows.
    """
    f_px = float(P1[0, 0])
    B_mm = float(np.linalg.norm(T21))
    cy   = float(P1[1, 2])
    H    = maskL_rect.shape[0]

    rows_list, widths_list, d_list = [], [], []

    for row in range(H):
        colsL = np.where(maskL_rect[row])[0]
        colsR = np.where(maskR_rect[row])[0]
        if len(colsL) < 2 or len(colsR) < 2:
            continue
        width_px = int(colsL[-1]) - int(colsL[0])
        if width_px < 2:
            continue
        d_center = (float(colsL[0]) - float(colsR[0]) + float(colsL[-1]) - float(colsR[-1])) / 2.0
        if d_center <= 0:
            continue
        rows_list.append(row)
        widths_list.append(width_px)
        d_list.append(d_center)

    if len(rows_list) < 2:
        return np.nan

    rows_arr   = np.array(rows_list,   dtype=float)
    widths_arr = np.array(widths_list, dtype=float)
    d_arr      = np.array(d_list,      dtype=float)

    z_per_row = (f_px * B_mm / d_arr) / 1000.0   # mm → m
    z_median  = float(np.median(z_per_row))

    r_arr = (widths_arr / 2.0) * z_median / f_px
    y_arr = (rows_arr - cy)    * z_median / f_px

    order = np.argsort(y_arr)
    return float(np.abs(_np_trapz(np.pi * r_arr[order]**2, y_arr[order])))


def compute_mask_properties(
    masks_list: list[tuple[int, str, np.ndarray]],
    right_mask_map: dict[int, np.ndarray] | None,
    calib: dict | None,
) -> pd.DataFrame:
    """
    Compute per-frame morphology from a list of (frame_num, folder, mask).

    calib dict keys: P1, T21, map1x, map1y, map2x, map2y, size, dk

    Returns DataFrame with columns:
        frame_num, folder, area_px, width_px, length_px,
        volume_px3, volume_m3, volume_axisym_m3,
        width_right_px, length_right_px
    """
    import cv2

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    records = []
    n_stereo_ok = 0

    it = tqdm(masks_list, desc="Computing metrics") if tqdm else masks_list
    for i, (frame_num, folder, mask) in enumerate(it):
        binary = (mask > 0).astype(np.uint8)
        area = int(binary.sum())

        coords = np.argwhere(binary)
        if coords.size == 0:
            width = length = 0
            volume_px3 = 0.0
            volume_m3 = volume_axisym_m3 = np.nan
            width_right_px = length_right_px = np.nan
        else:
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            width  = int(x_coords.max() - x_coords.min() + 1)
            length = int(y_coords.max() - y_coords.min() + 1)

            # Pixel-based volume: revolve left silhouette around vertical axis
            row_min, row_max = int(y_coords.min()), int(y_coords.max())
            vol_px = 0.0
            for row in range(row_min, row_max + 1):
                row_cols = np.where(binary[row, :])[0]
                if len(row_cols) == 0:
                    continue
                r = (row_cols[-1] - row_cols[0] + 1) / 2.0
                vol_px += np.pi * r * r
            volume_px3 = vol_px

            volume_m3 = volume_axisym_m3 = np.nan
            width_right_px = length_right_px = np.nan

            if right_mask_map is not None and calib is not None:
                if frame_num in right_mask_map:
                    maskR_raw = right_mask_map[frame_num]
                    binaryR   = (maskR_raw > 0).astype(np.uint8)
                    coordsR   = np.argwhere(binaryR)
                    if coordsR.size > 0:
                        yR, xR = coordsR[:, 0], coordsR[:, 1]
                        width_right_px  = float(xR.max() - xR.min() + 1)
                        length_right_px = float(yR.max() - yR.min() + 1)

                    try:
                        calib_size = tuple(int(x) for x in calib["size"])
                        maskL_full = _unletterbox((binary  * 255).astype(np.uint8), calib_size)
                        maskR_full = _unletterbox((binaryR * 255).astype(np.uint8), calib_size)

                        maskL_rect = cv2.remap(maskL_full, calib["map1x"], calib["map1y"],
                                               cv2.INTER_NEAREST) > 0
                        maskR_rect = cv2.remap(maskR_full, calib["map2x"], calib["map2y"],
                                               cv2.INTER_NEAREST) > 0

                        # Original stereo volume: edge-disparity approach
                        from swarm_assembly_methods.figures.figtraj.boundary import _extract_boundary_3d
                        bnd = _extract_boundary_3d(
                            maskL_rect, maskR_rect, calib["P1"], calib["T21"] * 1000.0,
                            min_width_px=5, flat_z=False)
                        rad = bnd["radius"]
                        ac  = bnd["axis_center"]
                        if len(rad) > 1:
                            dy = np.abs(np.diff(ac[:, 1]))
                            # boundary.py returns metres already
                            volume_m3 = float(np.sum(np.pi * rad[:-1]**2 * dy))

                        # Axisymmetric volume: more robust
                        volume_axisym_m3 = compute_axisym_volume_metric(
                            maskL_rect, maskR_rect, calib["P1"], calib["T21"] * 1000.0)

                        if np.isfinite(volume_axisym_m3):
                            n_stereo_ok += 1
                    except Exception as e:
                        pass  # stereo failed for this frame

        records.append({
            "frame_num":        frame_num,
            "folder":           folder,
            "area_px":          area,
            "volume_px3":       volume_px3,
            "volume_m3":        volume_m3,
            "volume_axisym_m3": volume_axisym_m3,
            "width_px":         width,
            "length_px":        length,
            "width_right_px":   width_right_px,
            "length_right_px":  length_right_px,
        })

    print(f"Metrics: {len(records)} frames, {n_stereo_ok} with stereo volume.")
    return pd.DataFrame(records)


def build_calib_dict(cfg: dict) -> dict | None:
    """
    Build calibration dict for stereo volume computation from config.

    Returns None if calibration keys are missing or files don't exist.
    """
    import cv2
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from swarm_assembly_methods.calibration.io_utils import load_intrinsics_json, load_extrinsics_json

    cal_cfg = cfg.get("calibration", {})
    intr_l = cal_cfg.get("intrinsics_left")
    intr_r = cal_cfg.get("intrinsics_right")
    extr   = cal_cfg.get("extrinsics")

    if not (intr_l and intr_r and extr):
        return None
    for p in (intr_l, intr_r, extr):
        if not Path(p).exists():
            print(f"  Calibration file not found: {p}")
            return None

    K1, d1, size_wh = load_intrinsics_json(Path(intr_l))
    K2, d2, _       = load_intrinsics_json(Path(intr_r))
    R, T_mm, _dk, _ = load_extrinsics_json(Path(extr))

    size_wh = tuple(int(x) for x in size_wh)
    R1, R2, P1, P2, *_ = cv2.stereoRectify(
        K1, d1, K2, d2, size_wh, R, T_mm,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, size_wh, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, size_wh, cv2.CV_32FC1)

    B_mm = float(np.linalg.norm(T_mm))
    print(f"Calibration loaded: baseline={B_mm:.1f} mm, size={size_wh}")

    return dict(
        P1=P1, T21=T_mm, size=size_wh,
        map1x=map1x, map1y=map1y,
        map2x=map2x, map2y=map2y,
    )

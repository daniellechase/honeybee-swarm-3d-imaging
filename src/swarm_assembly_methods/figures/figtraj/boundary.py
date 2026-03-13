"""
Extract 3D swarm boundary from stereo mask files.

Ported from swarmboundarydisparity_v2.py (swarm-self-assembly repo).
Returns outline, axis_center, and radius arrays in metres, ready to pass
to plot_combined_quiver_and_projections as boundary_* arguments.
"""

import json
import numpy as np
import cv2
from pathlib import Path

# Letterbox parameters for GoPro footage (1280×720 with 160px side padding)
_LB_SIZE_WH  = (1280, 720)
_LB_PAD_LEFT = 160
_LB_PAD_RIGHT = 160


def _load_mask(path):
    """Load binary mask from .npy or .npz (key='mask')."""
    raw = np.load(path)
    arr = raw["mask"] if hasattr(raw, "files") else raw
    return (arr > 0).astype(np.uint8) * 255


def _unletterbox(mask_u8, calib_size_wh):
    """Undo GoPro letterboxing and resize to calibration image size."""
    h, w = mask_u8.shape[:2]
    if (w, h) == calib_size_wh:
        return mask_u8
    if (w, h) == _LB_SIZE_WH:
        crop = mask_u8[:, _LB_PAD_LEFT: _LB_SIZE_WH[0] - _LB_PAD_RIGHT]
        return cv2.resize(crop, calib_size_wh, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(mask_u8, calib_size_wh, interpolation=cv2.INTER_NEAREST)


def _rectify_mask(mask_full, K, d, R_rect, P_rect, size_wh):
    map_x, map_y = cv2.initUndistortRectifyMap(K, d, R_rect, P_rect, size_wh, cv2.CV_32FC1)
    return cv2.remap(mask_full, map_x, map_y, cv2.INTER_NEAREST) > 0


def _extract_boundary_3d(maskL_rect, maskR_rect, P1, T21_mm, min_width_px=0, flat_z=True, flat_top=False):
    """
    Extract 3D boundary from rectified binary masks.

    For each scanline, finds leftmost/rightmost foreground pixels in both
    masks, computes disparity → depth → 3D coordinates (metres).

    Returns dict: outline, axis_center, radius, rows  (all in metres)
    """
    H = maskL_rect.shape[0]

    rows_list, left_disp, right_disp, left_x, right_x, widths = [], [], [], [], [], []

    for y in range(H):
        colsL = np.where(maskL_rect[y])[0]
        colsR = np.where(maskR_rect[y])[0]
        if len(colsL) == 0 or len(colsR) == 0:
            continue
        rows_list.append(y)
        left_disp.append(int(colsL[0])  - int(colsR[0]))
        right_disp.append(int(colsL[-1]) - int(colsR[-1]))
        left_x.append(int(colsL[0]))
        right_x.append(int(colsL[-1]))
        widths.append(int(colsL[-1]) - int(colsL[0]))

    rows_arr   = np.array(rows_list)
    left_disp  = np.array(left_disp)
    right_disp = np.array(right_disp)
    left_x     = np.array(left_x)
    right_x    = np.array(right_x)
    widths     = np.array(widths)

    keep = widths >= min_width_px
    rows_arr = rows_arr[keep]; left_disp = left_disp[keep]; right_disp = right_disp[keep]
    left_x = left_x[keep];    right_x   = right_x[keep]

    f_px = float(P1[0, 0])
    B_mm = float(np.linalg.norm(T21_mm))
    cx   = float(P1[0, 2])
    cy   = float(P1[1, 2])

    left_depth  = np.where(left_disp  > 0, f_px * B_mm / left_disp,  np.nan)
    right_depth = np.where(right_disp > 0, f_px * B_mm / right_disp, np.nan)

    if flat_z:
        z_const     = np.nanmedian(np.concatenate([left_depth, right_depth]))
        left_depth  = np.where(np.isfinite(left_depth),  z_const, np.nan)
        right_depth = np.where(np.isfinite(right_depth), z_const, np.nan)

    def _to_3d(x_px, depth):
        X = (x_px - cx) * depth / f_px
        Y = (rows_arr - cy) * depth / f_px
        return np.column_stack([X, Y, depth])

    left_3d  = _to_3d(left_x,  left_depth)
    right_3d = _to_3d(right_x, right_depth)

    valid    = np.isfinite(left_depth) & np.isfinite(right_depth)
    rows_v   = rows_arr[valid]
    left_3d  = left_3d[valid]
    right_3d = right_3d[valid]

    center_X = (left_3d[:, 0] + right_3d[:, 0]) / 2.0
    center_Y = left_3d[:, 1]
    center_Z = (left_3d[:, 2] + right_3d[:, 2]) / 2.0
    axis_center = np.column_stack([center_X, center_Y, center_Z])
    radius = np.sqrt((right_3d[:, 0] - left_3d[:, 0])**2 +
                     (right_3d[:, 2] - left_3d[:, 2])**2) / 2.0

    if flat_top:
        # Clip outline to start at the widest row (max x extent).
        # Everything above is dropped; the new top edge is flat at that row's y.
        i_max = int(np.argmax(radius))
        outline = np.vstack([left_3d[i_max:], right_3d[i_max:][::-1], left_3d[i_max:i_max+1]])
        print(f"  flat_top: clipping to widest row (index {i_max}/{len(radius)-1}), "
              f"y_top={left_3d[i_max,1]/1000:.6f} m")
    else:
        outline = np.vstack([left_3d, right_3d[::-1], left_3d[:1]])

    # mm → metres
    return dict(
        outline=outline / 1000.0,
        axis_center=axis_center / 1000.0,
        radius=radius / 1000.0,
        rows=rows_v,
    )


def load_boundary(mask_left_path, mask_right_path,
                  intrinsics_left, intrinsics_right, extrinsics,
                  flat_z=True, min_width_px=0, flat_top=False):
    """
    Load swarm boundary from mask files + calibration.

    Parameters
    ----------
    mask_left_path, mask_right_path : str or Path   .npy or .npz mask files
    intrinsics_left, intrinsics_right : str or Path  intrinsics JSON files
    extrinsics : str or Path                         extrinsics JSON file
    flat_z : bool   collapse boundary to median depth plane (recommended)

    Returns
    -------
    dict with outline, axis_center, radius (all in metres), or None if files missing
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from swarm_assembly_methods.calibration.io_utils import load_intrinsics_json

    mask_left_path  = Path(mask_left_path)
    mask_right_path = Path(mask_right_path)

    if not mask_left_path.exists() or not mask_right_path.exists():
        print(f"  Warning: boundary mask file(s) not found — skipping boundary overlay")
        return None

    K1, d1, size_wh = load_intrinsics_json(Path(intrinsics_left))
    K2, d2, _       = load_intrinsics_json(Path(intrinsics_right))

    extr = json.load(open(extrinsics))
    R   = np.array(extr["R_cam2_from_cam1"], dtype=np.float64)
    T   = np.array(extr["T_cam2_from_cam1_mm"], dtype=np.float64).reshape(3, 1)
    size_wh = tuple(int(x) for x in size_wh)

    # Rectification
    R1, R2, P1, P2, *_ = cv2.stereoRectify(
        K1, d1, K2, d2, size_wh, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    maskL_full = _unletterbox(_load_mask(mask_left_path),  size_wh)
    maskR_full = _unletterbox(_load_mask(mask_right_path), size_wh)
    maskL_rect = _rectify_mask(maskL_full, K1, d1, R1, P1, size_wh)
    maskR_rect = _rectify_mask(maskR_full, K2, d2, R2, P2, size_wh)

    bnd = _extract_boundary_3d(maskL_rect, maskR_rect, P1, T,
                               min_width_px=min_width_px, flat_z=flat_z, flat_top=flat_top)

    print(f"  Boundary: {len(bnd['outline'])} outline points, "
          f"Z={bnd['outline'][:,2].min():.3f}–{bnd['outline'][:,2].max():.3f} m")
    return bnd

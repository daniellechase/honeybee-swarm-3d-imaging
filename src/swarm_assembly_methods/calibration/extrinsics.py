"""Core logic for step 4: stereo extrinsics calibration.

If extrinsics.dk_sweep is set, calibrates over a range of dk values and picks
the one with the lowest stereo RMS. Otherwise uses extrinsics.dk directly.
"""

import json
import os
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .board import get_board_params, grid_object_pts
from .config import get_output_paths
from .detection import create_detector, detect_tags, dets_to_dict
from .io_utils import list_images, load_intrinsics_json, pair_by_offset


def _collect_detections(filesL, filesR, dk, max_pairs, min_common, board, detector, max_width):
    """Detect tags in all pairs for a given dk, return (objpoints, imgpoints1, imgpoints2)."""
    pairs = pair_by_offset(filesL, filesR, dk)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    objpoints, imgpoints1, imgpoints2 = [], [], []
    for pL, pR, fL, fR in pairs:
        imgL = cv2.imread(str(pL), cv2.IMREAD_COLOR)
        imgR = cv2.imread(str(pR), cv2.IMREAD_COLOR)
        if imgL is None or imgR is None:
            continue

        detL = dets_to_dict(detect_tags(imgL, detector, max_width), board)
        detR = dets_to_dict(detect_tags(imgR, detector, max_width), board)

        common = sorted(set(detL.keys()) & set(detR.keys()))
        if len(common) < min_common:
            continue

        frame_obj, frame_L, frame_R = [], [], []
        for tid in common:
            frame_obj.append(grid_object_pts(tid, board))
            frame_L.append(detL[tid])
            frame_R.append(detR[tid])

        objpoints.append(np.vstack(frame_obj).astype(np.float32))
        imgpoints1.append(np.vstack(frame_L).astype(np.float32))
        imgpoints2.append(np.vstack(frame_R).astype(np.float32))

    return objpoints, imgpoints1, imgpoints2


def _stereo_calibrate(objpoints, imgpoints1, imgpoints2, K1, dist1, K2, dist2, image_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        K1, dist1, K2, dist2,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return rms, R, T


def run_extrinsics(cfg: dict[str, Any]):
    board = get_board_params(cfg["session"]["board_type"])
    side = cfg["session"]["stereo_side"]
    det_cfg = cfg["detection"]
    ext_cfg = cfg.get("extrinsics", {})
    paths = get_output_paths(cfg)

    K1, dist1, size1 = load_intrinsics_json(paths["intrinsics_left"])
    K2, dist2, size2 = load_intrinsics_json(paths["intrinsics_right"])
    image_size = size1

    dk = ext_cfg.get("dk", 0)
    dk_sweep = ext_cfg.get("dk_sweep", None)  # e.g. [-7, -6, -5, -4, -3]
    min_common = ext_cfg.get("min_common_tags", 2)
    max_pairs = ext_cfg.get("max_pairs", None)

    detector = create_detector(det_cfg.get("n_threads", 4))
    max_width = det_cfg.get("max_processing_width", 1920)

    filesL = list_images(paths["frames_left"])
    filesR = list_images(paths["frames_right"])
    if not filesL or not filesR:
        raise RuntimeError("No images found in left or right frame folder.")

    dk_values = dk_sweep if dk_sweep else [dk]

    best_rms, best_R, best_T, best_dk, best_usable = None, None, None, None, None

    for dk_val in dk_values:
        print(f"\n--- dk={dk_val} ---")
        objpoints, imgpoints1, imgpoints2 = _collect_detections(
            filesL, filesR, dk_val, max_pairs, min_common, board, detector, max_width
        )
        usable = len(objpoints)
        print(f"Usable pairs: {usable}")
        if usable < 10:
            print(f"Too few pairs ({usable}), skipping.")
            continue

        rms, R, T = _stereo_calibrate(objpoints, imgpoints1, imgpoints2, K1, dist1, K2, dist2, image_size)
        print(f"Stereo RMS: {rms:.4f}")

        if best_rms is None or rms < best_rms:
            best_rms, best_R, best_T, best_dk, best_usable = rms, R, T, dk_val, usable

    if best_rms is None:
        raise RuntimeError("No dk value produced enough usable pairs.")

    print(f"\n{'='*40}")
    print(f"Best dk: {best_dk}  RMS: {best_rms:.4f}")
    print(f"R:\n{best_R}")
    print(f"T (mm): {best_T.ravel()}")

    out_path = paths["extrinsics"]
    os.makedirs(out_path.parent, exist_ok=True)

    out = {
        "convention": "X_cam2 = R * X_cam1 + T",
        "cam1_id": f"{side}_left",
        "cam2_id": f"{side}_right",
        "pairing": {
            "mode": "right_frame_id = left_frame_id + dk",
            "dk": int(best_dk),
            "MIN_COMMON_TAGS": int(min_common),
            "MAX_PAIRS": None if max_pairs is None else int(max_pairs),
        },
        "board_units": "mm",
        "image_size_W_H": [int(image_size[0]), int(image_size[1])],
        "stereo_rms": float(best_rms),
        "R_cam2_from_cam1": best_R.tolist(),
        "T_cam2_from_cam1_mm": best_T.ravel().tolist(),
        "num_usable_pairs": int(best_usable),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
    return best_dk

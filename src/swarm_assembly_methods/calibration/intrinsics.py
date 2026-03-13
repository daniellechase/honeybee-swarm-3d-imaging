"""Core logic for step 2: calculate per-camera intrinsics from AprilTag images."""

import json
import os
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .board import get_board_params, grid_object_pts
from .config import get_output_paths
from .detection import create_detector, detect_tags


def _calibrate_one_camera(image_dir, board, detector, max_width, show, every_nth=1):
    objpoints, imgpoints = [], []
    from .io_utils import list_images
    files = list_images(image_dir)
    if every_nth > 1:
        files = files[::every_nth]
    print(f"Scanning {len(files)} files in {image_dir}")

    last_img = None
    for path in tqdm(files):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        last_img = img
        dets = detect_tags(img, detector, max_width)
        if len(dets) < 1:
            continue

        frame_obj, frame_img = [], []
        for det in dets:
            tid = det.tag_id
            if not (board.id_offset <= tid < board.id_offset + board.rows * board.cols):
                continue
            frame_obj.extend(grid_object_pts(tid, board))
            frame_img.extend(det.corners.astype(np.float32))

            if show:
                for p in det.corners.astype(int):
                    cv2.circle(img, tuple(p), 4, (0, 255, 0), 2)

        if len(frame_obj) >= 4:
            objpoints.append(np.asarray(frame_obj, np.float32))
            imgpoints.append(np.asarray(frame_img, np.float32))

        if show:
            h, w = img.shape[:2]
            if w > 1920:
                scale = 1920 / w
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            try:
                cv2.imshow("detections", img)
                if cv2.waitKey(1) == 27:
                    show = False
                    cv2.destroyAllWindows()
            except cv2.error:
                show = False

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    if len(objpoints) < 3:
        raise RuntimeError(f"Need >= 3 valid images; found {len(objpoints)}")

    print(f"Using {len(objpoints)} images for calibration.")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, last_img.shape[1::-1], None, None, flags=0, criteria=criteria,
    )
    print(f"RMS reprojection error: {rms}")
    print(f"K:\n{K}")
    print(f"Dist coeffs: {dist.ravel()}")
    return K, dist, rms, last_img.shape[1::-1]  # (W, H)


def _save_intrinsics(out_path, K, dist, rms, image_size):
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "K": K.tolist(),
                "distCoeffs": dist.ravel().tolist(),
                "rms": float(rms),
                "image_size": list(image_size),
            },
            fh,
            indent=2,
        )
    print(f"Saved {out_path}")


def run_intrinsics_one_camera(cfg: dict[str, Any], camera: str):
    """Calibrate a single camera ('left' or 'right'). Intended to be called in its own process."""
    cam_map = {
        "left":  ("LEFT",  "frames_left",  "intrinsics_left"),
        "right": ("RIGHT", "frames_right", "intrinsics_right"),
    }
    cam_label, frames_key, intr_key = cam_map[camera]

    board = get_board_params(cfg["session"]["board_type"])
    det_cfg = cfg["detection"]
    detector = create_detector(det_cfg.get("n_threads", 4))
    max_width = det_cfg.get("max_processing_width", 1920)
    show = det_cfg.get("show_detections", False)
    every_nth = det_cfg.get("every_nth_frame", 1)
    paths = get_output_paths(cfg)

    img_dir = paths[frames_key]
    out_file = paths[intr_key]
    if not img_dir.exists():
        print(f"Skipping {cam_label}: {img_dir} does not exist.")
        return
    print(f"\n=== Calibrating {cam_label} camera ===")
    K, dist, rms, img_size = _calibrate_one_camera(img_dir, board, detector, max_width, show, every_nth)
    _save_intrinsics(out_file, K, dist, rms, img_size)


def run_intrinsics(cfg: dict[str, Any]):
    """Calibrate both cameras. Each runs in its own process to isolate segfaults."""
    import multiprocessing as mp
    for camera in ["left", "right"]:
        p = mp.Process(target=run_intrinsics_one_camera, args=(cfg, camera))
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {camera} camera calibration exited with code {p.exitcode}")

"""Core logic for step 1: export calibration frames from video.

Each camera is exported independently over the full time window — no delta
applied at export time. Pairing happens later at the extrinsics step via
pair_by_offset(dk).
"""

import os
from typing import Any

import cv2

from .config import get_output_paths


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _extract_frames(video_path: str, frame_indices: list[int], output_dir: str, prefix: str):
    _ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in frame_indices:
        if not (0 <= idx < total):
            print(f"Warning: {prefix} frame {idx} out of range (0-{total-1}): skipping")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"Warning: cv2 failed at {prefix} frame {idx}; skipping")
            continue
        out_path = os.path.join(output_dir, f"{prefix}_{idx:06d}.JPG")
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    cap.release()


def run_export_frames(cfg: dict[str, Any], camera: str = "both"):
    """
    Export calibration frames from video.

    Parameters
    ----------
    camera : "left", "right", or "both" (default)
    """
    root = cfg["session"]["data_root"]
    side = cfg["session"]["stereo_side"]
    fe = cfg["frame_export"]
    fps = fe.get("fps", 60)

    start_frame = fe["start_time_sec"] * fps
    end_frame = fe["end_time_sec"] * fps
    frames = list(range(start_frame, end_frame))

    paths = get_output_paths(cfg)

    all_cameras = [
        ("left",  "left_video",  "frames_left",  f"{side}_left_frame"),
        ("right", "right_video", "frames_right", f"{side}_right_frame"),
    ]
    cameras_to_run = [c for c in all_cameras if camera == "both" or c[0] == camera]

    for cam_label, video_key, dir_key, prefix in cameras_to_run:
        video_path = str(cfg["cameras"][video_key])
        out_dir = str(paths[dir_key])

        if not os.path.isfile(video_path):
            print(f"Video not found for {cam_label}: {video_path} — skipping.")
            continue

        print(f"Extracting {cam_label} frames ({len(frames)} frames)...")
        _extract_frames(video_path, frames, out_dir, prefix)

    print("Done exporting frames.")

"""
Load binary masks from NPZ folders with parallel I/O and tqdm progress.

Each mask file is loaded in a thread pool so disk I/O for hundreds of
folders does not freeze the process.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def extract_frame_number(filename: str) -> int:
    """Return the last integer in a filename (before extension)."""
    stem = os.path.splitext(filename)[0]
    nums = re.findall(r"(\d+)", stem)
    if not nums:
        raise ValueError(f"No frame number in filename: {filename}")
    return int(nums[-1])


# ---------------------------------------------------------------------------
# Video length detection
# ---------------------------------------------------------------------------

def resolve_frames_per_video(
    cfg: dict,
    folders: Sequence[str],
) -> list[int]:
    """
    Return frames_per_video from config.

    Sources in priority order:
      1. data.frames_per_video  — explicit list in config
      2. data.video_base_left   — read frame counts from MP4 files via cv2

    Raises RuntimeError if neither is available or any video file is missing.
    """
    import cv2

    data_cfg = cfg.get("data", {})

    manual = data_cfg.get("frames_per_video")
    if manual is not None:
        print(f"Using manually-set frames_per_video: {manual}")
        return list(manual)

    video_base = data_cfg.get("video_base_left")
    if not video_base:
        raise RuntimeError(
            "frames_per_video is required but not set.\n"
            "Either set data.frames_per_video explicitly, "
            "or set data.video_base_left so frame counts can be read from the MP4 files."
        )

    print("Reading frame counts from video files...")
    video_ext = data_cfg.get("video_ext", ".MP4")
    counts = []
    missing = []
    for folder in folders:
        vid_name = folder.replace("_Masks_Npz", "") + video_ext
        vid_path = os.path.join(video_base, vid_name)
        if not os.path.isfile(vid_path):
            missing.append(folder)
            counts.append(None)
            continue
        cap = cv2.VideoCapture(vid_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        counts.append(n)
        print(f"  {folder}: {n} frames")

    if missing:
        raise RuntimeError(
            f"Could not find video files for {len(missing)} folder(s):\n"
            + "\n".join(f"  {f}" for f in missing)
            + "\nSet data.frames_per_video manually to override."
        )

    print(f"frames_per_video: {counts}")
    return counts


# ---------------------------------------------------------------------------
# Left mask loading (parallel)
# ---------------------------------------------------------------------------

def _load_one_mask(folder_path: str, fname: str, cumulative_offset: int, folder: str):
    """Load a single NPZ mask file. Returns (global_frame, folder, mask)."""
    local_frame = extract_frame_number(fname)
    global_frame = cumulative_offset + local_frame
    data = np.load(os.path.join(folder_path, fname))
    mask = data["mask"]
    return (global_frame, folder, mask)


def load_masks_from_base(
    base_path: str | Path,
    folders: Sequence[str],
    load_intervals: Sequence[float],
    frames_per_video: Sequence[int],
    n_workers: int = 8,
) -> list[tuple[int, str, np.ndarray]]:
    """
    Load left-camera masks from all folders, subsampled by load_intervals.

    Uses a thread pool for parallel I/O.  Returns list of
    (global_frame_num, folder_name, mask_array) sorted by global frame.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    tasks = []
    cumulative_offset = 0
    for idx, (folder, interval) in enumerate(zip(folders, load_intervals)):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: folder not found, skipping: {folder_path}")
            if idx < len(frames_per_video):
                cumulative_offset += frames_per_video[idx]
            continue

        files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".npz"))
        files = files[::max(1, int(interval))]
        print(f"  {folder}: {len(files)} masks (interval={interval}, offset={cumulative_offset})")

        for fname in files:
            tasks.append((folder_path, fname, cumulative_offset, folder))

        if idx < len(frames_per_video):
            cumulative_offset += frames_per_video[idx]

    print(f"Loading {len(tasks)} mask files ({n_workers} threads)...")

    all_masks: list[tuple[int, str, np.ndarray]] = []
    it = tqdm(total=len(tasks), desc="Loading masks") if tqdm else None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_load_one_mask, fp, fn, off, fld): i
            for i, (fp, fn, off, fld) in enumerate(tasks)
        }
        results = [None] * len(tasks)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            if it:
                it.update(1)

    if it:
        it.close()

    all_masks = [r for r in results if r is not None]
    all_masks.sort(key=lambda x: x[0])
    return all_masks


# ---------------------------------------------------------------------------
# Right mask loading (match to desired global frames)
# ---------------------------------------------------------------------------

def load_matching_right_masks(
    base_path: str | Path,
    folders: Sequence[str],
    frames_per_video: Sequence[int],
    desired_global_frames: np.ndarray,
    tolerance: int = 0,
    n_workers: int = 8,
) -> list[tuple[int, str, np.ndarray]]:
    """
    Load right-camera masks matched to desired_global_frames (= left + dk).

    Returns list of (desired_frame, folder, mask) for matched frames.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    desired = np.array(sorted(set(int(gf) for gf in desired_global_frames)), dtype=int)
    if desired.size == 0:
        print("No desired_global_frames provided.")
        return []

    # First pass: build a map of (global_frame -> folder, fname, path) for all right files
    file_index: list[tuple[int, str, str, str]] = []  # (global_frame, folder, fname, path)
    cumulative_offset = 0
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            if idx < len(frames_per_video):
                cumulative_offset += frames_per_video[idx]
            continue
        files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".npz"))
        for fname in files:
            local_frame = extract_frame_number(fname)
            global_frame = cumulative_offset + local_frame
            file_index.append((global_frame, folder, fname, folder_path))
        if idx < len(frames_per_video):
            cumulative_offset += frames_per_video[idx]

    if not file_index:
        print("No right mask files found.")
        return []

    # Match each right file to the nearest desired frame within tolerance
    right_frames = np.array([gf for gf, *_ in file_index], dtype=int)
    print(f"Matching {len(desired)} desired frames from {len(right_frames)} right files "
          f"(tolerance={tolerance})...")

    # For each desired frame, find the closest right frame
    match_map: dict[int, tuple[int, str, str, str]] = {}  # desired_frame -> file info
    for desired_frame in desired:
        diffs = np.abs(right_frames - desired_frame)
        j = int(np.argmin(diffs))
        if int(diffs[j]) <= tolerance:
            best_diff = int(diffs[j])
            gf, folder, fname, path = file_index[j]
            if desired_frame not in match_map or best_diff < abs(match_map[desired_frame][0] - desired_frame):
                match_map[desired_frame] = (gf, folder, fname, path)

    print(f"  {len(match_map)} right frames matched.")

    # Load matched files in parallel
    to_load = list(match_map.items())
    it = tqdm(total=len(to_load), desc="Loading right masks") if tqdm else None

    def _load(desired_frame, info):
        gf, folder, fname, path = info
        data = np.load(os.path.join(path, fname))
        return (desired_frame, folder, data["mask"])

    results: list[tuple[int, str, np.ndarray]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_load, df, info): df
            for df, info in to_load
        }
        for future in as_completed(futures):
            results.append(future.result())
            if it:
                it.update(1)

    if it:
        it.close()

    results.sort(key=lambda x: x[0])
    print(f"  Loaded {len(results)} matching right masks.")
    return results

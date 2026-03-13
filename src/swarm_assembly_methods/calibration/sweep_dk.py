"""Core logic for step 3: sweep dk (frame offset) to find optimal stereo sync."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import get_output_paths
from .io_utils import load_intrinsics_json, load_extrinsics_json


# ---------- helpers ----------

def _undistort_points_xyt(xyt: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    if xyt.size == 0:
        return xyt.reshape(0, 3)
    xy = xyt[:, :2].astype(np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(xy, K, dist.reshape(-1, 1), P=K).reshape(-1, 2)
    return np.hstack([und, xyt[:, 2:3]]).astype(float)


def _load_npy_as_xyt(npy_path: Path, order: str = "fxy") -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    if order == "fxy":
        t, x, y = arr[:, 0].astype(int), arr[:, 1].astype(float), arr[:, 2].astype(float)
    else:
        x, y, t = arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(int)
    return np.column_stack([x, y, t]).astype(float)


def _video_prefix(npy_path: Path) -> str:
    """Extract GoPro video prefix (e.g. 'GH43') from filename."""
    m = re.search(r"(GH\d{2})", npy_path.stem, re.IGNORECASE)
    return m.group(1).upper() if m else npy_path.stem


def _pair_npy_files(cam1_dir: Path, cam2_dir: Path):
    """Pair left/right npy files by GoPro video prefix (e.g. GH43)."""
    left_files = {_video_prefix(f): f for f in sorted(cam1_dir.glob("*.npy"))}
    right_files = {_video_prefix(f): f for f in sorted(cam2_dir.glob("*.npy"))}
    common = sorted(set(left_files) & set(right_files))
    return [(prefix, left_files[prefix], right_files[prefix]) for prefix in common]


def _skew(tvec):
    return np.array(
        [[0, -tvec[2], tvec[1]], [tvec[2], 0, -tvec[0]], [-tvec[1], tvec[0], 0]], dtype=float
    )


def _normalize_pixels(xyt_pix, K):
    Kinv = np.linalg.inv(K)
    homog = np.hstack([xyt_pix[:, :2], np.ones((xyt_pix.shape[0], 1))])
    return (Kinv @ homog.T).T[:, :2]


def _dist2line(p1_norm, p2_norm, E):
    n1, n2 = p1_norm.shape[0], p2_norm.shape[0]
    if n1 == 0 or n2 == 0:
        return np.empty((n2, n1), dtype=float)
    pts1_h = np.hstack([p1_norm, np.ones((n1, 1))])
    lines = pts1_h @ E.T
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    x2, y2 = p2_norm[:, 0:1], p2_norm[:, 1:2]
    num = np.abs(a[None, :] * x2 + b[None, :] * y2 + c[None, :])
    den = np.sqrt(a[None, :] ** 2 + b[None, :] ** 2)
    den[den < 1e-12] = 1e-12
    return num / den


def _count_matches_one_frame(pts1, pts2, E, K, thresh_pixels):
    if pts1.size == 0 or pts2.size == 0:
        return 0
    p1n = _normalize_pixels(pts1, K)
    p2n = _normalize_pixels(pts2, K)
    dmat = _dist2line(p1n, p2n, E)
    fy = float(K[1, 1]) if abs(K[1, 1]) > 1e-9 else 1.0
    thresh_norm = thresh_pixels / fy
    cost = dmat.copy()
    cost[cost > thresh_norm] = thresh_norm + 1e7
    row_ind, col_ind = linear_sum_assignment(cost)
    return int(np.sum(cost[row_ind, col_ind] <= thresh_norm))


def _build_frame_index(xyt):
    idx = defaultdict(list)
    for i, t in enumerate(xyt[:, 2].astype(int)):
        idx[int(t)].append(i)
    return idx


def _sweep_one_pair(df1, df2, E, K, dk0, dk_radius, thresh, max_frames):
    """Sweep dk for one video pair. Returns {dk: score} and best_dk."""
    idx1 = _build_frame_index(df1)
    t1 = np.array(sorted(idx1.keys()), dtype=int)

    scores = {}
    for dk in range(dk0 - dk_radius, dk0 + dk_radius + 1):
        df2s = df2.copy()
        df2s[:, 2] += dk
        idx2 = _build_frame_index(df2s)
        common = np.intersect1d(t1, np.array(sorted(idx2.keys()), dtype=int))
        if common.size == 0:
            scores[dk] = 0
            continue
        if len(common) > max_frames:
            common = np.random.choice(common, max_frames, replace=False)
        score = sum(
            _count_matches_one_frame(df1[idx1[int(tt)]], df2s[idx2[int(tt)]], E, K, thresh)
            for tt in common
        )
        scores[dk] = score
        print(f"  dk={dk:4d}: common={len(common):4d}  score={score:6d}  avg={score/max(len(common),1):.2f}")

    best_dk = max(scores, key=lambda k: scores[k])
    return scores, best_dk


# ---------- public API ----------

def run_sweep_dk(cfg: dict[str, Any]) -> dict[str, int]:
    """Sweep dk for all video pairs in cam1_npy/cam2_npy folders.
    Returns {video_prefix: best_dk}.
    """
    root = cfg["session"]["data_root"]
    sk = cfg["sweep_dk"]
    paths = get_output_paths(cfg)

    K, dist, _ = load_intrinsics_json(paths["intrinsics_left"])

    cam1_dir = Path(sk["cam1_npy"]) if Path(sk["cam1_npy"]).is_absolute() else root / sk["cam1_npy"]
    cam2_dir = Path(sk["cam2_npy"]) if Path(sk["cam2_npy"]).is_absolute() else root / sk["cam2_npy"]

    pairs = _pair_npy_files(cam1_dir, cam2_dir)
    if not pairs:
        raise RuntimeError(f"No matching npy pairs found in {cam1_dir} and {cam2_dir}.")
    print(f"Found {len(pairs)} video pair(s): {[p for p, _, _ in pairs]}")

    # Build E matrix
    extrinsics_path = paths["extrinsics"]
    if extrinsics_path.exists():
        R, T, _, _ = load_extrinsics_json(extrinsics_path)
        E = _skew(T.ravel()) @ R
        print(f"Using real extrinsics from: {extrinsics_path}")
    else:
        baseline = sk.get("baseline", 0.5)
        E = _skew(np.array([baseline, 0.0, 0.0])) @ np.eye(3)
        print("No extrinsics found — using synthetic E (baseline assumption).")

    npy_order = sk.get("npy_order", "fxy")
    dk0 = sk["dk0"]
    dk_radius = sk.get("dk_radius", 3)
    thresh = sk.get("thresh_pixels", 6.0)
    max_frames = sk.get("max_frames", 80)
    t_start = sk.get("t_start")
    t_end = sk.get("t_end")
    seed = sk.get("seed", 42)
    np.random.seed(seed)

    results = {}  # prefix -> best_dk

    for prefix, f1, f2 in pairs:
        print(f"\n{'='*50}")
        print(f"Video pair: {prefix}  ({f1.name}  /  {f2.name})")
        print(f"Sweeping dk in [{dk0-dk_radius}, {dk0+dk_radius}], thresh={thresh}px")

        df1 = _undistort_points_xyt(_load_npy_as_xyt(f1, npy_order), K, dist)
        df2 = _undistort_points_xyt(_load_npy_as_xyt(f2, npy_order), K, dist)

        if t_start is not None and t_end is not None:
            df1 = df1[(df1[:, 2] >= t_start) & (df1[:, 2] <= t_end)]

        if df1.size == 0 or df2.size == 0:
            print("  No detections in time window — skipping.")
            continue

        _, best_dk = _sweep_one_pair(df1, df2, E, K, dk0, dk_radius, thresh, max_frames)
        results[prefix] = best_dk
        print(f"  => Best dk for {prefix}: {best_dk}")

    print(f"\n{'='*50}")
    print("Summary:")
    for prefix, dk in results.items():
        print(f"  {prefix}: dk={dk}")

    return results

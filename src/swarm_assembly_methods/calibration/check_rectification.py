"""Core logic for step 5: check rectification quality on AprilTag correspondences."""

import os
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .board import get_board_params
from .config import get_output_paths
from .detection import create_detector, detect_tags, dets_to_dict
from .io_utils import list_images, load_extrinsics_json, load_intrinsics_json, pair_by_offset
from .rectification import compute_rectification, rectify_points


def _build_correspondences(dictL, dictR):
    common = sorted(set(dictL.keys()) & set(dictR.keys()))
    if not common:
        return np.empty((0, 2)), np.empty((0, 2)), []
    ptsL = np.vstack([dictL[tid] for tid in common])
    ptsR = np.vstack([dictR[tid] for tid in common])
    return ptsL, ptsR, common


def _draw_debug(imgL_rect, imgR_rect, ptsLr, ptsRr, dy, out_path, max_draw=80):
    visL, visR = imgL_rect.copy(), imgR_rect.copy()
    n = min(max_draw, ptsLr.shape[0], ptsRr.shape[0])
    for i in range(n):
        p1 = tuple(np.round(ptsLr[i]).astype(int))
        p2 = tuple(np.round(ptsRr[i]).astype(int))
        y1 = int(round(ptsLr[i, 1]))
        cv2.circle(visL, p1, 4, (0, 255, 0), 2)
        cv2.circle(visR, p2, 4, (0, 255, 0), 2)
        cv2.line(visL, (0, y1), (visL.shape[1] - 1, y1), (0, 255, 0), 1)
        y2 = int(round(ptsRr[i, 1]))
        cv2.line(visR, (0, y2), (visR.shape[1] - 1, y2), (0, 255, 0), 1)

    stacked = np.hstack([visL, visR])
    txt = f"dy mean/std/maxabs: {dy.mean():.2f} / {dy.std():.2f} / {np.max(np.abs(dy)):.2f} px"
    cv2.putText(stacked, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), stacked)


def run_check_rectification(cfg: dict[str, Any]):
    board = get_board_params(cfg["session"]["board_type"])
    det_cfg = cfg["detection"]
    rect_cfg = cfg.get("rectification", {})
    paths = get_output_paths(cfg)

    K1, d1, _ = load_intrinsics_json(paths["intrinsics_left"])
    K2, d2, _ = load_intrinsics_json(paths["intrinsics_right"])
    R, Tmm, right_offset, _ = load_extrinsics_json(paths["extrinsics"])

    detector = create_detector(det_cfg.get("n_threads", 4))
    max_width = det_cfg.get("max_processing_width", 1920)
    min_common = cfg.get("extrinsics", {}).get("min_common_tags", 2)
    alpha = rect_cfg.get("alpha", 0.0)

    max_pairs = cfg.get("extrinsics", {}).get("max_pairs", 1)

    filesL = list_images(paths["frames_left"])
    filesR = list_images(paths["frames_right"])
    pairs = pair_by_offset(filesL, filesR, right_offset)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    print(f"Found {len(pairs)} pairs to check (RIGHT_OFFSET={right_offset})")

    out_dir = paths["rectification_debug"]
    os.makedirs(out_dir, exist_ok=True)

    dy_all = []
    dy_per_frame = []  # (fL, fR, mean_dy, std_dy, n_corners)
    used = 0
    saved_debug = False

    for pL, pR, fL, fR in tqdm(pairs, desc="Rectification dy"):
        imgL = cv2.imread(str(pL), cv2.IMREAD_COLOR)
        imgR = cv2.imread(str(pR), cv2.IMREAD_COLOR)
        if imgL is None or imgR is None:
            continue
        if imgL.shape[:2] != imgR.shape[:2]:
            continue

        H, W = imgL.shape[:2]
        detL = dets_to_dict(detect_tags(imgL, detector, max_width), board)
        detR = dets_to_dict(detect_tags(imgR, detector, max_width), board)
        ptsL, ptsR, common_ids = _build_correspondences(detL, detR)
        if len(common_ids) < min_common:
            continue

        (R1, R2, P1, P2), (m1x, m1y, m2x, m2y) = compute_rectification(
            K1, d1, K2, d2, R, Tmm, W, H, alpha=alpha,
        )

        ptsLr = rectify_points(ptsL, K1, d1, R1, P1)
        ptsRr = rectify_points(ptsR, K2, d2, R2, P2)
        dy = ptsLr[:, 1] - ptsRr[:, 1]
        dy_all.append(dy)
        dy_per_frame.append((fL, fR, float(dy.mean()), float(dy.std()), len(dy)))
        used += 1

        if not saved_debug:
            imgL_rect = cv2.remap(imgL, m1x, m1y, cv2.INTER_LINEAR)
            imgR_rect = cv2.remap(imgR, m2x, m2y, cv2.INTER_LINEAR)
            _draw_debug(imgL_rect, imgR_rect, ptsLr, ptsRr, dy, out_dir / f"debug_pair_L{fL}_R{fR}.png")
            saved_debug = True

    if used == 0:
        raise RuntimeError("No usable pairs with enough common tags.")

    dy_all = np.concatenate(dy_all)

    print(f"\n=== Per-frame mean dy (fL, fR, mean, std, n_corners) ===")
    for fL, fR, mean, std, n in dy_per_frame:
        print(f"  L{fL:06d} R{fR:06d}  dy_mean={mean:+.2f}  std={std:.2f}  n={n}")

    frame_means = np.array([x[2] for x in dy_per_frame])
    print(f"\n=== Overall summary ({used} pairs, {dy_all.size} corners) ===")
    print(f"Per-frame mean dy:  mean={frame_means.mean():.2f}  std={frame_means.std():.2f}  range=[{frame_means.min():.2f}, {frame_means.max():.2f}]")
    print(f"All corners dy:     mean={dy_all.mean():.2f}  std={dy_all.std():.2f}  maxabs={np.max(np.abs(dy_all)):.2f} px")

    plt.figure(figsize=(6, 4))
    plt.hist(dy_all, bins=80)
    plt.title("Rectified dy = yL - yR (AprilTag corners)")
    plt.xlabel("dy (px)")
    plt.ylabel("count")
    plt.tight_layout()
    hist_path = out_dir / "dy_hist.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved histogram: {hist_path}")



"""
Interactive board boundary localization.

Two-step workflow per camera-pair key:

  Step 1 — X extents  (left camera, rectified)
      Show the rectified left frame. Click any number of points along the
      left edge and the right edge of the board (labelled separately).
      Press [Enter] after each edge group. The rectified pixel x-coordinates
      are stored; metric X is computed after Step 2 supplies Z depth.

  Step 2 — Z center  (rectified stereo pair, side by side)
      Show rectified left + right frames. Click N corresponding point pairs
      on the board (same physical point in each image). Press [Enter] after
      each image. Each pair is triangulated → 3-D point in left-camera frame.
      Z_center = mean Z of all triangulated points.
      X coordinates from triangulation are also used to refine x_min / x_max.

Output JSON written to the path specified in the config:
  {
    "GH430142": {"x_min_m": 0.05, "x_max_m": 0.55,
                 "x_center_m": 0.30, "z_center_m": 0.95},
    ...
  }

The quiver pipeline can load this file to set x_range and z boundaries.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import load_extrinsics_json, load_intrinsics_json
from .rectification import compute_rectification


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frame(video_path: str | Path, frame_idx: int, fps: float = 60.0) -> np.ndarray:
    """Extract a single frame from a video using ffmpeg (fast input seek)."""
    video_path = Path(video_path)
    tmp = Path(os.getenv("TEMP", "/tmp")) / f"board_loc_{os.getpid()}_{video_path.stem}_{frame_idx}.png"
    timestamp = frame_idx / fps
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{timestamp:.6f}",
        "-i", str(video_path),
        "-frames:v", "1", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    img = cv2.imread(str(tmp), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    os.unlink(tmp)
    return img


def _rectify_image(img: np.ndarray, map1, map2) -> np.ndarray:
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Triangulation (raw pixel clicks → 3-D point in left-camera frame)
# ---------------------------------------------------------------------------

def _triangulate(pt_left, pt_right, K1, d1, K2, d2, R, T_mm):
    p1 = cv2.undistortPoints(
        np.array([[pt_left]], dtype=np.float64), K1, d1, P=K1).reshape(2)
    p2 = cv2.undistortPoints(
        np.array([[pt_right]], dtype=np.float64), K2, d2, P=K2).reshape(2)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T_mm])
    Xh = cv2.triangulatePoints(P1, P2, p1.reshape(2, 1), p2.reshape(2, 1))
    return (Xh[:3] / Xh[3]).ravel()   # mm


# ---------------------------------------------------------------------------
# Step-1 click collector: two groups (left edge / right edge) in left image
# ---------------------------------------------------------------------------

class _XEdgeCollector:
    """Click left-edge points, press Enter, click right-edge points, press Enter."""

    _GROUPS = ["LEFT edge of board", "RIGHT edge of board"]
    _COLORS = ["#2196F3", "#FF5722"]

    def __init__(self, ax, img_bgr, status_text):
        self.ax = ax
        self.group_clicks: list[list[tuple]] = [[], []]
        self.current = 0
        self.done = False
        self._status = status_text
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        self._update_ui()

    def _update_ui(self):
        if self.done:
            n0, n1 = len(self.group_clicks[0]), len(self.group_clicks[1])
            self._status.set_text(
                f"Done — left edge: {n0} pts, right edge: {n1} pts. Close to continue.")
        else:
            g = self._GROUPS[self.current]
            n = len(self.group_clicks[self.current])
            self._status.set_text(
                f"Step 1 — Click {g} points [{n} so far].  "
                f"[Enter] = done with this edge   [d] = delete last")
        self.ax.set_title(
            f"Step 1: X extents\n"
            f"Active: {self._GROUPS[self.current] if not self.done else 'done'}",
            fontsize=10, color="white")
        self.ax.figure.canvas.draw_idle()

    def on_click(self, event):
        if self.done or event.inaxes is not self.ax or event.button != 1:
            return
        toolbar = event.canvas.toolbar
        if toolbar is not None and toolbar.mode != "":
            return
        x, y = event.xdata, event.ydata
        self.group_clicks[self.current].append((x, y))
        n = len(self.group_clicks[self.current])
        self.ax.scatter(x, y, c=self._COLORS[self.current], s=60,
                        marker="o", zorder=5, edgecolors="white", linewidths=0.8)
        self.ax.annotate(str(n), (x, y), color="white", fontsize=8,
                         xytext=(5, 5), textcoords="offset points")
        print(f"  {self._GROUPS[self.current]} pt {n}: ({x:.1f}, {y:.1f})")
        self._update_ui()

    def on_key(self, event):
        if event.key == "enter":
            n = len(self.group_clicks[self.current])
            if n == 0:
                print(f"  No points for {self._GROUPS[self.current]} — click at least one.")
                return
            if self.current == 0:
                self.current = 1
                print(f"  Left edge done ({n} pts). Now click RIGHT edge.")
            else:
                self.done = True
                print(f"  Right edge done ({n} pts). Step 1 complete.")
            self._update_ui()
        elif event.key == "d":
            pts = self.group_clicks[self.current]
            if pts:
                pts.pop()
                print(f"  Deleted last pt from {self._GROUPS[self.current]}")
                self._update_ui()


# ---------------------------------------------------------------------------
# Step-2 click collector: stereo correspondence for Z depth
# ---------------------------------------------------------------------------

class _StereoClickCollector:
    """Click N corresponding points in left then right rectified image."""

    _LABELS = ["LEFT (rectified)", "RIGHT (rectified)"]
    _COLORS = ["#2196F3", "#64B5F6"]

    def __init__(self, axes, images_bgr, status_text):
        self.axes = axes
        self.panel_clicks: list[list[tuple]] = [[], []]
        self.current = 0
        self.done = False
        self._status = status_text
        for ax, img in zip(axes, images_bgr):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
        self._update_ui()

    def _update_ui(self):
        for i, ax in enumerate(self.axes):
            n = len(self.panel_clicks[i])
            active = (not self.done and i == self.current)
            color = self._COLORS[i] if active else "#aaaaaa"
            prefix = ">> " if active else "   "
            ax.set_title(f"{prefix}{self._LABELS[i]}  [{n} pts]",
                         color=color, fontsize=10)
        if self.done:
            self._status.set_text(
                f"Done — {len(self.panel_clicks[0])} point pairs. Close to compute.")
        else:
            n = len(self.panel_clicks[self.current])
            self._status.set_text(
                f"Step 2 — Click {self._LABELS[self.current]} [{n} pts].  "
                f"[Enter] = next image   [d] = delete last")
        self.axes[0].figure.canvas.draw_idle()

    def on_click(self, event):
        if self.done or event.inaxes is None or event.button != 1:
            return
        toolbar = event.canvas.toolbar
        if toolbar is not None and toolbar.mode != "":
            return
        try:
            panel = self.axes.index(event.inaxes)
        except ValueError:
            return
        if panel != self.current:
            print(f"  Please click in: {self._LABELS[self.current]}")
            return
        x, y = event.xdata, event.ydata
        self.panel_clicks[panel].append((x, y))
        n = len(self.panel_clicks[panel])
        self.axes[panel].scatter(x, y, c=self._COLORS[panel], s=60,
                                 marker="o", zorder=5, edgecolors="white", linewidths=0.8)
        self.axes[panel].annotate(str(n), (x, y), color="white", fontsize=8,
                                  xytext=(5, 5), textcoords="offset points")
        print(f"  {self._LABELS[panel]} pt {n}: ({x:.1f}, {y:.1f})")
        self._update_ui()

    def on_key(self, event):
        if event.key == "enter":
            n = len(self.panel_clicks[self.current])
            if n == 0:
                print(f"  No points in {self._LABELS[self.current]} — click at least one.")
                return
            if self.current == 0:
                n_r = len(self.panel_clicks[1])
                if n_r > 0 and n_r != n:
                    print(f"  Warning: left has {n} pts but right already has {n_r}.")
                self.current = 1
                print(f"  Left done ({n} pts). Now click RIGHT image.")
            else:
                n_l = len(self.panel_clicks[0])
                if n != n_l:
                    print(f"  Warning: right has {n} pts but left has {n_l}. "
                          f"Press Enter again to accept mismatch.")
                self.done = True
                print(f"  Step 2 complete — {min(n, n_l)} pairs will be triangulated.")
            self._update_ui()
        elif event.key == "d":
            pts = self.panel_clicks[self.current]
            if pts:
                pts.pop()
                print(f"  Deleted last pt from {self._LABELS[self.current]}")
                self._update_ui()


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def _process_pair(
    key: str,
    pair_cfg: dict,
    K1, d1, K2, d2,
    R, T_mm,
    map1x, map1y, map2x, map2y,
    P1_rect,
    data_root: str | None,
) -> dict:
    """Run the two-step interactive UI for one camera-pair key. Returns result dict."""

    def _rp(p):
        if p and not Path(p).is_absolute() and data_root:
            return str(Path(data_root) / p)
        return p

    left_video  = _rp(pair_cfg["left_video"])
    right_video = _rp(pair_cfg["right_video"])
    left_frame  = int(pair_cfg["left_frame"])
    right_frame = int(pair_cfg.get("right_frame", left_frame))

    print(f"\n{'='*60}")
    print(f"Pair: {key}")
    print(f"  Left  video: {left_video}  frame {left_frame}")
    print(f"  Right video: {right_video}  frame {right_frame}")

    fps = float(pair_cfg.get("fps", 60.0))
    imgL_raw = _extract_frame(left_video,  left_frame,  fps)
    imgR_raw = _extract_frame(right_video, right_frame, fps)

    imgL_rect = _rectify_image(imgL_raw, map1x, map1y)
    imgR_rect = _rectify_image(imgR_raw, map2x, map2y)

    # ------------------------------------------------------------------
    # Step 1: X extents — left rectified image only
    # ------------------------------------------------------------------
    print("\nStep 1: click the LEFT edge then RIGHT edge of the board.")
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8), facecolor="#1e1e1e")
    fig1.patch.set_facecolor("#1e1e1e")
    ax1.set_facecolor("#1e1e1e")
    status1 = fig1.text(0.5, 0.01, "", ha="center", color="white", fontsize=10,
                        transform=fig1.transFigure)
    fig1.suptitle(f"Board localize — {key}  (Step 1: X extents)",
                  color="white", fontsize=12)

    col1 = _XEdgeCollector(ax1, imgL_rect, status1)
    cid_click = fig1.canvas.mpl_connect("button_press_event", col1.on_click)
    cid_key   = fig1.canvas.mpl_connect("key_press_event",   col1.on_key)
    plt.show()
    fig1.canvas.mpl_disconnect(cid_click)
    fig1.canvas.mpl_disconnect(cid_key)
    plt.close(fig1)

    left_edge_px  = [pt[0] for pt in col1.group_clicks[0]]
    right_edge_px = [pt[0] for pt in col1.group_clicks[1]]
    if not left_edge_px or not right_edge_px:
        raise RuntimeError(f"Step 1 incomplete for {key}: need at least one point per edge.")

    # ------------------------------------------------------------------
    # Step 2: Z center — rectified stereo pair
    # ------------------------------------------------------------------
    print("\nStep 2: click corresponding points in LEFT then RIGHT rectified images.")
    fig2, axes2 = plt.subplots(1, 2, figsize=(22, 8), facecolor="#1e1e1e")
    fig2.patch.set_facecolor("#1e1e1e")
    for ax in axes2:
        ax.set_facecolor("#1e1e1e")
    status2 = fig2.text(0.5, 0.01, "", ha="center", color="white", fontsize=10,
                        transform=fig2.transFigure)
    fig2.suptitle(f"Board localize — {key}  (Step 2: Z depth via stereo)",
                  color="white", fontsize=12)

    col2 = _StereoClickCollector(list(axes2), [imgL_rect, imgR_rect], status2)
    cid_click2 = fig2.canvas.mpl_connect("button_press_event", col2.on_click)
    cid_key2   = fig2.canvas.mpl_connect("key_press_event",   col2.on_key)
    plt.show()
    fig2.canvas.mpl_disconnect(cid_click2)
    fig2.canvas.mpl_disconnect(cid_key2)
    plt.close(fig2)

    pts_l = col2.panel_clicks[0]
    pts_r = col2.panel_clicks[1]
    n_pairs = min(len(pts_l), len(pts_r))
    if n_pairs == 0:
        raise RuntimeError(f"Step 2 incomplete for {key}: need at least one stereo pair.")

    # Triangulate in rectified image coords → left-camera frame (mm → m)
    # For rectified images the projection matrices P1_rect, P2_rect apply.
    # We use the original K/d + R/T with the rectified pixel coords since
    # cv2.undistortPoints with identity distortion handles rectified pixels.
    pts3d = []
    for pl, pr in zip(pts_l[:n_pairs], pts_r[:n_pairs]):
        X_mm = _triangulate(pl, pr, K1, np.zeros(5), K2, np.zeros(5),
                            R, T_mm.reshape(3, 1))
        pts3d.append(X_mm / 1000.0)   # → metres
    pts3d = np.array(pts3d)           # (N, 3)  X, Y, Z in left-cam frame

    z_offset_m = float(pair_cfg.get("z_offset_m", 0.0))
    z_center_m = float(np.mean(pts3d[:, 2])) + z_offset_m
    Z_for_x    = z_center_m

    # Convert rectified pixel x → metric X using P1_rect:  X = (x_px - cx) / fx * Z
    fx = float(P1_rect[0, 0])
    cx = float(P1_rect[0, 2])
    x_left_px_mean  = float(np.mean(left_edge_px))
    x_right_px_mean = float(np.mean(right_edge_px))
    x_min_m = (x_left_px_mean  - cx) / fx * Z_for_x
    x_max_m = (x_right_px_mean - cx) / fx * Z_for_x
    x_center_m = (x_min_m + x_max_m) / 2.0

    result = dict(
        x_min_m    = round(x_min_m,    4),
        x_max_m    = round(x_max_m,    4),
        x_center_m = round(x_center_m, 4),
        z_center_m = round(z_center_m, 4),
        # diagnostics
        n_stereo_pairs      = n_pairs,
        z_values_m          = [round(float(p[2]), 4) for p in pts3d],
        left_edge_px_mean   = round(x_left_px_mean, 1),
        right_edge_px_mean  = round(x_right_px_mean, 1),
    )

    print(f"\n  Result for {key}:")
    for k, v in result.items():
        print(f"    {k}: {v}")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_board_localize(cfg: dict, data_root: str | None = None) -> None:
    """
    Run the interactive board localization tool from a config dict.

    Expected config structure:
        intrinsics_left:  "data/calibration/.../left_intrinsics.json"
        intrinsics_right: "data/calibration/.../right_intrinsics.json"
        extrinsics:       "data/calibration/.../extrinsics.json"
        output:           "data/calibration/S02/0722/gate/board_extents.json"
        pairs:
          - key: GH430142
            left_video:   "..."
            right_video:  "..."
            left_frame:   10200
            right_frame:  10195   # optional, defaults to left_frame
    """
    def _rp(p):
        if p and not Path(p).is_absolute() and data_root:
            return str(Path(data_root) / p)
        return p

    K1, d1, _  = load_intrinsics_json(Path(_rp(cfg["intrinsics_left"])))
    K2, d2, _  = load_intrinsics_json(Path(_rp(cfg["intrinsics_right"])))
    R, T_mm, _, size_wh = load_extrinsics_json(Path(_rp(cfg["extrinsics"])))
    W, H = int(size_wh[0]), int(size_wh[1])

    (R1, R2, P1_rect, P2_rect), (map1x, map1y, map2x, map2y) = compute_rectification(
        K1, d1, K2, d2, R, T_mm, W, H, alpha=0.0)

    output_path = Path(_rp(cfg["output"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results so partial runs can be resumed
    existing: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded existing results from {output_path} ({len(existing)} keys).")

    for pair_cfg in cfg["pairs"]:
        key = str(pair_cfg["key"])
        if key in existing:
            print(f"\nSkipping {key} — already in {output_path}. "
                  f"Delete the key from the JSON to redo it.")
            continue
        result = _process_pair(
            key, pair_cfg,
            K1, d1, K2, d2,
            R, T_mm,
            map1x, map1y, map2x, map2y,
            P1_rect,
            data_root=data_root,
        )
        existing[key] = result
        # Save after every pair so progress is not lost
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Saved → {output_path}")

    print(f"\nAll pairs done. Results in: {output_path}")

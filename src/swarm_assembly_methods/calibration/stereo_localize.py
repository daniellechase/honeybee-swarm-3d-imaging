"""Stereo localization: align coordinate systems of two stereo pairs.

For each pair (gate, shed) the user picks N corresponding physical points by
clicking in all four images (pair_left, pair_right for each pair).  Each click
quad gives one triangulated 3D point in that pair's left-camera frame.  A rigid
transform (R, T) mapping gate_left coords → shed_left coords is solved from the
N point correspondences via SVD.

Output JSON convention:
    X_shed_left = R @ X_gate_left + T
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import load_extrinsics_json, load_intrinsics_json


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frame(video_path: Path, frame_idx: int, fps: float = 60.0) -> np.ndarray:
    tmp = Path(os.getenv("TEMP", "/tmp")) / f"tmp_loc_{os.getpid()}_{video_path.stem}_{frame_idx}.png"
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
        raise RuntimeError(f"ffmpeg failed for {video_path} frame {frame_idx}")
    os.unlink(tmp)
    return img


def _load_pair_frames(pair_cfg: dict[str, Any], fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_img, right_img) for a stereo pair."""
    left_video = Path(pair_cfg["left_video"])
    right_video = Path(pair_cfg["right_video"])
    t_sec = float(pair_cfg["t_sec"])

    _, _, dk, _ = load_extrinsics_json(Path(pair_cfg["extrinsics"]))

    left_frame = round(t_sec * fps)
    right_frame = left_frame + dk

    print(f"  left  frame {left_frame}  ({left_video.name})")
    print(f"  right frame {right_frame}  ({right_video.name})  dk={dk}")

    imgL = _extract_frame(left_video, left_frame, fps)
    imgR = _extract_frame(right_video, right_frame, fps)
    return imgL, imgR


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def _triangulate_point(
    pt_left: tuple[float, float],
    pt_right: tuple[float, float],
    K1: np.ndarray, dist1: np.ndarray,
    K2: np.ndarray, dist2: np.ndarray,
    R: np.ndarray, T: np.ndarray,
) -> np.ndarray:
    """Triangulate one point given raw pixel clicks in left and right images.

    Returns 3-D point in the left-camera frame (mm units, same as T).
    """
    p1 = cv2.undistortPoints(
        np.array([[pt_left]], dtype=np.float64), K1, dist1, P=K1
    ).reshape(2)
    p2 = cv2.undistortPoints(
        np.array([[pt_right]], dtype=np.float64), K2, dist2, P=K2
    ).reshape(2)

    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T])

    X_h = cv2.triangulatePoints(P1, P2, p1.reshape(2, 1), p2.reshape(2, 1))
    X = (X_h[:3] / X_h[3]).ravel()
    return X


# ---------------------------------------------------------------------------
# Rigid transform (Procrustes / SVD)
# ---------------------------------------------------------------------------

def _solve_rigid_transform(
    src: np.ndarray,   # (N, 3) points in gate_left frame
    dst: np.ndarray,   # (N, 3) points in shed_left frame
) -> tuple[np.ndarray, np.ndarray]:
    """Solve X_dst = R @ X_src + T  (no scaling).

    Returns (R, T) where T has shape (3,).
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    A = src - src_mean
    B = dst - dst_mean

    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct for reflection
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    T = dst_mean - R @ src_mean
    return R, T


# ---------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------

_PANEL_LABELS = ["gate  LEFT", "gate  RIGHT", "shed  LEFT", "shed  RIGHT"]
_PANEL_COLORS = ["#2196F3", "#64B5F6", "#FF5722", "#FF8A65"]   # blue / orange


class _ClickCollector:
    """Collect point correspondences panel by panel.

    Workflow:
        - Active panel is highlighted. Zoom/pan freely, then click N points.
        - Press Enter to advance to the next panel.
        - After all 4 panels are done, press Enter (or close) to finish.
        - Press [d] to delete the last click in the current panel.
    """

    def __init__(self, axes, images_bgr, status_text):
        self.axes = axes
        self.n_panels = len(axes)
        self.panel_clicks = [[] for _ in range(self.n_panels)]   # clicks per panel
        self.scatter_handles = [[] for _ in range(self.n_panels)]
        self.current_panel = 0
        self.done = False
        self._status = status_text

        self._draw_images(images_bgr)
        self._update_ui()

    def _draw_images(self, images_bgr):
        for ax, img in zip(self.axes, images_bgr):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")

    def _update_ui(self):
        for i, ax in enumerate(self.axes):
            n = len(self.panel_clicks[i])
            label = _PANEL_LABELS[i]
            if not self.done and i == self.current_panel:
                color = _PANEL_COLORS[i]
                weight = "bold"
                prefix = ">> "
            elif i < self.current_panel or self.done:
                color = "#aaaaaa"
                weight = "normal"
                prefix = "   "
            else:
                color = "white"
                weight = "normal"
                prefix = "   "
            ax.set_title(f"{prefix}{label}  [{n} pts]", color=color,
                         fontweight=weight, fontsize=10)

        if self.done:
            n = len(self.panel_clicks[0])
            msg = f"Done — {n} point(s) collected. Close window to compute."
        else:
            n = len(self.panel_clicks[self.current_panel])
            msg = (
                f"Panel {self.current_panel + 1}/4: {_PANEL_LABELS[self.current_panel]}  "
                f"[{n} pts clicked]   Zoom/pan freely, then click points.  "
                f"[Enter] = next panel   [d] = delete last"
            )
        self._status.set_text(msg)

    def on_click(self, event):
        if self.done or event.inaxes is None or event.button != 1:
            return
        toolbar = event.canvas.toolbar
        if toolbar is not None and toolbar.mode != "":
            return
        try:
            panel_idx = self.axes.index(event.inaxes)
        except ValueError:
            return
        if panel_idx != self.current_panel:
            print(f"  Please click in: {_PANEL_LABELS[self.current_panel]}")
            return

        x, y = event.xdata, event.ydata
        pt_num = len(self.panel_clicks[panel_idx]) + 1
        self.panel_clicks[panel_idx].append((x, y))

        sc = self.axes[panel_idx].scatter(
            x, y, c=_PANEL_COLORS[panel_idx], s=60,
            marker="o", zorder=5, linewidths=0.8, edgecolors="white",
        )
        self.axes[panel_idx].annotate(
            str(pt_num), (x, y),
            color="white", fontsize=8, fontweight="bold",
            xytext=(5, 5), textcoords="offset points",
        )
        self.scatter_handles[panel_idx].append(sc)
        print(f"  {_PANEL_LABELS[panel_idx]} pt {pt_num}: ({x:.1f}, {y:.1f})")

        self._update_ui()
        event.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "enter":
            self._advance_panel(event.canvas)
        elif event.key == "d":
            self._delete_last(event.canvas)

    def _advance_panel(self, canvas):
        if self.done:
            return
        n = len(self.panel_clicks[self.current_panel])
        if n == 0:
            print(f"  No points clicked in {_PANEL_LABELS[self.current_panel]} — click at least one.")
            return
        # Check consistency with previous panels
        if self.current_panel > 0:
            prev_n = len(self.panel_clicks[self.current_panel - 1])
            if n != prev_n:
                print(f"  Warning: {_PANEL_LABELS[self.current_panel]} has {n} pts "
                      f"but previous panel had {prev_n}. Continue anyway? "
                      f"Press Enter again to confirm or [d] to delete.")
        print(f"  Panel {self.current_panel + 1} done ({n} pts). ", end="")
        if self.current_panel < self.n_panels - 1:
            self.current_panel += 1
            print(f"Now click in: {_PANEL_LABELS[self.current_panel]}")
        else:
            self.done = True
            print("All panels done. Close window to compute.")
        self._update_ui()
        canvas.draw_idle()

    def _delete_last(self, canvas):
        if self.done:
            return
        clicks = self.panel_clicks[self.current_panel]
        handles = self.scatter_handles[self.current_panel]
        if not clicks:
            print(f"  Nothing to delete in {_PANEL_LABELS[self.current_panel]}.")
            return
        clicks.pop()
        handles.pop().remove()
        # Remove the annotation number
        n = len(clicks) + 1
        ax = self.axes[self.current_panel]
        texts = [t for t in ax.texts if t.get_text() == str(n)]
        for t in texts:
            t.remove()
        print(f"  Deleted last click in {_PANEL_LABELS[self.current_panel]} "
              f"({len(clicks)} remaining)")
        self._update_ui()
        canvas.draw_idle()

    def build_quads(self) -> list[tuple]:
        """Zip panel clicks into quads; trim to shortest panel if mismatched."""
        n = min(len(c) for c in self.panel_clicks)
        return [
            (self.panel_clicks[0][i], self.panel_clicks[1][i],
             self.panel_clicks[2][i], self.panel_clicks[3][i])
            for i in range(n)
        ]


def _collect_correspondences(
    gate_left: np.ndarray,
    gate_right: np.ndarray,
    shed_left: np.ndarray,
    shed_right: np.ndarray,
) -> list[tuple]:
    """Open interactive matplotlib window; return list of click quads."""
    fig, axes_arr = plt.subplots(2, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#1e1e1e")
    for ax in axes_arr.flat:
        ax.set_facecolor("#1e1e1e")

    axes = [axes_arr[0, 0], axes_arr[0, 1], axes_arr[1, 0], axes_arr[1, 1]]
    images = [gate_left, gate_right, shed_left, shed_right]

    status_text = fig.text(0.5, 0.01, "", ha="center", color="lightgrey", fontsize=9)

    collector = _ClickCollector(axes, images, status_text)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    cid_click = fig.canvas.mpl_connect("button_press_event", collector.on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", collector.on_key)

    plt.show()

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    plt.close(fig)

    return collector.build_quads()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stereo_localize(cfg: dict[str, Any]):
    fps = float(cfg.get("fps", 60.0))
    gate_cfg = cfg["gate"]
    shed_cfg = cfg["shed"]
    out_path = Path(cfg["output"])

    # Load calibration
    K_gL, d_gL, _ = load_intrinsics_json(Path(gate_cfg["intrinsics_left"]))
    K_gR, d_gR, _ = load_intrinsics_json(Path(gate_cfg["intrinsics_right"]))
    R_gate, T_gate, _, _ = load_extrinsics_json(Path(gate_cfg["extrinsics"]))

    K_sL, d_sL, _ = load_intrinsics_json(Path(shed_cfg["intrinsics_left"]))
    K_sR, d_sR, _ = load_intrinsics_json(Path(shed_cfg["intrinsics_right"]))
    R_shed, T_shed, _, _ = load_extrinsics_json(Path(shed_cfg["extrinsics"]))

    # Extract frames
    print("Extracting gate frames...")
    gate_imgL, gate_imgR = _load_pair_frames(gate_cfg, fps)
    print("Extracting shed frames...")
    shed_imgL, shed_imgR = _load_pair_frames(shed_cfg, fps)

    # Collect clicks
    print("\nOpening interactive window...")
    print("Click gate_left → gate_right → shed_left → shed_right for each point.")
    print("Press [d] to delete the last click. Close window when done.\n")

    quads = _collect_correspondences(gate_imgL, gate_imgR, shed_imgL, shed_imgR)

    if len(quads) < 3:
        raise RuntimeError(f"Need at least 3 point correspondences, got {len(quads)}.")

    print(f"\nTriangulating {len(quads)} point(s)...")
    pts_gate, pts_shed = [], []

    for i, (gL_pt, gR_pt, sL_pt, sR_pt) in enumerate(quads):
        Xg = _triangulate_point(gL_pt, gR_pt, K_gL, d_gL, K_gR, d_gR, R_gate, T_gate)
        Xs = _triangulate_point(sL_pt, sR_pt, K_sL, d_sL, K_sR, d_sR, R_shed, T_shed)
        pts_gate.append(Xg)
        pts_shed.append(Xs)
        print(f"  pt {i+1}:  gate {Xg.round(1)}  shed {Xs.round(1)}")

    pts_gate = np.array(pts_gate)
    pts_shed = np.array(pts_shed)

    R, T = _solve_rigid_transform(pts_gate, pts_shed)

    # Residuals
    pts_pred = (R @ pts_gate.T).T + T
    residuals = np.linalg.norm(pts_pred - pts_shed, axis=1)
    print(f"\nResiduals (mm): {residuals.round(2)}")
    print(f"Mean residual: {residuals.mean():.2f} mm")

    out = {
        "convention": "X_shed_left = R @ X_gate_left + T",
        "cam1_id": "gate_left",
        "cam2_id": "shed_left",
        "units": "mm",
        "R_shed_from_gate": R.tolist(),
        "T_shed_from_gate_mm": T.tolist(),
        "n_points": len(quads),
        "mean_residual_mm": float(residuals.mean()),
        "residuals_mm": residuals.tolist(),
        "clicks": [
            {
                "gate_left": list(gL), "gate_right": list(gR),
                "shed_left": list(sL), "shed_right": list(sR),
                "X_gate_mm": Xg, "X_shed_mm": Xs,
            }
            for (gL, gR, sL, sR), Xg, Xs in zip(quads, pts_gate.tolist(), pts_shed.tolist())
        ],
    }

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

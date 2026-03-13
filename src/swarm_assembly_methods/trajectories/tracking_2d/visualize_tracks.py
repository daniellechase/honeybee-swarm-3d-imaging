"""
Visualize a random sample of tracks as a grid of subplots.
Each subplot shows one track: path in blue, start in green, end in red.
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import load_tracks


def plot_track_grid(tracks_path, n_tracks=25, seed=42, out_path=None):
    """
    Load tracks from a parquet file and plot a grid of randomly sampled tracks.

    Parameters
    ----------
    tracks_path : str or Path
        Path to a _tracks.npy file saved by save_tracks().
    n_tracks : int
        Number of tracks to sample (shown in an approximately square grid).
    seed : int
        Random seed for reproducibility.
    out_path : str or Path or None
        Where to save the figure. Defaults to tracks_path with _grid.png suffix.
    """
    tracks_path = Path(tracks_path)
    tracks = load_tracks(tracks_path)

    if not tracks:
        print(f"No tracks found in {tracks_path}")
        return

    tids = list(tracks.keys())

    # rank by mean frame-to-frame pixel displacement (real points only)
    def _mean_speed(tid):
        pts = [(pt[0], pt[1], pt[2]) for pt in tracks[tid]
               if (pt[3] if len(pt) > 3 else 0) == 0]
        if len(pts) < 2:
            return 0.0
        diffs = [np.sqrt((pts[k][1]-pts[k-1][1])**2 + (pts[k][2]-pts[k-1][2])**2)
                 for k in range(1, len(pts)) if pts[k][0] - pts[k-1][0] == 1]
        return float(np.mean(diffs)) if diffs else 0.0

    n_top = 5
    sorted_by_speed  = sorted(tids, key=_mean_speed, reverse=True)
    sorted_by_length = sorted(tids, key=lambda t: len(tracks[t]), reverse=True)
    fast    = sorted_by_speed[:n_top]
    longest = [t for t in sorted_by_length if t not in fast][:n_top]
    priority = fast + longest
    rest     = [t for t in tids if t not in set(priority)]
    rng      = random.Random(seed)
    random_pick = rng.sample(rest, min(max(n_tracks - len(priority), 0), len(rest)))
    sample   = priority + random_pick

    cols = int(np.ceil(np.sqrt(len(sample))))
    rows = int(np.ceil(len(sample) / cols))
    cell = 2.5  # inches per cell — equal width and height → square subplots

    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell),
                             squeeze=False)
    axes = axes.reshape(-1)

    for ax, tid in zip(axes, sample):
        traj   = tracks[tid]
        pts    = np.array([(x, y) for _, x, y, *_ in traj])
        interp = np.array([pt[3] if len(pt) > 3 else 0 for pt in traj], dtype=bool)
        real   = ~interp

        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", linewidth=0.8, alpha=0.8)
        if real.any():
            ax.scatter(pts[real, 0], pts[real, 1], color="steelblue", s=4, alpha=0.6, zorder=3)
        if interp.any():
            ax.scatter(pts[interp, 0], pts[interp, 1], color="black", s=4, alpha=0.5, zorder=3)
        ax.scatter(*pts[0], color="green", s=20, zorder=5)
        ax.scatter(*pts[-1], color="red",   s=20, zorder=5)
        tag = " ★" if tid in fast else (" ↕" if tid in longest else "")
        ax.set_title(f"tid {tid}  n={len(pts)}{tag}", fontsize=6)
        ax.set_box_aspect(1)
        ax.invert_yaxis()
        ax.tick_params(labelsize=5)

    for ax in axes[len(sample):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{tracks_path.stem}  —  {len(sample)} of {len(tids)} tracks  "
        f"(blue=detected, black=interpolated, green=start, red=end)",
        fontsize=9,
    )
    plt.tight_layout()

    if out_path is None:
        out_path = tracks_path.with_name(tracks_path.stem + "_grid.png")
    out_path = Path(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def run_visualize_tracks(cfg):
    """
    Run track visualization for all 2D track files matching the gh_files filter.

    Reads from output.tracks_2d_dir / {left,right}_camera / *_tracks.npy.
    Saves *_grid.png alongside each tracks file.
    """
    from pathlib import Path

    det_cfg = cfg.get("detections", {})
    vis_cfg = cfg.get("visualize", {})
    n_tracks = vis_cfg.get("n_sample_tracks", 25)
    seed = vis_cfg.get("seed", 42)
    gh_filter = det_cfg.get("gh_files", None)

    out_dir = Path(cfg["output"]["output_dir"])
    track_files = sorted(out_dir.glob("*_tracks.parquet"))

    if gh_filter is not None:
        stems = [s.upper() for s in gh_filter]
        track_files = [f for f in track_files
                       if any(s in f.stem.upper() for s in stems)]

    for tf in track_files:
        print(f"\nVisualizing {tf.name}")
        plot_track_grid(tf, n_tracks=n_tracks, seed=seed)

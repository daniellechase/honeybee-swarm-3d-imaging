"""
Project a sample of 3D tracks onto XY, XZ, YZ planes.

Layout: 2 tracks per row, each track occupying 3 square subplots (XY, XZ, YZ).
Total columns = 6, rows = ceil(n / 2).
Real detections = steelblue, interpolated gaps = black, start = green, end = red.
"""

import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# (col_a, col_b, xlabel, ylabel, invert_y)
# Y is positive-down; Z is depth (into page as the 2nd horizontal axis in 3D).
_PLANES = [
    ("X", "Y", "X (m)", "Y (m)", True),   # front view — invert Y (down+)
    ("X", "Z", "X (m)", "Z (m)", False),  # top view
    ("Z", "Y", "Z (m)", "Y (m)", True),   # side view — invert Y (down+)
]
_COLOR  = "steelblue"
_PANELS_PER_TRACK = 4   # XY, XZ, ZY, 3D
_TRACKS_PER_ROW   = 2
_NCOLS            = _PANELS_PER_TRACK * _TRACKS_PER_ROW   # 8
_CELL_SIZE        = 2.8  # inches per subplot (square)


def _draw_track(ax, grp, col_a, col_b, xlabel, ylabel, invert_y=False):
    a      = grp[col_a].to_numpy()
    b      = grp[col_b].to_numpy()
    interp = grp["xL"].isna().to_numpy()

    ax.plot(a, b, color=_COLOR, linewidth=0.8, alpha=0.7)
    if (~interp).any():
        ax.scatter(a[~interp], b[~interp], color=_COLOR, s=6, zorder=3, alpha=0.8)
    if interp.any():
        ax.scatter(a[interp], b[interp], color="black", s=6, zorder=4, alpha=0.5)
    ax.scatter(a[0],  b[0],  color="green", s=25, zorder=5)
    ax.scatter(a[-1], b[-1], color="red",   s=25, zorder=5)
    if invert_y:
        ax.invert_yaxis()
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.tick_params(labelsize=5)


def plot_3d_projections(df, n_tracks=25, seed=42, fps=60, out_path=None):
    """
    Plot n_tracks randomly sampled 3D trajectories.

    Layout: 2 tracks per row × 3 planes = 6 columns; ceil(n/2) rows.

    Parameters
    ----------
    df        : pd.DataFrame  columns: traj_id, t, X, Y, Z, xL (NaN = interpolated)
    n_tracks  : int
    seed      : int
    out_path  : Path
    """
    tids = df["traj_id"].unique().tolist()

    # rank by mean 3D speed across consecutive matched frames
    def _mean_speed_3d(tid):
        grp = df[df["traj_id"] == tid].sort_values("t")
        xyz = grp[["X", "Y", "Z"]].to_numpy()
        t   = grp["t"].to_numpy()
        dt  = np.diff(t).astype(float)
        seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        speeds = seg[dt > 0] / (dt[dt > 0] / fps)
        return float(speeds.mean()) if len(speeds) else 0.0

    n_top = 5
    sorted_by_speed  = sorted(tids, key=_mean_speed_3d, reverse=True)
    sorted_by_length = sorted(tids, key=lambda t: len(df[df["traj_id"] == t]), reverse=True)
    fast     = sorted_by_speed[:n_top]
    longest  = [t for t in sorted_by_length if t not in fast][:n_top]
    priority = fast + longest
    rest     = [t for t in tids if t not in set(priority)]
    rng      = random.Random(seed)
    random_pick = rng.sample(rest, min(max(n_tracks - len(priority), 0), len(rest)))
    sample   = priority + random_pick
    n        = len(sample)

    n_rows = math.ceil(n / _TRACKS_PER_ROW)
    fig = plt.figure(figsize=(_NCOLS * _CELL_SIZE, n_rows * _CELL_SIZE))
    gs  = fig.add_gridspec(n_rows, _NCOLS, hspace=0.4, wspace=0.4)

    for idx, tid in enumerate(sample):
        row      = idx // _TRACKS_PER_ROW
        col_base = (idx % _TRACKS_PER_ROW) * _PANELS_PER_TRACK
        grp      = df[df["traj_id"] == tid].sort_values("t")
        interp   = grp["xL"].isna().to_numpy()

        # 2D projection panels
        tag = " ★" if tid in fast else (" ↕" if tid in longest else "")
        plane_labels = ["XY", "XZ", "ZY"]
        for p_idx, (col_a, col_b, xlabel, ylabel, inv_y) in enumerate(_PLANES):
            ax = fig.add_subplot(gs[row, col_base + p_idx])
            _draw_track(ax, grp, col_a, col_b, xlabel, ylabel, invert_y=inv_y)
            ax.set_title(f"tid {tid}{tag} — {plane_labels[p_idx]}", fontsize=6)

        # 3D panel: X left-right, Z into page (mpl Y), Y vertical positive-down (mpl Z inverted)
        ax3d = fig.add_subplot(gs[row, col_base + 3], projection="3d")
        X = grp["X"].to_numpy()
        Y = grp["Y"].to_numpy()
        Z = grp["Z"].to_numpy()
        ax3d.plot(X, Z, Y, color=_COLOR, linewidth=0.8, alpha=0.7)
        if (~interp).any():
            ax3d.scatter(X[~interp], Z[~interp], Y[~interp],
                         color=_COLOR, s=6, zorder=3, alpha=0.8)
        if interp.any():
            ax3d.scatter(X[interp], Z[interp], Y[interp],
                         color="black", s=6, zorder=4, alpha=0.5)
        ax3d.scatter(X[0],  Z[0],  Y[0],  color="green", s=25, zorder=5)
        ax3d.scatter(X[-1], Z[-1], Y[-1], color="red",   s=25, zorder=5)
        ax3d.invert_zaxis()
        ax3d.set_xlabel("X", fontsize=5)
        ax3d.set_ylabel("Z (depth)", fontsize=5)
        ax3d.set_zlabel("Y", fontsize=5)
        ax3d.tick_params(labelsize=4)
        ax3d.set_title(f"tid {tid}{tag} — 3D", fontsize=6)
        ax3d.set_box_aspect([1, 1, 1])

    # hide unused panels (last row may have only 1 track)
    for idx in range(n, n_rows * _TRACKS_PER_ROW):
        row      = idx // _TRACKS_PER_ROW
        col_base = (idx % _TRACKS_PER_ROW) * _PANELS_PER_TRACK
        for p_idx in range(_PANELS_PER_TRACK):
            fig.add_subplot(gs[row, col_base + p_idx]).set_visible(False)

    fig.suptitle(
        f"{n} of {len(tids)} tracks  "
        f"(blue=detected, black=interpolated, green=start, red=end)",
        fontsize=8, y=1.002,
    )
    out_path = Path(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3D projection plot: {out_path}")


def _track_extent(grp):
    """
    Compute a square axis extent centred on the track's midpoint,
    with side length = max range across X, Y, Z (plus 10% padding).
    Returns (xlim, ylim, zlim) each as (lo, hi).
    """
    X = grp["X"].to_numpy()
    Y = grp["Y"].to_numpy()
    Z = grp["Z"].to_numpy()
    cx, cy, cz = X.mean(), Y.mean(), Z.mean()
    half = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()) / 2.0 * 1.1 + 1e-6
    return (cx - half, cx + half), (cy - half, cy + half), (cz - half, cz + half)


_ALL_TRACKS_PER_ROW = 5   # 3D-only layout: 5 panels per row


def _draw_page(tids_page, df, page_num, n_total):
    """
    Draw one page of 3D-only panels (no projections).
    Each panel uses a square extent sized to that track's own max XYZ range,
    so the true scale of motion is preserved and comparable across tracks.
    """
    n      = len(tids_page)
    n_rows = math.ceil(n / _ALL_TRACKS_PER_ROW)
    cell   = _CELL_SIZE * 1.4   # slightly larger since no projection panels
    fig    = plt.figure(figsize=(_ALL_TRACKS_PER_ROW * cell, n_rows * cell))
    gs     = fig.add_gridspec(n_rows, _ALL_TRACKS_PER_ROW, hspace=0.4, wspace=0.4)

    for idx, tid in enumerate(tids_page):
        row = idx // _ALL_TRACKS_PER_ROW
        col = idx %  _ALL_TRACKS_PER_ROW
        grp    = df[df["traj_id"] == tid].sort_values("t")
        interp = grp["xL"].isna().to_numpy()
        xlim, ylim, zlim = _track_extent(grp)

        ax3d = fig.add_subplot(gs[row, col], projection="3d")
        X = grp["X"].to_numpy()
        Y = grp["Y"].to_numpy()
        Z = grp["Z"].to_numpy()
        ax3d.plot(X, Z, Y, color=_COLOR, linewidth=0.8, alpha=0.7)
        if (~interp).any():
            ax3d.scatter(X[~interp], Z[~interp], Y[~interp],
                         color=_COLOR, s=6, zorder=3, alpha=0.8)
        if interp.any():
            ax3d.scatter(X[interp], Z[interp], Y[interp],
                         color="black", s=6, zorder=4, alpha=0.5)
        ax3d.scatter(X[0],  Z[0],  Y[0],  color="green", s=25, zorder=5)
        ax3d.scatter(X[-1], Z[-1], Y[-1], color="red",   s=25, zorder=5)
        ax3d.set_xlim(xlim); ax3d.set_ylim(zlim); ax3d.set_zlim(ylim)
        ax3d.invert_zaxis()
        ax3d.set_xlabel("X", fontsize=5)
        ax3d.set_ylabel("Z (depth)", fontsize=5)
        ax3d.set_zlabel("Y", fontsize=5)
        ax3d.tick_params(labelsize=4)
        ax3d.set_title(f"tid {tid}", fontsize=6)
        ax3d.set_box_aspect([1, 1, 1])

    # hide unused cells in last row
    for idx in range(n, n_rows * _ALL_TRACKS_PER_ROW):
        row = idx // _ALL_TRACKS_PER_ROW
        col = idx %  _ALL_TRACKS_PER_ROW
        fig.add_subplot(gs[row, col]).set_visible(False)

    fig.suptitle(
        f"Page {page_num} — tracks {tids_page[0]}–{tids_page[-1]}  "
        f"({n_total} total)  "
        f"(blue=detected, black=interpolated, green=start, red=end)",
        fontsize=7, y=1.002,
    )
    return fig


def plot_all_3d_projections(df, n_per_page=25, out_path=None):
    """
    Save every 3D trajectory to a multi-page PDF, n_per_page tracks per page.

    Tracks are sorted by traj_id.  Output is a .pdf file.

    Parameters
    ----------
    df         : pd.DataFrame  columns: traj_id, t, X, Y, Z, xL
    fps        : float
    n_per_page : int   tracks per PDF page (default 25)
    out_path   : Path  should end in .pdf
    """
    out_path = Path(out_path)
    tids     = sorted(df["traj_id"].unique().tolist())
    n_total  = len(tids)

    with PdfPages(out_path) as pdf:
        for page_num, start in enumerate(range(0, n_total, n_per_page), start=1):
            tids_page = tids[start : start + n_per_page]
            fig = _draw_page(tids_page, df, page_num, n_total)
            pdf.savefig(fig, bbox_inches="tight", dpi=120)
            plt.close(fig)

    n_pages = math.ceil(n_total / n_per_page)
    print(f"  Saved all-tracks PDF ({n_total} tracks, {n_pages} pages): {out_path}")

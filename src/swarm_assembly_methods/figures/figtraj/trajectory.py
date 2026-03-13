"""
Visualize 3D trajectories overlaid on rectified video frames.

Ported from visualization.py (swarm-self-assembly repo).
Produces per-trajectory PDFs: frame overlay, speed overlay, and 3D velocity plot.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from pathlib import Path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_video_frame(video_path, frame_idx, fallback_shape=(1440, 1920)):
    if video_path is None:
        h, w = fallback_shape
        return np.full((h, w, 3), 255, dtype=np.uint8)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: could not open video {video_path}, using blank frame.")
        h, w = fallback_shape
        cap.release()
        return np.full((h, w, 3), 255, dtype=np.uint8)
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    if not ok:
        cap.release()
        raise ValueError(f"Could not seek to frame {frame_idx} in {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def _rectify_frame(frame_bgr, K, dist, R_rect, P_rect):
    h, w = frame_bgr.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, R_rect, P_rect, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)


def _set_equal_3d_axes(ax, x, y, z):
    max_range = max(np.ptp(x), np.ptp(y), np.ptp(z)) / 2.0
    if max_range == 0:
        max_range = 0.1
    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def _interpolate_track_xy_synced(traj, max_gap=5):
    """
    traj: iterable of (t, x, y) with integer frame indices t.
    Returns list of (t, x, y, is_interp).
    """
    traj = sorted([(int(t), float(x), float(y)) for (t, x, y) in traj], key=lambda a: a[0])
    if len(traj) < 2:
        return [(t, x, y, False) for t, x, y in traj]

    T = np.array([p[0] for p in traj], dtype=int)
    X = np.array([p[1] for p in traj], dtype=float)
    Y = np.array([p[2] for p in traj], dtype=float)

    out = []
    for i in range(len(T) - 1):
        t0, t1 = T[i], T[i + 1]
        x0, x1 = X[i], X[i + 1]
        y0, y1 = Y[i], Y[i + 1]
        out.append((t0, x0, y0, False))
        gap = t1 - t0 - 1
        if gap > 0 and gap <= max_gap:
            for tt in range(t0 + 1, t1):
                a = (tt - t0) / (t1 - t0)
                out.append((tt, x0 + a * (x1 - x0), y0 + a * (y1 - y0), True))
    out.append((int(T[-1]), float(X[-1]), float(Y[-1]), False))
    return out


def _draw_trajectory_on_frame(frame_bgr, traj, color_start, color_end,
                               radius, thickness,
                               show_interp=True, color_values=None, colormap="viridis"):
    import matplotlib.cm as cm
    out = frame_bgr.copy()
    n = len(traj)
    if n == 0:
        return out

    parsed = []
    for item in traj:
        if len(item) == 3:
            f, x, y = item; is_interp = False
        else:
            f, x, y, is_interp = item
        parsed.append((int(f), float(x), float(y), bool(is_interp)))

    pts = [(int(round(x)), int(round(y))) for _, x, y, _ in parsed]
    interps = [is_interp for _, _, _, is_interp in parsed]

    if color_values is not None:
        cv_arr = np.asarray(color_values, dtype=float)
        cv_min = np.nanmin(cv_arr)
        cv_max = np.nanmax(cv_arr)
        cv_range = cv_max - cv_min if cv_max > cv_min else 1.0
        cmap = cm.get_cmap(colormap)
        def _bgr(i):
            frac = (cv_arr[i] - cv_min) / cv_range
            rgba = cmap(float(np.clip(frac, 0, 1)))
            return (int(255 * rgba[2]), int(255 * rgba[1]), int(255 * rgba[0]))
    else:
        cmap = cm.get_cmap("viridis")
        def _bgr(i):
            frac = i / max(n - 1, 1)
            rgba = cmap(frac)
            return (int(255 * rgba[2]), int(255 * rgba[1]), int(255 * rgba[0]))

    for i in range(n):
        bgr = _bgr(i)
        if i > 0:
            cv2.line(out, pts[i - 1], pts[i], bgr, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, pts[i], radius, bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, pts[i], radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    cv2.circle(out, pts[0],  radius + 4, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(out, pts[-1], radius + 4, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def plot_trajectory_on_video_frames(
    video_path_left,
    video_path_right,
    K1, d1, R1, P1r,
    K2, d2, R2, P2r,
    df3d,
    track_L,
    track_R=None,
    dk=0,
    traj_id=None,
    pick="longest",
    n_trajectories=1,
    length_min=None,
    length_max=None,
    t_min=None,
    t_max=None,
    random_seed=None,
    trail_color_start=(255, 255, 0),
    trail_color_end=(0, 0, 255),
    point_radius=5,
    line_thickness=2,
    figsize=(20, 7),
    dpi=150,
    debug_interp=False,
    save_dir=None,
    prefix="",
):
    """
    Visualize 3D trajectory(ies) overlaid on rectified GoPro video frames.

    For each trajectory, produces up to three PDFs:
      {prefix}_trajectory_{tid}.pdf          — left/right frame + 3D scatter
      {prefix}_trajectory_{tid}_speed.pdf    — speed-colored frame overlay
      {prefix}_trajectory_{tid}_velocity.pdf — 3D speed scatter

    Parameters
    ----------
    track_L : dict  {tid: ndarray (N, 3) [frame, x_rect, y_rect]}
        Rectified left 2D tracks.  traj_id == left track tid.
    save_dir : Path or None
        Directory to save PDFs.  None = current directory.
    prefix : str
        Filename prefix (e.g. GH430142).
    """
    plt.rcParams.update({
        "axes.labelsize": 16, "axes.titlesize": 16,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
    })
    save_dir = Path(save_dir) if save_dir is not None else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)

    traj_lengths = df3d.groupby("traj_id").size()

    valid_ids = traj_lengths.index
    if t_min is not None or t_max is not None:
        t_bounds = df3d.groupby("traj_id")["t"].agg(["min", "max"])
        if t_min is not None:
            t_bounds = t_bounds[t_bounds["max"] >= t_min]
        if t_max is not None:
            t_bounds = t_bounds[t_bounds["min"] <= t_max]
        valid_ids = t_bounds.index

    candidates = traj_lengths[traj_lengths.index.isin(valid_ids)]
    if length_min is not None:
        candidates = candidates[candidates >= length_min]
    if length_max is not None:
        candidates = candidates[candidates <= length_max]

    if pick == "longest":
        traj_ids = candidates.sort_values(ascending=False).head(n_trajectories).index.tolist()
    elif pick == "random":
        rng = np.random.default_rng(random_seed)
        pool = candidates.index.tolist()
        traj_ids = rng.choice(pool, size=min(n_trajectories, len(pool)), replace=False).tolist()
    elif pick == "id":
        if traj_id is None:
            raise ValueError("traj_id must be supplied when pick='id'")
        traj_ids = [traj_id]
    else:
        raise ValueError("pick must be 'longest', 'random', or 'id'")

    has_px_cols = {"xL", "yL", "xR", "yR"}.issubset(df3d.columns)

    has_orig_col = "traj_id_orig" in df3d.columns

    for tid in traj_ids:
        traj_df = df3d[df3d["traj_id"] == tid].sort_values("t")
        if len(traj_df) == 0:
            all_ids = sorted(df3d["traj_id"].unique())
            print(f"  [traj_id={tid}] No points. Available IDs (first 20): {all_ids[:20]}")
            continue

        # resolve the left track ID (original tracker ID before compaction)
        track_L_key = int(traj_df["traj_id_orig"].iloc[0]) if has_orig_col else tid
        if track_L_key not in track_L:
            print(f"  [traj_id={tid}] left track key {track_L_key} not found in track_L, skipping.")
            continue

        # --- left track ---
        traj_L_raw  = track_L[track_L_key]
        traj_L_dense = _interpolate_track_xy_synced(traj_L_raw, max_gap=3)
        Lmap = {t: (x, y, interp) for (t, x, y, interp) in traj_L_dense}

        # --- right track ---
        if has_px_cols:
            Rmap = {}
            for row in traj_df[["t", "xR", "yR"]].itertuples(index=False):
                is_interp = np.isnan(row.xR) or np.isnan(row.yR)
                Rmap[int(row.t)] = (row.xR, row.yR, True) if is_interp else (row.xR, row.yR, False)
            r_real = [(t, xy[0], xy[1]) for t, xy in Rmap.items() if not xy[2]]
            if len(r_real) >= 2:
                r_t = np.array([r[0] for r in r_real], float)
                r_x = np.array([r[1] for r in r_real], float)
                r_y = np.array([r[2] for r in r_real], float)
                # fill existing NaN entries
                for t_key, (xv, yv, is_interp) in list(Rmap.items()):
                    if is_interp:
                        Rmap[t_key] = (float(np.interp(t_key, r_t, r_x)),
                                       float(np.interp(t_key, r_t, r_y)),
                                       True)
                # add entries for Lmap frames not yet in Rmap (within matched range)
                for t_key in Lmap:
                    if t_key not in Rmap and r_t[0] <= t_key <= r_t[-1]:
                        Rmap[t_key] = (float(np.interp(t_key, r_t, r_x)),
                                       float(np.interp(t_key, r_t, r_y)),
                                       True)
        else:
            if track_R is None or "tidR" not in traj_df.columns:
                print(f"  [traj_id={tid}] No right pixel data available, skipping.")
                continue
            tidR = int(traj_df["tidR"].iloc[0])
            if tidR not in track_R:
                print(f"  [traj_id={tid}] tidR={tidR} not in track_R, skipping.")
                continue
            traj_R_dense = _interpolate_track_xy_synced(track_R[tidR], max_gap=3)
            Rmap = {t: (x, y, interp) for (t, x, y, interp) in traj_R_dense}

        t_common = sorted(t for t in Lmap if t in Rmap and not np.isnan(Rmap[t][0]))
        traj_L_plot = [(t, *Lmap[t]) for t in t_common]
        traj_R_plot = [(t, *Rmap[t]) for t in t_common]

        last_frame = int(traj_df["t"].iloc[-1])
        frame_L = _read_video_frame(video_path_left, last_frame)
        frame_L_rect = _rectify_frame(frame_L, K1, d1, R1, P1r) if video_path_left is not None else frame_L
        frame_R = _read_video_frame(video_path_right, last_frame + int(dk))
        frame_R_rect = _rectify_frame(frame_R, K2, d2, R2, P2r) if video_path_right is not None else frame_R

        img_L = _draw_trajectory_on_frame(
            frame_L_rect, traj_L_plot, trail_color_start, trail_color_end,
            point_radius, line_thickness, show_interp=debug_interp)
        img_R = _draw_trajectory_on_frame(
            frame_R_rect, traj_R_plot, trail_color_start, trail_color_end,
            point_radius, line_thickness, show_interp=debug_interp)

        # --- dense 3D for plotting ---
        t3 = traj_df["t"].to_numpy(int)
        X3 = traj_df["X"].to_numpy(float)
        Y3 = traj_df["Y"].to_numpy(float)
        Z3 = traj_df["Z"].to_numpy(float)
        if len(t3) >= 2:
            t_dense = np.arange(t3.min(), t3.max() + 1)
            X = np.interp(t_dense, t3, X3)
            Y = np.interp(t_dense, t3, Y3)
            Z = np.interp(t_dense, t3, Z3)
        else:
            t_dense, X, Y, Z = t3, X3, Y3, Z3

        t_color = (t_dense - t_dense[0]) / 60.0

        # --- panel figure: left frame | right frame | 3D ---
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs  = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0)
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB))
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB))
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[2], projection="3d")
        sc = ax3.scatter(X, Z, Y, c=t_color, cmap="viridis", s=60, alpha=1,
                         vmin=0, vmax=t_color.max())
        ax3.scatter(X[0],  Z[0],  Y[0],  s=100, c="none", marker="o",
                    edgecolors=(0, 1, 0), linewidths=1.5, label="start", depthshade=False)
        ax3.scatter(X[-1], Z[-1], Y[-1], s=100, c="none", marker="o",
                    edgecolors="red",    linewidths=1.5, label="end",   depthshade=False)
        cbar = fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.1, shrink=0.7)
        cbar.set_label("Time (s)", labelpad=10)

        _set_equal_3d_axes(ax3, X, Z, Y)
        ax3.invert_zaxis()
        ax3.tick_params(axis="z", pad=8)
        ax3.set_xlabel("X (m)", labelpad=10)
        ax3.set_ylabel("Z (m)", labelpad=10)
        ax3.set_zlabel("Y (m)", labelpad=20)
        ax3.view_init(elev=18, azim=-60)
        ax3.set_box_aspect((1, 1, 1))
        ax3.xaxis.set_major_locator(MaxNLocator(4))
        ax3.yaxis.set_major_locator(MaxNLocator(4))
        ax3.zaxis.set_major_locator(MaxNLocator(4))

        # wall projections
        proj_alpha = 0.25
        vmax_c = t_color.max() if t_color.max() > 0 else 1.0
        xlim3 = ax3.get_xlim(); ylim3 = ax3.get_ylim(); zlim3 = ax3.get_zlim()
        ax3.scatter(X, Z, zs=zlim3[0], zdir="z",
                    c=t_color, cmap="viridis", vmin=0, vmax=vmax_c,
                    s=10, alpha=proj_alpha, depthshade=False)
        ax3.scatter(X, Y, zs=ylim3[1], zdir="y",
                    c=t_color, cmap="viridis", vmin=0, vmax=vmax_c,
                    s=10, alpha=proj_alpha, depthshade=False)
        ax3.scatter(Z, Y, zs=xlim3[0], zdir="x",
                    c=t_color, cmap="viridis", vmin=0, vmax=vmax_c,
                    s=10, alpha=proj_alpha, depthshade=False)

        if debug_interp:
            is_interp = ~np.isin(t_dense, t3)
            if is_interp.sum() > 0:
                ax3.scatter(X[is_interp], Z[is_interp], Y[is_interp],
                            c="red", marker="x", s=40, linewidths=1.5,
                            depthshade=False, zorder=100, label="interpolated")

        plt.tight_layout()
        p = save_dir / f"{prefix}_trajectory_{tid}.pdf"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print(f"  Saved: {p}")
        plt.close(fig)


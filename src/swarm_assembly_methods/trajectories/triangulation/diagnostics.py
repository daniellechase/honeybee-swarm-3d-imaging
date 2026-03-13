"""
Diagnostic histograms:
  Row 1: frame-to-frame pixel displacement (left 2D, right 2D), 3D displacement
  Row 2: Vx, Vy, Vz — smoothed (blue) vs raw (orange) overlaid
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _2d_deltas(tracks):
    """
    Frame-to-frame Euclidean pixel displacement for real (non-interpolated) points.

    Parameters
    ----------
    tracks : dict  {tid: [(frame, x, y, interp), ...]}
    """
    deltas = []
    for traj in tracks.values():
        pts = [(pt[0], pt[1], pt[2]) for pt in traj
               if (pt[3] if len(pt) > 3 else 0) == 0]   # real only
        for k in range(1, len(pts)):
            df = pts[k][0] - pts[k-1][0]
            if df == 1:   # consecutive frames only
                dx = pts[k][1] - pts[k-1][1]
                dy = pts[k][2] - pts[k-1][2]
                deltas.append(np.sqrt(dx*dx + dy*dy))
    return np.array(deltas)


def _3d_velocities(df, xyz_cols, fps):
    """
    Per-frame velocities (m/s) along each axis.

    Parameters
    ----------
    df       : pd.DataFrame  with traj_id, t, and xyz_cols
    xyz_cols : tuple of 3 column names, e.g. ("X", "Y", "Z")
    fps      : float

    Returns
    -------
    vx, vy, vz : np.ndarray  (consecutive matched frames, dt==1 only)
    """
    vx, vy, vz = [], [], []
    for _, grp in df.groupby("traj_id"):
        grp = grp.sort_values("t")
        matched = grp["xL"].notna().to_numpy()
        t   = grp["t"].to_numpy()
        x   = grp[xyz_cols[0]].to_numpy()
        y   = grp[xyz_cols[1]].to_numpy()
        z   = grp[xyz_cols[2]].to_numpy()
        for k in range(1, len(t)):
            if not (matched[k] and matched[k-1]):
                continue
            dt = (t[k] - t[k-1]) / fps
            if dt <= 0:
                continue
            vx.append((x[k] - x[k-1]) / dt)
            vy.append((y[k] - y[k-1]) / dt)
            vz.append((z[k] - z[k-1]) / dt)
    return np.array(vx), np.array(vy), np.array(vz)


def _3d_distances(df, xyz_cols, fps):
    """
    Per-frame 3D displacement (metres/frame) — normalised by frame gap so
    values are directly comparable to max_3d_dist regardless of gaps.
    """
    dists = []
    for _, grp in df.groupby("traj_id"):
        grp     = grp.sort_values("t")
        matched = grp["xL"].notna().to_numpy()
        xyz     = grp[xyz_cols].to_numpy()
        t       = grp["t"].to_numpy()
        # only consecutive matched-frame pairs
        mask = matched[:-1] & matched[1:]
        if not mask.any():
            continue
        dt         = np.diff(t).astype(float)[mask]
        diff_norms = np.linalg.norm(np.diff(xyz, axis=0), axis=1)[mask]
        dists.append(diff_norms / np.maximum(dt, 1))
    return np.concatenate(dists) if dists else np.array([])


def _2d_acceleration(tracks):
    """
    Second-order finite difference Δ²x(t) = x(t+1) - 2x(t) + x(t-1) for real
    (non-interpolated) points on strictly consecutive frames.

    Returns
    -------
    acc_x, acc_y : np.ndarray  (one value per interior real point triplet)
    """
    acc_x, acc_y = [], []
    for traj in tracks.values():
        pts = [(pt[0], pt[1], pt[2]) for pt in traj
               if (pt[3] if len(pt) > 3 else 0) == 0]   # real only
        for k in range(1, len(pts) - 1):
            if pts[k][0] - pts[k-1][0] == 1 and pts[k+1][0] - pts[k][0] == 1:
                acc_x.append(pts[k+1][1] - 2*pts[k][1] + pts[k-1][1])
                acc_y.append(pts[k+1][2] - 2*pts[k][2] + pts[k-1][2])
    return np.array(acc_x), np.array(acc_y)


def _3d_vel_accel_paired(df, xyz_cols, fps):
    """
    For each interior matched frame in each trajectory, compute:
      speed       = ||X(t+1) - X(t-1)|| / (2/fps)   [central difference, m/s]
      accel_mag   = ||X(t+1) - 2X(t) + X(t-1)|| * fps²  [m/s²]

    Returns
    -------
    speeds, accels : np.ndarray  (same length, one per valid triplet)
    """
    speeds, accels = [], []
    for _, grp in df.groupby("traj_id"):
        grp     = grp.sort_values("t")
        matched = grp["xL"].notna().to_numpy()
        t       = grp["t"].to_numpy()
        xyz     = grp[list(xyz_cols)].to_numpy()
        for k in range(1, len(t) - 1):
            if not (matched[k-1] and matched[k] and matched[k+1]):
                continue
            if t[k] - t[k-1] != 1 or t[k+1] - t[k] != 1:
                continue
            delta = xyz[k+1] - xyz[k-1]
            d2    = xyz[k+1] - 2*xyz[k] + xyz[k-1]
            speeds.append(np.linalg.norm(delta) * fps / 2.0)
            accels.append(np.linalg.norm(d2) * fps**2)
    return np.array(speeds), np.array(accels)


def _y_error_vs_vel(df, xyz_cols, fps):
    """
    For each matched frame with a valid velocity estimate, return:
      y_err  = |yL - yR|  (epipolar residual, pixels)
      speed  = ||X(t+1) - X(t-1)|| / (2/fps)   (central difference, m/s)

    Returns
    -------
    y_errs, speeds : np.ndarray
    """
    y_errs, speeds = [], []
    for _, grp in df.groupby("traj_id"):
        grp     = grp.sort_values("t")
        matched = grp["xL"].notna().to_numpy()
        t       = grp["t"].to_numpy()
        xyz     = grp[list(xyz_cols)].to_numpy()
        yL      = grp["yL"].to_numpy()
        yR      = grp["yR"].to_numpy()
        for k in range(1, len(t) - 1):
            if not (matched[k-1] and matched[k] and matched[k+1]):
                continue
            if t[k] - t[k-1] != 1 or t[k+1] - t[k] != 1:
                continue
            delta = xyz[k+1] - xyz[k-1]
            y_errs.append(abs(yL[k] - yR[k]))
            speeds.append(np.linalg.norm(delta) * fps / 2.0)
    return np.array(y_errs), np.array(speeds)


def _3d_accel_components(df, xyz_cols, fps):
    """
    Per-axis 3D acceleration Δ²X(t) = X(t+1) - 2X(t) + X(t-1), scaled to m/s².
    Only consecutive matched frames (dt=1) are used.

    Returns
    -------
    ax, ay, az : np.ndarray
    """
    ax, ay, az = [], [], []
    for _, grp in df.groupby("traj_id"):
        grp     = grp.sort_values("t")
        matched = grp["xL"].notna().to_numpy()
        t       = grp["t"].to_numpy()
        xyz     = grp[list(xyz_cols)].to_numpy()
        for k in range(1, len(t) - 1):
            if not (matched[k-1] and matched[k] and matched[k+1]):
                continue
            if t[k] - t[k-1] != 1 or t[k+1] - t[k] != 1:
                continue
            d2 = xyz[k+1] - 2*xyz[k] + xyz[k-1]
            ax.append(d2[0] * fps**2)
            ay.append(d2[1] * fps**2)
            az.append(d2[2] * fps**2)
    return np.array(ax), np.array(ay), np.array(az)


def _hist(ax, data, label, color, bins=80, alpha=0.6):
    if len(data) == 0:
        return
    ax.hist(data, bins=bins, color=color, alpha=alpha, label=label, density=True)
    mean = np.mean(data)
    std  = np.std(data)
    stat_label = f"μ={mean:.2f}  σ={std:.2f}"
    ax.axvline(mean, color=color, linestyle="--", linewidth=1.0, alpha=0.9,
               label=stat_label)


def plot_diagnostics(tracks_L, tracks_R, df3d, fps, out_path, prefix="", max_3d_dist=None,
                     pred_residuals_L=None, pred_residuals_R=None):
    """
    Plot and save diagnostic histograms.

    Parameters
    ----------
    tracks_L, tracks_R : dict  {tid: [(frame, x, y, interp), ...]}  2D tracks
    df3d               : pd.DataFrame  3D tracks (must have X_raw/Y_raw/Z_raw if smoothed)
    fps                : float
    out_path           : Path
    prefix             : str  for figure title
    """
    has_raw = "X_raw" in df3d.columns

    fig, axes = plt.subplots(6, 4, figsize=(18, 24))

    # ---- Row 0: 2D delta pixels + 3D displacement (col 3 unused) ----
    delta_L = _2d_deltas(tracks_L)
    delta_R = _2d_deltas(tracks_R)
    dist_sm = _3d_distances(df3d, ["X", "Y", "Z"], fps)

    _hist(axes[0, 0], delta_L, "left",  "steelblue")
    axes[0, 0].set_title("Left cam: frame-to-frame Δpx")
    axes[0, 0].set_xlabel("pixels")
    axes[0, 0].set_ylabel("density")

    _hist(axes[0, 1], delta_R, "right", "steelblue")
    axes[0, 1].set_title("Right cam: frame-to-frame Δpx")
    axes[0, 1].set_xlabel("pixels")

    _hist(axes[0, 2], dist_sm, "smoothed", "steelblue")
    if has_raw:
        dist_raw = _3d_distances(df3d, ["X_raw", "Y_raw", "Z_raw"], fps)
        _hist(axes[0, 2], dist_raw, "raw", "darkorange")
    if max_3d_dist is not None:
        axes[0, 2].axvline(max_3d_dist, color="red", linestyle="--",
                           linewidth=1.2, label=f"gate={max_3d_dist}m")
    axes[0, 2].set_title("3D displacement per frame (m/frame)")
    axes[0, 2].set_xlabel("metres/frame")

    # col 3: prediction residuals for left and right overlaid
    ax = axes[0, 3]
    if pred_residuals_L is not None and len(pred_residuals_L):
        _hist(ax, pred_residuals_L, "left",  "steelblue")
    if pred_residuals_R is not None and len(pred_residuals_R):
        _hist(ax, pred_residuals_R, "right", "darkorange")
    ax.set_title("2D match: dist to prediction (px)")
    ax.set_xlabel("pixels")

    # ---- Row 1: Vx, Vy, Vz smoothed vs raw (col 3 unused) ----
    vx_sm, vy_sm, vz_sm = _3d_velocities(df3d, ("X",     "Y",     "Z"),     fps)
    labels_vel = [("Vx (m/s)", vx_sm), ("Vy (m/s)", vy_sm), ("Vz (m/s)", vz_sm)]

    if has_raw:
        vx_r, vy_r, vz_r = _3d_velocities(df3d, ("X_raw", "Y_raw", "Z_raw"), fps)
        raw_vel = [vx_r, vy_r, vz_r]
    else:
        raw_vel = [None, None, None]

    for col, (xlabel, v_sm) in enumerate(labels_vel):
        ax = axes[1, col]
        _hist(ax, v_sm, "smoothed", "steelblue")
        if raw_vel[col] is not None:
            _hist(ax, raw_vel[col], "raw", "darkorange")
        ax.set_title(f"3D {xlabel}")
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel("density")

    # col 3: 3D prediction error histogram
    ax = axes[1, 3]
    if "pred_error" in df3d.columns:
        pe = df3d["pred_error"].dropna().to_numpy()
        _hist(ax, pe, None, "darkorchid")
    ax.set_title("3D prediction error (m/frame)")
    ax.set_xlabel("metres/frame")

    # ---- Row 2: 2D acceleration Δ²x, Δ²y for left and right cameras ----
    acc_Lx, acc_Ly = _2d_acceleration(tracks_L)
    acc_Rx, acc_Ry = _2d_acceleration(tracks_R)

    accel_panels = [
        (axes[2, 0], acc_Lx, "Left cam: Δ²x (px)"),
        (axes[2, 1], acc_Ly, "Left cam: Δ²y (px)"),
        (axes[2, 2], acc_Rx, "Right cam: Δ²x (px)"),
        (axes[2, 3], acc_Ry, "Right cam: Δ²y (px)"),
    ]
    for i, (ax, data, title) in enumerate(accel_panels):
        _hist(ax, data, None, "mediumpurple")
        ax.set_title(title)
        ax.set_xlabel("pixels")
        if i == 0:
            ax.set_ylabel("density")

    # ---- Row 3: scatter plots ----
    xyz_sm = ("X", "Y", "Z")

    # Panel 0: 3D speed vs 3D acceleration magnitude
    sp, ac = _3d_vel_accel_paired(df3d, xyz_sm, fps)
    ax = axes[3, 0]
    if len(sp):
        ax.scatter(sp, ac, s=1, alpha=0.3, color="steelblue", rasterized=True)
    ax.set_title("3D speed vs acceleration")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("accel magnitude (m/s²)")

    # Panel 1: |yL - yR| vs 3D speed
    ye, sp2 = _y_error_vs_vel(df3d, xyz_sm, fps)
    ax = axes[3, 1]
    if len(ye):
        ax.scatter(ye, sp2, s=1, alpha=0.3, color="darkorange", rasterized=True)
    ax.set_title("|y error| vs 3D speed")
    ax.set_xlabel("|yL − yR| (px)")
    ax.set_ylabel("speed (m/s)")

    # Panel 2: speed vs disparity (xL - xR)
    ax = axes[3, 2]
    matched_mask = df3d["xL"].notna() & df3d["xR"].notna()
    if matched_mask.any():
        disp = (df3d.loc[matched_mask, "xL"] - df3d.loc[matched_mask, "xR"]).to_numpy()
        # get per-row speed via central difference — build a lookup {(traj_id, t): speed}
        sp_lookup = {}
        for _, grp in df3d.groupby("traj_id"):
            grp  = grp.sort_values("t")
            msk  = grp["xL"].notna().to_numpy()
            t_arr = grp["t"].to_numpy()
            xyz_arr = grp[["X", "Y", "Z"]].to_numpy()
            idx_arr = grp.index.tolist()
            for k in range(1, len(t_arr) - 1):
                if not (msk[k-1] and msk[k] and msk[k+1]):
                    continue
                if t_arr[k] - t_arr[k-1] != 1 or t_arr[k+1] - t_arr[k] != 1:
                    continue
                sp_lookup[(grp["traj_id"].iloc[k], t_arr[k])] = (
                    np.linalg.norm(xyz_arr[k+1] - xyz_arr[k-1]) * fps / 2.0
                )
        matched_df = df3d[matched_mask]
        sp_vals = np.array([
            sp_lookup.get((row.traj_id, row.t), np.nan)
            for row in matched_df.itertuples()
        ])
        valid = ~np.isnan(sp_vals)
        if valid.any():
            ax.scatter(sp_vals[valid], disp[valid], s=1, alpha=0.3,
                       color="teal", rasterized=True)
    ax.set_title("speed vs disparity")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("disparity xL−xR (px)")

    # Panel 3: |vz| vs sqrt(vx²+vy²) scatter  (reuse vx_sm/vy_sm/vz_sm from row 1)
    v_horiz = np.sqrt(vx_sm**2 + vy_sm**2)
    v_vert  = np.abs(vz_sm)
    ax = axes[3, 3]
    if len(v_horiz):
        ax.scatter(v_horiz, v_vert, s=1, alpha=0.3, color="crimson", rasterized=True)
    ax.set_title("|vz| vs horizontal speed")
    ax.set_xlabel("√(vx²+vy²) (m/s)")
    ax.set_ylabel("|vz| (m/s)")

    # ---- Row 4: 3D acceleration components ----
    a3x, a3y, a3z = _3d_accel_components(df3d, xyz_sm, fps)
    for col, (data, title) in enumerate([
        (a3x, "3D Δ²X (m/s²)"),
        (a3y, "3D Δ²Y (m/s²)"),
        (a3z, "3D Δ²Z (m/s²)"),
    ]):
        ax = axes[4, col]
        _hist(ax, data, None, "seagreen")
        ax.set_title(title)
        ax.set_xlabel("m/s²")
        if col == 0:
            ax.set_ylabel("density")
    # col 3: overlaid histograms of |vz| and sqrt(vx²+vy²)
    ax = axes[4, 3]
    _hist(ax, v_horiz, "√(vx²+vy²)", "steelblue")
    _hist(ax, v_vert,  "|vz|",        "crimson")
    ax.set_title("|vz| vs horiz speed (hist)")
    ax.set_xlabel("m/s")

    # ---- Row 5: prediction error vs speed scatter ----
    if "pred_error" in df3d.columns:
        pe_vals = df3d["pred_error"].to_numpy()
        # build speed lookup (central difference, m/s) same as above
        sp_lookup2 = {}
        for _, grp in df3d.groupby("traj_id"):
            grp     = grp.sort_values("t")
            msk     = grp["xL"].notna().to_numpy()
            t_arr   = grp["t"].to_numpy()
            xyz_arr = grp[["X", "Y", "Z"]].to_numpy()
            for k in range(1, len(t_arr) - 1):
                if not (msk[k-1] and msk[k] and msk[k+1]):
                    continue
                if t_arr[k] - t_arr[k-1] != 1 or t_arr[k+1] - t_arr[k] != 1:
                    continue
                sp_lookup2[(grp["traj_id"].iloc[k], t_arr[k])] = (
                    np.linalg.norm(xyz_arr[k+1] - xyz_arr[k-1]) * fps / 2.0
                )
        sp_pe = np.array([
            sp_lookup2.get((row.traj_id, row.t), np.nan)
            for row in df3d.itertuples()
        ])
        valid = ~np.isnan(pe_vals) & ~np.isnan(sp_pe)
        ax = axes[5, 0]
        if valid.any():
            ax.scatter(sp_pe[valid], pe_vals[valid], s=1, alpha=0.3,
                       color="darkorchid", rasterized=True)
        ax.set_title("prediction error vs speed")
        ax.set_xlabel("speed (m/s)")
        ax.set_ylabel("pred error (m/frame)")

    for col, (err_col, title) in enumerate(
        [("pred_error_x", "X error vs speed"),
         ("pred_error_y", "Y error vs speed"),
         ("pred_error_z", "Z error vs speed")], start=1
    ):
        ax = axes[5, col]
        if err_col in df3d.columns:
            err_vals = df3d[err_col].to_numpy()
            valid2 = ~np.isnan(err_vals) & ~np.isnan(sp_pe)
            if valid2.any():
                ax.scatter(sp_pe[valid2], err_vals[valid2], s=1, alpha=0.3,
                           color="darkorchid", rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("speed (m/s)")
        ax.set_ylabel("pred error (m/frame)")

    for ax in axes.flat:
        if not ax.get_visible():
            continue
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)

    fig.suptitle(f"{prefix} — tracking diagnostics", fontsize=10)
    plt.tight_layout()
    out_path = Path(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved diagnostics: {out_path}")

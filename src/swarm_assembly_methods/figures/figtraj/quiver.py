"""
Quiver and trajectory-count plots for 3D bee tracks.

Two main entry points:
    plot_combined_quiver_and_projections(df, cfg)
    plot_traj_count_3proj(df, cfg, metric="unique")

Both expect df to have columns: traj_id, t, X, Y, Z, vx, vy, vz, speed_mps.
Use compute_velocities_3d(df, fps) to add those columns first.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  Coordinate-system alignment (shed → gate)                           #
# ------------------------------------------------------------------ #

def load_localization(path):
    """Load localization JSON.  Returns (R, T_m) where T_m is in metres.

    Convention stored in the file: X_shed = R @ X_gate + T_mm/1000
    So to go shed → gate:  X_gate = R.T @ (X_shed - T_m)
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    R   = np.array(d["R_shed_from_gate"], dtype=np.float64)
    T_m = np.array(d["T_shed_from_gate_mm"], dtype=np.float64) / 1000.0
    return R, T_m


def transform_shed_to_gate(df: pd.DataFrame, R: np.ndarray, T_m: np.ndarray) -> pd.DataFrame:
    """Return a copy of df with X/Y/Z and vx/vy/vz rotated into the gate_left frame.

    Positions:   X_gate = R.T @ (X_shed - T_m)
    Velocities:  v_gate = R.T @ v_shed   (rotation only)
    speed_mps is recomputed from the transformed velocities.

    traj_id values are prefixed with "shed_" to avoid collisions with gate data.
    """
    df = df.copy()

    # --- positions ---
    pts = df[["X", "Y", "Z"]].to_numpy(float)          # (N, 3)
    pts_gate = (R.T @ (pts - T_m).T).T
    df["X"] = pts_gate[:, 0]
    df["Y"] = pts_gate[:, 1]
    df["Z"] = pts_gate[:, 2]

    # --- velocities ---
    vel_cols = ["vx", "vy", "vz"]
    if all(c in df.columns for c in vel_cols):
        vels = df[vel_cols].to_numpy(float)              # (N, 3)
        mask = np.isfinite(vels).all(axis=1)
        vels_gate = np.full_like(vels, np.nan)
        vels_gate[mask] = (R.T @ vels[mask].T).T
        df["vx"] = vels_gate[:, 0]
        df["vy"] = vels_gate[:, 1]
        df["vz"] = vels_gate[:, 2]
        df["speed_mps"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)

    # --- traj_id namespace ---
    df["traj_id"] = "shed_" + df["traj_id"].astype(str)

    return df


# ------------------------------------------------------------------ #
#  Velocity computation                                                #
# ------------------------------------------------------------------ #

def compute_velocities_3d(df, fps):
    """
    Add vx, vy, vz, speed_mps columns to a 3D track DataFrame.

    Velocities are computed between consecutive matched (non-interpolated)
    frames only.  Interpolated rows and the first row of each trajectory
    receive NaN.

    Parameters
    ----------
    df  : pd.DataFrame  columns: traj_id, t, X, Y, Z, xL
    fps : float

    Returns
    -------
    df with vx, vy, vz, speed_mps columns added (in-place copy)
    """
    df = df.copy()
    df["vx"] = np.nan
    df["vy"] = np.nan
    df["vz"] = np.nan

    for tid, grp in df.groupby("traj_id"):
        grp = grp.sort_values("t")
        matched = grp["xL"].notna().to_numpy()
        t = grp["t"].to_numpy()
        x = grp["X"].to_numpy()
        y = grp["Y"].to_numpy()
        z = grp["Z"].to_numpy()
        idx = grp.index.to_numpy()

        for k in range(1, len(t)):
            if not (matched[k] and matched[k - 1]):
                continue
            dt = (t[k] - t[k - 1]) / fps
            if dt <= 0:
                continue
            df.loc[idx[k], "vx"] = (x[k] - x[k - 1]) / dt
            df.loc[idx[k], "vy"] = (y[k] - y[k - 1]) / dt
            df.loc[idx[k], "vz"] = (z[k] - z[k - 1]) / dt

    df["speed_mps"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2)
    return df


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _apply_range(df, col, rng):
    if rng is None:
        return df
    lo, hi = float(rng[0]), float(rng[1])
    if lo > hi:
        lo, hi = hi, lo
    return df[(df[col] >= lo) & (df[col] <= hi)]


def _finite_minmax(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(a)), float(np.max(a))
    return (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)


def _set_centered_span(ax, x, y, span):
    xlo, xhi = _finite_minmax(x)
    ylo, yhi = _finite_minmax(y)
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    h = 0.5 * float(span)
    ax.set_xlim(cx - h, cx + h)
    ax.set_ylim(cy - h, cy + h)


def binned_field_2d(x, y, u, v, speed, bins=(40, 40), min_count=5, bin_size_m=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    u = np.asarray(u, float); v = np.asarray(v, float)
    sp = np.asarray(speed, float)

    if bin_size_m is not None:
        bs = float(bin_size_m)
        nx = max(1, math.ceil((np.nanmax(x) - np.nanmin(x)) / bs))
        ny = max(1, math.ceil((np.nanmax(y) - np.nanmin(y)) / bs))
        bins = (nx, ny)

    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins[0] + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), bins[1] + 1)
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, bins[0] - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, bins[1] - 1)

    U_sum = np.zeros((bins[1], bins[0]), float)
    V_sum = np.zeros((bins[1], bins[0]), float)
    S_sum = np.zeros((bins[1], bins[0]), float)
    Un_sum = np.zeros((bins[1], bins[0]), float)
    Vn_sum = np.zeros((bins[1], bins[0]), float)
    C = np.zeros((bins[1], bins[0]), int)

    for k in range(len(x)):
        if np.isfinite(u[k]) and np.isfinite(v[k]) and np.isfinite(sp[k]):
            j, i = iy[k], ix[k]
            U_sum[j, i] += u[k]; V_sum[j, i] += v[k]; S_sum[j, i] += sp[k]
            C[j, i] += 1
            m2d = math.hypot(u[k], v[k])
            if m2d > 1e-12:
                Un_sum[j, i] += u[k] / m2d
                Vn_sum[j, i] += v[k] / m2d

    U = np.full_like(U_sum, np.nan); V = np.full_like(V_sum, np.nan)
    S = np.full_like(S_sum, np.nan); Coh = np.full_like(U_sum, np.nan)

    ok = C >= min_count
    U[ok] = U_sum[ok] / C[ok]; V[ok] = V_sum[ok] / C[ok]; S[ok] = S_sum[ok] / C[ok]
    Coh[ok] = np.sqrt((Un_sum[ok] / C[ok]) ** 2 + (Vn_sum[ok] / C[ok]) ** 2)

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_cent, y_cent)
    return Xc, Yc, U, V, S, C, x_edges, y_edges, Coh


def _binned_field_3d(x, y, z, u, v, w, bins_3d=(10, 10, 10), min_count=5, bin_size_m=None):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    u = np.asarray(u, float); v = np.asarray(v, float); w = np.asarray(w, float)

    if bin_size_m is not None:
        bs = float(bin_size_m)
        nx = max(1, math.ceil((np.nanmax(x) - np.nanmin(x)) / bs))
        ny = max(1, math.ceil((np.nanmax(y) - np.nanmin(y)) / bs))
        nz = max(1, math.ceil((np.nanmax(z) - np.nanmin(z)) / bs))
        bins_3d = (nx, ny, nz)

    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins_3d[0] + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), bins_3d[1] + 1)
    z_edges = np.linspace(np.nanmin(z), np.nanmax(z), bins_3d[2] + 1)
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, bins_3d[0] - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, bins_3d[1] - 1)
    iz = np.clip(np.digitize(z, z_edges) - 1, 0, bins_3d[2] - 1)

    shape = (bins_3d[0], bins_3d[1], bins_3d[2])
    U_sum = np.zeros(shape, float); V_sum = np.zeros(shape, float); W_sum = np.zeros(shape, float)
    C = np.zeros(shape, int)

    for k in range(len(x)):
        if np.isfinite(u[k]) and np.isfinite(v[k]) and np.isfinite(w[k]):
            U_sum[ix[k], iy[k], iz[k]] += u[k]
            V_sum[ix[k], iy[k], iz[k]] += v[k]
            W_sum[ix[k], iy[k], iz[k]] += w[k]
            C[ix[k], iy[k], iz[k]] += 1

    U = np.full(shape, np.nan); V = np.full(shape, np.nan); W = np.full(shape, np.nan)
    ok = C >= min_count
    U[ok] = U_sum[ok] / C[ok]; V[ok] = V_sum[ok] / C[ok]; W[ok] = W_sum[ok] / C[ok]

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    Xc, Yc, Zc = np.meshgrid(x_cent, y_cent, z_cent, indexing="ij")
    return Xc, Yc, Zc, U, V, W, C, x_edges, y_edges, z_edges


def _voxel_project_traj_count(x, y, z, tids, traj_len_map, bin_size_m, projection):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    tids = np.asarray(tids)
    bs = float(bin_size_m)

    x_edges = np.linspace(np.nanmin(x), np.nanmax(x),
                          max(2, math.ceil((np.nanmax(x) - np.nanmin(x)) / bs) + 1))
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y),
                          max(2, math.ceil((np.nanmax(y) - np.nanmin(y)) / bs) + 1))
    z_edges = np.linspace(np.nanmin(z), np.nanmax(z),
                          max(2, math.ceil((np.nanmax(z) - np.nanmin(z)) / bs) + 1))
    nx, ny, nz = len(x_edges) - 1, len(y_edges) - 1, len(z_edges) - 1

    ix = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
    iz = np.clip(np.digitize(z, z_edges) - 1, 0, nz - 1)

    grid = [[[set() for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
    for k in range(len(x)):
        if np.isfinite(x[k]) and np.isfinite(y[k]) and np.isfinite(z[k]):
            grid[int(ix[k])][int(iy[k])][int(iz[k])].add(tids[k])

    if projection == "xy":
        n2d = np.full((ny, nx), np.nan); m2d = np.full((ny, nx), np.nan)
        for i in range(nx):
            for j in range(ny):
                col = set().union(*[grid[i][j][l] for l in range(nz)])
                if col:
                    n2d[j, i] = len(col) / nz
                    m2d[j, i] = np.mean([traj_len_map.get(t, 0) for t in col])
        xe, ye = x_edges, y_edges
    elif projection == "zy":
        n2d = np.full((ny, nz), np.nan); m2d = np.full((ny, nz), np.nan)
        for j in range(ny):
            for l in range(nz):
                col = set().union(*[grid[i][j][l] for i in range(nx)])
                if col:
                    n2d[j, l] = len(col) / nx
                    m2d[j, l] = np.mean([traj_len_map.get(t, 0) for t in col])
        xe, ye = z_edges, y_edges
    else:  # "xz"
        n2d = np.full((nz, nx), np.nan); m2d = np.full((nz, nx), np.nan)
        for i in range(nx):
            for l in range(nz):
                col = set().union(*[grid[i][j][l] for j in range(ny)])
                if col:
                    n2d[l, i] = len(col) / ny
                    m2d[l, i] = np.mean([traj_len_map.get(t, 0) for t in col])
        xe, ye = x_edges, z_edges

    x_cent = 0.5 * (xe[:-1] + xe[1:])
    y_cent = 0.5 * (ye[:-1] + ye[1:])
    return n2d, m2d, xe, ye, x_cent, y_cent


def _filter_df(vel, x_range, y_range, z_range, t_range, t_col="t"):
    vel = _apply_range(vel, "X", x_range)
    vel = _apply_range(vel, "Y", y_range)
    vel = _apply_range(vel, "Z", z_range)
    if t_range is not None:
        vel = _apply_range(vel, t_col, t_range)
    return vel


# ------------------------------------------------------------------ #
#  2D projection panel helper                                          #
# ------------------------------------------------------------------ #

def _binned_speed_minmax(*panel_args_list, bin_size_m, min_count):
    """
    Pre-compute binned speed fields for a list of (x, y, u, v, speed) tuples
    and return (vmin, vmax) from the actual cell values that will be displayed.
    """
    all_s = []
    for x, y, u, v, speed in panel_args_list:
        _, _, _, _, S, *_ = binned_field_2d(x, y, u, v, speed,
                                            min_count=min_count,
                                            bin_size_m=bin_size_m)
        finite = S[np.isfinite(S)]
        if len(finite):
            all_s.append(finite)
    if not all_s:
        return 0.0, 1.0
    combined = np.concatenate(all_s)
    return float(np.nanmin(combined)), float(np.nanmax(combined))


def _draw_proj_panel(ax, x, y, u, v, speed, xlabel, ylabel,
                     bin_size_m, min_count, quiver_stride,
                     arrow_scale, arrow_width, arrow_color,
                     vmin, vmax, colormap, normalize_arrows,
                     ylabel_labelpad=10):
    Xc, Yc, U, V, S, C, x_edges, y_edges, Coh = binned_field_2d(
        x, y, u, v, speed, min_count=min_count, bin_size_m=bin_size_m)

    im = ax.imshow(S, origin="lower",
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   aspect="equal", vmin=vmin, vmax=vmax, cmap=colormap)

    q = max(1, int(quiver_stride))
    if normalize_arrows == "flux":
        ax.quiver(Xc[::q, ::q], Yc[::q, ::q], U[::q, ::q], V[::q, ::q],
                  scale=arrow_scale, width=arrow_width, color=arrow_color)
    elif normalize_arrows == "coherence":
        mag = np.sqrt(U ** 2 + V ** 2); mag[mag < 1e-12] = np.nan
        Ud = U / mag * Coh; Vd = V / mag * Coh
        ax.quiver(Xc[::q, ::q], Yc[::q, ::q], Ud[::q, ::q], Vd[::q, ::q],
                  scale=arrow_scale, width=arrow_width, color=arrow_color)
    elif normalize_arrows:
        mag = np.sqrt(U ** 2 + V ** 2)
        Un = U / (mag + 1e-12); Vn = V / (mag + 1e-12)
        # scale_units="width" makes arrow_scale a direct size multiplier:
        # larger arrow_scale = longer arrows (opposite of default behaviour)
        ax.quiver(Xc[::q, ::q], Yc[::q, ::q], Un[::q, ::q], Vn[::q, ::q],
                  scale=1.0 / arrow_scale, scale_units="width",
                  width=arrow_width, color=arrow_color)
    else:
        ax.quiver(Xc[::q, ::q], Yc[::q, ::q], U[::q, ::q], V[::q, ::q],
                  angles="xy", scale_units="xy",
                  scale=arrow_scale, width=arrow_width, color=arrow_color)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=ylabel_labelpad)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_aspect("equal", adjustable="box")
    return im


def _draw_proj_panel_coherence(ax, x, y, u, v, speed, xlabel, ylabel,
                               bin_size_m, min_count, quiver_stride,
                               arrow_scale, arrow_width, arrow_color,
                               normalize_arrows, colormap,
                               ylabel_labelpad=10):
    """Like _draw_proj_panel but heatmap shows coherence |P| instead of mean speed."""
    Xc, Yc, U, V, S, C, x_edges, y_edges, Coh = binned_field_2d(
        x, y, u, v, speed, min_count=min_count, bin_size_m=bin_size_m)

    im = ax.imshow(Coh, origin="lower",
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   aspect="equal", vmin=0, vmax=1, cmap=colormap)

    q = max(1, int(quiver_stride))
    if normalize_arrows == "flux":
        ax.quiver(Xc[::q, ::q], Yc[::q, ::q], U[::q, ::q], V[::q, ::q],
                  scale=arrow_scale, width=arrow_width, color=arrow_color)
    elif normalize_arrows:
        mag = np.sqrt(U ** 2 + V ** 2)
        Un = U / (mag + 1e-12); Vn = V / (mag + 1e-12)
        ax.quiver(Xc[::q, ::q], Yc[::q, ::q], Un[::q, ::q], Vn[::q, ::q],
                  scale=1.0 / arrow_scale, scale_units="width",
                  width=arrow_width, color=arrow_color)
    else:
        ax.quiver(Xc[::q, ::q], Yc[::q, ::q], U[::q, ::q], V[::q, ::q],
                  angles="xy", scale_units="xy",
                  scale=arrow_scale, width=arrow_width, color=arrow_color)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=ylabel_labelpad)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_aspect("equal", adjustable="box")
    return im


def _overlay_board_circle(
    ax_xy, ax_zy, ax_xz,
    x_min_m, x_max_m, x_center_m, z_center_m,
    y_center_m=0.0,
    rect_height_m=0.02,
    fill_alpha=1.0,
    color="black", linewidth=1.5, linestyle="--",
):
    """
    Overlay the physical board as a dashed circle on the projection panels.

    Board is a circle in the XZ plane (vertical, facing the camera) at y=y_center_m:
      XZ panel : full circle centered at (x_center_m, z_center_m), radius = (x_max-x_min)/2
      XY panel : horizontal dashed line from x_min to x_max at y = y_center_m
      ZY panel : vertical dashed line at z = z_center_m spanning z_center ± r
    """
    r = (x_max_m - x_min_m) / 2.0
    theta = np.linspace(0, 2 * np.pi, 256)

    ax_xz.plot(x_center_m + r * np.cos(theta),
               z_center_m + r * np.sin(theta),
               color=color, linewidth=linewidth, linestyle=linestyle, zorder=6)
    rect_h = rect_height_m / 2.0

    def _draw_rect(ax, x0, x1, yc, h):
        import matplotlib.colors as _mc
        from matplotlib.patches import Rectangle as _Rect
        fc = _mc.to_rgba(color, alpha=fill_alpha)
        # extends upward (y-axis inverted: yc - h is visually above yc)
        ax.add_patch(_Rect((x0, yc - h), x1 - x0, h,
                           facecolor=fc, edgecolor=color,
                           linewidth=linewidth, linestyle=linestyle,
                           zorder=6))

    _draw_rect(ax_xy, x_min_m, x_max_m, y_center_m, rect_h)
    _draw_rect(ax_zy, z_center_m - r, z_center_m + r, y_center_m, rect_h)


def _overlay_boundary(ax_xy, ax_zy, ax_xz, boundary_outline,
                       boundary_axis_center, boundary_radius,
                       boundary_color, boundary_linewidth, boundary_alpha):
    if boundary_outline is None:
        return
    bo = np.asarray(boundary_outline)
    ax_xy.plot(bo[:, 0], bo[:, 1], color=boundary_color,
               linewidth=boundary_linewidth, alpha=boundary_alpha)
    if boundary_axis_center is not None and boundary_radius is not None:
        ac = np.asarray(boundary_axis_center)
        br = np.asarray(boundary_radius)
        max_r = np.max(br)
        cx_mean, cz_mean = np.mean(ac[:, 0]), np.mean(ac[:, 2])
        theta = np.linspace(0, 2 * np.pi, 128)
        ax_xz.plot(cx_mean + max_r * np.cos(theta), cz_mean + max_r * np.sin(theta),
                   color=boundary_color, linewidth=boundary_linewidth, alpha=boundary_alpha)
        mean_x = np.mean(bo[:, 0])
        ax_zy.plot(cz_mean + (bo[:, 0] - mean_x), bo[:, 1],
                   color=boundary_color, linewidth=boundary_linewidth, alpha=boundary_alpha)
    else:
        ax_xz.plot(bo[:, 0], bo[:, 2], color=boundary_color,
                   linewidth=boundary_linewidth, alpha=boundary_alpha)
        ax_zy.plot(bo[:, 2], bo[:, 1], color=boundary_color,
                   linewidth=boundary_linewidth, alpha=boundary_alpha)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def plot_combined_quiver_and_projections(
    df,
    # spatial / time filters
    x_range=None, y_range=None, z_range=None, t_range=None, t_col="t",
    speed_min=None, speed_max=None,
    # 2D projection params
    bin_size_m=0.1,
    min_count=10,
    quiver_stride=1,
    arrow_scale=1.5,
    arrow_width=0.008,
    arrow_color_2d="black",
    vmin=None, vmax=None,
    colormap="plasma",
    normalize_arrows="flux",
    # boundary overlay (optional)
    boundary_outline=None,
    boundary_axis_center=None,
    boundary_radius=None,
    boundary_color="black",
    boundary_linewidth=2,
    boundary_alpha=1.0,
    # 3D quiver params
    bin_size_m_3d=0.1,
    arrow_length=0.05,
    arrow_color_3d="black",
    elev=10, azim=-65,
    proj_face_alpha=0.08,
    show_grid=True,
    # projection colors
    proj_colors=("#e05c5c", "#5ca0e0", "#5cc45c"),
    board_extents=None,
    # figure
    figsize=(26, 6),
    dpi=150,
    save_path=None,
):
    """
    Combined figure: 3D quiver (left) + XY / ZY / XZ velocity projection panels (right).

    Parameters
    ----------
    df         : pd.DataFrame  with X, Y, Z, vx, vy, vz, speed_mps, traj_id, t
    save_path  : str or Path, optional
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    vel = df.dropna(subset=["vx", "vy", "vz", "speed_mps"]).copy()
    vel = _filter_df(vel, x_range, y_range, z_range, t_range, t_col)
    if speed_min is not None:
        vel = vel[vel["speed_mps"] >= speed_min]
    if speed_max is not None:
        vel = vel[vel["speed_mps"] <= speed_max]

    if len(vel) == 0:
        print("  No data after filtering — skipping quiver plot.")
        return None

    _vy_flip = -vel["vy"]
    if vmin is not None and vmax is not None:
        _vmin, _vmax = float(vmin), float(vmax)
    else:
        _vmin, _vmax = _binned_speed_minmax(
            (vel["X"], vel["Y"], vel["vx"], _vy_flip, vel["speed_mps"]),
            (vel["Z"], vel["Y"], vel["vz"], _vy_flip, vel["speed_mps"]),
            (vel["X"], vel["Z"], vel["vx"], vel["vz"],  vel["speed_mps"]),
            bin_size_m=bin_size_m, min_count=min_count,
        )
        if vmin is not None: _vmin = float(vmin)
        if vmax is not None: _vmax = float(vmax)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 2], wspace=0.05)
    gs_inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0, 1], wspace=0.35)

    ax3d = fig.add_subplot(gs_outer[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs_inner[0, 0])
    ax_zy = fig.add_subplot(gs_inner[0, 1])
    ax_xz = fig.add_subplot(gs_inner[0, 2])

    # ---- 3D quiver ----
    # Remap: plot_x=X, plot_y=Z, plot_z=Y (so Z=depth on Y axis, Y=vertical on Z axis)
    Xc, Yc, Zc, U, V, W, C, *_ = _binned_field_3d(
        vel["X"].values, vel["Z"].values, vel["Y"].values,
        vel["vx"].values, vel["vz"].values, vel["vy"].values,
        min_count=min_count, bin_size_m=bin_size_m_3d)

    ok = np.isfinite(U)
    xf, yf, zf = Xc[ok], Yc[ok], Zc[ok]
    uf, vf, wf = U[ok], V[ok], W[ok]
    mag = np.sqrt(uf ** 2 + vf ** 2 + wf ** 2) + 1e-12
    uf, vf, wf = uf / mag * arrow_length, vf / mag * arrow_length, wf / mag * arrow_length

    if len(xf) > 0:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        head_frac = 0.25; head_w = 0.15
        triangles = []; sx, sy, sz = [], [], []
        for i in range(len(xf)):
            d = np.array([uf[i], vf[i], wf[i]])
            d_len = np.linalg.norm(d)
            if d_len < 1e-12:
                continue
            d_hat = d / d_len
            o = np.array([xf[i], yf[i], zf[i]])
            tip = o + d; base = tip - d_hat * (d_len * head_frac)
            ref = np.array([0, 0, 1]) if abs(d_hat[2]) < 0.9 else np.array([1, 0, 0])
            perp = np.cross(d_hat, ref); perp = perp / np.linalg.norm(perp) * (d_len * head_w)
            triangles.append([base + perp, base - perp, tip])
            sx += [o[0], base[0], np.nan]; sy += [o[1], base[1], np.nan]; sz += [o[2], base[2], np.nan]

        ax3d.plot3D(sx, sy, sz, color=arrow_color_3d, linewidth=0.8)
        heads = Poly3DCollection(triangles, zsort="average")
        heads.set_facecolor(arrow_color_3d); heads.set_edgecolor(arrow_color_3d)
        ax3d.add_collection3d(heads)

    # ---- swarm surface (revolve boundary profile around vertical axis) ----
    if boundary_axis_center is not None and boundary_radius is not None:
        ac  = np.asarray(boundary_axis_center)   # (N, 3) metres [X, Y_down, Z_depth]
        rad = np.asarray(boundary_radius)         # (N,)
        theta = np.linspace(0, 2 * np.pi, 36)
        n = len(ac)
        # Remap to 3D plot coords: plot_x=cam_X, plot_y=cam_Z, plot_z=cam_Y
        Xs = ac[:, 0:1] + rad[:, None] * np.cos(theta)[None, :]
        Ys = ac[:, 2:3] + rad[:, None] * np.sin(theta)[None, :]
        Zs = np.tile(ac[:, 1:2], (1, len(theta)))
        # Add degenerate caps to close the surface
        cap_top = np.array([[ac[0, 0],  ac[0, 2],  ac[0, 1]]]  * len(theta))
        cap_bot = np.array([[ac[-1, 0], ac[-1, 2], ac[-1, 1]]] * len(theta))
        Xs = np.vstack([cap_top[:, 0:1].T, Xs, cap_bot[:, 0:1].T])
        Ys = np.vstack([cap_top[:, 1:2].T, Ys, cap_bot[:, 1:2].T])
        Zs = np.vstack([cap_top[:, 2:3].T, Zs, cap_bot[:, 2:3].T])
        ax3d.plot_surface(Xs, Ys, Zs, alpha=0.5, color="steelblue", edgecolor="none")

    # equal aspect
    pts = np.column_stack([vel["X"], vel["Z"], vel["Y"]])
    lo, hi = pts.min(0), pts.max(0)
    mid = 0.5 * (lo + hi); half = 0.5 * max(hi - lo)
    ax3d.set_xlim(mid[0]-half, mid[0]+half)
    ax3d.set_ylim(mid[1]-half, mid[1]+half)
    ax3d.set_zlim(mid[2]+half, mid[2]-half)   # inverted Z: large Y at bottom
    ax3d.set_xlabel("X (m)", fontsize=16, labelpad=10)
    ax3d.set_ylabel("Z (m)", fontsize=16, labelpad=10)
    ax3d.set_zlabel("Y (m)", fontsize=16, labelpad=22)
    ax3d.zaxis.set_tick_params(pad=8)
    ax3d.tick_params(axis="both", labelsize=14)
    ax3d.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3d.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3d.zaxis.set_major_locator(plt.MaxNLocator(3))
    ax3d.view_init(elev=elev, azim=azim)
    if not show_grid:
        ax3d.grid(False); ax3d.set_axis_off()

    if proj_colors is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection as P3D
        c_xy, c_xz, c_zy = proj_colors
        x0, x1 = mid[0]-half, mid[0]+half
        y0, y1 = mid[1]-half, mid[1]+half
        z0, z1 = mid[2]-half, mid[2]+half
        def _face3d(verts, color):
            poly = P3D([verts], alpha=proj_face_alpha)
            poly.set_facecolor(color); poly.set_edgecolor("none")
            ax3d.add_collection3d(poly)
            xs = [v[0] for v in verts] + [verts[0][0]]
            ys = [v[1] for v in verts] + [verts[0][1]]
            zs = [v[2] for v in verts] + [verts[0][2]]
            ax3d.plot3D(xs, ys, zs, color=color, linewidth=2)
        _face3d([(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)], c_xy)
        _face3d([(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)], c_xz)
        _face3d([(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)], c_zy)

    # ---- 2D projection panels ----
    im1 = _draw_proj_panel(ax_xy, vel["X"], vel["Y"], vel["vx"], _vy_flip,
                           vel["speed_mps"], "X (m)", "Y (m)",
                           bin_size_m, min_count, quiver_stride,
                           arrow_scale, arrow_width, arrow_color_2d,
                           _vmin, _vmax, colormap, normalize_arrows)
    _draw_proj_panel(ax_zy, vel["Z"], vel["Y"], vel["vz"], _vy_flip,
                     vel["speed_mps"], "Z (m)", "Y (m)",
                     bin_size_m, min_count, quiver_stride,
                     arrow_scale, arrow_width, arrow_color_2d,
                     _vmin, _vmax, colormap, normalize_arrows)
    im3 = _draw_proj_panel(ax_xz, vel["X"], vel["Z"], vel["vx"], vel["vz"],
                           vel["speed_mps"], "X (m)", "Z (m)",
                           bin_size_m, min_count, quiver_stride,
                           arrow_scale, arrow_width, arrow_color_2d,
                           _vmin, _vmax, colormap, normalize_arrows,
                           ylabel_labelpad=2)

    xlo, xhi = _finite_minmax(vel["X"]); ylo, yhi = _finite_minmax(vel["Y"])
    zlo, zhi = _finite_minmax(vel["Z"])
    span = max(xhi-xlo, yhi-ylo, zhi-zlo)
    _set_centered_span(ax_xy, vel["X"], vel["Y"], span)
    _set_centered_span(ax_zy, vel["Z"], vel["Y"], span)
    _set_centered_span(ax_xz, vel["X"], vel["Z"], span)
    ax_xy.set_box_aspect(1); ax_zy.set_box_aspect(1); ax_xz.set_box_aspect(1)
    ax_xy.invert_yaxis(); ax_zy.invert_yaxis()

    _overlay_boundary(ax_xy, ax_zy, ax_xz, boundary_outline, boundary_axis_center,
                      boundary_radius, boundary_color, boundary_linewidth, boundary_alpha)
    if board_extents:
        _overlay_board_circle(ax_xy, ax_zy, ax_xz, **board_extents)

    if proj_colors is not None:
        c_xy, c_xz, c_zy = proj_colors
        for spine in ax_xy.spines.values(): spine.set_edgecolor(c_xy); spine.set_linewidth(3)
        for spine in ax_zy.spines.values(): spine.set_edgecolor(c_zy); spine.set_linewidth(3)
        for spine in ax_xz.spines.values(): spine.set_edgecolor(c_xz); spine.set_linewidth(3)

    cbar = fig.colorbar(im3, ax=[ax_xy, ax_zy, ax_xz], fraction=0.015, pad=0.02, shrink=0.8)
    cbar.set_label("Speed (m/s)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved quiver plot: {save_path}")
    plt.close()
    return fig


def plot_traj_count_3proj(
    df,
    metric="unique",          # "unique" = n trajectories per bin; "density" = mean traj length
    x_range=None, y_range=None, z_range=None, t_range=None, t_col="t",
    bin_size_m=0.1,
    colormap="YlGnBu",
    vmax=None,
    boundary_outline=None,
    boundary_axis_center=None,
    boundary_radius=None,
    boundary_color="black",
    boundary_linewidth=2,
    boundary_alpha=1.0,
    proj_colors=("#e05c5c", "#5ca0e0", "#5cc45c"),
    board_extents=None,
    figsize=(26, 6),
    dpi=150,
    save_path=None,
):
    """
    3-panel (XY / ZY / XZ) heatmap of unique trajectory count or mean length per bin.

    Parameters
    ----------
    df      : pd.DataFrame  with X, Y, Z, traj_id
    metric  : "unique" → number of distinct trajectories per bin
              "density" → mean total trajectory length for trajectories passing through
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    vel = df.copy()
    vel = _filter_df(vel, x_range, y_range, z_range, t_range, t_col)
    if len(vel) == 0:
        print("  No data after filtering — skipping traj count plot.")
        return None

    traj_len_map = df.groupby("traj_id").size().to_dict()
    tids = vel["traj_id"].values
    x3, y3, z3 = vel["X"].to_numpy(float), vel["Y"].to_numpy(float), vel["Z"].to_numpy(float)

    precomp_xy = _voxel_project_traj_count(x3, y3, z3, tids, traj_len_map, bin_size_m, "xy")
    precomp_zy = _voxel_project_traj_count(x3, y3, z3, tids, traj_len_map, bin_size_m, "zy")
    precomp_xz = _voxel_project_traj_count(x3, y3, z3, tids, traj_len_map, bin_size_m, "xz")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 2], wspace=0.05)
    gs_inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0, 1], wspace=0.35)
    ax_xy = fig.add_subplot(gs_inner[0, 0])
    ax_zy = fig.add_subplot(gs_inner[0, 1])
    ax_xz = fig.add_subplot(gs_inner[0, 2])

    cbar_label = "mean trajectory length (frames)" if metric == "density" else "# trajectories"

    def _draw(ax, precomp, xlabel, ylabel):
        n_unique, mean_len, xe, ye, _, _ = precomp
        grid = mean_len if metric == "density" else n_unique
        vm = vmax if vmax is not None else np.nanmax(grid)
        im = ax.imshow(grid, origin="lower",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       aspect="equal", vmin=0, vmax=vm, cmap=colormap)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        return im

    im1 = _draw(ax_xy, precomp_xy, "X (m)", "Y (m)")
    _draw(ax_zy, precomp_zy, "Z (m)", "Y (m)")
    im3 = _draw(ax_xz, precomp_xz, "X (m)", "Z (m)")

    xlo, xhi = _finite_minmax(x3); ylo, yhi = _finite_minmax(y3); zlo, zhi = _finite_minmax(z3)
    span = max(xhi-xlo, yhi-ylo, zhi-zlo)
    _set_centered_span(ax_xy, x3, y3, span)
    _set_centered_span(ax_zy, z3, y3, span)
    _set_centered_span(ax_xz, x3, z3, span)
    if x_range: ax_xy.set_xlim(x_range); ax_xz.set_xlim(x_range)
    if y_range: ax_xy.set_ylim(y_range); ax_zy.set_ylim(y_range)
    if z_range: ax_zy.set_xlim(z_range); ax_xz.set_ylim(z_range)
    ax_xy.invert_yaxis(); ax_zy.invert_yaxis()

    _overlay_boundary(ax_xy, ax_zy, ax_xz, boundary_outline, boundary_axis_center,
                      boundary_radius, boundary_color, boundary_linewidth, boundary_alpha)
    if board_extents:
        _overlay_board_circle(ax_xy, ax_zy, ax_xz, **board_extents)

    if proj_colors is not None:
        c_xy, c_xz, c_zy = proj_colors
        for spine in ax_xy.spines.values(): spine.set_edgecolor(c_xy); spine.set_linewidth(3)
        for spine in ax_zy.spines.values(): spine.set_edgecolor(c_zy); spine.set_linewidth(3)
        for spine in ax_xz.spines.values(): spine.set_edgecolor(c_xz); spine.set_linewidth(3)

    cbar = fig.colorbar(im3, ax=[ax_xy, ax_zy, ax_xz], fraction=0.015, pad=0.02, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved traj count plot ({metric}): {save_path}")
    plt.close()
    return fig


def plot_depth_sliced_quiver(
    df,
    # per-projection depth filter ranges (the "thin" axis for each panel)
    xy_z_range=None,   # XY projection: filter by Z to this range
    xz_y_range=None,   # XZ projection: filter by Y to this range
    zy_x_range=None,   # ZY projection: filter by X to this range
    # display ranges (same for all panels, matching regular quiver config)
    x_range=None, y_range=None, z_range=None,
    t_range=None, t_col="t",
    speed_min=None, speed_max=None,
    # appearance (same params as plot_combined_quiver_and_projections)
    bin_size_m=0.1,
    min_count=10,
    quiver_stride=1,
    arrow_scale=1.5,
    arrow_width=0.008,
    arrow_color_2d="black",
    vmin=None, vmax=None,
    colormap="plasma",
    normalize_arrows="flux",
    boundary_outline=None,
    boundary_axis_center=None,
    boundary_radius=None,
    boundary_color="black",
    boundary_linewidth=2,
    boundary_alpha=1.0,
    bin_size_m_3d=0.1,
    arrow_length=0.05,
    arrow_color_3d="black",
    elev=10, azim=-65,
    proj_face_alpha=0.08,
    show_grid=True,
    proj_colors=("#e05c5c", "#5ca0e0", "#5cc45c"),
    figsize=(26, 6),
    dpi=150,
    save_path=None,
):
    """
    Same 3D + 3-projection layout as plot_combined_quiver_and_projections, but each
    2D projection panel uses a different depth filter:
      - XY panel  → data filtered to xy_z_range  (narrow Z slice)
      - XZ panel  → data filtered to xz_y_range  (narrow Y slice)
      - ZY panel  → data filtered to zy_x_range  (narrow X slice)
    The display extents are set by x_range / y_range / z_range (same as regular config).
    The 3D panel uses all three depth filters simultaneously.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    def _prep(extra_x=None, extra_y=None, extra_z=None):
        """Filter df and apply speed filter, combining display range + depth range."""
        sub = df.dropna(subset=["vx", "vy", "vz", "speed_mps"]).copy()
        # display / spatial range
        sub = _filter_df(sub, x_range, y_range, z_range, t_range, t_col)
        # per-panel depth override
        if extra_x: sub = _apply_range(sub, "X", extra_x)
        if extra_y: sub = _apply_range(sub, "Y", extra_y)
        if extra_z: sub = _apply_range(sub, "Z", extra_z)
        if speed_min is not None: sub = sub[sub["speed_mps"] >= speed_min]
        if speed_max is not None: sub = sub[sub["speed_mps"] <= speed_max]
        return sub

    vel_xy = _prep(extra_z=xy_z_range)
    vel_xz = _prep(extra_y=xz_y_range)
    vel_zy = _prep(extra_x=zy_x_range)
    vel_3d = _prep()  # 3D panel: full display range (no depth slice)

    if all(len(v) == 0 for v in [vel_xy, vel_xz, vel_zy]):
        print("  No data after filtering — skipping depth-sliced quiver plot.")
        return None

    # use full display data for vmax and axis extents
    vel_all = _prep()
    if vmin is not None and vmax is not None:
        _vmin, _vmax = float(vmin), float(vmax)
    else:
        panels = []
        if len(vel_xy) > 0:
            panels.append((vel_xy["X"], vel_xy["Y"], vel_xy["vx"], -vel_xy["vy"], vel_xy["speed_mps"]))
        if len(vel_zy) > 0:
            panels.append((vel_zy["Z"], vel_zy["Y"], vel_zy["vz"], -vel_zy["vy"], vel_zy["speed_mps"]))
        if len(vel_xz) > 0:
            panels.append((vel_xz["X"], vel_xz["Z"], vel_xz["vx"], vel_xz["vz"],  vel_xz["speed_mps"]))
        _vmin, _vmax = _binned_speed_minmax(*panels, bin_size_m=bin_size_m, min_count=min_count) if panels else (0.0, 1.0)
        if vmin is not None: _vmin = float(vmin)
        if vmax is not None: _vmax = float(vmax)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 2], wspace=0.05)
    gs_inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0, 1], wspace=0.35)

    ax3d = fig.add_subplot(gs_outer[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs_inner[0, 0])
    ax_zy = fig.add_subplot(gs_inner[0, 1])
    ax_xz = fig.add_subplot(gs_inner[0, 2])

    # ---- 3D quiver (intersection of all depth slices) ----
    if len(vel_3d) > 0:
        Xc, Yc, Zc, U, V, W, C, *_ = _binned_field_3d(
            vel_3d["X"].values, vel_3d["Z"].values, vel_3d["Y"].values,
            vel_3d["vx"].values, vel_3d["vz"].values, vel_3d["vy"].values,
            min_count=min_count, bin_size_m=bin_size_m_3d)
        ok = np.isfinite(U)
        xf, yf, zf = Xc[ok], Yc[ok], Zc[ok]
        uf, vf, wf = U[ok], V[ok], W[ok]
        mag = np.sqrt(uf**2 + vf**2 + wf**2) + 1e-12
        uf, vf, wf = uf/mag*arrow_length, vf/mag*arrow_length, wf/mag*arrow_length
        if len(xf) > 0:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            head_frac = 0.25; head_w = 0.15
            triangles = []; sx, sy, sz = [], [], []
            for i in range(len(xf)):
                d = np.array([uf[i], vf[i], wf[i]])
                d_len = np.linalg.norm(d)
                if d_len < 1e-12: continue
                d_hat = d / d_len
                o = np.array([xf[i], yf[i], zf[i]])
                tip = o + d; base = tip - d_hat*(d_len*head_frac)
                ref = np.array([0,0,1]) if abs(d_hat[2]) < 0.9 else np.array([1,0,0])
                perp = np.cross(d_hat, ref); perp = perp/np.linalg.norm(perp)*(d_len*head_w)
                triangles.append([base+perp, base-perp, tip])
                sx += [o[0], base[0], np.nan]; sy += [o[1], base[1], np.nan]; sz += [o[2], base[2], np.nan]
            ax3d.plot3D(sx, sy, sz, color=arrow_color_3d, linewidth=0.8)
            heads = Poly3DCollection(triangles, zsort="average")
            heads.set_facecolor(arrow_color_3d); heads.set_edgecolor(arrow_color_3d)
            ax3d.add_collection3d(heads)

    # 3D axis limits from full display data
    ref = vel_all if len(vel_all) > 0 else vel_xy if len(vel_xy) > 0 else vel_xz
    if len(ref) > 0:
        pts = np.column_stack([ref["X"], ref["Z"], ref["Y"]])
        lo, hi = pts.min(0), pts.max(0)
        mid = 0.5*(lo+hi); half = 0.5*max(hi-lo)
        ax3d.set_xlim(mid[0]-half, mid[0]+half)
        ax3d.set_ylim(mid[1]-half, mid[1]+half)
        ax3d.set_zlim(mid[2]+half, mid[2]-half)

    ax3d.set_xlabel("X (m)", fontsize=16, labelpad=10)
    ax3d.set_ylabel("Z (m)", fontsize=16, labelpad=10)
    ax3d.set_zlabel("Y (m)", fontsize=16, labelpad=22)
    ax3d.zaxis.set_tick_params(pad=8)
    ax3d.tick_params(axis="both", labelsize=14)
    ax3d.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3d.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3d.zaxis.set_major_locator(plt.MaxNLocator(3))
    ax3d.view_init(elev=elev, azim=azim)
    if not show_grid:
        ax3d.grid(False); ax3d.set_axis_off()

    # ---- 2D projection panels — each with its own filtered data ----
    def _panel(ax, vel, cx, cy, cu, cv, xlabel, ylabel, ylabel_labelpad=10):
        if len(vel) == 0:
            ax.set_xlabel(xlabel, fontsize=16); ax.set_ylabel(ylabel, fontsize=16)
            return None
        _vy_flip = -vel["vy"] if cy == "Y" else vel[cv.replace("-", "")]
        u = vel[cu]; v = (-vel["vy"] if cu == "vx" and cy == "Y" else
                          -vel["vy"] if cu == "vz" and cy == "Y" else vel[cv])
        return _draw_proj_panel(ax, vel[cx], vel[cy], vel[cu],
                                (-vel["vy"] if cy == "Y" else vel[cv]),
                                vel["speed_mps"], xlabel, ylabel,
                                bin_size_m, min_count, quiver_stride,
                                arrow_scale, arrow_width, arrow_color_2d,
                                _vmin, _vmax, colormap, normalize_arrows,
                                ylabel_labelpad)

    im1 = _draw_proj_panel(ax_xy, vel_xy["X"], vel_xy["Y"], vel_xy["vx"], -vel_xy["vy"],
                           vel_xy["speed_mps"], "X (m)", "Y (m)",
                           bin_size_m, min_count, quiver_stride, arrow_scale, arrow_width,
                           arrow_color_2d, _vmin, _vmax, colormap, normalize_arrows) if len(vel_xy) > 0 else None
    _draw_proj_panel(ax_zy, vel_zy["Z"], vel_zy["Y"], vel_zy["vz"], -vel_zy["vy"],
                     vel_zy["speed_mps"], "Z (m)", "Y (m)",
                     bin_size_m, min_count, quiver_stride, arrow_scale, arrow_width,
                     arrow_color_2d, _vmin, _vmax, colormap, normalize_arrows) if len(vel_zy) > 0 else None
    im3 = _draw_proj_panel(ax_xz, vel_xz["X"], vel_xz["Z"], vel_xz["vx"], vel_xz["vz"],
                           vel_xz["speed_mps"], "X (m)", "Z (m)",
                           bin_size_m, min_count, quiver_stride, arrow_scale, arrow_width,
                           arrow_color_2d, _vmin, _vmax, colormap, normalize_arrows,
                           ylabel_labelpad=2) if len(vel_xz) > 0 else None

    # set display limits from x_range / y_range / z_range, enforcing equal (square) panels
    def _get_lim(rng, col):
        if rng:
            return float(rng[0]), float(rng[1])
        return _finite_minmax(ref[col]) if len(ref) > 0 else (0.0, 1.0)

    def _set_square_lim(ax, xlo, xhi, ylo, yhi):
        cx, cy = 0.5*(xlo+xhi), 0.5*(ylo+yhi)
        half = 0.5 * max(xhi-xlo, yhi-ylo)
        ax.set_xlim(cx-half, cx+half)
        ax.set_ylim(cy-half, cy+half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)

    xlo, xhi = _get_lim(x_range, "X")
    ylo, yhi = _get_lim(y_range, "Y")
    zlo, zhi = _get_lim(z_range, "Z")

    _set_square_lim(ax_xy, xlo, xhi, ylo, yhi)
    _set_square_lim(ax_zy, zlo, zhi, ylo, yhi)
    _set_square_lim(ax_xz, xlo, xhi, zlo, zhi)
    ax_xy.invert_yaxis(); ax_zy.invert_yaxis()

    _overlay_boundary(ax_xy, ax_zy, ax_xz, boundary_outline, boundary_axis_center,
                      boundary_radius, boundary_color, boundary_linewidth, boundary_alpha)

    if proj_colors is not None:
        c_xy, c_xz, c_zy = proj_colors
        for spine in ax_xy.spines.values(): spine.set_edgecolor(c_xy); spine.set_linewidth(3)
        for spine in ax_zy.spines.values(): spine.set_edgecolor(c_zy); spine.set_linewidth(3)
        for spine in ax_xz.spines.values(): spine.set_edgecolor(c_xz); spine.set_linewidth(3)

    ref_im = im3 or im1
    if ref_im is not None:
        cbar = fig.colorbar(ref_im, ax=[ax_xy, ax_zy, ax_xz], fraction=0.015, pad=0.02, shrink=0.8)
        cbar.set_label("Speed (m/s)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved depth-sliced quiver plot: {save_path}")
    plt.close()
    return fig


def plot_traj_count_histogram(
    df,
    x_range=None, y_range=None, z_range=None, t_range=None, t_col="t",
    bin_size_m=0.1,
    bins=10,
    color="#5ca0e0",
    figsize=(6, 4),
    dpi=150,
    save_path=None,
):
    """
    Histogram of unique trajectory counts per 3D voxel (non-empty voxels only).
    """
    vel = df.copy()
    vel = _filter_df(vel, x_range, y_range, z_range, t_range, t_col)
    if len(vel) == 0:
        print("  No data after filtering — skipping traj count histogram.")
        return None

    x = vel["X"].to_numpy(float)
    y = vel["Y"].to_numpy(float)
    z = vel["Z"].to_numpy(float)
    tids = vel["traj_id"].values
    bs = float(bin_size_m)

    x_edges = np.linspace(np.nanmin(x), np.nanmax(x),
                          max(2, math.ceil((np.nanmax(x) - np.nanmin(x)) / bs) + 1))
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y),
                          max(2, math.ceil((np.nanmax(y) - np.nanmin(y)) / bs) + 1))
    z_edges = np.linspace(np.nanmin(z), np.nanmax(z),
                          max(2, math.ceil((np.nanmax(z) - np.nanmin(z)) / bs) + 1))
    nx, ny, nz = len(x_edges) - 1, len(y_edges) - 1, len(z_edges) - 1

    ix = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
    iz = np.clip(np.digitize(z, z_edges) - 1, 0, nz - 1)

    grid = [[[set() for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
    for k in range(len(x)):
        if np.isfinite(x[k]) and np.isfinite(y[k]) and np.isfinite(z[k]):
            grid[int(ix[k])][int(iy[k])][int(iz[k])].add(tids[k])

    counts = [len(grid[i][j][l])
              for i in range(nx) for j in range(ny) for l in range(nz)
              if grid[i][j][l]]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(counts, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    mean_val = np.mean(counts)
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2,
               label=f"mean = {mean_val:.1f}")
    ax.set_xlabel("unique trajectories per voxel", fontsize=13)
    ax.set_ylabel("number of voxels", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved traj count histogram: {save_path}")
    plt.close()
    return fig


def plot_coherence_quiver(
    df,
    x_range=None, y_range=None, z_range=None, t_range=None, t_col="t",
    speed_min=None, speed_max=None,
    bin_size_m=0.1,
    min_count=10,
    quiver_stride=1,
    arrow_scale=1.5,
    arrow_width=0.008,
    arrow_color_2d="black",
    colormap="plasma",
    normalize_arrows="flux",
    boundary_outline=None,
    boundary_axis_center=None,
    boundary_radius=None,
    boundary_color="black",
    boundary_linewidth=2,
    boundary_alpha=1.0,
    proj_colors=("#e05c5c", "#5ca0e0", "#5cc45c"),
    board_extents=None,
    figsize=(26, 6),
    dpi=150,
    save_path=None,
    # unused kwargs accepted for compatibility
    **_kwargs,
):
    """
    3-panel (XY / ZY / XZ) figure: heatmap colored by coherence |P| ∈ [0,1]
    with the same velocity arrows as the regular quiver plot overlaid.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    vel = df.dropna(subset=["vx", "vy", "vz", "speed_mps"]).copy()
    vel = _filter_df(vel, x_range, y_range, z_range, t_range, t_col)
    if speed_min is not None:
        vel = vel[vel["speed_mps"] >= speed_min]
    if speed_max is not None:
        vel = vel[vel["speed_mps"] <= speed_max]
    if len(vel) == 0:
        print("  No data after filtering — skipping coherence quiver plot.")
        return None

    _vy_flip = -vel["vy"]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 2], wspace=0.05)
    gs_inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0, 1], wspace=0.35)
    ax_3d = fig.add_subplot(gs_outer[0, 0])   # placeholder — leave blank, match layout
    ax_3d.set_visible(False)
    ax_xy = fig.add_subplot(gs_inner[0, 0])
    ax_zy = fig.add_subplot(gs_inner[0, 1])
    ax_xz = fig.add_subplot(gs_inner[0, 2])

    common = dict(bin_size_m=bin_size_m, min_count=min_count,
                  quiver_stride=quiver_stride, arrow_scale=arrow_scale,
                  arrow_width=arrow_width, arrow_color=arrow_color_2d,
                  normalize_arrows=normalize_arrows, colormap=colormap)

    im1 = _draw_proj_panel_coherence(ax_xy, vel["X"], vel["Y"],  vel["vx"], _vy_flip,  vel["speed_mps"], "X (m)", "Y (m)",  **common)
    _draw_proj_panel_coherence(      ax_zy, vel["Z"], vel["Y"],  vel["vz"], _vy_flip,  vel["speed_mps"], "Z (m)", "Y (m)",  **common, ylabel_labelpad=10)
    im3 = _draw_proj_panel_coherence(ax_xz, vel["X"], vel["Z"],  vel["vx"], vel["vz"], vel["speed_mps"], "X (m)", "Z (m)",  **common)

    x3, y3, z3 = vel["X"].to_numpy(float), vel["Y"].to_numpy(float), vel["Z"].to_numpy(float)
    span = max(*[hi - lo for lo, hi in [_finite_minmax(x3), _finite_minmax(y3), _finite_minmax(z3)]])
    _set_centered_span(ax_xy, x3, y3, span)
    _set_centered_span(ax_zy, z3, y3, span)
    _set_centered_span(ax_xz, x3, z3, span)
    ax_xy.invert_yaxis(); ax_zy.invert_yaxis()

    _overlay_boundary(ax_xy, ax_zy, ax_xz, boundary_outline, boundary_axis_center,
                      boundary_radius, boundary_color, boundary_linewidth, boundary_alpha)
    if board_extents:
        _overlay_board_circle(ax_xy, ax_zy, ax_xz, **board_extents)

    if proj_colors is not None:
        c_xy, c_xz, c_zy = proj_colors
        for spine in ax_xy.spines.values(): spine.set_edgecolor(c_xy); spine.set_linewidth(3)
        for spine in ax_zy.spines.values(): spine.set_edgecolor(c_zy); spine.set_linewidth(3)
        for spine in ax_xz.spines.values(): spine.set_edgecolor(c_xz); spine.set_linewidth(3)

    cbar = fig.colorbar(im3, ax=[ax_xy, ax_zy, ax_xz], fraction=0.015, pad=0.02, shrink=0.8)
    cbar.set_label(r"Coherence $|\mathbf{P}|$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved coherence quiver plot: {save_path}")
    plt.close()
    return fig


def plot_density_coherence(
    df,
    x_range=None, y_range=None, z_range=None, t_range=None, t_col="t",
    speed_min=None, speed_max=None,
    bin_size_m=0.1,
    min_count=10,
    quiver_stride=1,
    arrow_scale=1.0,
    arrow_width=0.008,
    arrow_color="black",
    colormap="YlGnBu",
    vmin=None, vmax=None,
    boundary_outline=None,
    boundary_axis_center=None,
    boundary_radius=None,
    boundary_color="black",
    boundary_linewidth=2,
    boundary_alpha=1.0,
    proj_colors=("#e05c5c", "#5ca0e0", "#5cc45c"),
    board_extents=None,
    figsize=(26, 6),
    dpi=150,
    save_path=None,
):
    """
    3-panel (XY / ZY / XZ) trajectory density heatmap with coherence arrows overlaid.

    Arrow direction: mean in-plane velocity direction (unit vector sum / N).
    Arrow length:    |P| = |mean unit velocity| ∈ [0, 1]  — NOT speed-projected.
                     Arrows are unit-normalised then rescaled by coherence, so length
                     encodes collective alignment rather than mean speed.

    Parameters
    ----------
    arrow_scale : multiplier applied to |P| to set arrow size in data units.
                  A value of ~bin_size_m works well so a fully coherent bin
                  (|P|=1) spans roughly one bin width.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    vel = df.copy()
    vel = _filter_df(vel, x_range, y_range, z_range, t_range, t_col)
    if speed_min is not None:
        spd = np.sqrt(vel["vx"]**2 + vel["vy"]**2 + vel["vz"]**2)
        vel = vel[spd >= speed_min]
    if speed_max is not None:
        spd = np.sqrt(vel["vx"]**2 + vel["vy"]**2 + vel["vz"]**2)
        vel = vel[spd <= speed_max]
    if len(vel) == 0:
        print("  No data after filtering — skipping density coherence plot.")
        return None

    x3 = vel["X"].to_numpy(float); y3 = vel["Y"].to_numpy(float); z3 = vel["Z"].to_numpy(float)
    vx = vel["vx"].to_numpy(float); vy = vel["vy"].to_numpy(float); vz = vel["vz"].to_numpy(float)
    spd3 = np.sqrt(vx**2 + vy**2 + vz**2)

    # Trajectory density (unique count) for the heatmap background
    traj_len_map = df.groupby("traj_id").size().to_dict()
    tids = vel["traj_id"].values
    precomp_xy = _voxel_project_traj_count(x3, y3, z3, tids, traj_len_map, bin_size_m, "xy")
    precomp_zy = _voxel_project_traj_count(x3, y3, z3, tids, traj_len_map, bin_size_m, "zy")
    precomp_xz = _voxel_project_traj_count(x3, y3, z3, tids, traj_len_map, bin_size_m, "xz")

    # Coherence binning for each projection
    # binned_field_2d returns: Xc, Yc, U, V, S, C, x_edges, y_edges, Coh
    bin_xy = binned_field_2d(x3, y3, vx, vy, spd3, min_count=min_count, bin_size_m=bin_size_m)
    bin_zy = binned_field_2d(z3, y3, vz, vy, spd3, min_count=min_count, bin_size_m=bin_size_m)
    bin_xz = binned_field_2d(x3, z3, vx, vz, spd3, min_count=min_count, bin_size_m=bin_size_m)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 2], wspace=0.05)
    gs_inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0, 1], wspace=0.35)
    ax_xy = fig.add_subplot(gs_inner[0, 0])
    ax_zy = fig.add_subplot(gs_inner[0, 1])
    ax_xz = fig.add_subplot(gs_inner[0, 2])

    def _draw(ax, precomp, bin_data, xlabel, ylabel):
        n_unique, mean_len, xe, ye, _, _ = precomp
        vm = vmax if vmax is not None else np.nanmax(n_unique)
        im = ax.imshow(n_unique, origin="lower",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       aspect="equal", vmin=0, vmax=vm, cmap=colormap)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)

        # Coherence arrows: direction = mean unit velocity, length = |P|
        Xc, Yc, U, V, S, C, _, _, Coh = bin_data
        mag = np.sqrt(U ** 2 + V ** 2)
        with np.errstate(invalid="ignore"):
            Un = np.where(mag > 1e-12, U / mag, np.nan)
            Vn = np.where(mag > 1e-12, V / mag, np.nan)
        Ua = Un * Coh * arrow_scale
        Va = Vn * Coh * arrow_scale
        q = max(1, int(quiver_stride))
        valid = np.isfinite(Ua) & np.isfinite(Va)
        ax.quiver(
            Xc[::q, ::q][valid[::q, ::q]],
            Yc[::q, ::q][valid[::q, ::q]],
            Ua[::q, ::q][valid[::q, ::q]],
            Va[::q, ::q][valid[::q, ::q]],
            angles="xy", scale_units="xy", scale=1.0,
            width=arrow_width, color=arrow_color,
        )
        return im

    im1 = _draw(ax_xy, precomp_xy, bin_xy, "X (m)", "Y (m)")
    _draw(ax_zy, precomp_zy, bin_zy, "Z (m)", "Y (m)")
    im3 = _draw(ax_xz, precomp_xz, bin_xz, "X (m)", "Z (m)")

    xlo, xhi = _finite_minmax(x3); ylo, yhi = _finite_minmax(y3); zlo, zhi = _finite_minmax(z3)
    span = max(xhi - xlo, yhi - ylo, zhi - zlo)
    _set_centered_span(ax_xy, x3, y3, span)
    _set_centered_span(ax_zy, z3, y3, span)
    _set_centered_span(ax_xz, x3, z3, span)
    ax_xy.invert_yaxis(); ax_zy.invert_yaxis()

    _overlay_boundary(ax_xy, ax_zy, ax_xz, boundary_outline, boundary_axis_center,
                      boundary_radius, boundary_color, boundary_linewidth, boundary_alpha)

    if board_extents:
        _overlay_board_circle(ax_xy, ax_zy, ax_xz, **board_extents)

    if proj_colors is not None:
        c_xy, c_xz, c_zy = proj_colors
        for spine in ax_xy.spines.values(): spine.set_edgecolor(c_xy); spine.set_linewidth(3)
        for spine in ax_zy.spines.values(): spine.set_edgecolor(c_zy); spine.set_linewidth(3)
        for spine in ax_xz.spines.values(): spine.set_edgecolor(c_xz); spine.set_linewidth(3)

    cbar = fig.colorbar(im3, ax=[ax_xy, ax_zy, ax_xz], fraction=0.015, pad=0.02, shrink=0.8)
    cbar.set_label("# trajectories", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved density+coherence plot: {save_path}")
    plt.close()
    return fig

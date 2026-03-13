"""
Publication-quality figure generation for swarm morphology time series.

Output filenames:
  vol_mass_norm           — Normalized volume V/V₀ + mass M/M₀
  diam_len_norm           — Normalized diameter D/D₀ + length L/L₀
  bees_mass               — Flying bee count + mass (twin axis)
  scatter_combined        — D/D₀ vs L/L₀ scatter, disassembly + assembly
  mask_overlays/0, 1, ... — Swarm silhouette outlines at overlay times
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Line colors per physical quantity
COL_VOL    = "#1f77b4"
COL_DIAM   = "#2CA9B9"
COL_LEN    = "#1B3B6F"
COL_WEIGHT = "#D62728"
COL_BEES   = "#9467BD"

# Journal figure widths / heights (inches) — match original script exactly
FIG_WIDTH_SINGLE  = 3.5
FIG_WIDTH_1P5     = 5.0
FIG_WIDTH_DOUBLE  = 6.5
FIG_HEIGHT_SINGLE  = FIG_WIDTH_SINGLE * 0.55
FIG_HEIGHT_SCATTER = FIG_WIDTH_SINGLE * 0.35


# ---------------------------------------------------------------------------
# Global plot state (set per session in generate_all_figures)
# ---------------------------------------------------------------------------
_t_dis_end_min:   float | None = None
_t_asm_start_min: float | None = None


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          7,
        "axes.titlesize":     7,
        "axes.labelsize":     7,
        "xtick.labelsize":    6,
        "ytick.labelsize":    6,
        "legend.fontsize":    6,
        "legend.frameon":     False,
        "lines.linewidth":    1.2,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.edgecolor":     "black",
        "axes.grid":          False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _black_box(ax) -> None:
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("black")
        sp.set_linewidth(0.8)
    ax.grid(False)


def _phase_vlines(ax, lw: float = 0.6, color: str = "0.5", ls: str = "-") -> None:
    """Three gray vertical lines: dis start (t=0), dis end, asm start."""
    ax.axvline(0, color=color, linestyle=ls, linewidth=lw, zorder=3)
    if _t_dis_end_min is not None:
        ax.axvline(_t_dis_end_min, color=color, linestyle=ls, linewidth=lw, zorder=3)
    if (_t_asm_start_min is not None and
            abs(_t_asm_start_min - (_t_dis_end_min or 0)) > 1e-3):
        ax.axvline(_t_asm_start_min, color=color, linestyle=ls, linewidth=lw, zorder=3)


def _direct_label(ax, t_series, y_series, text, color, x_frac=0.92, dy_frac=0.0) -> None:
    xlim  = ax.get_xlim()
    x_val = xlim[0] + x_frac * (xlim[1] - xlim[0])
    subset = t_series[t_series <= x_val]
    if subset.empty:
        return
    idx   = subset.index[-1]
    y_val = float(y_series.loc[idx])
    ylim  = ax.get_ylim()
    y_val += dy_frac * (ylim[1] - ylim[0])
    ax.annotate(text, xy=(x_val, y_val), xytext=(4, 0),
                textcoords="offset points", color=color,
                fontsize=7, va="center", ha="left", clip_on=False)


def _add_weight_twinx(ax, weight_df, t_lo, t_hi, normalize=False):
    """Add mass trace as twin-x axis or normalized on the same axis."""
    if weight_df is None:
        return None
    wt = weight_df[(weight_df["time_min"] >= t_lo) & (weight_df["time_min"] <= t_hi)].copy()
    if len(wt) == 0:
        return None
    wt_vals = wt["weight"] * 1000.0
    if normalize:
        _idx = (wt["time_min"]).abs().idxmin()
        wt_t0 = float(wt_vals.loc[_idx])
        if wt_t0 > 0:
            wt_vals = wt_vals / wt_t0
        ax.plot(wt["time_min"], wt_vals, color=COL_WEIGHT, linewidth=1.0, zorder=4)
        return (wt["time_min"], wt_vals)
    else:
        ax2 = ax.twinx()
        ax2.set_ylabel("Mass (g)", color=COL_WEIGHT)
        ax2.plot(wt["time_min"], wt_vals, color=COL_WEIGHT, linewidth=1.2, zorder=4)
        ax2.tick_params(axis="y", labelcolor=COL_WEIGHT)
        _black_box(ax2)
        return ax2


def _save_fig(fig, name: str, save_dir: str | None, fmt: str = "pdf") -> None:
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.{fmt}")
    fig.savefig(path)
    print(f"  Saved: {path}")


def _add_mask_vlines(ax, overlay_frames, df) -> None:
    """Draw thin vertical lines at each overlay time."""
    for frame in overlay_frames:
        rows = df[df["frame_num"] == frame]
        if len(rows) == 0:
            nearest = (df["frame_num"] - frame).abs().idxmin()
            rows = df.loc[[nearest]]
        t_val = float(rows["time_min"].iloc[0])
        ax.axvline(t_val, color="black", linewidth=0.5, zorder=5)


def _decimate(sub_df: pd.DataFrame, interval_s: float, dt_s: float) -> pd.DataFrame:
    step = max(1, int(round(interval_s / max(dt_s, 1e-6))))
    return sub_df.iloc[::step].copy()


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def _smooth_phases(df, col, asm_mask, rw_dis, rw_asm):
    pre_mask = ~asm_mask
    s_dis = df.loc[pre_mask, col].rolling(rw_dis, center=True, min_periods=1).mean()
    if asm_mask.any():
        s_asm = df.loc[asm_mask, col].rolling(rw_asm, center=True, min_periods=1).mean()
        return pd.concat([s_dis, s_asm]).sort_index()
    return s_dis


def _get_norm_value(series, mode, t0_idx):
    if mode == "t0":
        v = series.iloc[t0_idx]
        return float(v) if pd.notna(v) and v != 0 else 1.0
    elif mode == "max":
        v = series.max()
        return float(v) if pd.notna(v) and v != 0 else 1.0
    elif mode == "mean":
        v = series.mean()
        return float(v) if pd.notna(v) and v != 0 else 1.0
    return 1.0


# ---------------------------------------------------------------------------
# Flying bee loading
# ---------------------------------------------------------------------------

def load_bee_counts(
    base_path: str,
    folders: list[str],
    frames_per_video: list[int],
) -> pd.DataFrame | None:
    """
    Load bee detection npy files, count detections per frame.

    Expects each npy file as rows [frame_id, x, y, ...].
    Returns DataFrame with columns [global_frame, count].
    """
    if base_path is None or not os.path.isdir(base_path):
        return None
    all_files = sorted(os.listdir(base_path))
    npy_files = [f for f in all_files if f.endswith(".npy")]
    if not npy_files:
        return None

    records = []
    cumulative_offset = 0
    for idx, folder in enumerate(folders):
        vid_name = folder.replace("_Masks_Npz", "")
        matching = [f for f in npy_files if vid_name in f]
        if matching:
            fpath = os.path.join(base_path, matching[0])
            data = np.load(fpath)
            frame_ids = data[:, 0].astype(int)
            unique_frames, counts = np.unique(frame_ids, return_counts=True)
            for fr, ct in zip(unique_frames, counts):
                records.append((cumulative_offset + fr, ct))
        if idx < len(frames_per_video):
            cumulative_offset += frames_per_video[idx]

    if not records:
        return None
    return pd.DataFrame(records, columns=["global_frame", "count"])


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def plot_fig1_vol_weight(
    df: pd.DataFrame,
    weight_df: pd.DataFrame | None,
    norm_vol: float,
    t_lo: float, t_hi: float,
    save_dir: str, fmt: str,
    overlay_frames: list[int] | None = None,
    zoomed: bool = False,
) -> None:
    """Figure 1: Normalized volume + mass vs time."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_1P5 if zoomed else FIG_WIDTH_SINGLE,
                                    FIG_HEIGHT_SINGLE), constrained_layout=True)
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)

    ax.plot(df["time_min"], df["vol_smooth"] / norm_vol,
            color=COL_VOL, linewidth=1.0, zorder=4)
    ax.set_xlim(t_lo, t_hi)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$V/V_0$", fontsize=7)
    _phase_vlines(ax)
    _black_box(ax)

    if weight_df is not None:
        wt = weight_df[(weight_df["time_min"] >= t_lo) & (weight_df["time_min"] <= t_hi)]
        if len(wt) > 0:
            wt_vals = wt["weight"] * 1000.0
            _idx = (wt["time_min"]).abs().idxmin()
            wt_t0 = float(wt_vals.loc[_idx])
            if wt_t0 > 0:
                wt_norm = wt_vals / wt_t0
                ax2 = ax.twinx()
                ax2.plot(wt["time_min"], wt_norm, color=COL_WEIGHT, linewidth=1.0, zorder=4)
                ax2.set_ylabel(r"$M/M_0$", color=COL_WEIGHT, fontsize=7)
                ax2.tick_params(axis="y", labelcolor=COL_WEIGHT, labelsize=5)
                ax2.set_ylim(ax.get_ylim())
                _black_box(ax2)
                if zoomed:
                    _direct_label(ax2, wt["time_min"], wt_norm, "Mass", COL_WEIGHT,
                                  x_frac=0.85, dy_frac=0.04)

    if overlay_frames and zoomed:
        _add_mask_vlines(ax, overlay_frames, df)

    suffix = "_dis" if zoomed else ""
    _save_fig(fig, f"vol_mass_norm{suffix}", save_dir, fmt)
    plt.close(fig)


def plot_fig2_diam_len(
    df: pd.DataFrame,
    weight_df: pd.DataFrame | None,
    norm_diam: float,
    norm_len: float,
    t_lo: float, t_hi: float,
    save_dir: str, fmt: str,
    overlay_frames: list[int] | None = None,
    zoomed: bool = False,
) -> None:
    """Figure 2: Normalized diameter + length vs time."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_1P5 if zoomed else FIG_WIDTH_SINGLE,
                                    FIG_HEIGHT_SINGLE), constrained_layout=True)
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)

    d_norm = df["diam_smooth"] / norm_diam
    l_norm = df["len_smooth"]  / norm_len

    ax.plot(df["time_min"], d_norm, color=COL_DIAM, linewidth=1.0, zorder=4)
    ax.plot(df["time_min"], l_norm, color=COL_LEN,  linewidth=1.0, zorder=4)
    ax.set_xlim(t_lo, t_hi)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"Normalized $X/X_0$")
    _phase_vlines(ax)
    _black_box(ax)
    _direct_label(ax, df["time_min"], d_norm, "Diameter", COL_DIAM, x_frac=0.03, dy_frac=-0.07)
    _direct_label(ax, df["time_min"], l_norm, "Height",   COL_LEN,  x_frac=0.03, dy_frac=0.05)

    if weight_df is not None and zoomed:
        _wt_ret = _add_weight_twinx(ax, weight_df, t_lo, t_hi, normalize=True)
        if isinstance(_wt_ret, tuple):
            _direct_label(ax, _wt_ret[0], _wt_ret[1], "Mass", COL_WEIGHT,
                          x_frac=0.85, dy_frac=0.04)

    if overlay_frames and zoomed:
        _add_mask_vlines(ax, overlay_frames, df)

    suffix = "_dis" if zoomed else ""
    _save_fig(fig, f"diam_len_norm{suffix}", save_dir, fmt)
    plt.close(fig)


def save_mask_overlays(
    overlay_frames: list[int],
    df: pd.DataFrame,
    mask_arrays: dict[int, np.ndarray],
    save_dir: str,
    fmt: str = "pdf",
    t_plot_end: float | None = None,
) -> None:
    """
    Save each mask silhouette as a separate vector contour PDF.

    All files use the same fixed canvas (full mask size) so that relative
    swarm sizes across time are preserved when the PDFs are compared.
    Saved to save_dir/mask_overlays/0.pdf, 1.pdf, ...
    """
    import cv2
    from matplotlib.patches import Polygon as MplPolygon

    if not mask_arrays:
        print("  No mask arrays — skipping mask overlay save.")
        return

    overlay_dir = os.path.join(save_dir, "mask_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # Fixed canvas from first available mask
    sample_mask = next(iter(mask_arrays.values()))
    H_full, W_full = sample_mask.shape[:2]
    scale = 3.0 / max(H_full, W_full)
    fig_w, fig_h = W_full * scale, H_full * scale

    for i, frame in enumerate(overlay_frames):
        if frame not in mask_arrays:
            cand = min(mask_arrays.keys(), key=lambda k: abs(k - frame), default=None)
            if cand is None or abs(cand - frame) > 300:
                print(f"  overlay {i}: frame {frame} not found, skipping")
                continue
            frame = cand

        mask = (mask_arrays[frame] > 0).astype(np.uint8)
        df_row = df[df["frame_num"] == frame]
        if len(df_row) == 0:
            idx = (df["frame_num"] - frame).abs().idxmin()
            df_row = df.loc[[idx]]
        t_val = float(df_row["time_min"].iloc[0])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Color outline using viridis to match the scatter colorbar
        if t_plot_end and t_plot_end > 0:
            import matplotlib.cm as cm
            color = cm.viridis(t_val / t_plot_end)
        else:
            color = "black"

        out_path = os.path.join(overlay_dir, f"{i}.{fmt}")
        fig_m, ax_m = plt.subplots(figsize=(fig_w, fig_h))
        fig_m.patch.set_facecolor("none")
        fig_m.patch.set_alpha(0.0)
        ax_m.set_facecolor("none")
        ax_m.patch.set_alpha(0.0)
        ax_m.set_xlim(0, W_full)
        ax_m.set_ylim(H_full, 0)   # flip y to match image coords
        ax_m.set_aspect("equal")
        ax_m.set_axis_off()
        for cnt in contours:
            pts = cnt.squeeze()
            if pts.ndim < 2 or len(pts) < 3:
                continue
            poly = MplPolygon(pts, closed=True, fill=False,
                              edgecolor=color, linewidth=1.2)
            ax_m.add_patch(poly)
        fig_m.savefig(out_path, bbox_inches="tight", pad_inches=0.05,
                      facecolor="none", edgecolor="none", transparent=True)
        plt.close(fig_m)
        print(f"  Mask overlay {i}: frame {frame}, t={t_val:.2f} min → {out_path}")


def _load_mask_for_frame(
    mask_base: str,
    folders: list[str],
    frames_per_video: list[int],
    global_frame: int,
) -> np.ndarray | None:
    """
    Load the mask array closest to global_frame from the NPZ folder hierarchy.

    Returns a 2-D boolean/uint8 array or None if not found.
    """
    if not mask_base or not folders or not frames_per_video:
        return None
    offset = 0
    for folder, fpv in zip(folders, frames_per_video):
        if offset + fpv > global_frame:
            local_frame = global_frame - offset
            folder_path = os.path.join(mask_base, folder)
            if not os.path.isdir(folder_path):
                return None
            npz_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".npz"))
            if not npz_files:
                return None
            # Find file whose embedded frame number is closest to local_frame
            def _frame_num(fname):
                import re
                # Match the LAST numeric run in the filename (before extension)
                nums = re.findall(r"(\d+)", fname.rsplit(".", 1)[0])
                return int(nums[-1]) if nums else 0
            best = min(npz_files, key=lambda f: abs(_frame_num(f) - local_frame))
            try:
                data = np.load(os.path.join(folder_path, best))
                return data["mask"]
            except Exception:
                return None
        offset += fpv
    return None



def plot_fig5b_flying_bees(
    bee_df: pd.DataFrame,
    weight_df: pd.DataFrame | None,
    mask_t0_frame: int,
    mask_fps: float,
    save_dir: str,
    fmt: str,
) -> None:
    """Figure 5b: Flying bee count + mass vs time."""
    bee_df = bee_df.copy()
    bee_df["time_s"]   = (bee_df["global_frame"] - mask_t0_frame) / mask_fps
    bee_df["time_min"] = bee_df["time_s"] / 60.0
    bee_df = bee_df.sort_values("time_min").reset_index(drop=True)

    dt = float(bee_df["time_s"].diff().abs().median()) if len(bee_df) > 2 else 1.0
    rw = max(1, int(round(5.0 / max(dt, 1e-6))))
    bee_df["count_smooth"] = bee_df["count"].rolling(rw, center=True, min_periods=1).mean()

    t_lo = float(bee_df["time_min"].min())
    t_hi = float(bee_df["time_min"].max())

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, FIG_HEIGHT_SINGLE), constrained_layout=True)
    ax.tick_params(labelsize=6)
    ax.plot(bee_df["time_min"], bee_df["count_smooth"],
            color=COL_BEES, linewidth=1.0, zorder=4)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylabel("# Flying bees", color=COL_BEES)
    ax.tick_params(axis="y", labelcolor=COL_BEES)
    ax.set_xlabel("Time (min)")
    _phase_vlines(ax)
    _black_box(ax)
    _add_weight_twinx(ax, weight_df, t_lo, t_hi)
    _save_fig(fig, "bees_mass", save_dir, fmt)
    plt.close(fig)


def plot_fig6c_scatter_combined(
    df: pd.DataFrame,
    norm_diam: float,
    norm_len: float,
    t_dis_end: float,
    t_asm_start: float,
    t_plot_end: float,
    mask_overlay_frames: list[int],
    save_dir: str,
    fmt: str,
    plot_interval_s: float,
    dt_dis_s: float,
    dt_asm_s: float,
) -> None:
    """Figure 6c: D/D₀ vs L/L₀ scatter — disassembly + assembly, gap excluded."""
    sub = _decimate(
        df[((df["time_min"] >= 0) & (df["time_min"] <= t_dis_end)) |
           ((df["time_min"] >= t_asm_start) & (df["time_min"] <= t_plot_end))],
        plot_interval_s, dt_asm_s,
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, FIG_HEIGHT_SINGLE), constrained_layout=True)

    sc = None
    if len(sub):
        x = sub["diam_smooth"] / norm_diam
        y = sub["len_smooth"]  / norm_len
        sc = ax.scatter(x, y, c=sub["time_min"], cmap="viridis",
                        vmin=0, vmax=t_plot_end, s=6, alpha=0.8, linewidths=0)
        ax.set_xlim(ax.get_xlim()[0], 2.2)
        ax.set_ylim(ax.get_ylim()[0], 1.4)
        ax.set_aspect("equal")

        # Overlay frame markers (open squares) — show for both phases
        for frame in mask_overlay_frames:
            row = df.loc[(df["frame_num"] - frame).abs().idxmin()]
            t_val = float(row["time_min"])
            in_dis = 0 <= t_val <= t_dis_end
            in_asm = t_asm_start <= t_val <= t_plot_end
            ox = float(row["diam_smooth"]) / norm_diam
            oy = float(row["len_smooth"])  / norm_len
            print(f"  overlay frame={frame} t={t_val:.2f}min D/D0={ox:.3f} L/L0={oy:.3f} in_dis={in_dis} in_asm={in_asm}")
            if in_dis or in_asm:
                ax.plot(ox, oy, "s", mec="black", mfc="none",
                        markersize=6, mew=0.6, zorder=10)

        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, shrink=1.0)
        cb.set_label("Time (min)", fontsize=6)
        cb.outline.set_edgecolor("black")

    ax.set_xlabel(r"$D/D_0$")
    ax.set_ylabel(r"$H/H_0$")
    _black_box(ax)

    _save_fig(fig, "scatter_combined", save_dir, fmt)
    plt.close(fig)


def save_boundary_frames(
    phases: dict,
    df: pd.DataFrame,
    mask_fps: float,
    video_base: str,
    folders: list[str],
    frames_per_video: list[int],
    save_dir: str,
    video_ext: str = ".MP4",
) -> None:
    """
    Extract and save a JPEG frame from the video at each phase boundary
    (disassembly start, disassembly end, assembly start).

    Files saved as: save_dir/boundary_frames/t0.jpg, dis_end.jpg, asm_start.jpg
    """
    import cv2

    boundaries = {
        "t0":        phases.get("mask_t0_frame"),
        "dis_end":   phases.get("disassembly_end_frame"),
        "asm_start": phases.get("assembly_start_frame"),
    }

    out_dir = os.path.join(save_dir, "boundary_frames")
    os.makedirs(out_dir, exist_ok=True)

    # Build cumulative offset table
    offsets = []
    cum = 0
    for n in frames_per_video:
        offsets.append(cum)
        cum += n

    for label, global_frame in boundaries.items():
        if global_frame is None:
            print(f"  boundary_frames: {label} not set, skipping")
            continue

        # Find which folder this frame belongs to
        folder_idx = len(folders) - 1
        for i in range(len(folders) - 1, -1, -1):
            if global_frame >= offsets[i]:
                folder_idx = i
                break

        local_frame = global_frame - offsets[folder_idx]
        folder      = folders[folder_idx]
        vid_name    = folder.replace("_Masks_Npz", "") + video_ext
        vid_path    = str(Path(video_base) / vid_name)

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  boundary_frames: could not open {vid_path}")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"  boundary_frames: could not read frame {local_frame} from {vid_name}")
            continue

        t_min = (global_frame - phases["mask_t0_frame"]) / mask_fps / 60.0
        out_path = os.path.join(out_dir, f"{label}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"  boundary_frames: {label} → {folder} local={local_frame} t={t_min:.2f}min → {out_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_figures(
    df: pd.DataFrame,
    weight_df: pd.DataFrame | None,
    mask_t0_frame: int,
    mask_fps: float,
    phases: dict,
    cfg: dict,
    bee_df: pd.DataFrame | None = None,
) -> None:
    """
    Generate all morphology figures and save to plots.save_dir.

    phases: dict with keys mask_t0_frame, disassembly_end_frame, assembly_start_frame
    """
    from swarm_assembly_methods.utils import resolve_save_dir
    plots_cfg = cfg.get("plots", {})
    save_dir  = str(resolve_save_dir(plots_cfg, "save_dir", "figures/figmorph"))
    fmt       = plots_cfg.get("format", "pdf")
    norm_mode = plots_cfg.get("norm_mode", "t0")
    rw_s_dis  = float(plots_cfg.get("rolling_window_s_disassembly", 1.0))
    rw_s_asm  = float(plots_cfg.get("rolling_window_s_assembly", 10.0))
    pi_s_dis  = float(plots_cfg.get("plot_interval_s_disassembly", 1.0))
    pi_s_asm  = float(plots_cfg.get("plot_interval_s_assembly", 1.0))
    t_plot_end_cfg = float(plots_cfg.get("t_plot_end", 30.0))
    t_pre_min      = float(plots_cfg.get("t_pre_min", 0.0))

    if plots_cfg.get("paper_style", True):
        apply_paper_style()

    # --- Phase time boundaries ---
    global _t_dis_end_min, _t_asm_start_min
    mask_fps_f = float(mask_fps)

    dis_end_frame   = phases.get("disassembly_end_frame")
    asm_start_frame = phases.get("assembly_start_frame")

    _t_dis_end_min   = (dis_end_frame   - mask_t0_frame) / mask_fps_f / 60.0 if dis_end_frame   else None
    _t_asm_start_min = (asm_start_frame - mask_t0_frame) / mask_fps_f / 60.0 if asm_start_frame else None
    _dis_end = _t_dis_end_min   if _t_dis_end_min   is not None else t_plot_end_cfg
    _asm_st  = _t_asm_start_min if _t_asm_start_min is not None else _dis_end

    # --- Time bounds (match original: show pre-t0 steady state) ---
    df_tmin = float(df["time_min"].min())
    df_tmax = float(df["time_min"].max())
    if weight_df is not None:
        wt_tmin = float(weight_df["time_min"].min())
        wt_tmax = float(weight_df["time_min"].max())
        t_pre_start = max(df_tmin, wt_tmin)   # earliest time both sources have data
        t_end       = min(df_tmax, wt_tmax, t_plot_end_cfg)
    else:
        t_pre_start = df_tmin
        t_end       = min(df_tmax, t_plot_end_cfg)
    # Optional hard clip
    if t_pre_min > 0:
        t_pre_start = max(t_pre_start, -t_pre_min)

    print(f"Plot window: [{t_pre_start:.2f}, {t_end:.2f}] min  "
          f"(dis_end={_dis_end:.2f}, asm_start={_asm_st:.2f})")

    # --- Assembly / disassembly masks ---
    asm_mask_col = df["frame_num"] >= (asm_start_frame if asm_start_frame
                                       else df["frame_num"].iloc[-1] + 1)

    # --- Compute per-phase smoothing row counts ---
    dt_dis = float(df.loc[~asm_mask_col, "time_s"].diff().abs().median()) if (~asm_mask_col).any() else 1.0
    dt_asm = float(df.loc[ asm_mask_col, "time_s"].diff().abs().median()) if  asm_mask_col.any()  else 1.0
    dt_dis = max(dt_dis, 1e-6)
    dt_asm = max(dt_asm, 1e-6)
    rw_dis = max(1, int(round(rw_s_dis / dt_dis))) if rw_s_dis > 0 else 1
    rw_asm = max(1, int(round(rw_s_asm / dt_asm))) if rw_s_asm > 0 else 1

    # --- Choose best volume column ---
    has_axisym = "volume_axisym_m3" in df.columns and df["volume_axisym_m3"].notna().sum() > 0
    has_metric = "volume_m3" in df.columns and df["volume_m3"].notna().sum() > 0
    if has_axisym:
        vol_col = "volume_axisym_m3"
    elif has_metric:
        vol_col = "volume_m3"
    else:
        vol_col = "volume_px3"

    # --- Average left+right morphology ---
    has_right = "width_right_px" in df.columns and df["width_right_px"].notna().sum() > 0
    if has_right:
        df["width_avg_px"]  = df[["width_px",  "width_right_px"]].mean(axis=1)
        df["length_avg_px"] = df[["length_px", "length_right_px"]].mean(axis=1)
    else:
        df["width_avg_px"]  = df["width_px"].astype(float)
        df["length_avg_px"] = df["length_px"].astype(float)

    # --- Smooth ---
    df["vol_smooth"]  = _smooth_phases(df, vol_col,         asm_mask_col, rw_dis, rw_asm)
    df["diam_smooth"] = _smooth_phases(df, "width_avg_px",  asm_mask_col, rw_dis, rw_asm)
    df["len_smooth"]  = _smooth_phases(df, "length_avg_px", asm_mask_col, rw_dis, rw_asm)

    # --- Normalization ---
    t0_idx    = int((df["time_s"].abs()).idxmin())
    norm_vol  = _get_norm_value(df["vol_smooth"],  norm_mode, t0_idx)
    norm_diam = _get_norm_value(df["diam_smooth"], norm_mode, t0_idx)
    norm_len  = _get_norm_value(df["len_smooth"],  norm_mode, t0_idx)
    print(f"Norm ({norm_mode}): vol={norm_vol:.5g}  diam={norm_diam:.1f}  len={norm_len:.1f}")

    # --- Mask overlay frames ---
    inp_cfg = cfg.get("input", {})
    folders_left     = inp_cfg.get("folders_left", [])
    frames_per_video = inp_cfg.get("frames_per_video", [])
    mask_base_left   = inp_cfg.get("mask_base_left")

    # Use pre-resolved global frames (from pipeline.py via overlay_times_local),
    # or fall back to computing from overlay_times_min.
    raw_overlay = cfg.get("overlay_frames_resolved") or []
    if not raw_overlay:
        for t_min in (cfg.get("overlay_times_min") or []):
            raw_overlay.append(mask_t0_frame + int(round(float(t_min) * mask_fps_f * 60.0)))

    # Snap each global frame to the nearest frame actually present in the parquet
    overlay_frames: list[int] = []
    for gf in raw_overlay:
        nearest_idx = (df["frame_num"] - gf).abs().idxmin()
        snapped = int(df.loc[nearest_idx, "frame_num"])
        overlay_frames.append(snapped)
        snap_s = abs(snapped - gf) / mask_fps_f
        if snap_s > 1.0:
            print(f"  WARNING: overlay frame {gf} snapped to {snapped} ({snap_s:.1f}s away)")
    print(f"Overlay frames: {overlay_frames}")

    # Load mask thumbnails for insets (optional)
    mask_arrays: dict[int, np.ndarray] = {}
    if mask_base_left and folders_left and frames_per_video and overlay_frames:
        print("Loading mask thumbnails for scatter insets...")
        for gf in overlay_frames:
            m = _load_mask_for_frame(mask_base_left, folders_left, frames_per_video, gf)
            if m is not None:
                mask_arrays[gf] = m
        print(f"  Loaded {len(mask_arrays)}/{len(overlay_frames)} mask thumbnails")
        print("-- Mask overlay PDFs --")
        save_mask_overlays(overlay_frames, df, mask_arrays, save_dir, fmt, t_plot_end=t_end)

    # =====================================================================
    # Figure 1: Normalized volume + mass
    # =====================================================================
    print("-- Figure 1: volume + mass (normalized) --")
    plot_fig1_vol_weight(df, weight_df, norm_vol,
                         t_pre_start, t_end, save_dir, fmt)

    # =====================================================================
    # Figure 2: Normalized diameter + length
    # =====================================================================
    print("-- Figure 2: diameter + length (normalized) --")
    plot_fig2_diam_len(df, weight_df, norm_diam, norm_len,
                       t_pre_start, t_end, save_dir, fmt)

    # =====================================================================
    # Flying bee count + mass
    # =====================================================================
    print("-- bees_mass --")
    if bee_df is not None:
        plot_fig5b_flying_bees(bee_df, weight_df, mask_t0_frame, mask_fps_f, save_dir, fmt)
    else:
        print("  No bee detection data — skipping bees_mass.")

    # =====================================================================
    # Figure 6c: Diameter vs length scatter (disassembly + assembly combined)
    # =====================================================================
    print("-- Figure 6c: scatter combined --")
    if _t_dis_end_min is not None and asm_mask_col.any():
        plot_fig6c_scatter_combined(
            df, norm_diam, norm_len,
            t_dis_end=_dis_end,
            t_asm_start=_asm_st,
            t_plot_end=t_end,
            mask_overlay_frames=overlay_frames,
            save_dir=save_dir, fmt=fmt,
            plot_interval_s=pi_s_dis,
            dt_dis_s=dt_dis,
            dt_asm_s=dt_asm,
        )
    else:
        print("  Missing disassembly end or assembly phase — skipping fig6c.")

    # =====================================================================
    # Boundary frames: save a video frame at t0, dis_end, asm_start
    # =====================================================================
    video_base = inp_cfg.get("mask_base_left") and cfg.get("input", {}).get("mask_base_left")
    video_base = cfg.get("input", {}).get("mask_base_left")  # reuse mask_base_left path key
    # Use video_base_left if available, else skip
    from swarm_assembly_methods.utils import resolve_session_paths
    _vbase = None
    raw_root = cfg.get("raw_root")
    sess     = cfg.get("session", {})
    if raw_root and sess.get("subject") and sess.get("date") and sess.get("stereo_side"):
        _paths = resolve_session_paths(raw_root, cfg.get("processed_root","data"),
                                       cfg.get("figures_root","figures"),
                                       str(sess["subject"]), str(sess["date"]),
                                       str(sess["stereo_side"]), sess.get("run_id"))
        _vbase = cfg.get("input", {}).get("video_base_left") or _paths.get("video_base_left")
    if _vbase and folders_left and frames_per_video and save_dir:
        print("-- Boundary frames --")
        save_boundary_frames(
            phases=phases,
            df=df,
            mask_fps=mask_fps_f,
            video_base=_vbase,
            folders=folders_left,
            frames_per_video=frames_per_video,
            save_dir=save_dir,
            video_ext=cfg.get("input", {}).get("video_ext", ".MP4"),
        )
    else:
        print("  Boundary frames: video_base_left not available, skipping.")

    print(f"\nAll figures saved to: {save_dir}")

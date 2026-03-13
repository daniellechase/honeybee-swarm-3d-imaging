"""
Time-alignment utilities for morphology analysis.

Converts local (folder, frame) coordinates to global frames,
finds the disassembly onset, and aligns weight data to mask time axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def folder_local_to_global(
    folders: Sequence[str],
    folder_name: str,
    local_frame: int,
    frames_per_video: Sequence[int],
) -> int:
    """Convert (folder_name, local_frame) to a global frame number."""
    offset = 0
    for folder, length in zip(folders, frames_per_video):
        if folder == folder_name:
            return offset + int(local_frame)
        offset += length
    raise ValueError(f"Folder {folder_name!r} not found in folders list")


def find_steepest_decline(
    values: np.ndarray,
    smooth_window: int = 30,
    frame_nums: np.ndarray | None = None,
) -> int:
    """Return index of most negative derivative (steepest drop)."""
    s = pd.Series(values, dtype=float).rolling(
        window=smooth_window, center=True, min_periods=1).mean()
    if frame_nums is not None:
        fn = pd.Series(frame_nums, dtype=float)
        dframe = fn.diff().replace(0, np.nan)
        deriv = s.diff() / dframe
    else:
        deriv = s.diff()
    return int(deriv.idxmin())


def _estimate_boundaries_from_arrays(
    frames: np.ndarray,
    values: np.ndarray,
    rows_per_s: float,
    smooth_s: float = 10.0,
    onset_drop_frac: float = 0.02,
    recovery_frac: float = 0.02,
    assembly_frac: float = 0.05,
    post_steep_search_s: float = 300.0,
    onset_drop_abs: float | None = None,
    recovery_abs: float | None = None,
    assembly_abs: float | None = None,
    assembly_end_abs: float | None = None,
    baseline_window_s: float = 5.0,
    save_dir: str | None = None,
) -> dict:
    """
    Core boundary-detection algorithm on any declining-then-recovering signal.

    Parameters
    ----------
    frames : array of "frame" coordinates (may be seconds for weight, or
             global frame numbers for volume).
    values : signal values (e.g. volume or mass).
    rows_per_s : how many rows correspond to one second of real time.

    Thresholds — two modes (absolute takes priority if set):
      *_abs  : absolute value in signal units (e.g. kg for mass).
               onset_drop_abs  — signal must drop this much below baseline for t0.
               recovery_abs    — signal must be within this much above min for dis_end.
               assembly_abs    — signal must rise this much above min for asm_start.
      *_frac : fraction of total_drop = baseline - min (used if *_abs not set).

    Returns
    -------
    dict with keys:
        t0_frame, dis_end_frame, asm_start_frame, steep_frame,
        baseline, min_val, total_drop
    """
    smooth_rows = max(3, int(smooth_s * rows_per_s))
    post_rows   = max(10, int(post_steep_search_s * rows_per_s))

    s = pd.Series(values, dtype=float).rolling(
        window=smooth_rows, center=True, min_periods=1).mean()
    frame_s = pd.Series(frames, dtype=float)
    dframe  = frame_s.diff().replace(0.0, np.nan)
    deriv   = s.diff() / dframe

    steep_idx  = int(deriv.idxmin())
    steep_frame = frames[steep_idx]

    # Baseline: max of smoothed signal in baseline_window_s before the steepest point
    pre_rows = max(1, int(baseline_window_s * rows_per_s))
    window_start = max(0, steep_idx - pre_rows)
    baseline = float(s.iloc[window_start:steep_idx].max())

    # Trough: smoothed global minimum (used as dis_end and recovery reference)
    search_end = min(len(frames), steep_idx + post_rows)
    trough_idx = int(s.idxmin())   # global minimum of smoothed signal
    min_val    = float(s.iloc[trough_idx])
    total_drop = baseline - min_val

    if total_drop <= 0:
        return dict(
            t0_frame=frames[0], dis_end_frame=frames[-1],
            asm_start_frame=None, steep_frame=steep_frame,
            baseline=baseline, min_val=min_val, total_drop=0.0,
        )

    print(f"  [boundary detection] baseline={baseline:.6f}  min={min_val:.6f}  "
          f"total_drop={total_drop:.6f}  steep_frame={steep_frame}")

    # --- t0: walk backwards from steep_idx ---
    # Last frame still within threshold of baseline (drop barely started)
    onset_thr   = onset_drop_abs if onset_drop_abs is not None else onset_drop_frac * total_drop
    onset_level = baseline - onset_thr
    print(f"  [t0]         onset_thr={onset_thr:.6f}  onset_level={onset_level:.6f}"
          f"  ({'abs' if onset_drop_abs is not None else 'frac'})")
    t0_idx = 0
    for i in range(steep_idx - 1, -1, -1):
        if float(s.iloc[i]) >= onset_level:
            t0_idx = i
            break
    print(f"  [t0]         steep_idx={steep_idx}  t0_idx={t0_idx}"
          f"  s[steep_idx-1]={float(s.iloc[steep_idx-1]) if steep_idx>0 else 'n/a':.6f}"
          f"  s[t0_idx]={float(s.iloc[t0_idx]):.6f}")
    t0_frame = frames[t0_idx]

    # --- Disassembly end: walk backwards from trough, last frame where signal
    #     is still recovery_threshold above the minimum. recovery_mass=0 → trough. ---
    rec_thr       = recovery_abs if recovery_abs is not None else recovery_frac * total_drop
    dis_end_level = min_val + rec_thr
    print(f"  [dis_end]    rec_thr={rec_thr:.6f}  dis_end_level={dis_end_level:.6f}"
          f"  ({'abs' if recovery_abs is not None else 'frac'})")
    dis_end_idx = trough_idx   # default: trough (recovery_mass=0)
    for i in range(trough_idx - 1, steep_idx - 1, -1):
        if float(s.iloc[i]) > dis_end_level:
            dis_end_idx = i
            break
    dis_end_frame = frames[dis_end_idx]
    print(f"  [dis_end]    → frame {dis_end_frame}")

    # --- Assembly start: first frame after the trough where signal rises
    #     above assembly_abs (absolute level) or assembly_frac * total_drop
    #     above the trough (fraction mode). ---
    asm_thr     = assembly_abs if assembly_abs is not None else min_val + assembly_frac * total_drop
    recover_thr = asm_thr if assembly_abs is not None else asm_thr
    print(f"  [asm_start]  recover_thr={recover_thr:.6f}"
          f"  ({'abs level' if assembly_abs is not None else 'frac above trough'})")
    asm_start_frame = None
    for i in range(trough_idx + 1, len(frames)):
        if float(s.iloc[i]) >= recover_thr:
            asm_start_frame = frames[i]
            break

    # --- Assembly end: first frame after trough where signal rises within
    #     assembly_end_abs of the peak value after the trough. ---
    asm_end_frame = None
    if assembly_end_abs is not None:
        peak_after_trough = float(s.iloc[trough_idx:].max())
        asm_end_level = peak_after_trough - assembly_end_abs
        print(f"  [asm_end]    peak_after_trough={peak_after_trough:.6f}  asm_end_level={asm_end_level:.6f}")
        for i in range(trough_idx + 1, len(frames)):
            if float(s.iloc[i]) >= asm_end_level:
                asm_end_frame = frames[i]
                break
        print(f"  [asm_end]    → frame {asm_end_frame}")


    return dict(
        t0_frame=t0_frame,
        dis_end_frame=dis_end_frame,
        asm_start_frame=asm_start_frame,
        asm_end_frame=asm_end_frame,
        steep_frame=steep_frame,
        baseline=baseline,
        min_val=min_val,
        total_drop=total_drop,
    )


def estimate_phase_boundaries(
    df: pd.DataFrame,
    fps: float,
    smooth_s: float = 10.0,
    onset_drop_frac: float = 0.02,
    recovery_frac: float = 0.02,
    assembly_frac: float = 0.05,
    post_steep_search_s: float = 300.0,
) -> dict:
    """
    Auto-estimate phase boundaries from the volume signal.

    Returns dict with keys:
        mask_t0_frame       (int)   global frame of disassembly onset
        disassembly_end_min (float) minutes after t0 when disassembly ends
        assembly_start_min  (float | None) minutes after t0 when assembly begins

    All three *_frac thresholds are fractions of total_drop = baseline - min_vol
    (always positive, 0–1).
    """
    frames = df["frame_num"].values
    vol    = df["volume_px3"].ffill().bfill().values

    if len(frames) < 4:
        return dict(mask_t0_frame=int(frames[0]),
                    disassembly_end_min=np.nan,
                    assembly_start_min=None)

    median_dframe = float(np.median(np.diff(frames))) if len(frames) > 1 else 1.0
    rows_per_s    = fps / max(median_dframe, 1.0)

    b = _estimate_boundaries_from_arrays(
        frames, vol, rows_per_s,
        smooth_s=smooth_s,
        onset_drop_frac=onset_drop_frac,
        recovery_frac=recovery_frac,
        assembly_frac=assembly_frac,
        post_steep_search_s=post_steep_search_s,
    )

    t0_frame = int(b["t0_frame"])
    dend_min = (int(b["dis_end_frame"]) - t0_frame) / fps / 60.0
    astart_min = (
        (int(b["asm_start_frame"]) - t0_frame) / fps / 60.0
        if b["asm_start_frame"] is not None else None
    )

    return dict(
        mask_t0_frame=t0_frame,
        disassembly_end_min=round(dend_min, 2),
        assembly_start_min=round(astart_min, 2) if astart_min is not None else None,
    )


def align_time_axes(
    df: pd.DataFrame,
    weight_df: pd.DataFrame | None,
    mask_t0_frame: int,
    mask_fps: float,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, int, dict | None]:
    """
    Add time_s and time_min columns to df and weight_df, and return
    mass-derived phase boundaries (if available).

    weight_t0 comes from cfg["phases"]["weight_t0"] ("auto" or seconds).

    Auto mode — align on mass:
        Phase boundaries (onset, disassembly end, assembly start) are detected
        from the steady → decline → recovery pattern in the weight/mass signal.
        The mask time axis is then shifted so its steepest volume drop aligns
        with the mass steepest drop, giving both signals a shared t=0.

    Returns
    -------
    (df, weight_df, mask_t0_frame, mass_phases)

    mass_phases is a dict with keys t0_s, dis_end_s, asm_start_s (all in
    seconds relative to weight t0) when auto mode is used, else None.
    These are converted to mask global frames by the caller.
    """
    phases_cfg    = cfg.get("phases", {})
    weight_t0_cfg = phases_cfg.get("weight_t0") or cfg.get("session", {}).get("weight_t0", "auto")

    mass_phases: dict | None = None

    if weight_df is not None and weight_t0_cfg == "auto":
        weight_df = weight_df.copy()

        # weight frame_num is in seconds; rows are ~1 row/s
        w_frames = weight_df["frame_num"].values
        w_vals   = weight_df["weight"].values
        w_rows_per_s = 1.0 / max(float(np.median(np.diff(w_frames))), 1e-6) if len(w_frames) > 1 else 1.0

        def _opt_float(key):
            v = phases_cfg.get(key)
            return float(v) if v is not None else None

        wb = _estimate_boundaries_from_arrays(
            w_frames, w_vals, w_rows_per_s,
            smooth_s             = float(phases_cfg.get("smooth_s",             10.0)),
            onset_drop_frac      = float(phases_cfg.get("onset_drop_frac",      0.02)),
            recovery_frac        = float(phases_cfg.get("recovery_frac",        0.02)),
            assembly_frac        = float(phases_cfg.get("assembly_frac",        0.05)),
            onset_drop_abs       = _opt_float("onset_drop_mass"),
            recovery_abs         = _opt_float("recovery_mass"),
            assembly_abs         = _opt_float("assembly_mass"),
            assembly_end_abs     = _opt_float("assembly_end_mass"),
            baseline_window_s    = float(phases_cfg.get("baseline_window_s",    5.0)),
            save_dir             = cfg.get("plots", {}).get("save_dir"),
        )

        weight_t0      = float(wb["t0_frame"])   # seconds; mass onset = t=0
        weight_steep_s = float(wb["steep_frame"])

        print(
            f"Mass-based boundaries:"
            f"  onset={weight_t0:.1f}s"
            f"  dis_end={wb['dis_end_frame']:.1f}s"
            f"  asm_start={wb['asm_start_frame']}s"
            f"  asm_end={wb['asm_end_frame']}s"
            f"  (steep at {weight_steep_s:.1f}s)"
        )

        # Align mask: shift so mask steep drop coincides with mass steep drop
        vol            = df["volume_px3"].ffill().bfill().values
        mask_steep_idx = find_steepest_decline(vol, 10, df["frame_num"].values)
        mask_steep_frame = int(df.iloc[mask_steep_idx]["frame_num"])

        # dt_steep: seconds between mass onset (t=0) and mass steep drop
        dt_steep_s = weight_steep_s - weight_t0
        # mask_t0_frame: mask frame that corresponds to mass onset
        mask_t0_frame = mask_steep_frame - int(round(dt_steep_s * mask_fps))

        print(f"Mask alignment: mask_steep_frame={mask_steep_frame}  "
              f"dt_steep={dt_steep_s:.1f}s  → mask_t0_frame={mask_t0_frame}")

        weight_df["time_s"]   = weight_df["frame_num"] - weight_t0
        weight_df["time_min"] = weight_df["time_s"] / 60.0

        # Return mass phase boundaries in seconds-after-t0 for caller to convert
        asm_start_s = (float(wb["asm_start_frame"]) - weight_t0
                       if wb["asm_start_frame"] is not None else None)
        asm_end_s   = (float(wb["asm_end_frame"])   - weight_t0
                       if wb["asm_end_frame"]   is not None else None)

        if asm_start_s is not None and asm_end_s is not None:
            print(f"  → assembly duration: {(asm_end_s - asm_start_s)/60:.2f} min "
                  f"(start={asm_start_s/60:.2f} min, end={asm_end_s/60:.2f} min after t0)")

        mass_phases = dict(
            dis_end_s   = float(wb["dis_end_frame"]) - weight_t0,
            asm_start_s = asm_start_s,
            asm_end_s   = asm_end_s,
        )

    elif weight_df is not None:
        weight_df = weight_df.copy()
        weight_t0 = float(weight_t0_cfg)
        print(f"Manual weight_t0 = {weight_t0}s")
        weight_df["time_s"]   = weight_df["frame_num"] - weight_t0
        weight_df["time_min"] = weight_df["time_s"] / 60.0

    df = df.copy()
    df["time_s"]   = (df["frame_num"] - mask_t0_frame) / mask_fps
    df["time_min"] = df["time_s"] / 60.0

    return df, weight_df, mask_t0_frame, mass_phases

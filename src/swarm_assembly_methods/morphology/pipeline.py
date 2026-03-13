"""
Morphology compute pipeline entry point.

Orchestrates:
  1. Load calibration (optional, for metric volumes)
  2. Resolve frames_per_video
  3. Load / cache mask metrics
  4. Save diagnostic plot (volume vs global frame) to help set phase boundaries
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from swarm_assembly_methods.utils import update_yaml_field
from .alignment import estimate_phase_boundaries
from .config import get_cache_path, use_cache
from .loading import load_masks_from_base, load_matching_right_masks, resolve_frames_per_video
from .metrics import compute_mask_properties, build_calib_dict


def _save_diagnostic(
    df: pd.DataFrame,
    cache_path: Path,
    weight_csv: str | None,
    fps: float,
) -> None:
    """
    Save a raw volume-vs-frame diagnostic plot alongside the metrics parquet.

    Both panels share the same x-axis (global frame, left camera).
    Weight (whose CSV frame_num is in seconds) is converted to frames using fps
    and aligned to the start of the mask recording.

    Estimated phase boundaries are annotated as vertical lines and saved
    to *_estimated_boundaries.yaml for easy copy-paste into figures config.
    """
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Auto-estimate phase boundaries ----
    est = estimate_phase_boundaries(df, fps)
    t0_frame      = est["mask_t0_frame"]
    dend_min      = est["disassembly_end_min"]
    astart_min    = est["assembly_start_min"]

    print(
        f"\nEstimated phase boundaries:\n"
        f"  mask_t0_frame       = {t0_frame}\n"
        f"  disassembly_end_min = {dend_min}\n"
        f"  assembly_start_min  = {astart_min}\n"
        f"Copy these into configs/figures/figmorph/*.yaml (session section)."
    )

    # Save YAML snippet
    est_out = cache_path.with_name(cache_path.stem + "_estimated_boundaries.yaml")
    with open(est_out, "w") as f:
        yaml.dump(
            {"session": {
                "mask_t0_frame":       t0_frame,
                "disassembly_end_min": dend_min,
                "assembly_start_min":  astart_min,
            }},
            f, default_flow_style=False, sort_keys=False,
        )
    print(f"Estimated boundaries saved: {est_out}")

    # ---- Build plot ----
    # Weight x-axis: convert seconds → global frames using fps and first mask frame
    wdf = None
    has_weight = False
    if weight_csv and Path(weight_csv).exists():
        _w = pd.read_csv(weight_csv)
        if "frame_num" in _w.columns and "weight" in _w.columns:
            first_mask_frame = int(df["frame_num"].min())
            _w["frame_num_aligned"] = _w["frame_num"] * fps + first_mask_frame
            _w["weight_g"] = (_w["weight"] - _w["weight"].min()) * 1000
            wdf = _w
            has_weight = True

    n_rows = 2 if has_weight else 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 5 * n_rows),
        sharex=True, squeeze=False,
        gridspec_kw={"hspace": 0.10},
    )

    # Choose best available volume column
    vol_col = "volume_px3"
    for col in ("volume_axisym_m3", "volume_m3", "volume_px3"):
        if col in df.columns and df[col].notna().any():
            vol_col = col
            break

    # Vertical line positions (in global frames)
    dend_frame   = t0_frame + dend_min  * 60.0 * fps if np.isfinite(dend_min) else None
    astart_frame = t0_frame + astart_min * 60.0 * fps if astart_min is not None else None

    def _vlines(ax):
        ax.axvline(t0_frame,    color="black",  lw=1.0, ls="--", label=f"t0 (frame {t0_frame})")
        if dend_frame:
            ax.axvline(dend_frame,  color="orange", lw=1.0, ls="--",
                       label=f"disassembly end ({dend_min:.1f} min)")
        if astart_frame:
            ax.axvline(astart_frame, color="green", lw=1.0, ls="--",
                       label=f"assembly start ({astart_min:.1f} min)")

    ax1 = axes[0, 0]
    ax1.plot(df["frame_num"], df[vol_col].ffill().bfill(),
             linewidth=0.8, color="steelblue")
    _vlines(ax1)
    ax1.set_ylabel(vol_col)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title(
        "Estimated boundaries shown — verify and copy _estimated_boundaries.yaml "
        "into configs/figures/figmorph/*.yaml",
        fontsize=8,
    )

    if has_weight:
        ax2 = axes[1, 0]
        ax2.plot(wdf["frame_num_aligned"], wdf["weight_g"],
                 linewidth=0.8, color="crimson")
        _vlines(ax2)
        ax2.set_ylabel("Weight relative (g)")
        ax2.set_xlabel(
            f"Global frame (left camera)  [fps={fps}; "
            "weight aligned: frame 0 of weight = first mask frame]"
        )
    else:
        ax1.set_xlabel(f"Global frame (left camera)  [fps={fps}]")

    fig.tight_layout()
    out = cache_path.with_name(cache_path.stem + "_diagnostic.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved: {out}")


def run_morphology(cfg: dict, config_path=None) -> None:
    """Load masks and compute per-frame metrics; save to parquet cache + diagnostic plot."""
    data_cfg  = cfg.get("data", {})
    n_workers = data_cfg.get("n_workers", 8)

    mask_base_left  = data_cfg["mask_base_left"]
    mask_base_right = data_cfg.get("mask_base_right")
    folders_left    = data_cfg["folders_left"]
    folders_right   = data_cfg.get("folders_right", [])
    load_intervals  = data_cfg.get("load_intervals", [1] * len(folders_left))

    # ---- Calibration ----
    calib = build_calib_dict(cfg)

    # ---- frames_per_video ----
    print("Resolving frames_per_video...")
    frames_per_video = resolve_frames_per_video(cfg, folders_left)

    # ---- Metrics cache ----
    cache_path = get_cache_path(cfg)
    cache_ok = (
        use_cache(cfg)
        and cache_path is not None
        and Path(cache_path).exists()
    )

    if cache_ok:
        print(f"Cache already exists: {cache_path}  (set use_cache: false to recompute)")
        df = pd.read_parquet(cache_path)
    else:
        # ---- Load left masks ----
        print("\nLoading left masks...")
        masks_left = load_masks_from_base(
            mask_base_left, folders_left, load_intervals, frames_per_video, n_workers)
        print(f"Total left masks loaded: {len(masks_left)}")

        # ---- Load right masks (matched to left+dk) ----
        right_mask_map: dict[int, np.ndarray] | None = None
        if mask_base_right and calib is not None:
            # Per-video dk from config: calibration.dk_by_video, e.g. {"GH42": -5}
            cal_cfg = cfg.get("calibration", {})
            dk_by_video: dict[str, int] = {
                k.upper(): int(v)
                for k, v in cal_cfg.get("dk_by_video", {}).items()
            }

            # Build desired_right per frame and track desired -> left_frame mapping
            desired_right_to_left: dict[int, int] = {}
            per_frame_desired: list[int] = []
            for gf, folder, _mask in masks_left:
                gh_prefix = folder[:4].upper()  # "GH42" from "GH420142_Masks_Npz"
                dk = dk_by_video.get(gh_prefix, 0)
                desired = gf + dk
                desired_right_to_left[desired] = gf
                per_frame_desired.append(desired)

            desired_right = np.array(per_frame_desired, dtype=int)
            print("Loading right masks...")
            masks_right = load_matching_right_masks(
                mask_base_right, folders_right, frames_per_video,
                desired_global_frames=desired_right,
                tolerance=data_cfg.get("right_mask_tolerance", 60),
                n_workers=n_workers,
            )
            # Key by left_frame so metrics.py can look up by frame_num directly
            right_mask_map = {
                desired_right_to_left[gf]: m
                for gf, _fld, m in masks_right
                if gf in desired_right_to_left
            }
        elif mask_base_right:
            print("Right masks requested but no calibration — stereo volume disabled.")

        # ---- Compute metrics ----
        print("\nComputing per-frame metrics...")
        df = compute_mask_properties(masks_left, right_mask_map, calib)

        # ---- Save cache ----
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            print(f"Metrics cached to: {cache_path}")
        else:
            print("No cache path set — metrics not saved.")

    # ---- Write frames_per_video back to config YAML so future runs skip re-reading videos ----
    if config_path and frames_per_video:
        update_yaml_field(config_path, ["data", "frames_per_video"], list(frames_per_video))

    # ---- Diagnostic plot ----
    if cache_path:
        fps = cfg.get("session", {}).get("fps", 60)
        _save_diagnostic(df, Path(cache_path), data_cfg.get("weight_csv"), fps=fps)

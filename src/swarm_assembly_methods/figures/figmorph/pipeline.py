"""
Morphology figures pipeline entry point.

Reads a figures config (configs/figures/figmorph/*.yaml) which references
the pre-computed metrics parquet and contains all visualization settings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from swarm_assembly_methods.morphology.alignment import (
    align_time_axes, folder_local_to_global,
)
from swarm_assembly_methods.utils import resolve_save_dir, resolve_session_paths, update_yaml_field
from .figures import generate_all_figures, load_bee_counts


def run_morphology_figures(cfg: dict, config_path=None) -> None:
    """Generate morphology figures from a figures config dict."""
    sess_cfg   = cfg.get("session", {})
    inp        = cfg.get("input", {})
    phases_cfg = cfg.get("phases", {})
    mask_fps   = float(sess_cfg.get("fps", 60))

    # ---- Resolve paths from session params, then apply explicit input overrides ----
    raw_root       = cfg.get("raw_root")
    processed_root = cfg.get("processed_root", "data")
    figures_root   = cfg.get("figures_root", "figures")
    subject        = str(sess_cfg.get("subject", ""))
    date           = str(sess_cfg.get("date", ""))
    stereo_side    = str(sess_cfg.get("stereo_side", ""))

    run_id = sess_cfg.get("run_id") or None
    if raw_root and subject and date and stereo_side:
        auto_paths = resolve_session_paths(raw_root, processed_root, figures_root, subject, date, stereo_side, run_id)
    else:
        auto_paths = {}

    def _path(key: str) -> str | None:
        """Return explicit input override if set, otherwise auto-derived path."""
        return inp.get(key) or auto_paths.get(key)

    # ---- Load metrics parquet ----
    cache_path = _path("metrics_cache")
    if not cache_path or not Path(cache_path).exists():
        raise FileNotFoundError(
            f"metrics_cache not found: {cache_path!r}\n"
            "Run the morphology compute pipeline first."
        )
    df = pd.read_parquet(cache_path)
    print(f"Loaded metrics: {len(df)} rows, "
          f"frames [{df['frame_num'].min()}, {df['frame_num'].max()}]")

    # ---- Load weight data ----
    weight_df = None
    weight_csv = _path("weight_csv")
    if weight_csv and Path(weight_csv).exists():
        weight_df = pd.read_csv(weight_csv)
        weight_df["weight"] = weight_df["weight"] - weight_df["weight"].min()
        print(f"Loaded weight data: {len(weight_df)} rows")
    else:
        print("No weight CSV — weight overlay disabled.")

    # ---- folders / frames_per_video (needed for local→global conversion) ----
    folders_left     = inp.get("folders_left") or []
    frames_per_video = inp.get("frames_per_video") or []

    if not frames_per_video:
        # Try reading frame counts from video files directly
        video_base = _path("video_base_left")
        video_ext  = inp.get("video_ext", ".MP4")
        if video_base and folders_left:
            import cv2
            print("Reading frames_per_video from video files...")
            counts = []
            for folder in folders_left:
                vid_name = folder.replace("_Masks_Npz", "") + video_ext
                vid_path = str(Path(video_base) / vid_name)
                cap = cv2.VideoCapture(vid_path)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                counts.append(n)
                print(f"  {folder}: {n} frames")
            frames_per_video = counts

    def _local_to_global(key: str) -> int | None:
        """Resolve a [folder_name, local_frame] spec from phases config."""
        spec = phases_cfg.get(key)
        if not spec:
            return None
        folder_name, local_frame = str(spec[0]), int(spec[1])
        if not folders_left or not frames_per_video:
            raise ValueError(
                f"{key} is set but input.folders_left / input.frames_per_video "
                "are missing in the figures config. Copy them from your morphology "
                "compute config, or comment out the *_local specs and use "
                "mask_t0_frame / *_min instead."
            )
        try:
            gf = folder_local_to_global(folders_left, folder_name,
                                        local_frame, frames_per_video)
            print(f"  {key}: {folder_name} frame {local_frame} → global {gf}")
            return gf
        except ValueError as e:
            raise ValueError(
                f"{key} references folder {folder_name!r} not found in folders_left."
            ) from e

    # ---- Resolve phase boundaries ----
    # Priority for each: *_local > explicit frame/min > mass-based auto detection
    # t0 is resolved after align_time_axes (mass alignment may update mask_t0_frame)

    # disassembly start (t0) — initial value before mass alignment
    mask_t0_frame = _local_to_global("disassembly_start_local")
    if mask_t0_frame is None:
        raw = phases_cfg.get("mask_t0_frame")
        mask_t0_frame = int(raw) if (raw and raw != "auto") else 0
    print(f"Disassembly start (pre-alignment): frame {mask_t0_frame}")

    # Check for manual *_local overrides (highest priority — always respected)
    dis_end_frame_manual   = _local_to_global("disassembly_end_local")
    asm_start_frame_manual = _local_to_global("assembly_start_local")

    # Check for explicit frame/min overrides (second priority)
    def _explicit_dis_end(t0):
        dis_end_min = phases_cfg.get("disassembly_end_min")
        if dis_end_min is not None and dis_end_min != "auto":
            return t0 + int(float(dis_end_min) * mask_fps * 60)
        return None

    def _explicit_asm_start(t0):
        asm_start_min = phases_cfg.get("assembly_start_min")
        if asm_start_min is not None and asm_start_min != "auto":
            return t0 + int(float(asm_start_min) * mask_fps * 60)
        return None

    # ---- Align time axes ----
    # Mass-based auto mode: updates mask_t0_frame and returns mass phase boundaries.
    df, weight_df, mask_t0_frame, mass_phases = align_time_axes(
        df, weight_df, mask_t0_frame, mask_fps, cfg)

    # Resolve dis_end and asm_start in priority order:
    #   1. *_local manual override
    #   2. explicit min value from config (recomputed with updated mask_t0_frame)
    #   3. mass-based auto detection
    dis_end_frame   = dis_end_frame_manual
    asm_start_frame = asm_start_frame_manual

    if dis_end_frame is None:
        dis_end_frame = _explicit_dis_end(mask_t0_frame)
    if asm_start_frame is None:
        asm_start_frame = _explicit_asm_start(mask_t0_frame)

    if mass_phases is not None:
        if dis_end_frame is None and mass_phases["dis_end_s"] is not None:
            dis_end_frame = mask_t0_frame + int(round(mass_phases["dis_end_s"] * mask_fps))
            print(f"Mass-based disassembly_end_frame = {dis_end_frame}")
        if asm_start_frame is None and mass_phases["asm_start_s"] is not None:
            asm_start_frame = mask_t0_frame + int(round(mass_phases["asm_start_s"] * mask_fps))
            print(f"Mass-based assembly_start_frame = {asm_start_frame}")

    asm_end_min = None
    if mass_phases is not None and mass_phases.get("asm_end_s") is not None:
        asm_end_min = mass_phases["asm_end_s"] / 60.0
        print(f"Mass-based assembly_end_min = {asm_end_min:.2f}")

    phases = dict(
        mask_t0_frame=mask_t0_frame,
        disassembly_end_frame=dis_end_frame,
        assembly_start_frame=asm_start_frame,
        assembly_end_min=asm_end_min,
    )

    print(f"Time range: [{df['time_min'].min():.2f}, {df['time_min'].max():.2f}] min")
    if weight_df is not None:
        print(f"Weight time range: [{weight_df['time_min'].min():.2f}, {weight_df['time_min'].max():.2f}] min")
    print(f"mask_t0_frame={mask_t0_frame}  dis_end_frame={dis_end_frame}  asm_start_frame={asm_start_frame}")
    if dis_end_frame:
        print(f"  → dis_end_min={(dis_end_frame - mask_t0_frame)/mask_fps/60:.2f}  asm_start_min={(asm_start_frame - mask_t0_frame)/mask_fps/60:.2f}" if asm_start_frame else "")

    # ---- Load bee detection data (optional) ----
    bee_df = None
    bee_base_l = _path("bee_detect_base_left")
    bee_base_r = _path("bee_detect_base_right")
    folders_r  = inp.get("folders_right", [])
    # Use the already-resolved folders_left and frames_per_video
    if bee_base_l or bee_base_r:
        bee_left  = load_bee_counts(bee_base_l,  folders_left, frames_per_video) if bee_base_l else None
        bee_right = load_bee_counts(bee_base_r,  folders_r,    frames_per_video) if bee_base_r else None
        if bee_left is not None and bee_right is not None:
            bee_df = pd.merge(bee_left, bee_right, on="global_frame",
                              how="outer", suffixes=("_L", "_R"))
            bee_df["count"] = bee_df[["count_L", "count_R"]].mean(axis=1)
        else:
            bee_df = bee_left if bee_left is not None else bee_right

    # ---- Resolve overlay times (local folder+seconds → global frames) ----
    overlay_local = cfg.get("overlay_times_local") or []
    if overlay_local:
        resolved_overlay_frames = []
        for spec in overlay_local:
            folder_name, t_s = str(spec[0]), float(spec[1])
            local_frame = int(round(t_s * mask_fps))
            try:
                gf = folder_local_to_global(folders_left, folder_name,
                                            local_frame, frames_per_video)
                resolved_overlay_frames.append(gf)
                print(f"  overlay: {folder_name} @ {t_s:.1f}s → local frame {local_frame} → global {gf}")
            except ValueError as e:
                print(f"  WARNING: overlay {spec} — {e}")
        cfg["overlay_frames_resolved"] = resolved_overlay_frames
    elif cfg.get("overlay_times_min"):
        # Fallback: convert minutes-from-t0 to global frames
        cfg["overlay_frames_resolved"] = [
            mask_t0_frame + int(round(float(t) * mask_fps * 60))
            for t in cfg["overlay_times_min"]
        ]

    # ---- Generate figures ----
    # Inject all resolved paths into cfg so generate_all_figures can use them
    cfg.setdefault("input", {})
    cfg["input"]["folders_left"]     = list(folders_left)
    cfg["input"]["frames_per_video"] = list(frames_per_video)
    cfg["input"]["mask_base_left"]   = _path("mask_base_left")
    # Inject save_dir so generate_all_figures uses the auto-derived path
    # (only if not already set explicitly in plots)
    cfg.setdefault("plots", {})
    if not cfg["plots"].get("save_dir"):
        cfg["plots"]["save_dir"] = _path("save_dir")

    print("\nGenerating figures...")
    generate_all_figures(
        df=df,
        weight_df=weight_df,
        mask_t0_frame=mask_t0_frame,
        mask_fps=mask_fps,
        phases=phases,
        cfg=cfg,
        bee_df=bee_df,
    )

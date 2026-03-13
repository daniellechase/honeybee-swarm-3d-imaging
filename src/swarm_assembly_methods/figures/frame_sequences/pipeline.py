"""
Frame-sequence figure pipeline.

Reads a YAML config and produces strip figures (assembly and/or disassembly)
without writing intermediate frame images to disk.
"""

from __future__ import annotations

from pathlib import Path

from swarm_assembly_methods.utils import resolve_save_dir
from .extract import extract_frames
from .strip import build_strip, save_strip


def run_frame_sequence_figures(cfg: dict, config_path=None) -> None:
    """Generate frame-strip figures from config dict."""

    save_dir = resolve_save_dir(cfg, "save_dir", "figures/frame_sequences")
    save_dir.mkdir(parents=True, exist_ok=True)

    strip_cfg = cfg.get("strip", {})
    fig_width_in = float(strip_cfg.get("fig_width_in", 6.5))
    dpi          = int(strip_cfg.get("dpi", 300))
    pad_x        = int(strip_cfg.get("pad_x", 12))
    pad_top      = int(strip_cfg.get("pad_top", 12))
    pad_bottom   = int(strip_cfg.get("pad_bottom", 260))
    font_size    = int(strip_cfg.get("font_size", 182))
    seek_back    = float(strip_cfg.get("seek_back", 0.5))

    for name, seq_cfg in cfg.get("sequences", {}).items():
        print(f"\n=== Sequence: {name} ===")
        video_path = Path(seq_cfg["video"])
        times      = seq_cfg["times"]
        label_unit = seq_cfg.get("label_unit", "min")   # "min" or "sec"
        outputs    = seq_cfg.get("outputs", [f"{name}.pdf"])

        print(f"  Video: {video_path}")
        frames = extract_frames(video_path, times, seek_back=seek_back)

        strip = build_strip(
            frames,
            label_unit=label_unit,
            fig_width_in=fig_width_in,
            dpi=dpi,
            pad_x=pad_x,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            font_size=font_size,
        )

        for out_name in outputs:
            save_strip(strip, save_dir / out_name, dpi=dpi)

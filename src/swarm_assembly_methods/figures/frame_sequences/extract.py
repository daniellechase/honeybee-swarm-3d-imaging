"""
Extract single frames from a video at specified timestamps, in memory.

Returns PIL Images — no files written to disk.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

from PIL import Image


def get_video_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _parse(rate: str) -> float | None:
        rate = rate.strip()
        if not rate or rate == "0/0":
            return None
        if "/" in rate:
            n, d = rate.split("/")
            d = float(d)
            return float(n) / d if d else None
        return float(rate)

    avg = raw = None
    for line in res.stdout.splitlines():
        if line.startswith("avg_frame_rate="):
            avg = _parse(line.split("=", 1)[1])
        elif line.startswith("r_frame_rate="):
            raw = _parse(line.split("=", 1)[1])

    fps = avg or raw
    if fps is None:
        raise RuntimeError(f"Could not determine FPS for {video_path}")
    return fps


def time_to_seconds(t: str) -> float:
    """Supports HH:MM:SS or HH:MM:SS.mmm"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract_frames(
    video_path: str | Path,
    times: list[str],
    seek_back: float = 0.5,
) -> list[tuple[Image.Image, float]]:
    """
    Extract one frame per timestamp from video_path.

    Parameters
    ----------
    video_path : path to video file
    times      : list of timestamp strings "HH:MM:SS" or "HH:MM:SS.mmm"
    seek_back  : seconds to seek before target for reliable keyframe decode

    Returns
    -------
    List of (PIL.Image, t_sec) sorted by t_sec.
    """
    video_path = Path(video_path)
    fps = get_video_fps(video_path)
    print(f"  Video FPS: {fps:.6f}  ({video_path.name})")

    results: list[tuple[Image.Image, float]] = []

    for t_str in times:
        t_sec = time_to_seconds(t_str)
        ss_time = max(0.0, t_sec - seek_back)

        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-ss", f"{ss_time:.6f}",
            "-i", str(video_path),
            "-vf", f"select=gte(t\\,{seek_back:.6f})",
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "pipe:1",
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for t={t_str}:\n{res.stderr.decode()}"
            )
        img = Image.open(io.BytesIO(res.stdout)).convert("RGB")
        results.append((img, t_sec))
        print(f"  Extracted frame @ {t_str} ({t_sec:.3f} s)")

    results.sort(key=lambda x: x[1])
    return results

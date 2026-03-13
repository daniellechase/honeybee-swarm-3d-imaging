"""
Build a horizontal strip of frames with time labels below each panel.

Works entirely in memory — no intermediate files written.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _format_label(elapsed: float, unit: str) -> str:
    """Format elapsed time as '0 min', '2.0 min', '24 sec', etc."""
    if abs(elapsed - round(elapsed)) < 1e-6:
        return f"{int(round(elapsed))} {unit}"
    return f"{elapsed:.1f} {unit}"


def build_strip(
    frames: list[tuple[Image.Image, float]],
    label_unit: str = "min",       # "min" or "sec"
    fig_width_in: float = 6.5,
    dpi: int = 300,
    pad_x: int = 12,
    pad_top: int = 12,
    pad_bottom: int = 260,
    font_size: int = 26 * 7,
    text_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
) -> Image.Image:
    """
    Build a horizontal strip from a list of (PIL.Image, t_sec) pairs.

    Parameters
    ----------
    frames     : list of (image, absolute_time_seconds), sorted by time
    label_unit : "min" labels elapsed minutes; "sec" labels elapsed seconds
    fig_width_in, dpi : control output pixel width
    pad_*      : pixel spacing at DPI resolution
    """
    if not frames:
        raise ValueError("frames list is empty")

    images = [f[0] for f in frames]
    t_secs = [f[1] for f in frames]
    t0 = t_secs[0]

    if label_unit == "min":
        elapsed = [(t - t0) / 60.0 for t in t_secs]
    else:
        elapsed = [t - t0 for t in t_secs]

    n = len(images)
    w0, h0 = images[0].size
    for i, im in enumerate(images[1:], 1):
        if im.size != (w0, h0):
            raise RuntimeError(
                f"Frame size mismatch at index {i}: {im.size} vs {(w0, h0)}"
            )

    target_w_px = int(round(fig_width_in * dpi))
    available = target_w_px - (n - 1) * pad_x
    if available <= 0:
        raise RuntimeError("Not enough width; reduce pad_x or increase fig_width_in.")

    scale = available / (n * w0)
    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    print(f"  Resizing {w0}×{h0} → {new_w}×{new_h}  (scale={scale:.4f})")

    images_rs = [im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
                 for im in images]

    strip_w = n * new_w + (n - 1) * pad_x
    strip_h = pad_top + new_h + pad_bottom
    strip = Image.new("RGB", (strip_w, strip_h), bg_color)

    x = 0
    for im in images_rs:
        strip.paste(im, (x, pad_top))
        x += new_w + pad_x

    draw = ImageDraw.Draw(strip)
    font = _load_font(font_size)

    for i, e in enumerate(elapsed):
        x_center = i * (new_w + pad_x) + new_w // 2
        y_text = pad_top + new_h + pad_bottom // 2
        txt = _format_label(e, label_unit)
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x_center - tw // 2, y_text - th // 2), txt,
                  fill=text_color, font=font)

    return strip


def save_strip(strip: Image.Image, out_path: str | Path, dpi: int = 300) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = out_path.suffix.lstrip(".").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt == "PDF":
        strip.save(str(out_path), "PDF", resolution=float(dpi))
    else:
        strip.save(str(out_path), fmt)
    print(f"  Saved: {out_path}")

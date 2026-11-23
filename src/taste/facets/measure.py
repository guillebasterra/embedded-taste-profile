"""Local pixel statistics. Free, deterministic, no model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel

MEASURE_EDGE = 256
PALETTE_COLORS = 6


class Swatch(BaseModel):
    hex: str
    share: float  # fraction of pixels


class Measures(BaseModel):
    pin_id: str
    palette: list[Swatch]  # most common first
    lightness: float  # mean luma, 0 dark .. 1 light
    contrast: float  # std of luma, 0 flat .. ~0.5 hard
    saturation: float  # mean HSV saturation, 0 grey .. 1 vivid
    colorfulness: float  # Hasler-Suesstrunk, scaled to roughly 0..1
    warmth: float  # mean (R - B), -1 cool .. 1 warm
    busyness: float  # mean edge strength, 0 empty .. 1 dense detail
    aspect: float  # width / height of the original


def _load(path: Path) -> tuple[Image.Image, float]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        aspect = im.width / im.height if im.height else 1.0
        if im.mode in ("RGBA", "LA", "P"):
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[-1])
            im = flat
        else:
            im = im.convert("RGB")
        im.thumbnail((MEASURE_EDGE, MEASURE_EDGE), Image.LANCZOS)
        return im.copy(), aspect


def palette(im: Image.Image, n: int = PALETTE_COLORS) -> list[Swatch]:
    q = im.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    flat = q.getpalette() or []
    counts = sorted(((c, i) for c, i in q.getcolors() or []), reverse=True)
    total = sum(c for c, _ in counts) or 1
    out = []
    for count, idx in counts:
        r, g, b = flat[idx * 3: idx * 3 + 3]
        out.append(Swatch(hex=f"#{r:02x}{g:02x}{b:02x}", share=round(count / total, 3)))
    return out


def measure(pin_id: str, path: Path) -> Measures:
    im, aspect = _load(path)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    luma = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    sat = np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    edges = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edges = edges[1:-1, 1:-1]  # the filter lights up the frame border
    return Measures(
        pin_id=pin_id,
        palette=palette(im),
        lightness=round(float(luma.mean()), 3),
        contrast=round(float(luma.std()), 3),
        saturation=round(float(sat.mean()), 3),
        colorfulness=round(min(1.0, float(colorful) / 0.6), 3),
        warmth=round(float((r - b).mean()), 3),
        busyness=round(min(1.0, float(edges.mean()) * 4), 3),
        aspect=round(aspect, 3),
    )

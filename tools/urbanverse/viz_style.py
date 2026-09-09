#!/usr/bin/env python3
"""Small, dependency-free drawing helpers shared by evidence visualizers."""

from __future__ import annotations

from pathlib import Path

from PIL import ImageDraw, ImageFont


_REGULAR_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
_BOLD_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


def load_font(size: int, *, bold: bool = False):
    candidates = _BOLD_FONT_CANDIDATES if bold else _REGULAR_FONT_CANDIDATES
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    *,
    padding: int = 4,
    foreground: tuple[int, int, int] = (245, 247, 250),
    background: tuple[int, int, int] = (26, 31, 38),
) -> tuple[int, int, int, int]:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    rect = (box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding)
    draw.rectangle(rect, fill=background)
    draw.text((x, y), text, fill=foreground, font=font)
    return rect

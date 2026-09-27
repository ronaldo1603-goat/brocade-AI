"""Drawing detections on photographs (PIL only - works in scripts, notebooks and Gradio)."""
from __future__ import annotations

import colorsys
from functools import lru_cache
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from .base import ImageInput, load_image
from .geometry import Tile, YoloRow, yolo_to_xyxy

# the labs' colour-blind-safe palette, then golden-ratio hues for the other classes
LAB_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"]
GT_COLOR = "#ff2bd6"      # ground truth: dashed magenta (not in the class palette)
TILE_COLOR = "#ffffff"


@lru_cache(maxsize=None)
def class_color(class_id: int) -> str:
    """Stable, distinct colour per class id."""
    if class_id < len(LAB_PALETTE):
        return LAB_PALETTE[class_id]
    h = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


@lru_cache(maxsize=8)
def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy, text: str, color: str, font, img_h: int):
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    pad = max(1, th // 5)
    y0 = y - th - 2 * pad if y - th - 2 * pad >= 0 else y      # inside the box at the top edge
    draw.rectangle((x, y0, x + tw + 2 * pad, y0 + th + 2 * pad), fill=color)
    draw.text((x + pad, y0 + pad - t), text, fill="white", font=font)


def draw_detections(image: ImageInput, detections: Iterable, *, show_labels: bool = True,
                    show_scores: bool = True, tiles: Sequence[Tile] | None = None,
                    gt_rows: Sequence[YoloRow] | None = None, line_width: int | None = None,
                    max_side: int | None = 1600) -> Image.Image:
    """Return a copy of ``image`` with detections drawn on it.

    Args:
        detections: a ``DetectionResult`` or any iterable of ``Detection``.
        tiles: draw the tile grid used at inference (teaching aid for tiling).
        gt_rows: optional ground truth as YOLO rows, drawn dashed-magenta underneath.
        max_side: downscale large photos first so lines and labels stay readable
            (a 4896 px photo shown at 800 px would make 2 px lines invisible).
    """
    img = load_image(image).copy()
    W0, H0 = img.size
    scale = 1.0
    if max_side and max(W0, H0) > max_side:
        scale = max_side / max(W0, H0)
        img = img.resize((round(W0 * scale), round(H0 * scale)), Image.BILINEAR)
    W, H = img.size
    lw = line_width or max(2, round(max(W, H) / 500))
    font = _font(max(11, round(max(W, H) / 90)))
    draw = ImageDraw.Draw(img)

    if tiles:
        for t in tiles:
            draw.rectangle([v * scale for v in t.box], outline=TILE_COLOR, width=max(1, lw // 2))

    if gt_rows:
        for _, cx, cy, w, h in gt_rows:
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
            _dashed_rect(draw, (x1, y1, x2, y2), GT_COLOR, max(1, lw // 2 + 1))

    dets = sorted(detections, key=lambda d: d.score)       # best on top
    for d in dets:
        box = [v * scale for v in d.box]
        color = class_color(d.class_id)
        draw.rectangle(box, outline=color, width=lw)
        if show_labels:
            text = f"{d.class_name} {d.score:.2f}" if show_scores else d.class_name
            _label(draw, (box[0], box[1]), text, color, font, H)
    return img


def _dashed_rect(draw, box, color, width, dash=8):
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        n = max(1, int(length // dash))
        for k in range(0, n, 2):
            s, e = k / n, min(1.0, (k + 1) / n)
            draw.line((ax + (bx - ax) * s, ay + (by - ay) * s,
                       ax + (bx - ax) * e, ay + (by - ay) * e), fill=color, width=width)


def draw_ground_truth(image: ImageInput, rows: Sequence[YoloRow], class_names: list[str],
                      max_side: int | None = 1600) -> Image.Image:
    """Draw YOLO label rows with their class names (for label sanity checks)."""
    from .detector import Detection

    img = load_image(image)
    W, H = img.size
    dets = [Detection(yolo_to_xyxy(cx, cy, w, h, W, H), c, class_names[c], 1.0)
            for c, cx, cy, w, h in rows]
    return draw_detections(img, dets, show_scores=False, max_side=max_side)

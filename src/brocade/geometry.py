"""Box geometry shared by training notebooks and the inference package.

Everything here is the notebooks' code lifted out of the cells so that a model and
the code that decodes it can never drift apart. Two flavours of NMS live side by
side on purpose:

* ``iou`` / ``nms``           - the readable pure-Python versions from the mini-YOLO
                                lab (TODO 1 and TODO 2). Use them to learn and to test.
* ``nms_fast``                 - the same algorithm on top of ``torchvision.ops``; the
                                predictors use it because a tiled photograph can
                                produce thousands of candidate boxes.

Coordinate conventions used throughout the package:

* **YOLO**  ``(cx, cy, w, h)`` normalised to ``[0, 1]`` - the dataset label format.
* **xyxy**  ``(x1, y1, x2, y2)`` corners, either normalised or in pixels; functions
            say which in their docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

Box = tuple[float, float, float, float]
YoloRow = tuple[int, float, float, float, float]  # (class_id, cx, cy, w, h)


# --------------------------------------------------------------------------- #
# Format conversion                                                           #
# --------------------------------------------------------------------------- #
def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: float, H: float) -> Box:
    """Normalised YOLO box -> pixel corners, clamped to the image (Lab 4, TODO 1)."""
    x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
    x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
    return max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, W: float, H: float) -> Box:
    """Pixel corners -> normalised YOLO ``(cx, cy, w, h)``. Inverse of ``yolo_to_xyxy``."""
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def parse_yolo_file(path) -> list[YoloRow]:
    """Read one YOLO label file. A missing or empty file yields ``[]``."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return []
    rows: list[YoloRow] = []
    for line in path.read_text().strip().splitlines():
        p = line.split()
        if len(p) >= 5:
            rows.append((int(p[0]), *map(float, p[1:5])))
    return rows


# --------------------------------------------------------------------------- #
# IoU and NMS                                                                 #
# --------------------------------------------------------------------------- #
def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two xyxy boxes (any consistent units). Returns a float in [0, 1]."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes: Sequence[Sequence[float]], scores: Sequence[float],
        iou_threshold: float = 0.4) -> list[int]:
    """Greedy NMS, pure Python. Returns kept indices in decreasing-score order."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order if iou(boxes[best], boxes[i]) < iou_threshold]
    return keep


def nms_fast(boxes, scores, iou_threshold: float = 0.4, class_ids=None) -> list[int]:
    """Vectorised NMS (torchvision). Same contract as :func:`nms`.

    With ``class_ids`` given, suppression happens only *within* a class
    (``batched_nms``); without, it is class-agnostic like the lab's version.

    Note the one semantic difference: torchvision suppresses boxes with
    IoU **>** threshold, the lab's loop suppresses IoU **>=** threshold. They only
    disagree on exact ties, which never matter on real predictions.
    """
    from torchvision.ops import batched_nms
    from torchvision.ops import nms as tv_nms

    if len(boxes) == 0:
        return []
    b = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    s = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    if class_ids is None:
        keep = tv_nms(b, s, iou_threshold)
    else:
        keep = batched_nms(b, s, torch.as_tensor(class_ids, dtype=torch.int64), iou_threshold)
    return keep.tolist()


# --------------------------------------------------------------------------- #
# The mini-YOLO grid: encode (training) and decode (inference)                #
# --------------------------------------------------------------------------- #
def encode_targets(rows: Iterable[YoloRow], S: int, num_classes: int) -> torch.Tensor:
    """Place ground-truth boxes on an S x S grid -> tensor ``(5 + C, S, S)``.

    channel 0: objectness · 1-4: tx, ty (offset in cell), w, h (tile fractions)
    · 5+: one-hot class. Collision rule (B = 1): the larger box keeps the cell.
    """
    t = torch.zeros(5 + num_classes, S, S)
    for c, cx, cy, w, h in rows:
        gx = min(int(cx * S), S - 1)          # clamp: cx == 1.0 would index S
        gy = min(int(cy * S), S - 1)
        if t[0, gy, gx] == 1:
            if w * h <= t[3, gy, gx] * t[4, gy, gx]:
                continue
            t[5:, gy, gx] = 0
        t[0, gy, gx] = 1.0
        t[1, gy, gx] = cx * S - gx
        t[2, gy, gx] = cy * S - gy
        t[3, gy, gx] = w
        t[4, gy, gx] = h
        t[5 + int(c), gy, gx] = 1.0
    return t


def activate(pred: torch.Tensor) -> torch.Tensor:
    """Raw mini-YOLO head output ``(5 + C, S, S)`` -> objectness and box in [0, 1].

    Class logits are left raw: decoding only needs their argmax.
    """
    return torch.cat([torch.sigmoid(pred[:5]), pred[5:]], dim=0)


def decode_grid(grid: torch.Tensor, S: int | None = None, conf_threshold: float = 0.35
                ) -> tuple[list[Box], list[float], list[int]]:
    """Inverse of :func:`encode_targets`. ``grid`` must already be activated.

    Returns ``(boxes, scores, class_ids)`` with boxes as normalised xyxy, clipped
    to ``[0, 1]``. ``S`` defaults to the grid's own spatial size.
    """
    S = S or grid.shape[-1]
    boxes: list[Box] = []
    scores: list[float] = []
    class_ids: list[int] = []
    for gy in range(S):
        for gx in range(S):
            conf = float(grid[0, gy, gx])
            if conf < conf_threshold:
                continue
            tx, ty, w, h = (float(v) for v in grid[1:5, gy, gx])
            cx, cy = (gx + tx) / S, (gy + ty) / S
            boxes.append((max(0.0, cx - w / 2), max(0.0, cy - h / 2),
                          min(1.0, cx + w / 2), min(1.0, cy + h / 2)))
            scores.append(conf)
            class_ids.append(int(grid[5:, gy, gx].argmax()))
    return boxes, scores, class_ids


# --------------------------------------------------------------------------- #
# Tiling                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tile:
    """A rectangular window on the source photograph, in pixels."""
    x0: float
    y0: float
    x1: float
    y1: float
    row: int = 0
    col: int = 0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def box(self) -> Box:
        return (self.x0, self.y0, self.x1, self.y1)


def auto_tile_grid(W: int, H: int, tiles_on_long_side: int = 4) -> tuple[int, int]:
    """Pick ``(tiles_x, tiles_y)`` so tiles are close to square.

    The labs cut a 4896x3672 photo into 4x3 tiles of 1224x1224. ``auto`` keeps that
    tile *shape* for any resolution or orientation: 4 tiles along the long side and
    as many as fit on the short side (a portrait 1944x2592 becomes 3x4).
    """
    side = max(W, H) / tiles_on_long_side
    return max(1, round(W / side)), max(1, round(H / side))


def make_tiles(W: int, H: int, tiles_x: int, tiles_y: int, overlap: float = 0.0) -> list[Tile]:
    """Split a ``W x H`` image into a ``tiles_x x tiles_y`` grid of windows.

    ``overlap`` (fraction of a tile, 0 <= overlap < 1) grows every tile so that
    neighbours share a margin. With ``overlap=0`` this is exactly the labs' grid.
    Overlap recovers motifs cut in half by a tile border; the duplicates it
    creates are removed later by NMS over the whole photograph.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    tw, th = W / tiles_x, H / tiles_y
    mx, my = tw * overlap / 2, th * overlap / 2
    tiles = []
    for r in range(tiles_y):
        for c in range(tiles_x):
            x0, y0 = c * tw - mx, r * th - my
            x1, y1 = (c + 1) * tw + mx, (r + 1) * th + my
            tiles.append(Tile(max(0.0, x0), max(0.0, y0), min(float(W), x1),
                              min(float(H), y1), r, c))
    return tiles


def clip_rows_to_tile(rows: Iterable[YoloRow], W: int, H: int, tile: Tile,
                      min_frac: float = 0.25) -> list[YoloRow]:
    """Re-express image-level YOLO labels in the coordinates of one tile.

    Boxes are clipped to the tile; a box keeping less than ``min_frac`` of its
    area inside the tile is dropped (the labs use 0.25). Returns YOLO rows
    normalised to the tile.
    """
    kept: list[YoloRow] = []
    for c, cx, cy, w, h in rows:
        bx1, by1, bx2, by2 = yolo_to_xyxy(cx, cy, w, h, W, H)
        ix1, iy1 = max(bx1, tile.x0), max(by1, tile.y0)
        ix2, iy2 = min(bx2, tile.x1), min(by2, tile.y1)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) < min_frac * (bx2 - bx1) * (by2 - by1):
            continue
        kept.append((int(c), ((ix1 + ix2) / 2 - tile.x0) / tile.width,
                     ((iy1 + iy2) / 2 - tile.y0) / tile.height,
                     (ix2 - ix1) / tile.width, (iy2 - iy1) / tile.height))
    return kept


def tile_to_image(box_norm: Sequence[float], tile: Tile) -> Box:
    """Normalised xyxy inside ``tile`` -> pixel xyxy on the source photograph."""
    x1, y1, x2, y2 = box_norm
    return (tile.x0 + x1 * tile.width, tile.y0 + y1 * tile.height,
            tile.x0 + x2 * tile.width, tile.y0 + y2 * tile.height)

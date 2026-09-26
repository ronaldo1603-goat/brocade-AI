"""The lab's hand-checkable cases, plus the properties the inference path relies on."""
import random

import pytest
import torch

from brocade.geometry import (auto_tile_grid, clip_rows_to_tile, decode_grid, encode_targets,
                              iou, make_tiles, nms, nms_fast, tile_to_image, xyxy_to_yolo,
                              yolo_to_xyxy)


def test_iou_hand_cases():
    assert abs(iou((0, 0, 2, 2), (1, 1, 3, 3)) - 1 / 7) < 1e-9
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert iou((0, 0, 1, 1), (1, 0, 2, 1)) == 0.0          # touching edges


def test_nms_hand_cases():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
    assert nms(boxes, [0.9, 0.8, 0.7], 0.4) == [0, 2]
    assert nms(boxes, [0.8, 0.9, 0.7], 0.4) == [1, 2]
    assert nms(boxes, [0.9, 0.8, 0.7], 0.95) == [0, 1, 2]


def test_nms_fast_matches_reference():
    rng = random.Random(0)
    for _ in range(20):
        boxes, scores = [], []
        for _ in range(60):
            x, y = rng.uniform(0, 80), rng.uniform(0, 80)
            boxes.append((x, y, x + rng.uniform(5, 25), y + rng.uniform(5, 25)))
            scores.append(rng.random())
        assert nms_fast(boxes, scores, 0.45) == nms(boxes, scores, 0.45)


def test_nms_fast_class_aware():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11)]
    assert nms_fast(boxes, [0.9, 0.8], 0.4) == [0]
    assert sorted(nms_fast(boxes, [0.9, 0.8], 0.4, class_ids=[0, 1])) == [0, 1]
    assert nms_fast([], [], 0.4) == []


def test_encode_decode_roundtrip_and_edge():
    rows = [(3, 0.55, 0.30, 0.20, 0.10), (7, 0.10, 0.90, 0.15, 0.20),
            (1, 1.0, 1.0, 0.1, 0.1)]                           # cx == 1.0 must not crash
    t = encode_targets(rows, 7, 28)
    b, s, c = decode_grid(t, 7, 0.5)
    assert sorted(c) == [1, 3, 7]
    for _, cx, cy, w, h in rows[:2]:
        target = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        assert max(iou(target, x) for x in b) > 0.999


def test_encode_collision_keeps_larger():
    t = encode_targets([(3, 0.51, 0.51, 0.1, 0.1), (7, 0.55, 0.55, 0.3, 0.3)], 2, 8)
    assert t[0].sum() == 1 and t[5 + 7, 1, 1] == 1 and t[5 + 3].sum() == 0


def test_yolo_xyxy_inverse():
    x = yolo_to_xyxy(0.5, 0.4, 0.2, 0.3, 400, 200)
    assert pytest.approx(xyxy_to_yolo(*x, 400, 200)) == (0.5, 0.4, 0.2, 0.3)


def test_auto_tile_grid_matches_labs():
    assert auto_tile_grid(4896, 3672) == (4, 3)           # the labs' grid
    assert auto_tile_grid(1944, 2592) == (3, 4)           # portrait photo
    assert auto_tile_grid(4160, 3120) == (4, 3)


@pytest.mark.parametrize("overlap", [0.0, 0.2])
def test_tiles_cover_image(overlap):
    tiles = make_tiles(4896, 3672, 4, 3, overlap)
    assert len(tiles) == 12
    assert min(t.x0 for t in tiles) == 0 and max(t.x1 for t in tiles) == 4896
    if overlap == 0:
        assert all(abs(t.width - 1224) < 1e-6 and abs(t.height - 1224) < 1e-6 for t in tiles)
    else:
        assert tiles[1].x0 < tiles[0].x1                  # neighbours share a margin


def test_clip_then_map_back_is_identity():
    W, H = 4896, 3672
    rows = [(2, 0.30, 0.40, 0.03, 0.04)]                  # fully inside one tile
    for t in make_tiles(W, H, 4, 3):
        kept = clip_rows_to_tile(rows, W, H, t)
        if kept:
            c, cx, cy, w, h = kept[0]
            back = tile_to_image((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), t)
            assert iou(back, yolo_to_xyxy(*rows[0][1:], W, H)) > 0.9999
            break
    else:
        pytest.fail("box not found in any tile")
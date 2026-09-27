import json

import pytest
from PIL import Image

from brocade.detector import (BaseDetector, DetectionResult, MiniYoloDetector, is_ultralytics_weights,
                              load_detector, parse_tiles)
from brocade.geometry import clip_rows_to_tile, iou, parse_yolo_file, yolo_to_xyxy

from brocade.visualize import draw_detections

from conftest import CLASSES


class OracleDetector(BaseDetector):
    """Returns the ground truth of each tile - isolates tiling/mapping/NMS from any model."""
    backend = "oracle"

    def __init__(self, rows, W, H, **kw):
        self.rows, self.W, self.H, self.classes = rows, W, H, CLASSES
        self._init_detector(conf=0.1, iou=0.5, **kw)
        self._grid = None

    def predict(self, image, **kw):
        from brocade.geometry import make_tiles
        from brocade.detector import parse_tiles as pt
        tiles = kw.get("tiles", self.tiles)
        self._grid = iter(make_tiles(self.W, self.H, *pt(tiles, self.W, self.H),
                                     overlap=kw.get("overlap", self.overlap)))
        return super().predict(image, **kw)

    def detect_tiles(self, crops, conf, iou_thr):
        out = []
        for _ in crops:
            t = next(self._grid)
            kept = clip_rows_to_tile(self.rows, self.W, self.H, t, min_frac=0.999)
            out.append(([(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _, cx, cy, w, h in kept],
                        [0.9] * len(kept), [c for c, *_ in kept]))
        return out




def test_parse_tiles():
    assert parse_tiles("auto", 4896, 3672) == (4, 3)
    assert parse_tiles("auto", 448, 448) == (1, 1)         # a lab tile is not re-tiled
    assert parse_tiles("3x2", 10, 10) == (3, 2)
    assert parse_tiles(None, 10, 10) == (1, 1)
    assert parse_tiles((2, 5), 10, 10) == (2, 5)


def test_miniyolo_detector_runs(dataset, miniyolo_ckpt):
    det = load_detector(miniyolo_ckpt, device="cpu", conf=0.0)   # random weights: keep all cells
    assert isinstance(det, MiniYoloDetector) and det.classes == CLASSES
    res = det(dataset / "images" / "photo_1.jpg")
    assert isinstance(res, DetectionResult) and len(res.tiles) == 12
    assert res.num_candidates > 0 and len(res) <= res.num_candidates
    W, H = res.image_size
    assert all(0 <= d.box[0] <= d.box[2] <= W and 0 <= d.box[1] <= d.box[3] <= H for d in res)
    assert all(d.class_name in CLASSES for d in res)
    vis = draw_detections(dataset / "images" / "photo_1.jpg", res, tiles=res.tiles)
    assert max(vis.size) == 1600
    assert det.benchmark(n=2) > 0


def test_miniyolo_is_not_ultralytics(miniyolo_ckpt):
    assert not is_ultralytics_weights(miniyolo_ckpt)
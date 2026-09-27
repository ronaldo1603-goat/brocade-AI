"""Motif detection on full photographs.

Two backends share one pipeline:

* :class:`MiniYoloDetector`    - the from-scratch grid detector of the mini-YOLO lab.
* :class:`UltralyticsDetector` - a YOLOv8 / YOLO11 model fine-tuned with
  ``notebook/yolo11_finetune_lab.ipynb``.

Why the shared part is *tiling*
-------------------------------
Both detectors were trained on **tiles**: each 4896x3672 photograph was cut into
a 4x3 grid of square 1224x1224 windows, because a photo carries ~47 motifs on
average (and up to 1,600+). A model trained on tiles expects motifs at *tile*
scale, so a full photograph must be cut the same way at inference::

    photo ─▶ make_tiles ─▶ [tile_0 … tile_11] ─▶ detect_tiles (backend-specific)
          ─▶ tile coords → photo coords ─▶ NMS over the whole photo ─▶ DetectionResult

``detect_tiles`` is the only abstract step. Everything else - tiling, coordinate
mapping, the final NMS that removes duplicates at tile borders - lives here once.
"""
from __future__ import annotations

import time
import zipfile
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from .base import BasePredictor, ImageInput, load_image, resolve_device
from .geometry import (Box, Tile, activate, auto_tile_grid, decode_grid, make_tiles,
                       nms_fast, tile_to_image)

TileSpec = str | tuple[int, int] | None
TileOutput = tuple[list[Box], list[float], list[int]]   # normalised xyxy, scores, class ids

#: images whose long side is at most this many pixels are treated as a single tile
#: by ``tiles="auto"`` (the lab's own tiles are 448 px; photos are 2592+ px).
AUTO_TILE_MIN_SIDE = 1024


# --------------------------------------------------------------------------- #
# Results                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    box: Box            # pixel xyxy on the *source* image
    class_id: int
    class_name: str
    score: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = [round(v, 1) for v in self.box]
        d["score"] = round(self.score, 4)
        return d


@dataclass
class DetectionResult:
    image_size: tuple[int, int]                       # (W, H)
    detections: list[Detection]
    tiles: list[Tile] = field(default_factory=list)
    num_candidates: int = 0                           # boxes before the photo-level NMS
    timings_ms: dict[str, float] = field(default_factory=dict)
    backend: str = ""

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def counts(self) -> Counter:
        """Number of detections per motif name, most common first."""
        return Counter(d.class_name for d in self.detections)

    def filter(self, min_score: float) -> "DetectionResult":
        kept = [d for d in self.detections if d.score >= min_score]
        return DetectionResult(self.image_size, kept, self.tiles, self.num_candidates,
                               self.timings_ms, self.backend)

    def to_records(self) -> list[dict]:
        """Flat rows (one per detection) - ready for pandas / a Gradio table / JSON."""
        return [{"class_id": d.class_id, "class_name": d.class_name,
                 "score": round(d.score, 4),
                 "x1": round(d.box[0], 1), "y1": round(d.box[1], 1),
                 "x2": round(d.box[2], 1), "y2": round(d.box[3], 1)}
                for d in self.detections]

    def summary(self) -> str:
        W, H = self.image_size
        top = ", ".join(f"{n} x{c}" for n, c in self.counts().most_common(5)) or "-"
        return (f"{len(self)} motifs on {W}x{H} ({len(self.tiles)} tiles, "
                f"{self.num_candidates} candidates before NMS, "
                f"{self.timings_ms.get('total', 0):.0f} ms) | top: {top}")


# --------------------------------------------------------------------------- #
# Shared pipeline                                                             #
# --------------------------------------------------------------------------- #
def parse_tiles(spec: TileSpec, W: int, H: int) -> tuple[int, int]:
    """``"auto"`` | ``"4x3"`` | ``(4, 3)`` | ``None``/``"none"``/``"1x1"`` -> ``(tiles_x, tiles_y)``."""
    if spec is None or (isinstance(spec, str) and spec.lower() in {"none", "off", "1x1", "1"}):
        return 1, 1
    if isinstance(spec, str) and spec.lower() == "auto":
        if max(W, H) <= AUTO_TILE_MIN_SIDE:
            return 1, 1
        return auto_tile_grid(W, H)
    if isinstance(spec, str):
        nx, ny = spec.lower().replace("×", "x").split("x")
        return int(nx), int(ny)
    nx, ny = spec
    return int(nx), int(ny)


class BaseDetector(ABC):
    """Tiling + coordinate mapping + photo-level NMS. Subclasses implement ``detect_tiles``."""

    backend = "base"
    classes: list[str]

    def _init_detector(self, conf: float, iou: float, tiles: TileSpec = "auto",
                       overlap: float = 0.0, agnostic_nms: bool = True, max_det: int = 3000,
                       tile_batch: int = 16):
        self.conf, self.iou = conf, iou
        self.tiles, self.overlap = tiles, overlap
        self.agnostic_nms, self.max_det = agnostic_nms, max_det
        self.tile_batch = tile_batch

    @abstractmethod
    def detect_tiles(self, crops: list[Image.Image], conf: float, iou: float) -> list[TileOutput]:
        """Detect on each crop. Boxes are *normalised to the crop*, NMS'd per crop."""

    def predict(self, image: ImageInput, conf: float | None = None, iou: float | None = None,
                tiles: TileSpec = "default", overlap: float | None = None,
                agnostic_nms: bool | None = None) -> DetectionResult:
        """Detect motifs on one image. Unset arguments fall back to the constructor's."""
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
        tiles = self.tiles if tiles == "default" else tiles
        overlap = self.overlap if overlap is None else overlap
        agnostic = self.agnostic_nms if agnostic_nms is None else agnostic_nms

        t0 = time.perf_counter()
        img = load_image(image)
        W, H = img.size
        grid = make_tiles(W, H, *parse_tiles(tiles, W, H), overlap=overlap)
        crops = [img.crop(tuple(round(v) for v in t.box)) for t in grid]
        t1 = time.perf_counter()

        per_tile: list[TileOutput] = []
        for i in range(0, len(crops), self.tile_batch):
            per_tile += self.detect_tiles(crops[i:i + self.tile_batch], conf, iou)
        t2 = time.perf_counter()

        boxes, scores, cls_ids = [], [], []
        for tile, (b, s, c) in zip(grid, per_tile):
            boxes += [tile_to_image(bx, tile) for bx in b]
            scores += list(s)
            cls_ids += list(c)
        keep = nms_fast(boxes, scores, iou, None if agnostic else cls_ids)[: self.max_det]
        dets = [Detection(tuple(float(v) for v in boxes[i]), int(cls_ids[i]),
                          self.classes[int(cls_ids[i])], float(scores[i])) for i in keep]
        t3 = time.perf_counter()

        return DetectionResult(
            image_size=(W, H), detections=dets, tiles=grid, num_candidates=len(boxes),
            timings_ms={"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3,
                        "postprocess": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3},
            backend=self.backend)

    __call__ = predict

    def predict_many(self, images: Sequence[ImageInput], **kw) -> list[DetectionResult]:
        return [self.predict(im, **kw) for im in images]


# --------------------------------------------------------------------------- #
# Backend 1: the mini-YOLO from the lab                                       #
# --------------------------------------------------------------------------- #
class MiniYoloDetector(BaseDetector, BasePredictor):
    """The lab's ``model -> activate -> decode_grid -> nms`` chain, per tile.

    Preprocessing is the lab's ``TileDataset`` transform: ``Resize(224)`` +
    ``ToTensor`` with **no** normalisation - read from the checkpoint, not assumed.

    Accepted checkpoints: ``miniyolo_checkpoint.pt`` (dict with ``state_dict`` and
    ``classes``), a whole pickled model (``torch.save(model, "mini_yolo.pt")``) plus
    ``classes=".../classes_name.yaml"``, or one converted with ``brocade convert``.
    """

    backend = "miniyolo"

    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 classes: list[str] | str | Path | None = None, conf: float | None = None,
                 iou: float | None = None, tiles: TileSpec = None, overlap: float = 0.0,
                 agnostic_nms: bool = True, tile_batch: int = 32):
        BasePredictor.__init__(self, checkpoint, device, classes)
        if self.meta["arch"] != "MiniYolo":
            raise ValueError(f"{checkpoint} holds a {self.meta['arch']}, not a MiniYolo")
        self._init_detector(
            conf=conf if conf is not None else self.meta.get("conf_threshold", 0.35),
            iou=iou if iou is not None else self.meta.get("nms_iou", 0.4),
            tiles=tiles if tiles is not None else self.meta.get("tiles", "auto"),
            overlap=overlap, agnostic_nms=agnostic_nms, tile_batch=tile_batch)

    def postprocess(self, raw: torch.Tensor, conf: float = 0.35, iou: float = 0.4
                    ) -> list[TileOutput]:
        """Raw ``(B, 5+C, S, S)`` grids -> per-tile (boxes, scores, class ids) after NMS."""
        out = []
        for grid in raw.cpu():
            b, s, c = decode_grid(activate(grid), conf_threshold=conf)
            keep = nms_fast(b, s, iou)
            out.append(([b[i] for i in keep], [s[i] for i in keep], [c[i] for i in keep]))
        return out

    def detect_tiles(self, crops, conf, iou):
        return self.postprocess(self.forward(self.preprocess(crops)), conf=conf, iou=iou)


# --------------------------------------------------------------------------- #
# Backend 2: Ultralytics YOLOv8 / YOLO11                                      #
# --------------------------------------------------------------------------- #
class UltralyticsDetector(BaseDetector):
    """A fine-tuned Ultralytics model (``best.pt``, or an exported ``.onnx``).

    Ultralytics does its own letterbox preprocessing and per-image NMS, so only
    ``detect_tiles`` differs from the mini-YOLO path; the tiling and photo-level
    NMS are the same code. Class names come from the model itself.
    """

    backend = "ultralytics"

    def __init__(self, weights: str | Path, device: str | None = None,
                 conf: float = 0.25, iou: float = 0.5, imgsz: int | None = None,
                 tiles: TileSpec = "auto", overlap: float = 0.1, agnostic_nms: bool = True,
                 tile_batch: int = 16):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError("UltralyticsDetector needs `pip install ultralytics`") from e
        self.checkpoint_path = Path(weights)
        self.model = YOLO(str(weights), task="detect")
        self.device = resolve_device(device)
        names = self.model.names
        self.classes = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        overrides = getattr(self.model, "overrides", {}) or {}
        self.imgsz = int(imgsz or overrides.get("imgsz") or 640)
        self.meta = {"arch": overrides.get("model", self.checkpoint_path.name),
                     "classes": self.classes, "img_size": self.imgsz}
        self._init_detector(conf=conf, iou=iou, tiles=tiles, overlap=overlap,
                            agnostic_nms=agnostic_nms, tile_batch=tile_batch)

    @property
    def _ul_device(self):
        # Ultralytics wants 0 / "cpu" / "mps" rather than torch.device("cuda")
        return 0 if self.device.type == "cuda" else self.device.type

    def detect_tiles(self, crops, conf, iou):
        results = self.model.predict(crops, conf=conf, iou=iou, imgsz=self.imgsz,
                                     device=self._ul_device, verbose=False, max_det=1000)
        out = []
        for r in results:
            bx = r.boxes
            out.append(([tuple(b) for b in bx.xyxyn.cpu().tolist()],
                        bx.conf.cpu().tolist(), bx.cls.int().cpu().tolist()))
        return out

    def benchmark(self, n: int = 20) -> float:
        """Mean latency (ms) of one tile, after warm-up."""
        tile = Image.new("RGB", (self.imgsz, self.imgsz), (128, 128, 128))
        for _ in range(3):
            self.detect_tiles([tile], self.conf, self.iou)
        t0 = time.perf_counter()
        for _ in range(n):
            self.detect_tiles([tile], self.conf, self.iou)
        return (time.perf_counter() - t0) / n * 1000

    def __repr__(self) -> str:
        return (f"UltralyticsDetector({self.checkpoint_path.name}, classes={len(self.classes)}, "
                f"imgsz={self.imgsz}, device={self.device})")


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
_ULTRALYTICS_SUFFIXES = {".onnx", ".engine", ".torchscript", ".mlpackage", ".tflite", ".pb"}


def is_ultralytics_weights(path: str | Path) -> bool:
    """Cheap sniff: does this file come from Ultralytics? (no unpickling needed)"""
    path = Path(path)
    if path.suffix in _ULTRALYTICS_SUFFIXES or path.is_dir():
        return True
    if not path.exists():
        # e.g. "yolo11s.pt": Ultralytics downloads its official weights by name
        return path.name.lower().startswith(("yolo", "rtdetr"))
    try:
        with zipfile.ZipFile(path) as zf:
            pkl = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
            return pkl is not None and b"ultralytics" in zf.read(pkl)
    except zipfile.BadZipFile:
        return False


def load_detector(weights: str | Path, backend: str = "auto", **kwargs) -> BaseDetector:
    """Build the right detector for a weights file.

    ``backend="auto"`` recognises Ultralytics files (``best.pt``, ``.onnx`` ...) and
    treats everything else as a mini-YOLO checkpoint. Extra ``kwargs`` go to the
    detector's constructor (``conf``, ``iou``, ``tiles``, ``overlap``, ``device`` ...).
    """
    if backend == "auto":
        backend = "ultralytics" if is_ultralytics_weights(weights) else "miniyolo"
    if backend == "ultralytics":
        kwargs.pop("classes", None)
        return UltralyticsDetector(weights, **kwargs)
    if backend == "miniyolo":
        return MiniYoloDetector(weights, **kwargs)
    raise ValueError(f"unknown backend {backend!r} (auto | miniyolo | ultralytics)")

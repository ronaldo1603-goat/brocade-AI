"""Shared fixtures: a tiny synthetic 'textile' dataset and throwaway checkpoints.

The real dataset is 2.2 GB and CC BY-NC, so tests never touch it. Instead we draw
photographs of coloured shapes with YOLO labels - enough to exercise every code
path (tiling, coordinate mapping, NMS, checkpoint formats, CLI).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CLASSES = ["Chaska", "Llama", "Qenqo"]


def make_photo(path: Path, W=1600, H=1200, n=12, seed=0):
    """Draw n shapes (circle / square / triangle = class 0/1/2); write YOLO labels."""
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 140, size=(H, W, 3), dtype=np.uint8)
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    rows = []
    for _ in range(n):
        c = int(rng.integers(0, 3))
        s = int(rng.integers(60, 140))
        x, y = int(rng.integers(0, W - s)), int(rng.integers(0, H - s))
        col = [(220, 40, 40), (40, 200, 60), (40, 60, 220)][c]
        if c == 0:
            d.ellipse((x, y, x + s, y + s), fill=col)
        elif c == 1:
            d.rectangle((x, y, x + s, y + s), fill=col)
        else:
            d.polygon([(x + s / 2, y), (x, y + s), (x + s, y + s)], fill=col)
        rows.append((c, (x + s / 2) / W, (y + s / 2) / H, s / W, s / H))
    img.save(path, quality=92)
    path.with_suffix(".txt").write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows))
    return rows


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("chinchero")
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "classes_name.yaml").write_text(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    for i in range(3):
        p = root / "images" / f"photo_{i}.jpg"
        make_photo(p, seed=i)
        p.with_suffix(".txt").rename(root / "labels" / f"photo_{i}.txt")
    return root


@pytest.fixture(scope="session")
def miniyolo_ckpt(tmp_path_factory):
    from brocade.checkpoint import save_checkpoint
    from brocade.models import MiniYolo

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("weights") / "miniyolo_checkpoint.pt"
    save_checkpoint(MiniYolo(len(CLASSES)), path, classes=CLASSES, img_size=224,
                    tiles="auto", conf_threshold=0.35, nms_iou=0.4)
    return path

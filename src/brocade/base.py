"""``BasePredictor``: the part of inference every task shares.

Both predictors follow the same four-step template::

    image ──load_image──▶ PIL ──preprocess──▶ (B,3,H,W) ──forward──▶ raw ──postprocess──▶ answer
                                 (task-agnostic)          (no_grad, eval)   (task-specific)

The base class owns everything above the last arrow: picking a device, loading a
checkpoint together with its preprocessing recipe, turning a path / PIL image /
numpy array into a normalised batch, and running the forward pass in eval mode
under ``torch.no_grad``. Subclasses only decide what the raw output *means* - an
``argmax`` over logits for the classifier, grid decoding plus NMS for the detector.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .checkpoint import build_from_checkpoint, load_checkpoint

ImageInput = str | Path | Image.Image | np.ndarray


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """``None``/``"auto"`` -> CUDA if available, then Apple MPS, else CPU."""
    if device not in (None, "auto"):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(image: ImageInput) -> Image.Image:
    """Path / PIL / HxWx3 uint8 array -> RGB PIL image with EXIF rotation applied.

    Phone photos (two of the dataset's three cameras are phones) often store the
    pixels sideways plus an EXIF "rotate me" flag. Labels are drawn on the
    displayed orientation, so we apply the flag before anything else.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
        img = ImageOps.exif_transpose(img)
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8))
    else:
        raise TypeError(f"unsupported image type {type(image).__name__}")
    return img.convert("RGB")


class BasePredictor(ABC):
    """Load once, predict many times.

    Args:
        checkpoint: path to a brocade checkpoint (see ``checkpoint.py``).
        device: ``"cpu"``, ``"cuda"``, ``"mps"`` or ``None`` for automatic.
        classes: optional class names (list or ``classes_name.yaml``) for
            checkpoints saved without them.
    """

    #: default input side when a checkpoint does not record one
    default_img_size: int = 224

    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 classes: list[str] | str | Path | None = None):
        self.device = resolve_device(device)
        self.checkpoint_path = Path(checkpoint)
        self.meta: dict[str, Any] = {}
        self.model = self.load(checkpoint, classes)
        self.transform = self.build_transform()

    # ---- 1. load ------------------------------------------------------------
    def load(self, checkpoint, classes=None) -> torch.nn.Module:
        """Read the checkpoint, keep its metadata, return the model in eval mode."""
        ck = load_checkpoint(checkpoint, classes)
        self.meta = {k: v for k, v in ck.items() if k != "state_dict"}
        model = build_from_checkpoint(ck).to(self.device)
        model.eval()   # BatchNorm / Dropout switch to inference behaviour - not optional
        return model

    @property
    def classes(self) -> list[str]:
        return self.meta["classes"]

    @property
    def img_size(self) -> int:
        return int(self.meta.get("img_size") or self.default_img_size)

    # ---- 2. preprocess --------------------------------------------------------
    def build_transform(self) -> transforms.Compose:
        """The *eval* transform: resize -> tensor -> (normalise, if trained with it)."""
        steps = [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]
        if self.meta.get("mean") is not None and self.meta.get("std") is not None:
            steps.append(transforms.Normalize(self.meta["mean"], self.meta["std"]))
        return transforms.Compose(steps)

    def preprocess(self, images: ImageInput | list[ImageInput]) -> torch.Tensor:
        """One image or a list -> a ``(B, 3, H, W)`` float tensor on ``self.device``."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        return torch.stack([self.transform(load_image(im)) for im in images]).to(self.device)

    # ---- 3. forward -----------------------------------------------------------
    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    # ---- 4. postprocess (task-specific) ---------------------------------------
    @abstractmethod
    def postprocess(self, outputs: Any, **kwargs) -> Any:
        """Turn raw network output into the task's answer."""

    def predict(self, image: ImageInput, **kwargs) -> Any:
        """The full template: preprocess -> forward -> postprocess."""
        return self.postprocess(self.forward(self.preprocess(image)), **kwargs)

    __call__ = predict

    # ---- utilities ------------------------------------------------------------
    def benchmark(self, n: int = 20, batch: int = 1) -> float:
        """Mean forward latency in milliseconds on a random batch (after warm-up)."""
        x = torch.rand(batch, 3, self.img_size, self.img_size, device=self.device)
        for _ in range(3):
            self.forward(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            self.forward(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(arch={self.meta.get('arch')}, "
                f"classes={len(self.meta.get('classes', []))}, img_size={self.img_size}, "
                f"device={self.device})")

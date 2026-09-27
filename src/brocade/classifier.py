"""Motif classification: a crop in, a motif name out."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import BasePredictor, ImageInput


@dataclass
class Classification:
    name: str
    confidence: float
    topk: list[tuple[str, float]]

    def __str__(self) -> str:
        return f"{self.name} ({self.confidence:.1%})"


class ClassifierPredictor(BasePredictor):
    """Wraps ``motif_classifier.pt`` from the classification lab (LeNet or AlexNet).

    The checkpoint carries its own ``img_size``, ``mean`` and ``std``, so the
    transform used here is exactly the lab's ``eval_tf``.
    """

    def postprocess(self, logits: torch.Tensor, topk: int = 3) -> list[Classification]:
        probs = torch.softmax(logits, dim=1).cpu()
        k = min(topk, probs.shape[1])
        out = []
        for row in probs:
            vals, idx = row.topk(k)
            pairs = [(self.classes[i], float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
            out.append(Classification(pairs[0][0], pairs[0][1], pairs))
        return out

    def predict(self, image: ImageInput, topk: int = 3) -> Classification:
        """Classify one image (path, PIL or array)."""
        return super().predict(image, topk=topk)[0]

    def predict_batch(self, images: list[ImageInput], topk: int = 3) -> list[Classification]:
        return super().predict(images, topk=topk)

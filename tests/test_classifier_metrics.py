import numpy as np
import torch

from brocade.checkpoint import save_checkpoint
from brocade.classifier import ClassifierPredictor
from brocade.detector import Detection, DetectionResult

from brocade.models import LeNet


def test_classifier_artifact_like_the_lab(tmp_path, dataset):
    """Same keys as the lab's motif_classifier.pt: arch/classes/img_size/mean/std."""
    p = save_checkpoint(LeNet(3), tmp_path / "motif_classifier.pt", classes=["a", "b", "c"],
                        img_size=32, mean=[0.5, 0.4, 0.3], std=[0.2, 0.2, 0.2], val_acc=0.5)
    clf = ClassifierPredictor(p, device="cpu")
    assert clf.transform.transforms[-1].mean == [0.5, 0.4, 0.3]
    r = clf.predict(dataset / "images" / "photo_0.jpg", topk=2)
    assert r.name in {"a", "b", "c"} and len(r.topk) == 2
    assert abs(sum(p for _, p in clf.predict(dataset / "images" / "photo_0.jpg", topk=3).topk) - 1) < 1e-5
    assert len(clf.predict_batch([dataset / "images" / "photo_0.jpg"] * 4)) == 4


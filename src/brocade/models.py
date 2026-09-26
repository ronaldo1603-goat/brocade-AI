"""Model definitions, copied verbatim from the notebooks.

A ``state_dict`` only stores tensors keyed by layer name, so the class that
receives it must have *exactly* the layer names and shapes it was trained with.
That is why these definitions are frozen copies of the notebook cells: change a
layer here and every saved checkpoint stops loading.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Classification (cnn_brocade_datapipeline_lab.ipynb)                         #
# --------------------------------------------------------------------------- #
class LeNet(nn.Module):
    """LeNet-5, modern flavour: 32x32 RGB in, ``num_classes`` logits out."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


class AlexNet(nn.Module):
    """AlexNet built from the lab's layer table: 224x224 RGB in."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))


# --------------------------------------------------------------------------- #
# Detection (mini_yolo_detector_iou_nms_lab.ipynb)                            #
# --------------------------------------------------------------------------- #
def conv_block(cin: int, cout: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.BatchNorm2d(cout), nn.ReLU())


class LocNet(nn.Module):
    """One backbone, two heads: class logits + one box in [0, 1] xyxy."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(conv_block(3, 32), conv_block(32, 64),
                                      conv_block(64, 128),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cls_head = nn.Linear(128, num_classes)
        self.box_head = nn.Linear(128, 4)

    def forward(self, x):
        feats = self.backbone(x)
        return self.cls_head(feats), torch.sigmoid(self.box_head(feats))


class MiniYolo(nn.Module):
    """YOLO v1 minus anchors, B = 1: five stride-2 blocks, 224 -> 7x7 grid.

    Output is the *raw* grid ``(B, 5 + C, S, S)``; see ``geometry.activate``.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64), conv_block(64, 128),
            conv_block(128, 128), conv_block(128, 256))
        self.head = nn.Conv2d(256, 5 + num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.backbone(x))


MODELS: dict[str, type[nn.Module]] = {
    "LeNet": LeNet,
    "AlexNet": AlexNet,
    "LocNet": LocNet,
    "MiniYolo": MiniYolo,
}


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Instantiate a model by the ``arch`` name stored in a checkpoint."""
    try:
        return MODELS[arch](num_classes)
    except KeyError:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(MODELS)}") from None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

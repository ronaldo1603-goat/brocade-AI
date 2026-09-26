"""Reading and writing self-describing checkpoints.

The classification lab ends on a lesson worth repeating: weights alone are not a
model. Inference also needs the class names, the input size and the
normalisation statistics the network was trained with. A *brocade checkpoint*
is therefore a plain dict::

    {
        "arch": "MiniYolo",            # key into models.MODELS
        "state_dict": {...},           # tensors on CPU
        "classes": ["Aysa Cuty", ...], # index -> name, the order class_id refers to
        "img_size": 224,               # square input side
        "mean": None, "std": None,     # Normalize stats, or None if ToTensor only
        ...                            # task-specific extras (tiles, thresholds, ...)
    }

The notebooks saved models in three different ways over time; ``load_checkpoint``
accepts all of them and returns this normalised dict:

1. the dict above (``motif_classifier.pt``, ``miniyolo_checkpoint.pt``);
2. a whole pickled module - ``torch.save(model, "mini_yolo.pt")``;
3. a bare ``state_dict``.
"""
from __future__ import annotations

import ast
import contextlib
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from . import models as _models

PathLike = str | Path


def read_class_names(yaml_path: PathLike) -> list[str]:
    """Class names from the dataset's ``classes_name.yaml`` (or an Ultralytics ``data.yaml``).

    Uses the same dependency-free parse as the notebooks, and falls back to PyYAML
    for the multi-line ``names:`` mapping Ultralytics writes.
    """
    text = Path(yaml_path).read_text(encoding="utf-8")
    m = re.search(r"names:\s*(\[.*?\])", text, flags=re.S)
    if m:
        return list(ast.literal_eval(m.group(1)))
    import yaml  # optional dependency, only for mapping-style files

    names = yaml.safe_load(text)["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


@contextlib.contextmanager
def _notebook_classes_in_main():
    """Make ``torch.load`` able to unpickle modules saved from a notebook.

    ``torch.save(model)`` pickles a *reference* to the class, e.g.
    ``__main__.MiniYolo`` - the notebook's own namespace. In any other process
    ``__main__`` is a different module and unpickling fails. We temporarily
    expose our frozen copies of the classes under ``__main__``.
    """
    main = sys.modules.get("__main__")
    added = []
    if main is not None:
        for name, obj in {**_models.MODELS, "conv_block": _models.conv_block}.items():
            if not hasattr(main, name):
                setattr(main, name, obj)
                added.append(name)
    try:
        yield
    finally:
        for name in added:
            delattr(main, name)


def _torch_load(path: PathLike) -> Any:
    with _notebook_classes_in_main():
        # weights_only=False: our checkpoints carry Python lists/dicts next to the
        # tensors. Only load checkpoints you trust - unpickling can run code.
        return torch.load(path, map_location="cpu", weights_only=False)


def _guess_arch(state_dict: dict[str, torch.Tensor]) -> tuple[str, int]:
    """Infer ``(arch, num_classes)`` from the layer names of a state_dict."""
    keys = set(state_dict)
    if "head.weight" in keys:                                   # MiniYolo
        return "MiniYolo", state_dict["head.weight"].shape[0] - 5
    if "cls_head.weight" in keys:                               # LocNet
        return "LocNet", state_dict["cls_head.weight"].shape[0]
    first = state_dict.get("features.0.weight")
    if first is not None:                                       # LeNet (5x5) / AlexNet (11x11)
        arch = "AlexNet" if first.shape[-1] == 11 else "LeNet"
        last = max((k for k in keys if k.startswith("classifier.") and k.endswith(".weight")),
                   key=lambda k: int(k.split(".")[1]))
        return arch, state_dict[last].shape[0]
    raise ValueError("cannot infer the architecture from this state_dict's layer names")


def load_checkpoint(path: PathLike, classes: list[str] | PathLike | None = None) -> dict:
    """Load any of the three checkpoint styles and return a normalised dict.

    ``classes`` overrides / supplies the class names (a list, or a path to a
    ``classes_name.yaml``). If a checkpoint carries no names and none are
    given, generic ``class_0 ... class_{C-1}`` names are used.
    """
    raw = _torch_load(path)

    if isinstance(raw, nn.Module):                               # style 2
        ck = {"arch": type(raw).__name__, "state_dict": raw.state_dict()}
    elif isinstance(raw, dict) and "state_dict" in raw:          # style 1
        ck = dict(raw)
    elif isinstance(raw, dict) and all(torch.is_tensor(v) for v in raw.values()):  # style 3
        ck = {"state_dict": raw}
    else:
        raise ValueError(f"{path}: not a recognised brocade checkpoint "
                         f"(got {type(raw).__name__} with keys "
                         f"{list(raw)[:6] if isinstance(raw, dict) else '-'})")

    arch, n_cls = _guess_arch(ck["state_dict"])
    ck.setdefault("arch", arch)
    ck["num_classes"] = n_cls

    if classes is not None:
        ck["classes"] = read_class_names(classes) if isinstance(classes, (str, Path)) else list(classes)
    ck.setdefault("classes", [f"class_{i}" for i in range(n_cls)])
    if len(ck["classes"]) != n_cls:
        raise ValueError(f"checkpoint has {n_cls} output classes but {len(ck['classes'])} "
                         "class names were supplied - wrong classes_name.yaml?")
    ck["source_path"] = str(path)
    return ck


def build_from_checkpoint(ck: dict) -> nn.Module:
    """Instantiate ``ck['arch']`` and load its weights (strict)."""
    model = _models.build_model(ck["arch"], ck["num_classes"])
    model.load_state_dict(ck["state_dict"])
    return model


def save_checkpoint(model: nn.Module, path: PathLike, *, classes: list[str],
                    img_size: int, mean=None, std=None, **extra) -> Path:
    """Write a self-describing brocade checkpoint (see module docstring)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    art = {
        "arch": type(model).__name__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": list(classes),
        "img_size": int(img_size),
        "mean": mean, "std": std,
        "torch_version": torch.__version__,
        **extra,
    }
    torch.save(art, path)
    return path

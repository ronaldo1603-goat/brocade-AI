"""Buổi 1 - bước 13: load the notebooks' real weights OUTSIDE the notebooks.

Usage (from the repo root, venv activated):
    python scripts/check_weights.py                                   # scans weights/
    python scripts/check_weights.py --classes path/to/classes_name.yaml
    python scripts/check_weights.py --image path/to/tile_or_crop.jpg  # + one real prediction
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brocade.checkpoint import build_from_checkpoint, load_checkpoint   # noqa: E402
from brocade.geometry import activate, decode_grid, nms                 # noqa: E402
from brocade.models import count_parameters                             # noqa: E402

DEFAULT_SIZE = {"MiniYolo": 224, "LocNet": 96, "AlexNet": 224, "LeNet": 32}


def check(path: Path, classes, image):
    print(f"\n=== {path.name} " + "=" * max(0, 60 - len(path.name)))
    try:
        ck = load_checkpoint(path)
        if classes and ck["classes"][0] == "class_0":   # file has no names -> use the yaml
            ck = load_checkpoint(path, classes=classes)
    except Exception as e:                       # e.g. frcnn.pt, or names/yaml mismatch
        print(f"  SKIPPED: {type(e).__name__}: {e}")
        return False
    model = build_from_checkpoint(ck).eval()     # eval(): BatchNorm/Dropout in inference mode
    size = int(ck.get("img_size") or DEFAULT_SIZE[ck["arch"]])
    with torch.no_grad():
        out = model(torch.rand(1, 3, size, size))
    shape = tuple(out[0].shape if isinstance(out, tuple) else out.shape)

    print(f"  arch        : {ck['arch']}  ({count_parameters(model):,} parameters)")
    print(f"  classes     : {len(ck['classes'])} -> {ck['classes'][:4]} ...")
    print(f"  input       : 3x{size}x{size} | mean/std: {ck.get('mean')} / {ck.get('std')}")
    print(f"  output shape: {shape}")
    if ck["classes"][0] == "class_0":
        print("  WARNING     : no class names in this file -> pass --classes classes_name.yaml")

    if ck["arch"] == "MiniYolo":
        assert shape == (1, 5 + len(ck["classes"]), 7, 7), "unexpected grid shape"
    elif ck["arch"] in ("LeNet", "AlexNet"):
        assert shape == (1, len(ck["classes"])), "unexpected logits shape"

    if image:                                    # one real prediction, same transform as training
        steps = [transforms.Resize((size, size)), transforms.ToTensor()]
        if ck.get("mean") is not None:
            steps.append(transforms.Normalize(ck["mean"], ck["std"]))
        x = transforms.Compose(steps)(Image.open(image).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            raw = model(x)
        if ck["arch"] == "MiniYolo":
            boxes, scores, cls = decode_grid(activate(raw[0]), conf_threshold=0.35)
            keep = nms(boxes, scores, 0.4)
            print(f"  prediction  : {len(keep)} boxes after NMS on {Path(image).name}")
            for i in keep[:5]:
                print(f"      {ck['classes'][cls[i]]:<16s} {scores[i]:.2f}  "
                      f"{tuple(round(v, 3) for v in boxes[i])}")
        elif ck["arch"] in ("LeNet", "AlexNet"):
            p = torch.softmax(raw, 1)[0]
            top = p.topk(min(3, len(p)))
            print("  prediction  : " + ", ".join(
                f"{ck['classes'][i]} {v:.1%}" for v, i in zip(top.values.tolist(), top.indices.tolist())))
    print("  OK")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--classes", default=None,
                    help="classes_name.yaml - only used for files saved without class names")
    ap.add_argument("--image", default=None, help="a 448px tile (detector) or a motif crop (classifier)")
    args = ap.parse_args()

    files = sorted(Path(args.weights_dir).glob("*.pt")) + sorted(Path(args.weights_dir).glob("*.pth"))
    if not files:
        sys.exit(f"no .pt files in {Path(args.weights_dir).resolve()} - copy your checkpoints there")
    ok = sum(check(f, args.classes, args.image) for f in files)
    print(f"\n{ok}/{len(files)} checkpoints loaded outside the notebook")


if __name__ == "__main__":
    main()

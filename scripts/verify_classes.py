"""Check that a classes_name.yaml matches the class ids actually used in the labels.

    python scripts/verify_classes.py --yaml classes_name.yaml --labels D:/data/Iconography_Chinchero/labels
    python scripts/verify_classes.py --yaml classes_name.yaml --checkpoint weights/miniyolo_checkpoint.pt

A wrong order does not crash anything - every prediction is just silently given the
wrong motif name. So we check it against two independent pieces of evidence:
  1. box counts per class id vs. the counts the classification lab printed for the
     full dataset (its "rarest classes" output);
  2. the class list stored inside a checkpoint trained with the original yaml.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brocade import models as _models              # noqa: E402
from brocade.checkpoint import read_class_names   # noqa: E402

# printed by cnn_brocade_datapipeline_lab.ipynb, cell 15, on the full 1,358-image dataset
KNOWN_COUNTS = {"JuchuyMakiMaki": 298, "Weqontoy": 149, "Qente": 103, "ChunkuChili": 99}


def load_trusted(path):
    """torch.load that also opens notebook files saved with torch.save(model).

    Such a file pickles a reference to ``__main__.MiniYolo``; in this script
    ``__main__`` is verify_classes.py, so we expose the model classes there first
    (same trick as brocade.checkpoint). Only load files you trust.
    """
    import torch

    main_mod = sys.modules["__main__"]
    added = [n for n in {**_models.MODELS, "conv_block": _models.conv_block}
             if not hasattr(main_mod, n)]
    for n in added:
        setattr(main_mod, n, {**_models.MODELS, "conv_block": _models.conv_block}[n])
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        for n in added:
            delattr(main_mod, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="classes_name.yaml")
    ap.add_argument("--labels", help="folder of YOLO .txt label files (searched recursively)")
    ap.add_argument("--checkpoint", help="a .pt that stores its class list (e.g. miniyolo_checkpoint.pt)")
    args = ap.parse_args()

    names = read_class_names(args.yaml)
    print(f"{args.yaml}: {len(names)} names")
    ok, checked = True, False

    if args.labels:
        counts, files = Counter(), 0
        for f in Path(args.labels).rglob("*.txt"):
            files += 1
            for line in f.read_text().split("\n"):
                if line.strip():
                    counts[int(line.split()[0])] += 1
        checked = True
        print(f"\n{files} label files, {sum(counts.values()):,} boxes, "
              f"class ids used: {min(counts)}..{max(counts)}")
        if max(counts) >= len(names):
            print(f"  FAIL: labels use id {max(counts)} but the yaml has only {len(names)} names")
            ok = False
        print(f"\n  {'id':>3}  {'name':<16} {'boxes':>7}")
        for i, n in enumerate(names):
            flag = ""
            if n in KNOWN_COUNTS:
                match = counts[i] == KNOWN_COUNTS[n]
                flag = "  <- matches the notebook" if match else f"  <- MISMATCH (notebook: {KNOWN_COUNTS[n]})"
                ok &= match
            print(f"  {i:>3}  {n:<16} {counts[i]:>7}{flag}")
        if files < 1358:
            print("\n  note: fewer than 1,358 label files - the count check only holds on the full dataset")

    if args.checkpoint:
        ck = load_trusted(args.checkpoint)
        stored = ck.get("classes") if isinstance(ck, dict) else None
        checked = checked or bool(stored)
        if not stored:
            kind = "a whole pickled model (torch.save(model))" if not isinstance(ck, dict) else "a dict without 'classes'"
            print(f"\n{args.checkpoint}: {kind} - it stores NO class list, nothing to compare.\n"
                  "  Use a checkpoint saved as a dict with 'classes' (e.g. miniyolo_checkpoint.pt),\n"
                  "  or check with --labels instead.")
        elif len(stored) < len(names):
            # e.g. motif_classifier.pt: the lab kept only classes with >= 30 crops,
            # sorted by name -> can check spelling, not the id order
            unknown = [n for n in stored if n not in names]
            print(f"\n{args.checkpoint}: stores {len(stored)} of {len(names)} classes (a subset, "
                  "e.g. the classifier) - checking spelling only, not order")
            if unknown:
                ok = False
                print(f"  names not in the yaml: {unknown}")
            else:
                print("  every name exists in the yaml")
        elif stored == names:
            print(f"\n{args.checkpoint}: class list IDENTICAL to the yaml")
        else:
            ok = False
            print(f"\n{args.checkpoint}: class list DIFFERS from the yaml")
            for i, (a, b) in enumerate(zip(stored, names)):
                if a != b:
                    print(f"  id {i:>2}: checkpoint {a!r:<18} yaml {b!r}")
            print("  -> the checkpoint's list is the original: copy it into the yaml")

    if not checked:
        print("\nRESULT: NOTHING CHECKED - pass --labels, or a checkpoint that stores its classes")
        sys.exit(2)
    print("\nRESULT:", "OK" if ok else "CHECK FAILED - fix the yaml before training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

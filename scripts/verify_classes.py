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
from brocade.checkpoint import read_class_names   # noqa: E402

# printed by cnn_brocade_datapipeline_lab.ipynb, cell 15, on the full 1,358-image dataset
KNOWN_COUNTS = {"JuchuyMakiMaki": 298, "Weqontoy": 149, "Qente": 103, "ChunkuChili": 99}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="classes_name.yaml")
    ap.add_argument("--labels", help="folder of YOLO .txt label files (searched recursively)")
    ap.add_argument("--checkpoint", help="a .pt that stores its class list (e.g. miniyolo_checkpoint.pt)")
    args = ap.parse_args()

    names = read_class_names(args.yaml)
    print(f"{args.yaml}: {len(names)} names")
    ok = True

    if args.labels:
        counts, files = Counter(), 0
        for f in Path(args.labels).rglob("*.txt"):
            files += 1
            for line in f.read_text().split("\n"):
                if line.strip():
                    counts[int(line.split()[0])] += 1
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
        import torch
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        stored = ck.get("classes") if isinstance(ck, dict) else None
        if not stored:
            print(f"\n{args.checkpoint}: stores no class list - cannot compare")
        elif stored == names:
            print(f"\n{args.checkpoint}: class list IDENTICAL to the yaml")
        else:
            ok = False
            print(f"\n{args.checkpoint}: class list DIFFERS from the yaml")
            for i, (a, b) in enumerate(zip(stored, names)):
                if a != b:
                    print(f"  id {i:>2}: checkpoint {a!r:<18} yaml {b!r}")
            print("  -> the checkpoint's list is the original: copy it into the yaml")

    print("\nRESULT:", "OK" if ok else "CHECK FAILED - fix the yaml before training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

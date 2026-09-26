import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from brocade.checkpoint import load_checkpoint, read_class_names, save_checkpoint
from brocade.models import AlexNet, LeNet, MiniYolo

SRC = Path(__file__).resolve().parents[1] / "src"


def test_read_class_names(dataset):
    assert read_class_names(dataset / "classes_name.yaml") == ["Chaska", "Llama", "Qenqo"]


def test_read_ultralytics_style_yaml(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text("path: x\nnames:\n  0: Chaska\n  1: Llama\n")
    assert read_class_names(p) == ["Chaska", "Llama"]


@pytest.mark.parametrize("cls,arch", [(LeNet, "LeNet"), (AlexNet, "AlexNet"), (MiniYolo, "MiniYolo")])
def test_bare_state_dict_arch_is_inferred(tmp_path, cls, arch):
    p = tmp_path / "sd.pt"
    torch.save(cls(5).state_dict(), p)
    ck = load_checkpoint(p)
    assert ck["arch"] == arch and ck["num_classes"] == 5
    assert ck["classes"] == [f"class_{i}" for i in range(5)]


def test_notebook_pickled_model_loads(tmp_path, dataset):
    """Reproduce `torch.save(model, ...)` from a notebook, where the class lives in __main__."""
    out = tmp_path / "mini_yolo.pt"
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(SRC)!r})
        import torch, torch.nn as nn
        from brocade.models import conv_block
        class MiniYolo(nn.Module):              # defined in __main__, like in a notebook
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = nn.Sequential(conv_block(3, 32), conv_block(32, 64),
                    conv_block(64, 128), conv_block(128, 128), conv_block(128, 256))
                self.head = nn.Conv2d(256, 5 + num_classes, kernel_size=1)
            def forward(self, x): return self.head(self.backbone(x))
        torch.save(MiniYolo(3), {str(out)!r})
    """)
    subprocess.run([sys.executable, "-c", script], check=True)
    ck = load_checkpoint(out, classes=dataset / "classes_name.yaml")
    assert ck["arch"] == "MiniYolo" and ck["classes"][1] == "Llama"


def test_wrong_number_of_class_names(tmp_path):
    p = save_checkpoint(MiniYolo(3), tmp_path / "m.pt", classes=["a", "b", "c"], img_size=224)
    with pytest.raises(ValueError, match="3 output classes"):
        load_checkpoint(p, classes=["a", "b"])
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_model

from config import TOKENIZER_PATH
from model import Mallm, MallmConfig

EXPORTS = [
    ("checkpoints/best.pt", "hf/mallm-base"),
    ("checkpoints/best_sft.pt", "hf/mallm-instruct"),
]


def export(ckpt_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = MallmConfig(**ckpt["config"])
    model = Mallm(cfg)
    model.load_state_dict(ckpt["model"])

    save_model(model, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    shutil.copy(TOKENIZER_PATH, out / "tokenizer.json")
    shutil.copy("model.py", out / "model.py")


if __name__ == "__main__":
    for ckpt_path, out_dir in EXPORTS:
        export(ckpt_path, out_dir)

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer

import wandb
from config import (
    CKPT_DIR,
    PAD,
    SFT_MAX_LEN,
    SFT_TRAIN_DIR,
    SFT_VAL_DIR,
    TOKENIZER_PATH,
)
from model import Mallm, MallmConfig


def get_batch(ds: Dataset, B, pad_id, device):
    idx = torch.randint(0, len(ds), (B,))
    rows = ds[idx]

    max_l = max(len(i) for i in rows["input_ids"])

    ids = np.full((B, max_l), pad_id, dtype=np.int64)
    mask = np.zeros((B, max_l), dtype=np.int64)

    for i in range(B):
        n = len(rows["input_ids"][i])
        ids[i, :n] = rows["input_ids"][i]
        mask[i, :n] = rows["loss_mask"][i]

    x = ids[:, :-1]
    y = ids[:, 1:].copy()
    y[mask[:, 1:] == 0] = -1

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@dataclass
class SftConfig:
    batch_size: int = 12
    device: str = "cuda"
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    max_step: int = 10_000
    warm_up_steps: int = 200
    eval_interval: int = 500
    eval_batches_num: int = 20
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1


@torch.no_grad()
def evaluate_loss(ds: Dataset, model, pad_id: int, sft_cfg: SftConfig):
    model.eval()
    losses = []
    for _ in range(sft_cfg.eval_batches_num):
        x, y = get_batch(ds, B=sft_cfg.batch_size, pad_id=pad_id, device=sft_cfg.device)
        with torch.autocast(sft_cfg.device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def save_ckpt(mallm, ckpt_path, step, lowest_loss, optimizer_state):
    checkpoint = {
        "model": mallm.state_dict(),
        "config": asdict(mallm.config),
        "step": step,
        "lowest_loss": lowest_loss,
        "optimizer_state": optimizer_state,
    }
    torch.save(checkpoint, ckpt_path)


def get_lr(step, sft_cfg: SftConfig):
    if step < sft_cfg.warm_up_steps:
        return (step / sft_cfg.warm_up_steps) * sft_cfg.max_lr
    elif step >= sft_cfg.max_step:
        return sft_cfg.min_lr

    progress_percent = (step - sft_cfg.warm_up_steps) / (
        sft_cfg.max_step - sft_cfg.warm_up_steps
    )
    coefficient = 0.5 * (math.cos(math.pi * progress_percent) + 1)
    return coefficient * (sft_cfg.max_lr - sft_cfg.min_lr) + sft_cfg.min_lr


if __name__ == "__main__":
    torch.manual_seed(2026)

    train_ds = load_from_disk(SFT_TRAIN_DIR)
    val_ds = load_from_disk(SFT_VAL_DIR)
    assert isinstance(train_ds, Dataset)
    assert isinstance(val_ds, Dataset)
    sft_cfg = SftConfig()

    run_dir = Path(CKPT_DIR) / f"sft-{datetime.now():%m%d-%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pretrain_best_ckpt = torch.load("checkpoints/best.pt", map_location=sft_cfg.device)
    cfg = MallmConfig(**pretrain_best_ckpt["config"])
    cfg.block_size = SFT_MAX_LEN
    mallm = Mallm(cfg).to(sft_cfg.device)
    mallm.load_state_dict(pretrain_best_ckpt["model"])

    wandb.init(
        project="mallm", name=run_dir.name, config={**asdict(sft_cfg), **asdict(cfg)}
    )

    mallm_compiled = torch.compile(mallm)
    lowest_loss = float("inf")
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    pad_id = tok.token_to_id(PAD)

    decay_param = [param for param in list(mallm.parameters()) if param.dim() >= 2]
    none_decay_param = [param for param in list(mallm.parameters()) if param.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_param, "weight_decay": sft_cfg.weight_decay},
            {"params": none_decay_param, "weight_decay": 0.0},
        ],
        lr=sft_cfg.max_lr,
        betas=sft_cfg.betas,
        fused=(sft_cfg.device == "cuda"),
    )

    for step in range(sft_cfg.max_step):
        optimizer.zero_grad()
        lr = get_lr(step, sft_cfg)

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(
            train_ds, B=sft_cfg.batch_size, pad_id=pad_id, device=sft_cfg.device
        )
        with torch.autocast(sft_cfg.device, dtype=torch.bfloat16):
            logits, loss = mallm_compiled(x, y)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            mallm_compiled.parameters(), sft_cfg.grad_clip
        )

        optimizer.step()

        if step % sft_cfg.eval_interval == 0:
            loss_train = evaluate_loss(
                train_ds, mallm_compiled, pad_id=pad_id, sft_cfg=sft_cfg
            )
            loss_val = evaluate_loss(
                val_ds, mallm_compiled, pad_id=pad_id, sft_cfg=sft_cfg
            )
            print(
                f"step: {step},lr {optimizer.param_groups[0]['lr']}, loss: {loss_train}, valid loss: {loss_val}, grad_norm: {grad_norm.item()}"
            )
            wandb.log(
                {
                    "loss/train": loss_train,
                    "loss/val": loss_val,
                    "lr": lr,
                    "grad_norm": grad_norm.item(),
                },
                step=step,
            )

            if loss_val < lowest_loss:
                lowest_loss = loss_val
                save_ckpt(
                    mallm,
                    run_dir / "best.pt",
                    step + 1,
                    lowest_loss,
                    optimizer.state_dict(),
                )

            save_ckpt(
                mallm,
                run_dir / "last.pt",
                step + 1,
                lowest_loss,
                optimizer.state_dict(),
            )

    save_ckpt(
        mallm,
        run_dir / "last.pt",
        step=sft_cfg.max_step,
        lowest_loss=lowest_loss,
        optimizer_state=optimizer.state_dict(),
    )

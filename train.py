import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import (
    CKPT_DIR,
    TRAIN_BIN,
    VAL_BIN,
    VOCAB_SIZE,
)
from model import Mallm, MallmConfig


def get_batch(data, B, T, device):
    starts = torch.randint(0, len(data) - T, (B,))
    xs = []
    ys = []
    for start in starts:
        row_x = data[start : start + T].astype(np.int64)
        xs.append(row_x)
        row_y = data[start + 1 : start + 1 + T].astype(np.int64)
        ys.append(row_y)

    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys))

    return x.to(device), y.to(device)


@dataclass
class TrainConfig:
    batch_size: int = 32
    device: str = "cuda"
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    max_step: int = 1000
    warm_up_steps: int = 100
    eval_interval: int = 100
    eval_batches_num: int = 20
    grad_clip: float = 1.0


@torch.no_grad()
def evaluate_loss(data, model, block_size, tr_cfg: TrainConfig):
    model.eval()
    losses = []
    for _ in range(tr_cfg.eval_batches_num):
        x, y = get_batch(data, B=tr_cfg.batch_size, T=block_size, device=tr_cfg.device)
        with torch.autocast(tr_cfg.device, dtype=torch.bfloat16):
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


def get_lr(step, tr_cfg: TrainConfig):
    if step < tr_cfg.warm_up_steps:
        return (step / tr_cfg.warm_up_steps) * tr_cfg.max_lr
    elif step >= tr_cfg.max_step:
        return tr_cfg.min_lr

    progress_percent = (step - tr_cfg.warm_up_steps) / (
        tr_cfg.max_step - tr_cfg.warm_up_steps
    )
    coefficient = 0.5 * (math.cos(math.pi * progress_percent) + 1)
    return coefficient * (tr_cfg.max_lr - tr_cfg.min_lr) + tr_cfg.min_lr


if __name__ == "__main__":
    data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    data_val = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

    tr_cfg = TrainConfig()

    resume_from_dir = None

    if resume_from_dir:
        run_dir = Path(resume_from_dir)

        latest_ckpt = torch.load(run_dir / "last.pt")
        cfg = MallmConfig(**latest_ckpt["config"])
        lowest_loss = latest_ckpt["lowest_loss"]
        start_step = latest_ckpt["step"]
    else:
        run_dir = Path(CKPT_DIR) / datetime.now().strftime("%m%d-%H%M")
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = MallmConfig(vocab_size=VOCAB_SIZE)
        lowest_loss = float("inf")
        start_step = 0

    mallm = Mallm(cfg).to(tr_cfg.device)
    optimizer = torch.optim.AdamW(mallm.parameters(), lr=tr_cfg.max_lr)

    if resume_from_dir:
        mallm.load_state_dict(latest_ckpt["model"])
        optimizer.load_state_dict(latest_ckpt["optimizer_state"])

    for step in range(start_step, tr_cfg.max_step):
        optimizer.zero_grad()
        lr = get_lr(step, tr_cfg)

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(
            data, B=tr_cfg.batch_size, T=cfg.block_size, device=tr_cfg.device
        )
        with torch.autocast(tr_cfg.device, dtype=torch.bfloat16):
            logits, loss = mallm(x, y)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(mallm.parameters(), tr_cfg.grad_clip)

        optimizer.step()

        if step % tr_cfg.eval_interval == 0:
            loss_train = evaluate_loss(
                data, mallm, block_size=cfg.block_size, tr_cfg=tr_cfg
            )
            loss_val = evaluate_loss(
                data_val, mallm, block_size=cfg.block_size, tr_cfg=tr_cfg
            )
            print(
                f"step: {step},lr {optimizer.param_groups[0]['lr']}, loss: {loss_train}, valid loss: {loss_val}, grad_norm: {grad_norm.item()}"
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
        step=tr_cfg.max_step,
        lowest_loss=lowest_loss,
        optimizer_state=optimizer.state_dict(),
    )

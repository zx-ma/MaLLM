import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CKPT_PATH,
    TOKENIZER_PATH,
    TRAIN_BIN,
    TRAIN_TXT,
    VAL_BIN,
    VALID_TXT,
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


@torch.no_grad()
def evaluate_loss(data, model, loop_time, batch_size, block_size, device):
    model.eval()
    losses = []
    for _ in range(loop_time):
        x, y = get_batch(data, B=batch_size, T=block_size, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def save_ckpt(mallm, ckpt_path):
    checkpoint = {"model": mallm.state_dict(), "config": asdict(mallm.config)}
    torch.save(checkpoint, ckpt_path)


def get_lr(step, max_lr, min_lr, warm_up_steps, max_step):
    if step < warm_up_steps:
        return (step / warm_up_steps) * max_lr
    elif step >= max_step:
        return min_lr

    progress_percent = (step - warm_up_steps) / (max_step - warm_up_steps)
    coefficient = 0.5 * (math.cos(math.pi * progress_percent) + 1)
    return coefficient * (max_lr - min_lr) + min_lr


if __name__ == "__main__":
    data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    data_val = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

    cfg = MallmConfig(vocab_size=VOCAB_SIZE)

    batch_size = 32
    device = "cuda"
    Path(CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)

    max_lr = 3e-4
    min_lr = 3e-5
    warm_up_steps = 100
    max_step = 1000

    mallm = Mallm(cfg).to(device)

    optimizer = torch.optim.AdamW(mallm.parameters(), lr=max_lr)
    for step in range(max_step):
        optimizer.zero_grad()
        lr = get_lr(
            step,
            max_lr=max_lr,
            min_lr=min_lr,
            warm_up_steps=warm_up_steps,
            max_step=max_step,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(data, B=batch_size, T=cfg.block_size, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits, loss = mallm(x, y)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(mallm.parameters(), 1)

        optimizer.step()

        if step % 100 == 0:
            loss_train = evaluate_loss(
                data,
                mallm,
                20,
                batch_size=batch_size,
                block_size=cfg.block_size,
                device=device,
            )
            loss_val = evaluate_loss(
                data_val,
                mallm,
                20,
                batch_size=batch_size,
                block_size=cfg.block_size,
                device=device,
            )
            print(
                f"step: {step},lr {optimizer.param_groups[0]['lr']}, loss: {loss_train}, valid loss: {loss_val}, grad_norm: {grad_norm.item()}"
            )

            save_ckpt(mallm, CKPT_PATH)

    save_ckpt(mallm, CKPT_PATH)

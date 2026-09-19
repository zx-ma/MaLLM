import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TOKENIZER_PATH, TRAIN_BIN, TRAIN_TXT, VAL_BIN, VALID_TXT, VOCAB_SIZE
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
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


if __name__ == "__main__":
    data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    data_val = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

    cfg = MallmConfig(vocab_size=VOCAB_SIZE)

    batch_size = 32
    device = "cuda"

    mallm = Mallm(cfg).to(device)

    optimizer = torch.optim.AdamW(mallm.parameters(), lr=3e-4)
    for step in range(1000):
        optimizer.zero_grad()

        x, y = get_batch(data, B=batch_size, T=cfg.block_size, device=device)
        logits, loss = mallm(x, y)
        loss.backward()
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
            print(f"step: {step}, loss: {loss_train}, valid loss: {loss_val}")

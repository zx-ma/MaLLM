import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TOKENIZER_PATH, TRAIN_BIN, TRAIN_TXT, VAL_BIN, VALID_TXT, VOCAB_SIZE
from model import Mallm


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


if __name__ == "__main__":
    data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")

    block_size = 256
    device = "cuda"

    mallm = Mallm(
        vocab_size=VOCAB_SIZE, block_size=block_size, d=384, n_head=6, n_layer=6
    ).to(device)

    optimizer = torch.optim.AdamW(mallm.parameters(), lr=3e-4)
    for step in range(1000):
        optimizer.zero_grad()

        x, y = get_batch(data, B=32, T=block_size, device=device)
        logits, loss = mallm(x, y)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step: {step}, loss: {loss.item()}")

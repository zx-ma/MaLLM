from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cos_sin(T, head_dim, device):
    expon = torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim

    inv_freq = 1 / (10000**expon)
    positions = torch.arange(T, dtype=torch.float, device=device)
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    head_dim = x.size(-1)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    x_rotated = torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )

    return x_rotated


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.l1 = nn.Linear(d, 4 * d)
        self.l2 = nn.Linear(4 * d, d)
        self.ac1 = nn.GELU()

    def forward(self, x):
        x = self.l1(x)
        x = self.ac1(x)
        x = self.l2(x)

        return x


class Attention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.nh = n_head
        self.q_project = nn.Linear(d, d)
        self.k_project = nn.Linear(d, d)
        self.v_project = nn.Linear(d, d)

        self.fc = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.size()

        q = self.q_project(x)
        k = self.k_project(x)
        v = self.v_project(x)

        q = q.view(B, T, self.nh, -1).transpose(1, 2)
        k = k.view(B, T, self.nh, -1).transpose(1, 2)
        v = v.view(B, T, self.nh, -1).transpose(1, 2)

        # scores = q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5)
        # mask = torch.tril(torch.ones(T, T, device=x.device))
        # scores = scores.masked_fill(mask == 0, float("-inf"))
        # alpha = torch.softmax(scores, dim=-1)
        # out = alpha @ v

        the_cos, the_sin = rope_cos_sin(T, q.size(-1), x.device)
        q = apply_rope(q, the_cos, the_sin)
        k = apply_rope(k, the_cos, the_sin)

        out = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True)

        out = out.transpose(1, 2).reshape(B, T, -1)

        return self.fc(out)


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.attn = Attention(d, n_head)
        self.ln1 = nn.RMSNorm(d)
        self.mlp = MLP(d)
        self.ln2 = nn.RMSNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class MallmConfig:
    vocab_size: int
    d: int = 640
    block_size: int = 256
    n_head: int = 10
    n_layer: int = 16


class Mallm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.tok_emb = nn.Embedding(config.vocab_size, config.d)
        # self.pos_emb = nn.Embedding(config.block_size, config.d)

        self.blocks = nn.ModuleList(
            [Block(config.d, config.n_head) for _ in range(config.n_layer)]
        )

        self.ln_final = nn.RMSNorm(config.d)

        self.language_model_head = nn.Linear(config.d, config.vocab_size, bias=False)

        self.tok_emb.weight = self.language_model_head.weight

    def forward(self, index, target=None):
        B, T = index.size()
        x = self.tok_emb(index)

        # positions = torch.arange(T, device=index.device)
        # pos = self.pos_emb(positions)
        # x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        logits = self.language_model_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), target.reshape(-1), ignore_index=-1
            )

        return logits, loss

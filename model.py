import torch
import torch.nn as nn


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

        scores = q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))

        scores = scores.masked_fill(mask == 0, float("-inf"))

        alpha = torch.softmax(scores, dim=-1)

        out = alpha @ v
        out = out.transpose(1, 2).reshape(B, T, -1)

        return self.fc(out)


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.attn = Attention(d, n_head)
        self.ln1 = nn.LayerNorm(d)
        self.mlp = MLP(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Mallm(nn.module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

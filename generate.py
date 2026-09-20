from dataclasses import asdict

import torch
from tokenizers import Tokenizer

from config import CKPT_PATH, TOKENIZER_PATH
from model import Mallm, MallmConfig

device = "cuda"
checkpoint = torch.load("checkpoints/mallm-12l-8h.pt", map_location=device)
cfg = MallmConfig(**checkpoint["config"])
mallm = Mallm(cfg).to(device)

mallm.load_state_dict(checkpoint["model"])
mallm.eval()
print(cfg)
print(sum(p.numel() for p in mallm.parameters()))

tok = Tokenizer.from_file(TOKENIZER_PATH)
prompt = "Once upon a time"
ids = tok.encode(prompt).ids
index = torch.tensor([ids], device=device)


max_ouput_token = 200

eot_id = tok.token_to_id("<|endoftext|>")


with torch.no_grad():
    for _ in range(max_ouput_token):
        logits, _ = mallm(index)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = tok.decode([next_id.item()])

        index = torch.cat([index, next_id], dim=1)

        if next_id.item() == eot_id:
            break
        else:
            print(tok.decode([next_id.item()]), end="")

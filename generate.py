import torch
from tokenizers import Tokenizer

from config import TOKENIZER_PATH
from model import Mallm, MallmConfig

device = "cuda"
checkpoint = torch.load("checkpoints/best.pt", map_location=device)
cfg = MallmConfig(**checkpoint["config"])
mallm = Mallm(cfg).to(device)

mallm.load_state_dict(checkpoint["model"])
mallm.eval()
print(cfg)

tok = Tokenizer.from_file(TOKENIZER_PATH)
prompt = "Once upon a time"
ids = tok.encode(prompt).ids
index = torch.tensor([ids], device=device)


max_output_tokens = 600
temperature = 0.9
topk = 50
eot_id = tok.token_to_id("<|endoftext|>")

print(prompt, end="")
with torch.no_grad():
    for _ in range(max_output_tokens):
        logits, _ = mallm(index)

        logits = logits[:, -1, :]

        logits = logits / temperature
        values, _ = torch.topk(logits, topk)
        logits[logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = tok.decode([next_id.item()])

        index = torch.cat([index, next_id], dim=1)

        if next_id.item() == eot_id:
            break
        else:
            print(out, end="")

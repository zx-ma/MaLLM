import torch
from tokenizers import Tokenizer

from config import EOT, TOKENIZER_PATH
from model import Mallm, MallmConfig


@torch.no_grad()
def generate(model, tok, prompt_text, max_new_tokens, temperature, topk, device):
    eot_id = tok.token_to_id(EOT)
    ids = tok.encode(prompt_text).ids
    n_prompt = len(ids)
    index = torch.tensor([ids], device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(index)

        logits = logits[:, -1, :]

        logits = logits / temperature
        values, _ = torch.topk(logits, topk)
        logits[logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eot_id:
            break
        index = torch.cat([index, next_id], dim=1)

    return tok.decode(index[0, n_prompt:].tolist())


if __name__ == "__main__":
    device = "cuda"
    checkpoint = torch.load("checkpoints/best_sft.pt", map_location=device)
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    cfg = MallmConfig(**checkpoint["config"])
    mallm = Mallm(cfg).to(device)

    mallm.load_state_dict(checkpoint["model"])
    mallm.eval()
    # print(cfg)

    # prompt = "Once upon a time"
    prompt = (
        "<|user|>\n"
        "Summary: bob goes into the forest and finds a hidden lake.\n"
        "Words: brave, lake, map\n"
        "<|assistant|>\n"
    )
    ids = tok.encode(prompt).ids
    index = torch.tensor([ids], device=device)

    text = generate(
        mallm,
        tok=tok,
        prompt_text=prompt,
        max_new_tokens=600,
        temperature=0.9,
        topk=50,
        device=device,
    )

    # print(prompt + text)
    print(text)

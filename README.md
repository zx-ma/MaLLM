# mallm

84M-param LLM from scratch on one RTX 5070 Laptop (8 GB).

- **Model**: 16 layers, d 640, 10 heads, RoPE, RMSNorm, SDPA, GELU MLP, tied embeddings, 8k byte-level BPE
- **Pretrain**: TinyStories V2, 1.63 B tokens (3 epochs), 9.5 h, val loss 1.09
- **SFT**: TinyStories-Instruct, 160 k samples, loss on story only, 19 min, val loss 1.30 to ~1.03

```
<|user|>
Summary: ...
Words: ...
<|assistant|>
story<|endoftext|>
```

```bash
uv sync
uv run hf download roneneldan/TinyStories TinyStoriesV2-GPT4-{train,valid}.txt --repo-type dataset --local-dir data/raw
uv run hf download roneneldan/TinyStoriesInstruct TinyStories-Instruct-{train,valid}.txt --repo-type dataset --local-dir data/raw
uv run wandb login
uv run python train_tokenizer.py
uv run python prepare_pretrain.py && uv run python pretrain.py
uv run python prepare_sft.py && uv run python sft.py
uv run python generate.py
```

Config: `MallmConfig` (model.py), `TrainConfig` (pretrain.py), `SftConfig` (sft.py).

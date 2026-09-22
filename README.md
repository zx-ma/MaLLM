# mallm

## Result

| | |
|---|---|
| Params | 84,163,200 |
| Val loss | 1.0916 (ppl 2.98) |
| Time | 9 h 30 min, RTX 5070 Laptop 8 GB |

## Model

| | |
|---|---|
| Layers / dim / heads | 16 / 640 / 10 |
| Context | 256 |
| Vocab | 8192 (byte-level BPE) |
| Norm | RMSNorm, pre-norm |
| Position | RoPE, base 10000 |
| Attention | SDPA, causal |
| MLP | 4×, GELU |
| Embeddings | tied |

## Training

| | |
|---|---|
| Data | TinyStories V2 GPT-4, 542.7 M tokens |
| Steps | 199000 * 8192 tokens = 1.63 B (3 epochs) |
| Optimizer | AdamW fused, betas (0.9, 0.95), wd 0.1 on 2D params |
| LR | 5e-4 to 5e-5 cosine, 2000 warmup |
| Clip / precision | 1.0 / bf16 autocast, `torch.compile` |

## Files

```
config.py             paths, vocab size, special tokens
bpe.py                BPE from scratch (reference, unused)
train_tokenizer.py    trains tokenizer-8k.json
check_tokenizer.py    tokenizer checks
prepare.py            txt → uint16 .bin
model.py              model
train.py              pretraining
generate.py           sampling
```

## Run

```bash
uv sync

mkdir -p data/raw && cd data/raw
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ../..

uv run wandb login
uv run python train_tokenizer.py
uv run python prepare.py
uv run python train.py
uv run python generate.py
```

`MallmConfig` in `model.py`, `TrainConfig` in `train.py`.


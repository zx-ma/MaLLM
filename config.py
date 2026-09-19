TRAIN_TXT = "data/raw/TinyStoriesV2-GPT4-train.txt"
VALID_TXT = "data/raw/TinyStoriesV2-GPT4-valid.txt"

VOCAB_SIZE = 8192
SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]
TOKENIZER_PATH = f"tokenizer-{VOCAB_SIZE // 1024}k.json"

TRAIN_BIN = "data/processed/train.bin"
VAL_BIN = "data/processed/val.bin"

CKPT_PATH = "checkpoints/mallm.pt"

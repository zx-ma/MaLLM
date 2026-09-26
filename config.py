TRAIN_TXT = "data/raw/TinyStoriesV2-GPT4-train.txt"
VALID_TXT = "data/raw/TinyStoriesV2-GPT4-valid.txt"

VOCAB_SIZE = 8192

EOT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
PAD = "<|pad|>"
SPECIALS = [EOT, USER, ASSISTANT, PAD]

# SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]
TOKENIZER_PATH = f"tokenizer-{VOCAB_SIZE // 1024}k.json"

TRAIN_BIN = "data/processed/train.bin"
VAL_BIN = "data/processed/val.bin"

SFT_TRAIN_TXT = "data/raw/TinyStories-Instruct-train.txt"
SFT_VALID_TXT = "data/raw/TinyStories-Instruct-valid.txt"
SFT_TRAIN_DIR = "data/processed/sft_train"
SFT_VAL_DIR = "data/processed/sft_val"
SFT_MAX_LEN = 512

CKPT_DIR = "checkpoints"

from tokenizers import Tokenizer

TOKENIZER_PATH = "tokenizer-8k.json"
DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]


tok = Tokenizer.from_file(TOKENIZER_PATH)


with open(DATA_PATH, "r", encoding="utf-8") as f:
    sample = f.read(5000)

    enc = tok.encode(sample)
    assert tok.decode(enc.ids, skip_special_tokens=False) == sample

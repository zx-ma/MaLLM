from tokenizers import Tokenizer

from config import TOKENIZER_PATH, TRAIN_TXT

tok = Tokenizer.from_file(TOKENIZER_PATH)

with open(TRAIN_TXT, encoding="utf-8") as f:
    sample = f.read(5000)

enc = tok.encode(sample)
assert tok.decode(enc.ids, skip_special_tokens=False) == sample

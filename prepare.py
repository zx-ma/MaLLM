from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from config import TOKENIZER_PATH, TRAIN_BIN, TRAIN_TXT, VAL_BIN, VALID_TXT

Path(TRAIN_BIN).parent.mkdir(parents=True, exist_ok=True)


def batch2file(storys, tok, out_f):
    encs = tok.encode_batch(storys)
    ids = []
    for e in encs:
        ids.extend(e.ids)
    arr = np.array(ids, dtype=np.uint16)

    arr.tofile(out_f)
    return len(arr)


tok = Tokenizer.from_file(TOKENIZER_PATH)
lines = []
storys = []
n_docs = 0
total = 0

with open(VALID_TXT, encoding="utf-8") as f, open(VAL_BIN, mode="wb") as wf:
    for line in f:
        lines.append(line)
        if line.strip() == "<|endoftext|>":
            storys.append("".join(lines))
            lines = []
            if len(storys) >= 1000:
                token_num = batch2file(storys, tok, wf)
                total += token_num
                storys = []
            n_docs += 1

    if storys:
        total += batch2file(storys, tok, wf)

print(f"{n_docs=}")
print(f"{total=}")

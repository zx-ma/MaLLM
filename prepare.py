from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from config import TOKENIZER_PATH, TRAIN_BIN, TRAIN_TXT, VAL_BIN, VALID_TXT


def batch2file(storys, tok, out_f):
    encs = tok.encode_batch(storys)
    ids = []
    for e in encs:
        ids.extend(e.ids)
    arr = np.array(ids, dtype=np.uint16)

    arr.tofile(out_f)
    return len(arr)


def encode_file(tokenizer_path, text_path, bin_path):
    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(tokenizer_path)
    lines = []
    stories = []
    n_docs = 0
    total = 0

    with open(text_path, encoding="utf-8") as f, open(bin_path, mode="wb") as wf:
        for line in f:
            lines.append(line)
            if line.strip() == "<|endoftext|>":
                stories.append("".join(lines))
                lines = []
                if len(stories) >= 1000:
                    token_num = batch2file(stories, tok, wf)
                    total += token_num
                    stories = []
                n_docs += 1

        if stories:
            total += batch2file(stories, tok, wf)

    return n_docs, total


if __name__ == "__main__":
    n_docs, total = encode_file(TOKENIZER_PATH, TRAIN_TXT, TRAIN_BIN)
    print(f"n_docs:{n_docs}, total: {total}")

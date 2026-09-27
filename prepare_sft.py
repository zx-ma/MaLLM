import random
import re

from datasets import Dataset, Features, List, Value
from tokenizers import Tokenizer

from config import (
    ASSISTANT,
    EOT,
    SFT_MAX_LEN,
    SFT_TRAIN_DIR,
    SFT_TRAIN_TXT,
    SFT_VAL_DIR,
    SFT_VALID_TXT,
    TOKENIZER_PATH,
    USER,
)

FEATURES = Features(
    {
        "input_ids": List(Value("uint16")),
        "loss_mask": List(Value("uint8")),
    }
)


def to_dict(a_text: str, pattern: re.Pattern):
    list_res = pattern.split(a_text)
    res = {}
    i = 1
    while i + 1 < len(list_res):
        res[list_res[i]] = list_res[i + 1].strip()
        i += 2

    if "Story" not in res:
        return None

    return res


def encode(prompt, response, tok: Tokenizer):
    prompt_enc = tok.encode(prompt)
    p_ids = prompt_enc.ids
    response_enc = tok.encode(response)
    r_ids = response_enc.ids
    ids = p_ids + r_ids
    mask = [0] * len(p_ids) + [1] * len(r_ids)

    if len(ids) > SFT_MAX_LEN:
        return None
    return ids, mask


def to_prompt_response(a_dict):
    prompt = [
        USER,
    ]

    keys = [k for k in a_dict if k != "Story"]
    random.shuffle(keys)
    for key in keys:
        prompt.append(f"{key}: {a_dict[key]}")

    response = a_dict["Story"]

    prompt.append(f"{ASSISTANT}\n")
    response += EOT
    return "\n".join(prompt), response


def sft_data_generator(text_path, pattern, tok: Tokenizer):
    one_data_list = []
    with open(text_path, encoding="utf-8") as f_text:
        for line in f_text:
            if line.strip() == EOT:
                one_data_str = "".join(one_data_list)
                one_data_list = []
                a_dict = to_dict(one_data_str, pattern)
                if a_dict is None:
                    continue
                prompt, response = to_prompt_response(a_dict)
                res = encode(prompt, response, tok)

                if res is None:
                    continue

                ids, mask = res

                yield {"input_ids": ids, "loss_mask": mask}

            else:
                one_data_list.append(line)


if __name__ == "__main__":
    random.seed(2026)
    pattern = re.compile(
        r"^(Features|Summary|Words|Random sentence|Story):", flags=re.M
    )

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    for split, text_path, out_dir in [
        ("train", SFT_TRAIN_TXT, SFT_TRAIN_DIR),
        ("valid", SFT_VALID_TXT, SFT_VAL_DIR),
    ]:
        ds = Dataset.from_generator(
            sft_data_generator,
            gen_kwargs={"text_path": text_path, "pattern": pattern, "tok": tokenizer},
            features=FEATURES,
        )
        ds.save_to_disk(out_dir)
        print(split, len(ds))

from collections import Counter
from pathlib import Path


def get_pair_count(arr):
    return Counter(zip(arr, arr[1:]))


def my_merge(arr, pair, new_id):
    res = []
    idx = 0
    while idx < len(arr) - 1:
        cur_pair = (arr[idx], arr[idx + 1])
        if cur_pair == pair:
            res.append(new_id)
            idx += 2
        else:
            res.append(arr[idx])
            idx += 1

    if idx == len(arr) - 1:
        res.append(arr[idx])

    return res


def train(text: str, vocab_size):
    merged_dict = {}
    arr = list(text.encode("utf-8"))
    # vocab = {idx: bytes([idx]) for idx in range(256)}
    # print(f"{vocab=} \n")

    for i in range(vocab_size - 256):
        count_dic = get_pair_count(arr)
        if not count_dic:
            break

        pair = count_dic.most_common(1)[0][0]
        merged_dict[pair] = 256 + i
        # vocab[256 + i] = vocab[pair[0]] + vocab[pair[1]]

        # print(vocab[256 + i])
        arr = my_merge(arr, pair, 256 + i)

    return merged_dict


def build_vocab(merged_dict):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for pair, val in merged_dict.items():
        vocab[val] = vocab[pair[0]] + vocab[pair[1]]

    return vocab


def encode(text, merged_dic):
    arr = list(text.encode("utf-8"))
    for pair, val in merged_dic.items():
        arr = my_merge(arr, pair, val)

    return arr


def decode(arr, vocab):
    byte_arr = [vocab[index] for index in arr]
    text = b"".join(byte_arr).decode("utf-8")

    return text


if __name__ == "__main__":
    print(get_pair_count([1, 2, 3, 1, 2]))
    print(my_merge([1, 2, 3, 1, 2], (1, 2), 4))
    print(my_merge([1, 2, 3], (1, 2), 4))

    with Path("data", "TinyStoriesV2-GPT4-valid.txt").open(encoding="utf-8") as f:
        text = f.read(100000)
        merged_dic = train(text, 500)
        vocab = build_vocab(merged_dic)
        # print(vocab)
        encoded = encode(text, merged_dic)
        # print(encoded)
        decoded = decode(encoded, vocab)
        print(decoded)

    # print(train("hi", 300))

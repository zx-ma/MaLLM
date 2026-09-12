from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

data_path = "data/TinyStoriesV2-GPT4-valid.txt"
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
# print(tok.get_vocab_size())
bpe_trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)


tok.train([data_path], bpe_trainer)
tok.save("tokenizer-8k.json")

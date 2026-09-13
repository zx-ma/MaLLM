from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from config import SPECIALS, TOKENIZER_PATH, TRAIN_TXT, VOCAB_SIZE

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()

bpe_trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIALS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tok.train([TRAIN_TXT], bpe_trainer)
print("vocab_size", tok.get_vocab_size())

tok.save(TOKENIZER_PATH)

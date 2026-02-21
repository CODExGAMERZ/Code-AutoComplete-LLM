import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC

DATA_PATH = "data/processed/train.txt"
OUTPUT_PATH = "tokenizer/tokenizer.json"
VOCAB_SIZE = 8000


def main():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )

    tokenizer.train([DATA_PATH], trainer)

    tokenizer.save(OUTPUT_PATH)
    print(f"Tokenizer saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
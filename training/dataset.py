import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class CodeDataset(Dataset):
    def __init__(self, path, tokenizer_path, block_size=256):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokens = self.tokenizer.encode(text).ids
        self.block_size = block_size

        # Number of full blocks available
        self.num_blocks = (len(self.tokens) - 1) // self.block_size

        print(f"[DATASET] Total tokens: {len(self.tokens)}")
        print(f"[DATASET] Block size: {self.block_size}")
        print(f"[DATASET] Total training blocks: {self.num_blocks}")

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size

        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1:end + 1], dtype=torch.long)

        return x, y
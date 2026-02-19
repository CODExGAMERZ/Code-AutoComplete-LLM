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
        print(f"[DATASET] Total tokens: {len(self.tokens)} | Block size: {self.block_size}")

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.block_size])
        y = torch.tensor(self.tokens[idx + 1:idx + self.block_size + 1])
        return x, y

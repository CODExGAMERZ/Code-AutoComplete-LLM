import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
from tokenizers import Tokenizer
from model.ai import GPT

DEVICE = "cpu"
VOCAB_SIZE = 8000
BLOCK_SIZE = 256

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

model = GPT(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_layers=8,
    n_heads=8,
    n_embd=512
)

model.load_state_dict(torch.load("model/model_60M.pth", map_location=DEVICE))
model.eval()

def autocomplete(prompt, max_tokens=50):
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids).unsqueeze(0)

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(x)
        next_token = torch.argmax(logits[0, -1])
        x = torch.cat([x, next_token.view(1, 1)], dim=1)

    return tokenizer.decode(x[0].tolist())

print(autocomplete("def merge_sort(arr):"))

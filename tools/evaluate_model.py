import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from model.ai import GPT
from training.dataset import CodeDataset

DEVICE = "cpu"
CHECKPOINT_PATH = "model/checkpoints/latest_checkpoint.pth"

model = GPT(
    vocab_size=8000,
    block_size=256,
    n_layers=8,
    n_heads=8,
    n_embd=512,
).to(DEVICE)

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

dataset = CodeDataset(
    path="data/processed/train.txt",
    tokenizer_path="tokenizer/tokenizer.json",
    block_size=256,
)

loader = DataLoader(dataset, batch_size=16)

criterion = nn.CrossEntropyLoss()

total_loss = 0
total_batches = 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        total_loss += loss.item()
        total_batches += 1

avg_loss = total_loss / total_batches
perplexity = math.exp(avg_loss)

print("===== EVALUATION =====")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
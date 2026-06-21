import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import sys
import os
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from model.ai import GPT
from training.dataset import CodeDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "model/checkpoints/latest_checkpoint.pth"

MAX_BATCHES = 300

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

print("\nStarting evaluation...\n")

with torch.no_grad():
    loop = tqdm(loader, total=MAX_BATCHES)

    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()
        total_batches += 1

        avg_loss_so_far = total_loss / total_batches
        loop.set_postfix(avg_loss=avg_loss_so_far)

        if total_batches >= MAX_BATCHES:
            break

avg_loss = total_loss / total_batches
perplexity = math.exp(avg_loss)

print("\n===== FAST EVALUATION COMPLETE =====")
print(f"Batches Evaluated: {total_batches}")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
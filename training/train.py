import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader
from model.ai import GPT
from training.dataset import CodeDataset

DEVICE = "cpu"
VOCAB_SIZE = 8000
BLOCK_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 3e-4
NUM_WORKERS = 0

def main():
    dataset = CodeDataset(
        path="data/processed/train.txt",
        tokenizer_path="tokenizer/tokenizer.json",
        block_size=BLOCK_SIZE
    )

    if len(dataset) == 0:
        raise ValueError("Dataset too small")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    model = GPT(VOCAB_SIZE, BLOCK_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch+1} Step {step}/{len(loader)} Loss {loss.item():.4f}")

        print(f"Epoch {epoch+1} Avg Loss {(total_loss / len(loader)):.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/model.pth")

if __name__ == "__main__":
    main()
    print("Done")
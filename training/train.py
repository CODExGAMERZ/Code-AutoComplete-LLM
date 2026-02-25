import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import argparse

from model.ai import GPT
from training.dataset import CodeDataset

DEVICE = "cpu"
CHECKPOINT_PATH = "model/checkpoints/latest_checkpoint.pth"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    BLOCK_SIZE = 256
    VOCAB_SIZE = 8000

    train_dataset = CodeDataset(
        path="data/processed/train.txt",
        tokenizer_path="tokenizer/tokenizer.json",
        block_size=BLOCK_SIZE
    )

    val_dataset = CodeDataset(
        path="data/processed/val.txt",
        tokenizer_path="tokenizer/tokenizer.json",
        block_size=BLOCK_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("model/checkpoints", exist_ok=True)

    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        print("Resuming from checkpoint...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss = 0

            pbar = tqdm(train_loader)

            optimizer.zero_grad()

            for step, (x, y) in enumerate(pbar):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits, _ = model(x)

                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

                loss.backward()

                if (step + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    logits, _ = model(x)
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            perplexity = math.exp(val_loss)

            print(f"\nEpoch {epoch} Validation Loss: {val_loss:.4f} | Perplexity: {perplexity:.4f}")

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch
                },
                CHECKPOINT_PATH
            )

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch
            },
            CHECKPOINT_PATH
        )
        print("Checkpoint saved.")

    print("Training complete.")


if __name__ == "__main__":
    main()
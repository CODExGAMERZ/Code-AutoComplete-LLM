import os
import sys
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from model.ai import GPT
from training.dataset import CodeDataset

DEVICE = "cpu"

VOCAB_SIZE = 8000
BLOCK_SIZE = 256

CHECKPOINT_PATH = "model/checkpoints/latest_checkpoint.pth"
TRAIN_PATH = "data/processed/train.txt"
VAL_PATH = "data/processed/val.txt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_dataset = CodeDataset(
        path=TRAIN_PATH,
        tokenizer_path=TOKENIZER_PATH,
        block_size=BLOCK_SIZE,
    )

    val_dataset = CodeDataset(
        path=VAL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        block_size=BLOCK_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    global_step = 0

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]

    model.train()

    try:
        for epoch in range(start_epoch, args.epochs):
            progress = tqdm(train_loader)
            optimizer.zero_grad()

            for step, (x, y) in enumerate(progress):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )

                loss = loss / args.grad_accum
                loss.backward()

                if (step + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                progress.set_postfix(loss=loss.item() * args.grad_accum)

            val_loss = evaluate(model, val_loader, loss_fn)
            val_ppl = math.exp(val_loss)

            print(f"\nEpoch {epoch} Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.4f}\n")

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "step": global_step,
                },
                CHECKPOINT_PATH,
            )

    except KeyboardInterrupt:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
            },
            CHECKPOINT_PATH,
        )
        print("Training interrupted. Checkpoint saved.")

    print("Training complete.")

if __name__ == "__main__":
    main()
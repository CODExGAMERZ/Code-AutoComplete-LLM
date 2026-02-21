import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import csv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from training.dataset import CodeDataset
from model.ai import GPT

DEVICE = "cpu"
CHECKPOINT_PATH = "model/checkpoints/latest_checkpoint.pth"
LOG_PATH = "training_logs/loss_log.csv"

VOCAB_SIZE = 8000
BLOCK_SIZE = 256

cpu_count = os.cpu_count()
torch.set_num_threads(int(cpu_count * 0.75))
torch.set_num_interop_threads(2)


def save_checkpoint(model, optimizer, scheduler, epoch, step):
    os.makedirs("model/checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
    }, CHECKPOINT_PATH)


def load_checkpoint(model, optimizer, scheduler):
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt["epoch"], ckpt["step"]
    return 0, 0


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    os.makedirs("training_logs", exist_ok=True)

    dataset = CodeDataset(args.data, args.tokenizer, block_size=BLOCK_SIZE)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True
    )

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512,
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    total_steps = (len(dataloader) * args.epochs) // args.grad_accum
    scheduler = get_scheduler(optimizer, args.warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss()

    start_epoch, global_step = load_checkpoint(model, optimizer, scheduler)

    with open(LOG_PATH, "a", newline="") as log_file:
        writer = csv.writer(log_file)

        try:
            for epoch in range(start_epoch, args.epochs):
                model.train()
                loop = tqdm(dataloader, desc=f"Epoch {epoch}")

                optimizer.zero_grad()

                for step, (x, y) in enumerate(loop):
                    x, y = x.to(DEVICE), y.to(DEVICE)

                    logits = model(x)
                    loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )

                    loss = loss / args.grad_accum
                    loss.backward()

                    if (step + 1) % args.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        writer.writerow([global_step, loss.item() * args.grad_accum])
                        log_file.flush()

                    loop.set_postfix(loss=loss.item() * args.grad_accum)

                save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step)

        except KeyboardInterrupt:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/train.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)

    args = parser.parse_args()
    train(args)
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from model.ai import GPT
from training.dataset import CodeDataset

DEVICE = "cpu"
VOCAB_SIZE = 8000
BLOCK_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 3e-4
NUM_WORKERS = 0

CHECKPOINT_DIR = "model/checkpoints"
CHECKPOINT_EVERY = 5000


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = CodeDataset(
        path="data/processed/train.txt",
        tokenizer_path="tokenizer/tokenizer.json",
        block_size=BLOCK_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    global_step = 0

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
            global_step += 1

            if step % 10 == 0:
                print(
                    f"Epoch {epoch+1} "
                    f"Step {step}/{len(loader)} "
                    f"Loss {loss.item():.4f}"
                )

            if global_step % CHECKPOINT_EVERY == 0:
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"ckpt_e{epoch}_s{global_step}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    ckpt_path
                )
                print(f"💾 Saved checkpoint: {ckpt_path}")

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch+1} finished | Avg Loss {avg_loss:.4f}")

    torch.save(model.state_dict(), "model/model_60M_final.pth")
    print("🎉 Training finished. Final model saved.")


if __name__ == "__main__":
    main()

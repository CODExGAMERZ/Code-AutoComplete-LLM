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
EPOCHS = 10          # You can raise this if you want a longer run
LEARNING_RATE = 3e-4
NUM_WORKERS = 0

# Checkpoint settings
START_CHECKPOINT_PATH = "model/checkpoints/ckpt_e0_s5000.pth"
CHECKPOINT_DIR = "model/checkpoints"
CHECKPOINT_EVERY = 50000  # Save every 50,000 steps

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt["step"]

def save_checkpoint(model, optimizer, epoch, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(
        CHECKPOINT_DIR,
        f"ckpt_resume_e{epoch}_s{step}.pth"
    )
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        },
        path
    )
    print(f"💾 Saved checkpoint: {path}")

def main():
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

    start_epoch = 0
    start_step = 0

    # Load existing checkpoint if available
    if os.path.exists(START_CHECKPOINT_PATH):
        start_epoch, start_step = load_checkpoint(
            START_CHECKPOINT_PATH, model, optimizer
        )
        print(f"🔄 Resuming from epoch {start_epoch}, step {start_step}")

    global_step = start_step

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0

        for step, (x, y) in enumerate(loader, start=0):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            total_loss += loss.item()

            if global_step % 10 == 0:
                print(
                    f"Epoch {epoch+1} Step {global_step} "
                    f"Loss {loss.item():.4f}"
                )

            # Save checkpoint every 50,000 steps
            if global_step % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, optimizer, epoch, global_step)

        avg_loss = total_loss / len(loader)
        print(f"📊 Epoch {epoch+1} complete | Avg Loss {avg_loss:.4f}")

    # Final save after training
    final_path = os.path.join("model", "model_60M_final_resumed.pth")
    torch.save(model.state_dict(), final_path)
    print(f"🎉 Training complete — final model saved at {final_path}")


if __name__ == "__main__":
    main()

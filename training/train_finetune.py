import os
import sys
import json
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
EPOCHS = 2
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY = 5000

START_CHECKPOINT = "model/checkpoints/ckpt_algo_ft_s55000.pth"
CHECKPOINT_DIR = "model/checkpoints"


def main():
    os.makedirs("analysis", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = CodeDataset(
        path="data/processed/train_finetune.txt",
        tokenizer_path="tokenizer/tokenizer.json",
        block_size=BLOCK_SIZE
    )

    print(f"[DATASET] Total tokens: {len(dataset.tokens)} | Block size: {BLOCK_SIZE}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
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

    ckpt = torch.load(START_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    global_step = ckpt["step"]

    print(f"🔄 Fine-tuning from step {global_step}")

    loss_history = []

    try:
        for epoch in range(EPOCHS):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                logits = model(x)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                global_step += 1

                if global_step % 10 == 0:
                    print(f"FT Step {global_step} Loss {loss.item():.4f}")

                if global_step % CHECKPOINT_EVERY == 0:
                    ckpt_path = os.path.join(
                        CHECKPOINT_DIR,
                        f"ckpt_algo_ft_s{global_step}.pth"
                    )
                    torch.save(
                        {
                            "step": global_step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict()
                        },
                        ckpt_path
                    )
                    print(f"💾 Saved algo fine-tune checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        print("⛔ Training interrupted by user")

    finally:
        with open("analysis/loss_history.json", "w") as f:
            json.dump(loss_history, f)
        print("📊 Saved entropy (loss) history to analysis/loss_history.json")

        final_model_path = "model/model_60M_algo_finetuned.pth"
        torch.save(model.state_dict(), final_model_path)
        print(f"🎉 Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()

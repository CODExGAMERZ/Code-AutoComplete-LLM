import os
import sys
import torch
from torch.utils.data import DataLoader

torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from model.ai import GPT
from training.dataset import CodeDataset

DEVICE = "cpu"

VOCAB_SIZE = 8000
BLOCK_SIZE = 256
BATCH_SIZE = 6
LR = 1e-4
EPOCHS = 1

DATA_PATH = "data/processed/train_balanced.txt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
CHECKPOINT_PATH = "model/checkpoints/latest_checkpoint.pth"

def main():
    dataset = CodeDataset(
        path=DATA_PATH,
        tokenizer_path=TOKENIZER_PATH,
        block_size=BLOCK_SIZE,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512,
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"], strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    try:
        for epoch in range(EPOCHS):
            for step, (x, y) in enumerate(loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    except KeyboardInterrupt:
        print("Interrupted, saving checkpoint")

    torch.save(
        {
            "model_state": model.state_dict(),
        },
        CHECKPOINT_PATH,
    )

    print("Balanced fine-tune complete")

if __name__ == "__main__":
    main()
import argparse
import torch
import sys
import os
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from model.ai import GPT
from tokenizers import Tokenizer

DEVICE = "cpu"
VOCAB_SIZE = 8000
BLOCK_SIZE = 256


def clean_output(text: str) -> str:
    if "# =====" in text:
        text = text.split("# =====")[0]

    text = re.sub(r"dfs_\d+\(", "dfs(", text)
    text = re.sub(r"binary_search_\d+\(", "binary_search(", text)
    text = re.sub(r"Stack_\d+", "Stack", text)

    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if line.strip().startswith("def ") and len(cleaned) > 0:
            break
        if line.strip().startswith("class ") and len(cleaned) > 0:
            break
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def generate(model, tokenizer, prompt, max_tokens, temperature, top_k, rep_penalty):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x)
            logits = logits[:, -1, :]
            logits = logits / temperature

            for token_id in set(x[0].tolist()):
                logits[0, token_id] /= rep_penalty

            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, indices, values)
                logits = mask

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)

    text = tokenizer.decode(x[0].tolist())
    return clean_output(text)


def load_model(checkpoint_path):
    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512,
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-m", "--max_tokens", type=int, default=120)
    parser.add_argument("-t", "--temperature", type=float, default=0.2)
    parser.add_argument("-k", "--top_k", type=int, default=20)
    parser.add_argument("--rep_penalty", type=float, default=1.15)

    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    model = load_model(args.checkpoint)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rep_penalty=args.rep_penalty,
    )

    print(output)


if __name__ == "__main__":
    main()
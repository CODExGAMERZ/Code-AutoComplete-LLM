import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import argparse
from tokenizers import Tokenizer
from model.ai import GPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 8000
BLOCK_SIZE = 256

def generate(model, tokenizer, prompt, max_tokens=120, temperature=0.2, top_k=10):
    model.eval()

    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    past_kvs = None

    with torch.no_grad():
        logits, past_kvs = model(x)

        for _ in range(max_tokens):
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, ix = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            logits, past_kvs = model(next_token, past_kvs)
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("--mode", choices=["autocomplete", "creative"], default="autocomplete")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512
    )

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)

    if args.mode == "autocomplete":
        temperature = 0.2
        top_k = 10
    else:
        temperature = 0.8
        top_k = 50

    output = generate(model, tokenizer, args.prompt, temperature=temperature, top_k=top_k)

    print(output)


if __name__ == "__main__":
    main()
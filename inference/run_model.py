import argparse
import torch
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from tokenizers import Tokenizer
from model.ai import GPT

def load_model(checkpoint_path=None):
    DEVICE = "cpu"
    VOCAB_SIZE = 8000
    BLOCK_SIZE = 256

    print("🔹 Loading tokenizer...")
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    print("🔹 Building model architecture...")
    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512
    ).to(DEVICE)

    if checkpoint_path:
        print(f"🔹 Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("⚠ No checkpoint specified — using untrained weights")

    model.eval()
    return model, tokenizer

def sample_next_token(logits, temperature=0.8, top_k=50):
    logits = logits / temperature
    values, indices = torch.topk(logits, top_k)
    probs = torch.softmax(values, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    return indices[choice].item()

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50):
    print("🔹 Encoding prompt...")
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    print("🔹 Generating tokens...")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            logits = logits[0, -1]

            # temperature + top-k
            filtered_logits, filtered_indices = torch.topk(logits, top_k)
            filtered_logits = filtered_logits / temperature
            probs = torch.softmax(filtered_logits, dim=-1)
            next_id = filtered_indices[torch.multinomial(probs, num_samples=1)].item()

            x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

    decoded = tokenizer.decode(x[0].tolist())

    # remove weird spacing
    decoded = decoded.replace("  ", " ").replace(" .", ".").replace(" ( ", "(").replace(" )", ")")
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Code prompt to complete")
    parser.add_argument("-m", "--max_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("-k", "--top_k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()

    print("🔹 Loading model...")
    model, tokenizer = load_model(args.checkpoint)

    print("\n=== PROMPT ===")
    print(args.prompt)

    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print("\n=== GENERATED ===")
    print(output)

if __name__ == "__main__":
    main()

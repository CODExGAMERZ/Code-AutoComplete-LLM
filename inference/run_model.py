import os
import sys
import argparse
import torch
from tokenizers import Tokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from model.ai import GPT

DEVICE = "cpu"

VOCAB_SIZE = 8000
BLOCK_SIZE = 256


def apply_repetition_penalty(logits, generated_ids, penalty=1.1):
    for token_id in set(generated_ids):
        logits[token_id] /= penalty
    return logits


def sample_next_token(
    logits,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    generated_ids=None,
):
    logits = logits / temperature

    if generated_ids is not None:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    probs = torch.softmax(logits, dim=-1)

    if top_k > 0:
        values, indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs)
        probs[indices] = values
        probs /= probs.sum()

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False

        sorted_probs[cutoff] = 0
        sorted_probs /= sorted_probs.sum()

        idx = torch.multinomial(sorted_probs, 1)
        return sorted_idx[idx].item()

    return torch.multinomial(probs, 1).item()


def clean_python_spacing(text: str) -> str:
    replacements = {
        " .": ".",
        " ,": ",",
        " :": ":",
        " ;": ";",
        " )": ")",
        "( ": "(",
        "[ ": "[",
        " ]": "]",
        "{ ": "{",
        " }": "}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = text.replace("fi b on ac ci", "fibonacci")
    text = text.replace("di j k stra", "dijkstra")

    return text


def generate(
    model,
    tokenizer,
    prompt,
    max_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
):
    model.eval()

    input_ids = tokenizer.encode(prompt).ids
    generated = input_ids.copy()

    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x)
            next_logits = logits[0, -1]

            next_id = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated,
            )

            generated.append(next_id)

            x = torch.tensor([generated[-BLOCK_SIZE:]], dtype=torch.long).to(DEVICE)

    text = tokenizer.decode(generated)
    return clean_python_spacing(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-m", "--max_tokens", type=int, default=120)
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-k", "--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--rep_penalty", type=float, default=1.1)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512,
    ).to(DEVICE)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

    print("\n=== PROMPT ===")
    print(args.prompt)

    output = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.rep_penalty,
    )

    print("\n=== GENERATED ===")
    print(output)


if __name__ == "__main__":
    main()
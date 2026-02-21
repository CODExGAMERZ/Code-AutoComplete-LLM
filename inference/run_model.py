import argparse
import torch
import sys
import os
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from model.ai import GPT
from tokenizers import Tokenizer

DEVICE = "cpu"
VOCAB_SIZE = 8000
BLOCK_SIZE = 256

STOP_STRINGS = [
    "\n\n\n",
    "\ndef ",
    "\nclass ",
    "if __name__ ==",
]

TEMPLATE_DRIFT_PATTERNS = [
    "a, b = 0, 1",
    "result = []",
    "for _ in range(n)",
]


def clean_python(code: str) -> str:
    lines = code.splitlines()
    cleaned = []

    for line in lines:
        if "### TASK" in line or "### LANGUAGE" in line:
            break
        cleaned.append(line)

    code = "\n".join(cleaned)
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()


def generate(
    model,
    tokenizer,
    prompt,
    max_tokens,
    temperature,
    top_k,
    rep_penalty,
    deterministic=False,
):
    model.eval()

    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    is_class_context = "class " in prompt
    is_method_context = "self" in prompt
    is_recursive_context = "return" in prompt
    is_graph_context = "graph" in prompt or "dfs" in prompt.lower()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x)
            logits = logits[:, -1, :]

            if deterministic:
                temperature = 0.0
                top_k = 1

            if is_class_context or is_method_context:
                temperature = min(temperature, 0.15)

            if is_graph_context:
                temperature = min(temperature, 0.15)

            if temperature > 0:
                logits = logits / temperature

            for token_id in set(x[0].tolist()):
                logits[0, token_id] /= rep_penalty

            decoded_so_far = tokenizer.decode(x[0].tolist())

            for pattern in TEMPLATE_DRIFT_PATTERNS:
                if pattern in decoded_so_far:
                    logits *= 0.90

            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, indices, values)
                logits = mask

            probs = torch.softmax(logits, dim=-1)

            if deterministic:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, 1)

            x = torch.cat([x, next_id], dim=1)

            decoded = tokenizer.decode(x[0].tolist())

            for stop in STOP_STRINGS:
                if stop in decoded:
                    return clean_python(decoded.split(stop)[0])

    return clean_python(tokenizer.decode(x[0].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-m", "--max_tokens", type=int, default=120)
    parser.add_argument("-t", "--temperature", type=float, default=0.2)
    parser.add_argument("-k", "--top_k", type=int, default=10)
    parser.add_argument("--rep_penalty", type=float, default=1.2)
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layers=8,
        n_heads=8,
        n_embd=512,
    )

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(DEVICE)
    model.eval()

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rep_penalty=args.rep_penalty,
        deterministic=args.deterministic,
    )

    print(output)


if __name__ == "__main__":
    main()
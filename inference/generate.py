import torch
import torch.nn.functional as F

STOP_STRINGS = [
    "\n\n\n",
    "if __name__ ==",
    "# End",
    "\ndef ",
    "\nclass ",
]

def clean_python(code):
    """Clean up generated Python code by removing incomplete statements."""
    lines = code.split('\n')
    cleaned = []
    for line in lines:
        if any(stop in line for stop in STOP_STRINGS):
            break
        cleaned.append(line)
    return '\n'.join(cleaned)

def apply_repetition_penalty(logits, generated_ids, penalty=1.1):
    unique_tokens = set(generated_ids)
    for token_id in unique_tokens:
        logits[0, token_id] /= penalty
    return logits

def top_k_filtering(logits, top_k=40):
    if top_k <= 0:
        return logits
    values, indices = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(1)
    logits[logits < min_values] = -float("Inf")
    return logits

@torch.no_grad()
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
    is_dfs_context = "dfs" in prompt.lower()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x)
            logits = logits[:, -1, :]

            if deterministic:
                temperature = 0.0
                top_k = 1

            if temperature > 0:
                logits = logits / temperature

            for token_id in set(x[0].tolist()):
                logits[0, token_id] /= rep_penalty

            decoded_so_far = tokenizer.decode(x[0].tolist())

            if is_class_context and "a, b =" in decoded_so_far:
                logits *= 0.95

            if is_method_context:
                temperature = min(temperature, 0.15)

            if is_dfs_context:
                temperature = min(temperature, 0.15)

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

            if "\ndef " in decoded or "\nclass " in decoded:
                break

    return clean_python(tokenizer.decode(x[0].tolist()))
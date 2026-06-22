import os
import random

CLEAN_DIR = "data/cleaned"
OUTPUT = "data/processed/train.txt"
VAL_OUTPUT = "data/processed/val.txt"

VAL_SPLIT = 0.1

def build():
    all_code = []

    allowed_exts = (".py", ".c", ".h", ".java")
    for root, _, files in os.walk(CLEAN_DIR):
        for file in files:
            if file.endswith(allowed_exts):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if len(content) > 50:
                        all_code.append(content)

    random.shuffle(all_code)

    split = int(len(all_code) * (1 - VAL_SPLIT))
    train = all_code[:split]
    val = all_code[split:]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("".join(f"<bos>{doc}<eos>\n" for doc in train))

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        f.write("".join(f"<bos>{doc}<eos>\n" for doc in val))

    print("Train + Val files built.")

if __name__ == "__main__":
    build()
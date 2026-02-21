import os
import random

BASE_PATH = "data/processed/train.txt"
ALGO_PATH = "data/processed/algorithms.txt"
UTILITY_PATH = "data/processed/utilities.txt"
OUTPUT_PATH = "data/processed/train_balanced.txt"

def read_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

base_text = read_file(BASE_PATH)
algo_text = read_file(ALGO_PATH)
utility_text = read_file(UTILITY_PATH)

base_chunks = base_text.split("\n\n")
algo_chunks = algo_text.split("\n\n")
utility_chunks = utility_text.split("\n\n")

random.shuffle(base_chunks)
random.shuffle(algo_chunks)
random.shuffle(utility_chunks)

base_count = int(len(base_chunks) * 0.7)
algo_count = int(len(base_chunks) * 0.2)
utility_count = int(len(base_chunks) * 0.1)

balanced = (
    base_chunks[:base_count]
    + algo_chunks[:algo_count]
    + utility_chunks[:utility_count]
)

random.shuffle(balanced)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n\n".join(balanced))

print("Balanced corpus written to", OUTPUT_PATH)
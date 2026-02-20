import os

BASE_TRAIN = "data/processed/train.txt"
ALGO_TRAIN = "data/algorithms/algorithms.txt"
OUT_FILE = "data/processed/train_finetune.txt"

with open(BASE_TRAIN, "r", encoding="utf-8") as f:
    base = f.read()

with open(ALGO_TRAIN, "r", encoding="utf-8") as f:
    algo = f.read()

# Important: algorithms get repeated to increase weight
combined = base + "\n\n" + (algo * 5)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(combined)

print("✅ Fine-tuning corpus created")
print(f"Base chars: {len(base)}")
print(f"Algo chars: {len(algo)}")
print(f"Total chars: {len(combined)}")

import os

os.makedirs("data/processed", exist_ok=True)

out = []

for root, _, files in os.walk("data/raw"):
    for f in files:
        if f.endswith(".py"):
            try:
                with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                    out.append(file.read())
            except:
                pass

with open("data/processed/train.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(out))

print("✅ train.txt built from raw GitHub code")
print("Characters:", sum(len(x) for x in out))

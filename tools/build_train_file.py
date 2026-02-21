import os

SOURCE = "data/raw"
OUTPUT = "data/processed/train.txt"

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as out:
    for root, _, files in os.walk(SOURCE):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        out.write(content)
                        out.write("\n\n")
                except:
                    pass

print("train.txt rebuilt.")
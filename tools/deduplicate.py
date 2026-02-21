import os
import hashlib

ROOT = "data/raw"

hashes = {}
removed = 0

for root, _, files in os.walk(ROOT):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "rb") as f:
                content = f.read()
                h = hashlib.md5(content).hexdigest()

            if h in hashes:
                os.remove(path)
                removed += 1
            else:
                hashes[h] = path

print(f"Removed {removed} duplicate files.")
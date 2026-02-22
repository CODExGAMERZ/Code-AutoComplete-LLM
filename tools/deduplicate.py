import os
import hashlib

CLEAN_DIR = "data/cleaned"

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def deduplicate():
    hashes = set()
    removed = 0

    for root, _, files in os.walk(CLEAN_DIR):
        for file in files:
            path = os.path.join(root, file)
            h = file_hash(path)

            if h in hashes:
                os.remove(path)
                removed += 1
            else:
                hashes.add(h)

    print(f"Removed duplicates: {removed}")

if __name__ == "__main__":
    deduplicate()
import os
import shutil

RAW_DIR = "data/raw"
CLEAN_DIR = "data/cleaned"

REMOVE_DIRS = {
    "tests",
    "test",
    "docs",
    "doc",
    ".github",
    ".git",
    "examples",
    "scripts",
    "ci",
    "build",
    "dist",
    "__pycache__",
}

def should_remove(path):
    for r in REMOVE_DIRS:
        if r in path.lower():
            return True
    return False

def clean_repo():
    if os.path.exists(CLEAN_DIR):
        shutil.rmtree(CLEAN_DIR)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    for root, dirs, files in os.walk(RAW_DIR):
        if should_remove(root):
            continue

        for file in files:
            if not file.endswith(".py"):
                continue

            src_path = os.path.join(root, file)

            rel_path = os.path.relpath(src_path, RAW_DIR)
            dest_path = os.path.join(CLEAN_DIR, rel_path)

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)

    print("Cleaning complete.")

if __name__ == "__main__":
    clean_repo()
import os
import shutil

RAW_DIR = "data/raw"

REMOVE_DIR_NAMES = {
    "tests",
    "test",
    "docs",
    "doc",
    "examples",
    "example",
    ".github",
    ".git",
    "__pycache__",
    "scripts",
    "script",
    "migrations",
    "migration",
    "build",
    "dist",
    "ci",
    ".vscode",
    ".idea"
}

def should_remove_dir(name):
    return name.lower() in REMOVE_DIR_NAMES

def should_remove_file(name):
    name_lower = name.lower()
    if name_lower.startswith("test_"):
        return True
    if name_lower.endswith("_test.py"):
        return True
    if not name_lower.endswith(".py"):
        return True
    return False

def prune():
    removed_dirs = 0
    removed_files = 0

    for root, dirs, files in os.walk(RAW_DIR, topdown=True):
        dirs[:] = [d for d in dirs if not should_remove_dir(d)]

        for d in list(dirs):
            if should_remove_dir(d):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                removed_dirs += 1
                dirs.remove(d)

        for f in files:
            if should_remove_file(f):
                os.remove(os.path.join(root, f))
                removed_files += 1

    print(f"Removed dirs: {removed_dirs}")
    print(f"Removed files: {removed_files}")
    print("Pruning complete.")

if __name__ == "__main__":
    prune()
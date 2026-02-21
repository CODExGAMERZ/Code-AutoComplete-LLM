import os
import shutil

SOURCE_DIR = "data/raw"
DEST_DIR = "data/expanded"

IGNORE_FOLDERS = {
    "venv", "__pycache__", "migrations",
    "build", "dist", ".git", ".github",
    "node_modules", "tests", "docs"
}

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

for root, dirs, files in os.walk(SOURCE_DIR):
    dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

    for file in files:
        if file.endswith(".py"):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(DEST_DIR, file)

            try:
                shutil.copy2(src_path, dst_path)
            except:
                pass

print("Cleaning complete.")
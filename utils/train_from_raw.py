import os

RAW_DIR = "data/raw"
OUT_FILE = "data/processed/train.txt"

os.makedirs("data/processed", exist_ok=True)

def read_py_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return None

def collect_repo_files(repo_path):
    py_files = []
    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    return sorted(py_files)

all_repos_text = []

for repo_name in os.listdir(RAW_DIR):
    repo_path = os.path.join(RAW_DIR, repo_name)
    if not os.path.isdir(repo_path):
        continue

    repo_text = []
    repo_text.append(f"# ===== REPOSITORY: {repo_name} =====\n")

    py_files = collect_repo_files(repo_path)

    for file_path in py_files:
        content = read_py_file(file_path)
        if content is None:
            continue

        rel_path = os.path.relpath(file_path, repo_path)
        repo_text.append(f"\n# --- FILE: {rel_path} ---\n")
        repo_text.append(content)

    all_repos_text.append("\n".join(repo_text))

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_repos_text))

total_chars = sum(len(x) for x in all_repos_text)

print("✅ Built training corpus with repo-level structure")
print(f"Repositories processed: {len(all_repos_text)}")
print(f"Total characters: {total_chars}")

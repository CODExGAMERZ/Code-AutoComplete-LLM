import os

BASE_DIR = "data"

total_py_files = 0
total_c_files = 0
total_h_files = 0
total_java_files = 0
total_txt_files = 0
total_chars = 0

def count_file(path):
    global total_chars
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            total_chars += len(content)
    except:
        pass

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        full_path = os.path.join(root, file)

        if file.endswith(".py"):
            total_py_files += 1
            count_file(full_path)

        elif file.endswith(".c"):
            total_c_files += 1
            count_file(full_path)

        elif file.endswith(".h"):
            total_h_files += 1
            count_file(full_path)

        elif file.endswith(".java"):
            total_java_files += 1
            count_file(full_path)

        elif file.endswith(".txt"):
            total_txt_files += 1
            count_file(full_path)

approx_tokens = total_chars // 4

print("========== DATASET STATS ==========")
print(f"Total .py files: {total_py_files}")
print(f"Total .c files: {total_c_files}")
print(f"Total .h files: {total_h_files}")
print(f"Total .java files: {total_java_files}")
print(f"Total .txt files: {total_txt_files}")
print(f"Total characters: {total_chars:,}")
print(f"Approx tokens (~chars/4): {approx_tokens:,}")
print("===================================")
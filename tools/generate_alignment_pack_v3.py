import os
from pathlib import Path

OUTPUT_DIR = "data/cleaned/patterns_manual"

def generate_stack_variants():
    variants = []

    variants.append("""
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)
""")

    variants.append("""
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        if item is None:
            raise ValueError("Cannot push None")
        self._items.append(item)
""")

    variants.append("""
class Stack:
    def __init__(self):
        self._items = list()

    def push(self, item):
        self._items += [item]
""")

    return variants


def generate_dfs_variants():
    variants = []

    variants.append("""
def dfs(graph, node, visited):
    visited.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
""")

    variants.append("""
def dfs(graph, start, visited):
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph.get(node, []))
""")

    return variants


def generate_binary_search_variants():
    variants = []

    variants.append("""
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""")

    variants.append("""
def binary_search(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(arr) and arr[lo] == target else -1
""")

    return variants


def generate_file_io_variants():
    variants = []

    variants.append("""
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
""")

    variants.append("""
def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
""")

    variants.append("""
def append_file(path, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
""")

    return variants


def generate():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    blocks = []

    base_blocks = []
    base_blocks += generate_stack_variants()
    base_blocks += generate_dfs_variants()
    base_blocks += generate_binary_search_variants()
    base_blocks += generate_file_io_variants()

    for _ in range(500):
        for block in base_blocks:
            blocks.append(block)

    content = "\n".join(blocks)

    output_path = os.path.join(OUTPUT_DIR, "alignment_patterns.py")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Alignment v3 generated. Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    generate()
import os
from pathlib import Path

OUTPUT_DIR = "data/cleaned/patterns_manual"

REPEAT = 1200

def stack_variants():
    return ["""
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()

    def peek(self):
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0
""" for _ in range(REPEAT)]


def dfs_variants():
    return ["""
def dfs(graph, node, visited):
    visited.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
""" for _ in range(REPEAT)]


def binary_search_variants():
    return ["""
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""" for _ in range(REPEAT)]


def file_io_variants():
    return ["""
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
""" for _ in range(REPEAT)]


def generate():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    blocks = []
    blocks += stack_variants()
    blocks += dfs_variants()
    blocks += binary_search_variants()
    blocks += file_io_variants()

    content = "\n".join(blocks)

    output_path = os.path.join(OUTPUT_DIR, "alignment_patterns.py")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Alignment v2 generated. Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    generate()
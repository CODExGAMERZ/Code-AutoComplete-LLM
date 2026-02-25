# 🧠 Python Code Autocomplete LLM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Architecture](https://img.shields.io/badge/Architecture-GPT%20Decoder-green)
![Parameters](https://img.shields.io/badge/Parameters-33.5M-orange)
![Training](https://img.shields.io/badge/Training-CPU--Only-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A decoder-only GPT-style Transformer trained entirely from scratch for Python code autocompletion.

This project implements a true autoregressive decoder with KV-cache support and a fully local training pipeline.

---

# 🚀 Highlights

* ✅ Custom Causal Self-Attention (No TransformerEncoder)
* ✅ KV-cache for incremental decoding
* ✅ Resume-safe training loop
* ✅ Dual inference modes (Autocomplete / Creative)
* ✅ Hardened dataset cleaning pipeline
* ✅ CPU-only training

---

# 🏗️ Architecture

| Component       | Value            |
| --------------- | ---------------- |
| Model Type      | Decoder-only GPT |
| Layers          | 8                |
| Attention Heads | 8                |
| Embedding Size  | 512              |
| Context Length  | 256              |
| Parameters      | ~33,551,168      |
| KV Cache        | Enabled          |

This is a true causal language model — not an encoder repurposed for generation.

---

# 📊 Performance

## Dataset

* ~5.1M training tokens
* ~0.5M validation tokens
* Balanced alignment patterns

## Final Metrics

| Metric          | Value |
| --------------- | ----- |
| Validation Loss | 2.84  |
| Perplexity      | 17.20 |
| Epochs          | 2     |

Stable convergence for a 33M parameter CPU-trained model.

---

# ⚡ KV-Cache Decoding

Generation uses incremental decoding:

1. Full prompt processed once
2. Subsequent tokens reuse stored key/value tensors
3. No full-sequence recomputation

Result:

* Faster inference
* Lower latency
* Proper GPT-style behavior

---

# ✨ Example Outputs

## DFS

```python
def dfs(graph, node, visited):
```

```python
stack = [start]
while stack:
    node = stack.pop()
    if node not in visited:
        visited.add(node)
        stack.extend(graph.get(node, []))
```

---

## Stack

```python
class Stack:
    def push(self, item):
```

```python
self._items.append(item)
```

---

## Binary Search

```python
def binary_search(arr, target):
```

```python
lo, hi = 0, len(arr)
while lo < hi:
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        lo = mid + 1
    else:
        hi = mid
return -1
```

---

# 🧠 Inference Modes

## Autocomplete (Default)

* temperature = 0.2
* top_k = 10
* Deterministic behavior

## Creative

* temperature = 0.8
* top_k = 50
* More diverse generation

Run:

```bash
python inference/run_model.py \
  -c model/checkpoints/latest_checkpoint.pth \
  -p "def dfs(graph, node, visited):" \
  --mode autocomplete
```

---

# 📂 Project Structure

```
model/
tokenizer/
training/
inference/
tools/
data/
MODEL_CARD.md
README.md
```

---

# 🎓 What This Project Demonstrates

* Building a GPT decoder from scratch
* Implementing causal attention manually
* Integrating KV-cache
* Dataset curation and alignment engineering
* CPU-based LLM training

---

# 🚀 Roadmap

* Scale dataset to 20M+ tokens
* Increase parameter count
* Expand context window
* Add lightweight API server

---

# 📜 License

MIT License

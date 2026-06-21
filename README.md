# 🧠 Multilingual Code Autocomplete LLM (From Scratch)

A GPT-style **decoder-only Transformer** trained entirely from scratch for **Python, C, and Java** code autocompletion.

This project uses a **true causal GPT decoder architecture with KV-cache support and GPU acceleration**, enabling fast incremental generation and stable autoregressive behavior on local CPUs as well as Google Colab/Kaggle GPUs.

---

# 🚀 What’s New (Multilingual & GPU Upgrades)

✅ **Python, C, and Java Support**: Expanded preprocessor, cleaner, and tokenizer to fully support `.py`, `.c`, `.h`, and `.java` files.
✅ **Scaled Multilingual Dataset**: ~38.7M training tokens sourced from standard algorithms, CPython, Redis, and Apache Commons-Lang.
✅ **Dynamic GPU Acceleration**: Automated CUDA (`cuda`) GPU detection for training and evaluation.
✅ **Google Colab & Kaggle Support**: Fully configured notebooks and lightweight packaging.
✅ **VS Code Fork Integration**: Includes architecture plan to embed this LLM inside a custom VS Code editor ([VSCODE_FORK_PLAN.md](file:///c:/Users/codex/GitHub/Code-AutoComplete-LLM/VSCODE_FORK_PLAN.md)).
✅ **KV-cache & Causal Self-Attention**: Enabled for fast decoding.

---

# 🧩 System Overview

## 1️⃣ Data Pipeline

* Raw repositories collected
* Hardened cleaning removes:

  * tests
  * build files
  * compiled artifacts
  * duplicate files
* Curated alignment patterns added (balanced, non-repetitive)
* Train / validation split

## 2️⃣ Tokenizer

* Custom BPE tokenizer (vocab size: 8000)
* Trained on processed corpus
* Python-aware tokenization

## 3️⃣ Model Architecture

Decoder-only GPT-style architecture:

| Component       | Value                 |
| --------------- | --------------------- |
| Layers          | 8                     |
| Attention Heads | 8                     |
| Embedding Size  | 512                   |
| Context Length  | 256                   |
| Parameters      | ~33.5M                |
| Attention       | Causal Self-Attention |
| KV Cache        | ✅ Supported           |

Total Parameters: ~33,551,168

---

# ⚡ KV-Cache Support

Generation now uses incremental decoding:

* First forward pass processes full prompt
* Subsequent tokens reuse stored key/value tensors
* No full-sequence recomputation

Result:

* Faster inference
* Lower latency
* True GPT-style decoding behavior

---

# 🎯 Training Setup

* Optimizer: AdamW
* Loss: Cross-Entropy
* Gradient Accumulation Supported
* Resume-safe checkpointing

Training can be interrupted safely:

```bash
Ctrl+C
```

Restarting resumes automatically from the last checkpoint.

---

# 📊 Performance Benchmarks

## Dataset

* ~5.1M training tokens
* ~0.5M validation tokens
* Balanced curated alignment

## Final Metrics (Decoder Architecture)

| Metric                | Value |
| --------------------- | ----- |
| Epochs                | 2     |
| Final Validation Loss | 2.84  |
| Perplexity            | 17.20 |

For a 33M parameter CPU-trained model, this is strong stability.

---

# ✨ Example Outputs

## DFS

Input:

```python
def dfs(graph, node, visited):
```

Output:

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

Input:

```python
class Stack:
    def push(self, item):
```

Output:

```python
self._items.append(item)
```

---

## Binary Search

Input:

```python
def binary_search(arr, target):
```

Output:

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

## Autocomplete Mode (default)

* temperature = 0.2
* top_k = 10
* Deterministic
* Code-focused

## Creative Mode

* temperature = 0.8
* top_k = 50
* More diverse
* Useful for code generation

Run with:

```bash
python inference/run_model.py \
  -c model/checkpoints/latest_checkpoint.pth \
  -p "def dfs(graph, node, visited):" \
  --mode autocomplete
```

---

# 🖥️ CLI Usage (Simple Wrapper)

You can build a simple CLI wrapper:

```bash
python codellm.py autocomplete "def binary_search(arr, target):"
python codellm.py creative "Write a Python LRU cache implementation"
```

---

# 📂 Project Structure

```
AutoComplete-LLM/
│
├── model/
│   ├── ai.py
│   └── checkpoints/
│
├── tokenizer/
│   ├── tokenizer.json
│   └── train_tokenizer.py
│
├── training/
│   ├── dataset.py
│   └── train.py
│
├── inference/
│   └── run_model.py
│
├── tools/
│   ├── download_dataset.py
│   ├── hardened_clean.py
│   ├── build_train_file.py
│   ├── evaluate_model.py
│   └── generate_alignment_pack_v3.py
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── train_on_colab.ipynb
└── README.md
```

---

# 🎓 What This Project Demonstrates

* Full LLM lifecycle from scratch
* Decoder-only Transformer implementation
* Custom causal attention
* KV-cache integration
* Dataset curation & alignment engineering
* CPU-only training of 33M parameter model
* Practical engineering for small-scale LLM systems

---

# 📜 License

MIT License

---

# 🚀 Status

This model is:

* Stable
* Usable for Python autocomplete
* Structurally aligned
* KV-cache enabled
* Resume-safe trained

Further scaling would require:

* Larger dataset (20–50M tokens)
* GPU acceleration
* 60–120M parameter scale

But at current scale, this is a functional local Python code LLM.

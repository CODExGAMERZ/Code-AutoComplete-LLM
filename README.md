# Code-AutoComplete-LLM

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python%20%7C%20C%20%7C%20Java-multilingual-3776ab?logo=python&logoColor=white)
![Training](https://img.shields.io/badge/Training-Dual%20T4%20DDP-76b900?logo=nvidia&logoColor=white)
![Parameters](https://img.shields.io/badge/Parameters-29.5M%20→%20265M-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A decoder-only Transformer series trained **from scratch** for multilingual code autocompletion in **Python, C, and Java**. The project spans four architecture generations — from a 29.5M-parameter baseline up to a 265M Gemma-style model with Grouped-Query Attention, RoPE, and GeGLU — all designed to run on dual Kaggle T4 GPUs using Distributed Data Parallel training.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Evaluation & Tools](#evaluation--tools)
- [Known Issues & Fixes Applied](#known-issues--fixes-applied)
- [License](#license)

---

## Architecture Overview

Four model configurations are supported. Each generation introduces structural improvements while preserving backward compatibility with older checkpoints via automatic architecture detection at inference time.

| Hyperparameter | `llm_v1` | `llm_v2` | `llm_v3` | `gemma_v4` |
|:---|:---:|:---:|:---:|:---:|
| **Active Parameters** | 29.5M | 110.3M | 265.1M | **203.0M** (~235.7M saved) |
| **Normalization** | LayerNorm | RMSNorm | RMSNorm | **Dual RMSNorm** |
| **FFN Block** | GELU MLP | SwiGLU | SwiGLU | **GeGLU** |
| **Positional Encoding** | Absolute learned | Absolute learned | Absolute learned | **RoPE** |
| **Attention Heads (Q / KV)** | 8 / 8 | 12 / 12 | 16 / 16 | **16 / 4 (GQA)** |
| **Layers** | 8 | 12 | 18 | **18** |
| **Embedding Dimension** | 512 | 768 | 1024 | **1024** |
| **Context Length** | 256 | 1024 | 1024 | **1024** (extrapolatable via RoPE) |
| **Vocab Size** | 8,000 | 32,000 | 32,000 | **32,000** |
| **Logit Soft-Capping** | — | — | — | Attn: 50.0 / Output: 30.0 |
| **Status** | Trained | Trained | Trained | **Implemented — ready to train** |

### Key architectural features (gemma_v4)

- **Grouped-Query Attention (GQA)** — 16 query heads share 4 KV heads, reducing KV cache memory by 4×.
- **RoPE** — Rotary positional embeddings applied before KV cache concatenation, enabling length extrapolation at inference.
- **Dual RMSNorm** — Pre-norm on inputs and post-norm on residual projections for training stability.
- **GeGLU** — GELU-gated linear unit in the FFN, with hidden dimension aligned to a multiple of 256 for hardware efficiency.
- **Attention logit soft-capping** — `50 * tanh(score / 50)` prevents attention spikes without hard masking.
- **Output logit soft-capping** — `30 * tanh(logit / 30)` stabilizes final logit distributions.
- **Weight tying** — Token embedding and output projection share weights, saving ~32.7M parameters.
- **Self-contained checkpoints** — The compiled `tokenizer.json` is embedded directly inside every `.pth` file, preventing vocab mismatch on remote setups.

All models use **FlashAttention via `F.scaled_dot_product_attention`** (PyTorch 2.0+ SDPA) and KV caching at inference time.

---

## Project Structure

```
Code-AutoComplete-LLM/
│
├── model/
│   └── ai.py                       # All architectures: OldGPT (v1), GPT (v2/v3), GemmaGPT (v4)
│
├── tokenizer/
│   ├── tokenizer.json              # 32k BPE vocabulary (compiled)
│   └── train_tokenizer.py          # Trains the BPE tokenizer from train.txt
│
├── training/
│   ├── dataset.py                  # Memory-mapped .bin dataset loader
│   └── train.py                    # DDP training loop with cosine LR, AMP, gradient clipping
│
├── inference/
│   ├── run_model.py                # Interactive autocompletion with KV Cache
│   └── compare_models.py           # Side-by-side generation across checkpoints
│
├── tools/
│   ├── download_hf_dataset.py      # Streams Python/C/Java from codeparrot/github-code-clean
│   ├── hardened_clean.py           # Filters boilerplate, oversized files, and duplicates
│   ├── build_train_file.py         # Merges cleaned files into train.txt / val.txt with <bos>/<eos>
│   ├── prepare_dataset.py          # Pre-tokenizes text splits into memory-mapped .bin files
│   ├── evaluate_model.py           # Cross-entropy loss and perplexity on held-out data
│   ├── stress_test.py              # Generation throughput benchmark (tokens/sec)
│   ├── model_stats.py              # Parameter counts and config inspection
│   ├── plot_loss.py                # Training loss curve from CSV logs
│   ├── plot_comparison.py          # Multi-model loss decay comparison chart
│   ├── analyze_losses.py           # Step count, final loss, minimum loss across runs
│   ├── inspect_checkpoints.py      # Parameter counts and key comparison across checkpoints
│   ├── check_checkpoint_params.py  # File size and state dict key inspection
│   ├── test_kv_cache.py            # KV cache correctness test
│   ├── check_logits.py             # Logit distribution diagnostics
│   ├── validate_v4.py              # gemma_v4 architecture validation
│   ├── compare_tokenizers.py       # Vocab and encoding comparison across tokenizer versions
│   ├── inspect_dataset.py          # Token count and block statistics for a .bin file
│   ├── inspect_zip.py              # Dataset zip contents inspector
│   ├── download_dataset.py         # Legacy multi-repo GitHub downloader
│   ├── run_kaggle_pipeline.py      # Full Kaggle pipeline automation
│   ├── test_checkpoint_save.py     # Checkpoint save/load round-trip test
│   ├── deduplicate.py              # Removes identical files from data/cleaned/
│   └── prepare_dataset.py          # Tokenizes train.txt → train.bin (memmap format)
│
├── data/
│   ├── raw/                        # Downloaded source files (per language)
│   ├── cleaned/                    # Filtered source files
│   └── processed/                  # train.txt, val.txt, train.bin, val.bin
│
├── model/checkpoints/              # Saved .pth checkpoints
├── training_logs/                  # Per-model loss CSV logs
│
├── train_on_kaggle.ipynb           # Kaggle GPU training notebook (recommended)
├── train_on_colab.ipynb            # Google Colab training notebook
├── requirements.txt
├── MODEL_CARD.md
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/CODExGAMERZ/Code-AutoComplete-LLM.git
cd Code-AutoComplete-LLM
pip install -r requirements.txt
```

**Requirements:** `torch`, `tokenizers`, `tqdm`, `numpy`, `matplotlib`, `pandas`, `datasets`

> PyTorch 2.0+ is required for `F.scaled_dot_product_attention` (FlashAttention) and `torch.compile`.

---

## Data Pipeline

Run these steps in order. **Do not reuse pre-existing `.bin` or `.txt` files** from earlier runs — they may have been built from a buggy pipeline (see [Known Issues](#known-issues--fixes-applied)).

### Step 1 — Download

Streams Python, C, and Java source files from [`codeparrot/github-code-clean`](https://huggingface.co/datasets/codeparrot/github-code-clean) on Hugging Face:

```bash
python tools/download_hf_dataset.py --num_files 60000
```

This downloads 60,000 files per language (180,000 total). Increase `--num_files` for larger token budgets — the 29.5M–265M parameter range benefits significantly from 100M+ tokens. Files are saved under `data/raw/`.

### Step 2 — Clean

Filters out boilerplate, test suites, oversized files, and duplicates:

```bash
python tools/hardened_clean.py
```

Cleaned files are written to `data/cleaned/`. Filtering thresholds: `MIN_LINES=3`, `MAX_LINES=4000`. Only genuine test files are excluded (`test_*.py`, `*_test.py`, JUnit-pattern filenames) — source files with "test" appearing elsewhere in the name are kept.

### Step 3 — Build text splits

Merges cleaned files into `train.txt` and `val.txt`, wrapping every document in `<bos>…<eos>` boundary tokens:

```bash
python tools/build_train_file.py
```

### Step 4 — Train the tokenizer (optional but recommended)

Re-trains the BPE vocabulary on your current corpus. Skip this step if you want to reuse the bundled `tokenizer/tokenizer.json`:

```bash
python tokenizer/train_tokenizer.py
```

Produces a 32,000-token BPE vocabulary with special tokens `<pad>`, `<unk>`, `<bos>`, `<eos>`.

### Step 5 — Pre-tokenize to binary

Compiles the text splits into memory-mapped `.bin` files for O(1) load time during training:

```bash
python tools/prepare_dataset.py
```

Outputs `data/processed/train.bin` and `data/processed/val.bin`. After this step, verify the printed **total token count** — target at least 100–150M tokens for the 29.5M model, more for larger configs.

---

## Training

### Distributed (Dual T4 on Kaggle — recommended)

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=2 training/train.py \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 32 \
  --lr 3e-4 \
  --model_name gemma_v4
```

### Single GPU / CPU

```bash
python training/train.py \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 16 \
  --lr 3e-4 \
  --model_name gemma_v4
```

### Training arguments

| Argument | Default | Description |
|:---|:---:|:---|
| `--epochs` | `2` | Number of training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--grad_accum` | `32` | Gradient accumulation steps |
| `--lr` | `3e-4` | Peak learning rate |
| `--model_name` | `latest_checkpoint` | Checkpoint name (no `.pth`) |
| `--n_layers` | `18` | Transformer layers (new model only) |
| `--n_heads` | `16` | Attention heads (new model only) |
| `--n_embd` | `1024` | Embedding dimension (new model only) |
| `--block_size` | `1024` | Context length (new model only) |

Architecture arguments are only used when training a model from scratch. Resuming from an existing checkpoint reads the config directly from the `.pth` file.

**Outputs per run:**
- `model/checkpoints/<model_name>.pth` — checkpoint with embedded tokenizer
- `training_logs/<model_name>_loss.csv` — per-step loss log

### Training details

- **Optimizer**: AdamW
- **LR schedule**: Cosine annealing with 5% linear warmup
- **Gradient clipping**: `max_norm=1.0`
- **Precision**: FP16 AMP (`torch.amp.GradScaler`)
- **Compilation**: `torch.compile` (auto-enabled on CUDA, skipped gracefully on older PyTorch)
- **DDP backend**: NCCL (auto-detected via `RANK` env variable)

---

## Inference

### Interactive completion

```bash
python inference/run_model.py \
  --checkpoint model/checkpoints/gemma_v4.pth \
  --prompt "def binary_search(arr, target):" \
  --max_tokens 120 \
  --temperature 0.2 \
  --top_k 10
```

The script auto-detects model architecture (v1 LayerNorm, v2/v3 GPT, or gemma_v4 GemmaGPT) from checkpoint weights and loads the embedded tokenizer — no separate tokenizer path needed.

### Recommended sampling parameters

| Mode | Temperature | Top-K | Best for |
|:---|:---:|:---:|:---|
| **Autocomplete** | 0.2 | 10 | Structured syntax, API calls, multi-line completions |
| **Creative** | 0.8 | 50 | Algorithm generation, boilerplate scaffolding |

### Compare checkpoints

Run side-by-side generation across multiple checkpoints on the same prompt:

```bash
python inference/compare_models.py \
  --prompt "for i in range(" \
  --checkpoints model/checkpoints/llm_v2.pth model/checkpoints/gemma_v4.pth
```

---

## Evaluation & Tools

```bash
# Cross-entropy loss and perplexity on validation set
python tools/evaluate_model.py --checkpoint model/checkpoints/gemma_v4.pth

# Generation throughput (tokens/sec) and latency benchmark
python tools/stress_test.py --checkpoint model/checkpoints/gemma_v4.pth

# Parameter count and architecture config summary
python tools/model_stats.py --checkpoint model/checkpoints/gemma_v4.pth

# Plot training loss curve
python tools/plot_loss.py --log training_logs/gemma_v4_loss.csv

# Compare loss decay across multiple training runs
python tools/plot_comparison.py \
  --logs training_logs/llm_v2_loss.csv training_logs/gemma_v4_loss.csv
```

---

## Known Issues & Fixes Applied

These bugs were present in earlier runs and caused broken, incoherent output. All are resolved in this codebase. If you start from a previous dataset or checkpoint, read this section before retraining.

### 1. Missing document boundaries

`build_train_file.py` was joining files with bare `\n\n`, giving the model no signal that one file ended and an unrelated one began. Every file is now wrapped as `<bos>{content}<eos>`. Both tokens are encoded as single atomic IDs (2 and 3) by the BPE tokenizer.

### 2. Dataset was ~85% smaller than intended

The previous default downloaded only 10,000 files per language (30,000 total), yielding roughly 7.5M tokens — approximately 20–100× too little data for models in the 29.5M–110M parameter range. The default is now 60,000 files per language. Two secondary causes also discarded large fractions of valid data:

- `MAX_LINES` was 1,500 — silently dropped every source file longer than ~50 lines of average length. Raised to **4,000**.
- The test-file filter matched any filename containing the substring `"test"`, incorrectly rejecting files like `latest_utils.py` or `ContextTest.java`. Now only genuine test entrypoints are filtered (`test_*.py`, `*_test.py`, `tests.py`, JUnit-pattern class names).

### 3. Rebuild from scratch

Processed files from earlier runs should be discarded. Run the full pipeline from [Step 1](#step-1--download) to generate clean data. After `prepare_dataset.py`, check the printed **Total tokens** line before starting a training run.

---

## License

[MIT License](./LICENSE) — free to use, adapt, and redistribute.
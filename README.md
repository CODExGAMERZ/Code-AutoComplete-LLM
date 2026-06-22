# Code-AutoComplete-LLM

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python%20%7C%20C%20%7C%20Java-multilingual-3776ab?logo=python&logoColor=white)
![Model Size](https://img.shields.io/badge/Parameters-29.5M-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A lightweight decoder-only Transformer model trained **from scratch** on **402 million tokens** for multilingual code autocompletion in **Python, C, and Java**. The model is designed for efficient, offline code generation, using a classic GPT-style architecture with Multi-Head Attention, absolute learned positional embeddings, LayerNorm, and a GELU feedforward network.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Evaluation & Diagnostic Tools](#evaluation--diagnostic-tools)
- [License](#license)

---

## Architecture Overview

The model employs a custom GPT implementation optimized for code completion tasks.

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Total Parameters** | 29.5M | Lightweight footprint suitable for CPU/GPU local execution |
| **Normalization** | LayerNorm | Standard layer normalization applied pre-attention/FFN |
| **FFN Block** | GELU MLP | Feed-forward expansion ratio of 4× (`512 -> 2048 -> 512`) |
| **Positional Encoding** | Absolute learned | Standard spatial embedding up to context window |
| **Attention** | Multi-Head (MHA) | 8 attention heads with head dimension of 64 |
| **Layers** | 8 | Deeper block stacking for pattern recognition |
| **Embedding Dimension** | 512 | Vector space dimension for token projections |
| **Context Length** | 256 | Target window size for coding history context |
| **Vocab Size** | 8,000 | Token dictionary size optimized by custom BPE |

### Key Architectural Details
* **Autoregressive Generation**: Predicts the next token given preceding token context.
* **KV Caching**: Cache key and value projections at inference time for $O(1)$ autoregressive generation scaling.
* **Unified Vocab**: Single multilingual vocabulary trained on Python, C, and Java scripts.

---

## Project Structure

```
Code-AutoComplete-LLM/
│
├── model/
│   ├── ai.py                       # GPT model architecture definition
│   └── checkpoints/                # Saved PyTorch (.pth) checkpoints
│
├── tokenizer/
│   ├── tokenizer.json              # Trained 8k BPE vocabulary
│   └── train_tokenizer.py          # Trains the BPE tokenizer on processed train.txt
│
├── training/
│   ├── dataset.py                  # PyTorch CodeDataset loader
│   └── train.py                    # Training loop with checkpoints and loss logging
│
├── inference/
│   └── run_model.py                # Interactive local autocomplete and text generator
│
├── tools/
│   ├── download_dataset.py         # Downloads Python/C/Java zip archives from GitHub
│   ├── hardened_clean.py           # Filters files by size/lines, removes duplicates & comments
│   ├── deduplicate.py              # Removes duplicate files by md5 hash from cleaned/
│   ├── generate_alignment_pack_v3.py # Generates repeated programming patterns for training stability
│   ├── build_balanced_corpus.py    # Builds balanced training datasets in custom proportions
│   ├── build_train_file.py         # Merges cleaned code files into train.txt / val.txt with boundaries
│   ├── model_stats.py              # Reports total and trainable parameter counts of model definition
│   ├── check_checkpoint_params.py  # Reports parameter count inside saved model checkpoint
│   ├── evaluate_model.py           # Calculates average cross-entropy loss and perplexity on dataset
│   └── plot_loss.py                # Visualizes the training loss decay curve from CSV logs
│
├── data/
│   ├── raw/                        # Extracted raw repositories (git-ignored)
│   ├── cleaned/                    # Cleaned, formatted source code (git-ignored)
│   └── processed/                  # Compiled text splits (train.txt / val.txt) (git-ignored)
│
├── training_logs/                  # Loss log directory (contains loss_log.csv)
├── requirements.txt
├── MODEL_CARD.md
└── LICENSE
```

---

## Installation

Ensure you have Python 3.8+ and PyTorch installed on your system.

```bash
git clone https://github.com/CODExGAMERZ/Code-AutoComplete-LLM.git
cd Code-AutoComplete-LLM
pip install -r requirements.txt
```

> PyTorch 2.0+ is recommended for optimal attention speedups.

---

## Data Pipeline

Follow these steps sequentially to set up the dataset, train the BPE tokenizer, and compile train/validation splits.

### Step 1 — Download Datasets
Downloads raw source code archives (Python, C, Java repositories like `cpython`, `redis`, and standard algorithms libraries) from GitHub:
```bash
python tools/download_dataset.py
```
This extracts files to `data/raw/`.

### Step 2 — Clean the Source Code
Filters out oversized files, boilerplate, test files, non-ASCII heavy files, and strips long introductory comments:
```bash
python tools/hardened_clean.py
```
Cleaned source files are saved to `data/cleaned/`.

### Step 3 — Generate Alignment Patterns & Deduplicate (Optional)
To boost performance on common structures (e.g. stack implementation, binary search, file I/O, DFS) and deduplicate content:
```bash
python tools/generate_alignment_pack_v3.py
python tools/deduplicate.py
```

### Step 4 — Build Splits
Collects all cleaned documents and compiles them into `train.txt` and `val.txt` datasets (with a 90/10 split), wrapping each document in `<bos>` and `<eos>` tokens:
```bash
python tools/build_train_file.py
```

### Step 5 — Train the Tokenizer
Trains the Byte-Pair Encoding tokenizer on the training corpus split:
```bash
python tokenizer/train_tokenizer.py
```
This generates the vocabulary file `tokenizer/tokenizer.json` with a vocabulary size of 8,000.

---

## Training

To train the 29.5M parameter model from scratch or resume from a saved checkpoint:

```bash
python training/train.py --epochs 2 --batch_size 8 --grad_accum 4 --lr 3e-4
```

### Command-Line Arguments
* `--epochs`: Number of training epochs (default: `2`).
* `--batch_size`: Per-device batch size (default: `8`).
* `--grad_accum`: Number of gradient accumulation steps before updating weights (default: `4`).
* `--lr`: Peak learning rate for AdamW optimizer (default: `3e-4`).
* `--train_path`: Path to training file (default: `data/processed/train.txt`).
* `--val_path`: Path to validation file (default: `data/processed/val.txt`).

### Training Features
* **Checkpointing**: Saves checkpoints to `model/checkpoints/latest_checkpoint.pth` at the end of each epoch, which includes model weights, optimizer state, epoch, and global steps.
* **Auto-Resume**: If a checkpoint is detected in the checkpoint directory, training automatically resumes from the saved state.
* **Loss Logging**: Real-time training losses are logged to `training_logs/loss_log.csv` per step.

---

## Inference

Run interactive or batch autocompletions using your trained checkpoint:

```bash
python inference/run_model.py -c model/checkpoints/latest_checkpoint.pth -p "def binary_search(arr, target):" --mode autocomplete
```

### Inference Options
* `-c`, `--checkpoint`: Path to the PyTorch model checkpoint (e.g., `model/checkpoints/latest_checkpoint.pth`).
* `-p`, `--prompt`: The input code snippet/prompt for the model to autocomplete.
* `--mode`: The generation mode. Choose between:
  * `autocomplete` (default): Low temperature (`0.2`) and narrow top-k (`10`) for structured, precise coding completions.
  * `creative`: Higher temperature (`0.8`) and wider top-k (`50`) for generating alternative block setups.

---

## Evaluation & Diagnostic Tools

We provide several auxiliary scripts under the `tools/` folder to diagnose training performance and inspect checkpoints:

### 1. Calculate Parameter Statistics
Inspect model size directly from the class definition:
```bash
python tools/model_stats.py
```

### 2. Verify Saved Checkpoint Weight Parameters
Inspect the actual model checkpoint weights size:
```bash
python tools/check_checkpoint_params.py
```

### 3. Fast Validation Evaluation
Calculate the cross-entropy loss and perplexity score on the training/validation splits using the saved checkpoint:
```bash
python tools/evaluate_model.py
```

### 4. Plot Loss Decay
Generate and show the training loss curve based on the step logs:
```bash
python tools/plot_loss.py
```

---

## License

[MIT License](./LICENSE) — free to use, adapt, and redistribute.
# 🧠 Python Code Autocomplete LLM (From Scratch)

A **GPT-style Transformer language model** trained **entirely from scratch** to perform **Python code autocompletion**.
The project covers the **full LLM lifecycle**: data collection, tokenization, pretraining, algorithm fine-tuning, inference, and quantitative analysis.

This model was trained **locally on CPU**, without using external LLM APIs.

---

## 🚀 Key Features

* ~**60M parameter** decoder-only Transformer (GPT-style)
* Custom **BPE tokenizer** trained on Python source code
* Pretrained on **real GitHub repositories**
* Fine-tuned for **algorithmic reasoning**
* End-to-end **training, inference, and evaluation** pipeline
* **Entropy & loss analysis** with matplotlib + Desmos
* Fully **offline & reproducible**

---

## 🧩 How the System Works

1. **Data Collection**
   Open-source Python repositories are cloned into `data/raw/`

2. **Preprocessing**
   `.py` files are merged into a single training corpus

3. **Tokenization**
   A Byte Pair Encoding (BPE) tokenizer learns Python-specific tokens

4. **Pretraining**
   GPT-style Transformer trained with causal language modeling

5. **Algorithm Fine-Tuning**
   Model is adapted to generate classic algorithms (search, DP, sorting)

6. **Evaluation & Analysis**
   Loss curves, entropy metrics, and single-step confidence analysis

7. **Inference**
   Model performs next-token code autocompletion

---

## 📂 Project Structure

```
AutoComplete-LLm/
├── analysis/
│   ├── entropy_single_step.py
│   ├── plot_loss.py
│   ├── loss_history.json
│   └── loss_curve.png
│
├── data/
│   ├── algorithms/
│   │   └── algorithms.txt
│   ├── processed/
│   │   ├── train.txt
│   │   └── train_finetune.txt
│   └── raw/
│       └── fastapi/
│
├── model/
│   ├── ai.py
│   └── checkpoints/
│       ├── ckpt_e0_s50000.pth
│       └── ckpt_algo_ft_s55000.pth
│
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer.json
│
├── training/
│   ├── dataset.py
│   ├── train.py
│   └── train_finetune.py
│
├── inference/
│   └── run_model.py
│
├── utils/
│   ├── build_finetune_corpus.py
│   ├── train_from_raw.py
│   └── generate_big_train_txt.py
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## 🧠 Model Configuration

| Component       | Value                          |
| --------------- | ------------------------------ |
| Architecture    | GPT (Decoder-only Transformer) |
| Layers          | 8                              |
| Attention Heads | 8                              |
| Embedding Size  | 512                            |
| Context Length  | 256                            |
| Parameters      | ~60M                           |
| Optimizer       | AdamW                          |
| Loss Function   | Cross-Entropy                  |

---

## 🔬 Training Dynamics & Entropy Analysis

* Cross-entropy loss converges to **~0.25** during fine-tuning
* Loss curve shows **stable saturation** (no divergence)
* Single-step entropy analysis on algorithmic prompts:

  * Entropy ≈ **1.10**
  * Effective next-token vocabulary ≈ **3**

This indicates **strong model confidence** during structured code generation.

---

## ✨ Example Autocomplete

**Prompt**

```python
def fibonacci(n):
```

**Model Output**

```python
a, b = 0, 1
result = []
for _ in range(n):
    result.append(a)
    a, b = b, a + b
return result
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

```bash
python utils/train_from_raw.py
python tokenizer/train_tokenizer.py
python training/train.py
python training/train_finetune.py
```

---

## ▶️ Inference

```bash
python inference/run_model.py \
  -c model/checkpoints/ckpt_algo_ft_s55000.pth \
  -p "def fibonacci(n):"
```

---

## 🎯 What This Project Demonstrates

* Deep understanding of Transformer internals
* Token-level language modeling
* Fine-tuning for reasoning tasks
* Entropy-based performance analysis
* Practical ML engineering on limited hardware

---

## 📜 License

MIT License

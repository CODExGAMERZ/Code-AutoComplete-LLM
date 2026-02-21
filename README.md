# 🧠 Python Code Autocomplete LLM (From Scratch)

A GPT-style Transformer trained entirely from scratch for Python code autocompletion.

This project implements the full LLM lifecycle:

* Data collection
* Tokenizer training
* Model pretraining
* Efficient block-based training
* Entropy & perplexity evaluation
* CPU-only inference

No external LLM APIs were used.

---

## 🚀 Key Features

* ~33.5M parameter GPT-style decoder-only Transformer
* Custom 8000-token BPE tokenizer trained on Python code
* ~31M cleaned training tokens
* CPU-only training (~8–9 hours per full run)
* Unified training pipeline with checkpoint resume
* Cross-entropy & perplexity evaluation tools
* Loss curve visualization (matplotlib)
* Fully offline and reproducible

---

## 📂 Project Structure

```
AutoComplete-LLm/
├── data/                # Ignored (raw + processed data)
├── model/
│   ├── ai.py
│   └── checkpoints/     # Ignored
│
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer.json
│
├── training/
│   ├── dataset.py
│   └── train.py
│
├── inference/
│   ├── generate.py
│   ├── postprocess.py
│   └── run_model.py
│
├── tools/
│   ├── build_train_file.py
│   ├── clean_repos.py
│   ├── deduplicate.py
│   ├── evaluate_model.py
│   ├── inspect_dataset.py
│   ├── model_stats.py
│   └── plot_loss.py
│
├── MODEL_CARD.md
├── README.md
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
| Vocabulary      | 8000 (Custom BPE)              |
| Parameters      | 33,551,168                     |

---

## 🔬 Training Dynamics

The model is trained using causal language modeling with cross-entropy loss:

```
Loss = -log(P_model(correct_token))
```

Perplexity is computed as:

```
Perplexity = exp(loss)
```

Healthy training behavior observed:

* Rapid entropy collapse in early steps
* Smooth exponential decay
* No divergence
* Stable convergence

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
python tokenizer/train_tokenizer.py
python training/train.py
```

Training automatically resumes from the latest checkpoint.

---

## ▶️ Evaluation

Fast evaluation (~15 minutes):

```bash
python tools/evaluate_model.py
```

Plot loss curve:

```bash
python tools/plot_loss.py
```

---

## ▶️ Inference

```bash
python inference/run_model.py \
  -c model/checkpoints/latest_checkpoint.pth \
  -p "def fibonacci(n):"
```

---

## 🎯 What This Project Demonstrates

* Transformer internals implemented from scratch
* Token-level autoregressive modeling
* Efficient block-based training
* CPU-only large model training
* Practical ML engineering without GPUs
* Entropy & perplexity analysis

---

## 📜 License

MIT License

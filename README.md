# 🧠 Python Code Autocomplete LLM (From Scratch)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Platform](https://img.shields.io/badge/Platform-CPU--Only-lightgrey)
![GitHub stars](https://img.shields.io/github/stars/CODExGAMERZ/Code-AutoComplete-LLM?style=social)
![GitHub forks](https://img.shields.io/github/forks/CODExGAMERZ/Code-AutoComplete-LLM?style=social)
![GitHub issues](https://img.shields.io/github/issues/CODExGAMERZ/Code-AutoComplete-LLM)
![GitHub last commit](https://img.shields.io/github/last-commit/CODExGAMERZ/Code-AutoComplete-LLM)

A **GPT-style Transformer language model** trained **entirely from scratch** to perform **Python code autocompletion**, using **real open-source GitHub repositories** and **no external LLM APIs**.

This project demonstrates a **full end-to-end LLM pipeline** — data collection, tokenization, model training, and inference — designed to run on a **CPU-only consumer laptop**.

---

## 🚀 Features

* Decoder-only **GPT architecture** (causal language model)
* Custom **BPE tokenizer** trained on Python source code
* Fine-tuned on real GitHub repositories:

  * Flask
  * Requests
  * FastAPI
* Fully **local & offline** (privacy-preserving)
* End-to-end **training, inference, and autocomplete**
* Clean, reproducible project structure

---

## 🧩 How It Works

1. **Data Collection** – Open-source Python repositories are cloned into `data/raw/`
2. **Preprocessing** – All `.py` files are merged into a single training corpus
3. **Tokenization** – A custom BPE tokenizer learns Python-specific tokens
4. **Model Training** – A GPT-style Transformer is trained with causal masking
5. **Inference** – The model predicts the next tokens to autocomplete code

---

## 📂 Project Structure

```
AutoComplete-LLm/
│
├── data/
│   ├── raw/                 # GitHub Python repositories
│   │   ├── flask/
│   │   ├── requests/
│   │   └── fastapi/
│   └── processed/
│       └── train.txt        # Merged training corpus
│
├── model/
│   └── ai.py                # GPT model implementation
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
│   └── generate.py
│
├── utils/
│   ├── train_from_raw.py    # Builds train.txt from raw repos
│   ├── generate_big_train_txt.py
│   ├── config.py
│   └── helpers.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧠 Model Details

| Component       | Value                          |
| --------------- | ------------------------------ |
| Architecture    | GPT (Decoder-only Transformer) |
| Layers          | 6                              |
| Attention Heads | 8                              |
| Embedding Size  | 512                            |
| Context Length  | 256 tokens                     |
| Parameters      | ~45M                           |
| Optimizer       | AdamW                          |
| Loss Function   | Cross-Entropy                  |

---

## 📚 Dataset

The model is trained on **real Python source code** collected from open-source GitHub repositories.

Data pipeline:

1. Clone repositories into `data/raw/`
2. Merge and clean `.py` files into `data/processed/train.txt`
3. Train tokenizer and model on the merged corpus

All code is used **locally for educational and research purposes only**.

---

## ⚙️ Quick Start

```bash
pip install -r requirements.txt
python utils/train_from_raw.py
python tokenizer/train_tokenizer.py
python training/train.py
python inference/generate.py
```

---

## ✨ Inference (Autocomplete)

Example:

**Input**

```python
def binary_search(arr, target):
```

**Output**

```python
left = 0
right = len(arr) - 1
while left <= right:
    mid = (left + right) // 2
```

---

## 🎯 What This Project Demonstrates

* Deep understanding of Transformer internals
* Tokenization & sequence modeling
* Training language models from scratch
* Code-specific language modeling
* Practical ML engineering and debugging

---

## 📌 Future Improvements

* Top-k & temperature sampling
* Indentation-aware decoding
* VS Code autocomplete extension
* LoRA fine-tuning
* Byte-level tokenization

---

## 📜 License

MIT License

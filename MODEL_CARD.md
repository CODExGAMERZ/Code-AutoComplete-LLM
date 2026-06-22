# Model Card: Code-AutoComplete-LLM

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-Python%20%7C%20C%20%7C%20Java-3776ab)
![Parameters](https://img.shields.io/badge/Parameters-33.6M-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Model Summary

Code-AutoComplete-LLM is a 33.6M parameter decoder-only causal language model trained from scratch on **402 million tokens** for next-token prediction over Python, C, and Java source code. It is designed to run offline on consumer-grade local systems, requiring minimal resource footprints (under 150MB of RAM) for lightweight code autocompletion.

* **Intended Use**: Local, offline source code completion, autocomplete, and code intelligence.
* **Out of Scope**: General conversational interface, multi-turn dialogue, question-answering, code debugging, and safety-critical execution. The model is a base next-token predictor trained strictly on raw source file splits.

---

## Model Specification

| Hyperparameter | Value |
| :--- | :---: |
| **Total Parameters** | 33,551,168 (~33.6M) |
| **Trainable Parameters** | 33,551,168 |
| **Attention** | Standard Multi-Head Attention (MHA) |
| **Normalization** | LayerNorm (Pre-normalization) |
| **Activation** | GELU |
| **Positional Encoding** | Absolute learned embeddings |
| **Layers** | 8 |
| **Attention Heads** | 8 |
| **Head Dimension** | 64 |
| **Embedding Dimension** | 512 |
| **FFN Hidden Dimension** | 2048 (4 × `n_embd`) |
| **Context Length** | 256 tokens |
| **Vocab Size** | 8,000 |
| **Biases** | Enabled in all Linear layers |
| **Weight Tying** | Disabled |

---

## Architectural Details

The model is built using a clean PyTorch-based GPT style block structure:

### Token & Position Embeddings
The model maps token IDs to a dense vector space using a token embedding layer, and adds absolute positional embeddings representing the index of each token inside the context window:
* **Token Embeddings**: `nn.Embedding(8000, 512)` -> $4,096,000$ parameters.
* **Position Embeddings**: `nn.Embedding(256, 512)` -> $131,072$ parameters.

### Transformer Blocks
The network stack consists of 8 identical Transformer blocks. Each block applies LayerNorm before processing attention and feedforward sub-layers:
1. **Pre-Attention LayerNorm**: Standard `nn.LayerNorm(512)` with weight and bias ($1,024$ parameters).
2. **Causal Self-Attention**:
   * Uses a single fused projection layer for Q, K, and V: `nn.Linear(512, 1536)` ($787,968$ parameters).
   * Splits and reshapes inputs into 8 attention heads with a dimension of 64.
   * Employs causal masking to restrict token visibility to prior context.
   * Projects output back to the embedding space: `nn.Linear(512, 512)` ($262,656$ parameters).
3. **Pre-FFN LayerNorm**: Standard `nn.LayerNorm(512)` ($1,024$ parameters).
4. **Feedforward MLP**:
   * Expands embedding features to a higher-dimensional space: `nn.Linear(512, 2048)` ($1,050,624$ parameters).
   * Applies the GELU activation function.
   * Projects back to the embedding space: `nn.Linear(2048, 512)` ($1,049,088$ parameters).

*Total Parameters per Block*: $3,152,384$ parameters.
*Total for 8 stacked blocks*: $25,219,072$ parameters.

### Final Layer & Head
* **Final LayerNorm**: `nn.LayerNorm(512)` ($1,024$ parameters).
* **Language Modeling Head**: Projects final hidden features back to the token space for class probability distribution: `nn.Linear(512, 8000)` ($4,104,000$ parameters).

---

## Tokenizer

The vocabulary is trained on the processed codebase training split:
* **Algorithm**: Byte-Pair Encoding (BPE).
* **Library**: Hugging Face `tokenizers`.
* **Vocabulary Size**: 8,000.
* **Normalizer**: NFKC.
* **Pre-Tokenizer**: ByteLevel (without prefix space).
* **Special Tokens**: 
  * `<pad>` (ID: `0`)
  * `<unk>` (ID: `1`)
  * `<bos>` (ID: `2`)
  * `<eos>` (ID: `3`)

---

## Dataset

* **Source**: Curated source code files representing Python, C, and Java projects (including algorithms repositories, CPython libraries, Redis database code, and Apache Commons Lang).
* **Training Volume**: Trained on **402 million tokens** compiled from these sources.
* **Preprocessing Filters**:
  * Excludes files under 3 lines or over 4,000 lines.
  * Excludes non-ASCII heavy files (typically binary assets or base64-encoded files).
  * Excludes dedicated test files (`test_*.py`, `*_test.py`, test class suites).
  * Strips large initial header block comments / docstrings (over 20 lines) to avoid noise.
  * Removes exact duplicate files by checking MD5 file content hashes.
* **Dataset Format**: Wrapped in `<bos>{file content}<eos>` delimiters.

---

## Training Configuration

* **Optimizer**: AdamW.
* **Peak Learning Rate**: `3e-4`.
* **Loss Function**: Shifted token Cross-Entropy loss.
* **Batch Size**: 8 per step, with 4 gradient accumulation steps (effective batch size of 32).
* **Logging**: Tracks progress in `training_logs/loss_log.csv`.
* **Checkpoint File**: `model/checkpoints/latest_checkpoint.pth`.

---

## Limitations

* **Small Context Window**: The context size is restricted to 256 tokens, which may limit comprehension of very large files or long-range dependencies.
* **Limited Domain**: Restricted to Python, C, and Java source patterns. Performance on other languages (e.g., JavaScript, Rust) will be low.
* **Base Model**: Lacks reinforcement learning from human feedback (RLHF) or instruction fine-tuning. It behaves purely as a next-token autocomplete predictor and will not answer direct prompt questions.

---

## License

[MIT License](./LICENSE)
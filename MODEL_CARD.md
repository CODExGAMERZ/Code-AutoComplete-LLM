# Model Card: Code-AutoComplete-LLM

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-Python%20%7C%20C%20%7C%20Java-3776ab)
![Parameters](https://img.shields.io/badge/Parameters-29.5M%20→%20235.7M-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Model Summary

Code-AutoComplete-LLM is a series of four decoder-only causal language models trained from scratch for next-token prediction over Python, C, and Java source code. The series progresses from a 29.5M-parameter baseline (v1) to a 235.7M Gemma-style architecture (gemma_v4) incorporating Grouped-Query Attention, Rotary Positional Embeddings, GeGLU feed-forward blocks, and soft logit capping. All models share the same inference interface and checkpoint format; the loading code auto-detects architecture from saved weights.

**Intended use:** local code autocompletion, code generation research, architecture experimentation on consumer or cloud GPU hardware.

**Out of scope:** instruction following, chat, code execution, safety-critical applications. These are base language models trained on next-token prediction only.

---

## Model Family

| | `llm_v1` | `llm_v2` | `llm_v3` | `gemma_v4` |
|:---|:---:|:---:|:---:|:---:|
| **Total Parameters** | 29.5M | 110.3M | 265.1M | 235.7M |
| **Attention** | MHA | MHA | MHA | GQA (16Q / 4KV) |
| **Normalization** | LayerNorm | RMSNorm | RMSNorm | Dual RMSNorm |
| **FFN** | GELU MLP (4×) | SwiGLU | SwiGLU | GeGLU |
| **Positional Encoding** | Absolute learned | Absolute learned | Absolute learned | RoPE |
| **Logit Capping** | — | — | — | Attn 50.0 / Output 30.0 |
| **Layers** | 8 | 12 | 18 | 18 |
| **Heads (Q / KV)** | 8 / 8 | 12 / 12 | 16 / 16 | 16 / 4 |
| **Head Dimension** | 64 | 64 | 64 | 64 |
| **Embedding Dim** | 512 | 768 | 1024 | 1024 |
| **FFN Hidden Dim** | 2048 | 2048 | 2816 | 2816 |
| **Context Length** | 256 | 1024 | 1024 | 1024 |
| **Vocab Size** | 8,000 | 32,000 | 32,000 | 32,000 |
| **Biases** | Yes (all Linear) | No | No | No |
| **Weight Tying** | Yes | Yes | Yes | Yes |
| **Status** | Trained | Trained | Trained | Ready to train |

> **FFN hidden dimension** is computed as `⌈(2 × 4n / 3) / 256⌉ × 256` for v2/v3/gemma_v4, and `4n` for v1's plain GELU MLP. For n=768 this gives 2048; for n=1024 this gives 2816.

---

## Architecture Details

### llm_v1 — 29.5M (OldGPT)

The baseline architecture. Closest to the original GPT-2 design: standard LayerNorm, biased Linear projections throughout, GELU MLP with a 4× expansion ratio, and learned absolute position embeddings.

| Component | Spec |
|:---|:---|
| Layers | 8 |
| Attention | MHA, 8 heads, head_dim=64 |
| Norm | `nn.LayerNorm` (pre-norm, weight + bias) |
| FFN | `Linear(512→2048) → GELU → Linear(2048→512)`, biased |
| Position | `nn.Embedding(256, 512)` |
| Vocab | 8,000 (separate from v2/v3/v4 tokenizer) |

### llm_v2 — 110.3M (GPT)

First major upgrade. Drops LayerNorm in favour of RMSNorm, removes all biases from Linear layers (including QKV), upgrades the FFN to SwiGLU, and expands vocabulary to 32k BPE.

| Component | Spec |
|:---|:---|
| Layers | 12 |
| Attention | MHA, 12 heads, head_dim=64 |
| Norm | `RMSNorm` (pre-norm, weight only, ε=1e-6) |
| FFN | SwiGLU — `silu(W₁x) ⊙ W₂x`, then `W₃` — hidden 2048 |
| Position | `nn.Embedding(1024, 768)` |
| Biases | Removed from all Linear layers |

### llm_v3 — 265.1M (GPT)

Scaled v2: wider (n_embd 768→1024), deeper (12→18 layers), more attention heads (12→16). Architecture is otherwise identical to v2. The wider embedding increases the FFN hidden dimension to 2816.

### gemma_v4 — 235.7M (GemmaGPT)

The current target architecture. Borrows design choices from Gemma while remaining entirely custom. Key differences from v3:

**Grouped-Query Attention** — 16 query heads share 4 KV head pairs. During the forward pass, KV heads are expanded via `repeat_interleave` to match the query head count. This reduces KV cache memory by 4× at inference time.

```
Q projection:  n_embd → n_heads × head_dim     (1024 → 1024)
K projection:  n_embd → n_kv_heads × head_dim  (1024 → 256)
V projection:  n_embd → n_kv_heads × head_dim  (1024 → 256)
```

**Rotary Positional Embeddings (RoPE)** — Applied to Q and K after projection, before KV cache concatenation. Uses a pre-computed `(cos, sin)` cache up to `2 × block_size` positions, allowing some extrapolation beyond the training context length. `theta=10000`.

**GeGLU feed-forward** — Replaces SwiGLU's `silu` gate with `gelu(tanh approximate)`:

```
GeGLU(x) = W₃ · (gelu(W₁x) ⊙ W₂x)
```

**Dual RMSNorm** — Each transformer block applies RMSNorm both before the attention/FFN sub-layers (pre-norm) and on the output of each sub-layer before the residual addition (post-norm):

```python
x = x + post_norm1(attn(norm1(x)))
x = x + post_norm2(ffn(norm2(x)))
```

**Attention logit soft-capping:**
```python
scores = 50.0 * torch.tanh(scores / 50.0)
```
Applied after `QKᵀ / √d` and before the causal mask. Prevents attention score spikes without hard clipping.

**Output logit soft-capping:**
```python
logits = 30.0 * torch.tanh(logits / 30.0)
```
Applied to the final vocabulary projection before returning from `GemmaGPT.forward`.

**No position embedding table** — gemma_v4 has no `pos_emb` weight; position information is carried entirely by RoPE applied inside each attention layer.

**Parameter breakdown (gemma_v4):**

| Component | Parameters |
|:---|---:|
| Token embedding (32k × 1024) | 32,768,000 |
| 18 × GemmaBlock | 202,973,184 |
| Final RMSNorm | 1,024 |
| Output head (tied to embedding) | — |
| **Total (unique)** | **235,742,208** |

Per GemmaBlock: Q (1,048,576) + K (262,144) + V (262,144) + out_proj (1,048,576) + 4×RMSNorm (4,096) + GeGLU W₁/W₂/W₃ (3 × 2,883,584) = **11,276,288** params.

---

## Tokenizer

| Property | Value |
|:---|:---|
| Algorithm | Byte-Pair Encoding (BPE) |
| Library | HuggingFace `tokenizers` |
| Vocabulary size | 32,000 (v2/v3/gemma_v4) · 8,000 (v1) |
| Pre-tokenizer | ByteLevel (`add_prefix_space=False`) |
| Normalizer | NFKC |
| Decoder | ByteLevel |
| Special tokens | `<pad>` (0), `<unk>` (1), `<bos>` (2), `<eos>` (3) |
| Minimum frequency | 2 subwords |

The tokenizer is trained on `data/processed/train.txt` using `tokenizer/train_tokenizer.py`. From v2 onward, the compiled `tokenizer.json` is **embedded directly inside every `.pth` checkpoint** under the `tokenizer_json` key, so inference requires no separate tokenizer file.

---

## Dataset

| Property | Value |
|:---|:---|
| Primary source | `codeparrot/github-code-clean` (Hugging Face) |
| Languages | Python, C, Java |
| Files per language | 60,000 (default; configurable via `--num_files`) |
| Target token range | 100M–200M+ tokens |
| Minimum file lines | 3 |
| Maximum file lines | 4,000 |
| Document format | `<bos>{file content}<eos>` |
| Storage format | Memory-mapped `uint16` `.bin` files |
| Sampling | Non-overlapping fixed-stride blocks of `block_size` |

**Cleaning filters applied by `hardened_clean.py`:**
- Removes files shorter than 3 lines or longer than 4,000 lines.
- Removes files where >40% of characters are non-ASCII (binary assets, encoded data).
- Removes genuine test entrypoints only: `test_*.py`, `*_test.py`, `tests.py`, JUnit-style class names. Files with "test" elsewhere in the name (e.g., `latest_utils.py`, `ContextTest.java`) are kept.
- Removes duplicate files by content hash.

---

## Training Configuration

| Hyperparameter | Value |
|:---|:---|
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Peak learning rate | 3e-4 |
| LR schedule | Cosine annealing |
| Warmup | 5% of total gradient steps (linear) |
| Gradient clipping | `max_norm=1.0` |
| Precision | FP16 AMP (`torch.amp.GradScaler`) |
| Compilation | `torch.compile` (auto-enabled on CUDA, PyTorch 2.0+) |
| Per-device batch size | 2 (DDP) |
| Gradient accumulation | 32 steps |
| Effective batch size | 128 tokens-per-step × batch_size × world_size |
| Hardware | 2× NVIDIA T4 (16GB each, 30GB total VRAM) |
| DDP backend | NCCL via `torchrun --nproc_per_node=2` |
| Loss function | Cross-entropy on shifted token sequences |
| Validation | Per-epoch, averaged across all validation batches; perplexity reported |

Training is resumable from any checkpoint. On resume, architecture config (`vocab_size`, `block_size`, `n_layers`, `n_heads`, `n_embd`, `architecture_type`) is read from the checkpoint's `config` dict, and optimizer, scheduler, and scaler states are restored. If the saved `epoch` matches or exceeds `--epochs`, the epoch count is extended automatically to allow continuation.

---

## Checkpoint Format

Every `.pth` file saved by `training/train.py` is a Python dict with the following keys:

| Key | Type | Description |
|:---|:---|:---|
| `model_state` | `OrderedDict` | Model weights (raw model, not DDP-wrapped) |
| `optimizer_state` | `dict` | AdamW state (moments, step counts) |
| `scheduler_state` | `dict` | LambdaLR state |
| `scaler_state` | `dict` \| `None` | GradScaler state; `None` on CPU |
| `epoch` | `int` | Last completed epoch index |
| `global_step` | `int` | Cumulative gradient update steps |
| `tokenizer_json` | `str` \| `None` | Full contents of `tokenizer/tokenizer.json` |
| `config` | `dict` | Architecture metadata (see below) |

`config` dict keys: `vocab_size`, `block_size`, `n_layers`, `n_heads`, `n_embd`, `architecture_type` (`"gemma_v4"` or `"gpt"`).

---

## Architecture Auto-Detection

Inference scripts (`run_model.py`, `compare_models.py`) do not require you to specify which model version a checkpoint contains. Detection is done by inspecting weight keys:

| Weight key present | Architecture | Class |
|:---|:---|:---|
| `blocks.0.attn.q_proj.weight` | gemma_v4 (GQA, split projections) | `GemmaGPT` |
| `blocks.0.ln1.weight` | v1 (LayerNorm key) | `OldGPT` |
| neither of the above | v2 / v3 (RMSNorm, fused QKV) | `GPT` |

Config metadata (`n_embd`, `n_layers`, `n_heads`, `block_size`, `vocab_size`) is then read from the `config` dict if present, or inferred from weight tensor shapes as a fallback for older checkpoints.

---

## Inference

### Sampling parameters

| Parameter | Autocomplete mode | Creative mode |
|:---|:---:|:---:|
| `--temperature` | 0.2 | 0.8 |
| `--top_k` | 10 | 50 |
| `--max_tokens` | 120 | 200+ |

Low temperature + narrow top-k produces deterministic, syntactically tight completions. Higher temperature broadens the distribution and is better suited for generating novel structure from a short prompt.

### KV Cache

All architectures support autoregressive KV caching. The initial forward pass processes the full prompt and returns cached K/V tensors for all layers. Subsequent steps decode one token at a time, attending over the growing cache without recomputing earlier positions. Memory cost scales as:

```
2 × n_layers × n_kv_heads × (prompt_len + generated_len) × head_dim × dtype_bytes
```

For gemma_v4 at FP16: `2 × 18 × 4 × 1024 × 64 × 2 = 18.9MB` at full context.

### Prompt handling

Prompts longer than `block_size` are truncated to the last `block_size` tokens. Token IDs outside the model's vocabulary (can occur when using the 32k tokenizer with a v1 checkpoint) are clamped to `<unk>` (id=1). Generation halts on `<eos>` (id=3) regardless of `--max_tokens`.

---

## Limitations

- **No instruction following.** These are base models trained on next-token prediction over raw source files. They do not respond to natural language instructions, docstrings as queries, or fill-in-the-middle prompts without fine-tuning.
- **Context length.** All models were trained at `block_size=1024` (v1: 256). While RoPE in gemma_v4 allows some extrapolation, outputs beyond the training length are unreliable.
- **Language coverage.** The dataset is Python, C, and Java only. Output quality for other languages will be significantly lower.
- **Data volume.** For the 29.5M–265M parameter range, meaningful generalization requires at least 100M–200M training tokens. Runs on smaller corpora will produce syntactically broken or repetitive output regardless of architecture.
- **No evaluation benchmarks.** No standard coding benchmarks (HumanEval, MBPP, etc.) have been run against these checkpoints. The only available metrics are validation cross-entropy loss and perplexity from `tools/evaluate_model.py`.
- **Tokenizer mismatch.** v1 uses an 8k vocabulary trained on a different corpus. Do not mix v1 checkpoints with the 32k tokenizer or vice versa.

---

## License

[MIT License](./LICENSE)
# AutoComplete-LLM v2

## Model Overview

AutoComplete-LLM v2 is a 60M parameter GPT-style transformer trained for Python code autocomplete.

## Architecture

- Parameters: ~60M
- Layers: 8
- Attention Heads: 8
- Embedding Size: 512
- Context Window: 256 tokens
- Vocabulary Size: 8000 (Custom BPE)

## Training Details

- Dataset Size: ~14.5M tokens per epoch
- Total Training Tokens Seen: ~29M
- Optimizer: AdamW
- Learning Rate: 3e-4
- Scheduler: Linear Warmup + Decay
- Hardware: Ryzen AI 7 CPU, 32GB RAM
- Training Time: ~9 hours (CPU only)

## Final Metrics

- Average Cross Entropy Loss: <FILL>
- Perplexity: <FILL>

## Intended Use

- Python code autocomplete
- Educational experimentation
- Lightweight local inference
- Small-scale research

## Limitations

- Context limited to 256 tokens
- Not instruction-tuned
- May hallucinate long-range dependencies
- Not optimized for multi-file reasoning

## Future Improvements

- Increase context to 512
- Add structural fine-tuning dataset
- Implement KV-cache inference
- Quantization for faster CPU inferenc

## Reproducibility

The training dataset is not included in this repository due to size.
To reproduce training:
1. Clone recommended Python repositories.
2. Run cleaning pipeline.
3. Build train.txt.
4. Run training script.
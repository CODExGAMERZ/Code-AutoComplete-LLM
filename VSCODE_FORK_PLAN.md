# 🚀 Custom VS Code Fork: Multilingual IDE with Built-In Autocomplete LLM

This document outlines the architectural plan and roadmap to fork Microsoft's VS Code and build a personalized, lightweight IDE specifically optimized for **Python, C, and Java** that uses our custom 33M parameter LLM for offline AI autocompletion and help.

---

## 💡 Concept Overview
By building a customized version of VS Code, we can package our lightweight 33M parameter LLM directly into the IDE distribution. 
* **Zero Setup**: The user opens the IDE and instantly gets AI autocompletion without installing extra extensions or configuring API keys.
* **100% Offline**: Running a 33M parameter model takes under 150MB of RAM, making it fast and battery-friendly on standard CPUs.

---

## 🛠️ Roadmap & Architecture

### Phase 1: Forking VS Code
1. Clone the open-source upstream repository: `microsoft/vscode`.
2. Configure build tools (Node.js, Yarn, Python) to build the client locally.
3. Rename the IDE branding, icon, and logo (e.g., "Polyglot IDE" or "Antigravity Code").

### Phase 2: Built-in Local Inference Server
Rather than requiring the user to run Python scripts manually, we can embed a lightweight local model runner directly inside the IDE:
* **Option A (ONNX Runtime Node.js)**: Export our trained PyTorch model (`.pth`) to **ONNX format**. Run inference directly in the VS Code backend using `@microsoft/onnxruntime-node`. This removes the Python dependency completely!
* **Option B (Bundled Python Runner)**: Ship a lightweight, pre-compiled Python binary (like PyInstaller or micro-python) bundled inside the IDE package to load the PyTorch weights and host a local `localhost` endpoint.

### Phase 3: Integration into Code Editor UI
We can bind the local inference logic to the standard VS Code API:
1. **Inline Completion Provider**: Register a `vscode.InlineCompletionItemProvider`.
2. As the developer types, capture the document context (last 256 tokens) and trigger the ONNX/Python model.
3. Show suggestions in gray text inline (similar to GitHub Copilot).

---

## 🏁 How to Build & Test Locally

### 1. Export PyTorch Model to ONNX format (Python)
Add this script to your model directory to convert your PyTorch model to ONNX so it can be loaded inside JavaScript/Node.js:
```python
import torch
from model.ai import GPT

# Load model structure
model = GPT(vocab_size=8000, block_size=256)
ckpt = torch.load("latest_checkpoint.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state"])

# Sample input tensor (dummy tokens)
dummy_input = torch.zeros((1, 256), dtype=torch.long)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    input_names=["input"], 
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch", 1: "sequence"}}
)
print("ONNX model exported successfully!")
```

### 2. Loading ONNX in VS Code Extension (JavaScript)
```javascript
const ort = require('onnxruntime-node');

async function getCompletion(tokenIds) {
    const session = await ort.InferenceSession.create('model.onnx');
    const inputTensor = new ort.Tensor('int64', new BigInt64Array(tokenIds), [1, tokenIds.length]);
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    
    // Process logits and return predicted completion...
}
```

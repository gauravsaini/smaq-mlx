# SMAQ — Quick Start Guide

> **Folded Turbo SMAQ** compresses your LLM's KV cache to **~4.8× smaller** with only **+3% perplexity cost**.
> Works with any MLX model on Apple Silicon. No retraining, no fine-tuning — just plug in and compress.

---

## What You Need

| Requirement     | Minimum                           |
|-----------------|-----------------------------------|
| **Hardware**    | Apple Silicon Mac (M1/M2/M3/M4)   |
| **RAM**         | Enough to load your model + ~20%  |
| **Python**      | 3.9+                             |
| **Package Mgr** | `uv` (recommended) or `pip`      |

---

## 1. Install (One-Time Setup)

```bash
# Clone both repos
git clone https://github.com/gauravsaini/smaq.git
git clone https://github.com/gauravsaini/smaq-mlx.git

# Create a virtual environment
cd smaq
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install mlx mlx-lm datasets huggingface_hub
uv pip install -e ../smaq-mlx
```

---

## 2. Run the Full Evaluation (One Command)

This is the **easiest way to test** SMAQ on any MLX model:

```bash
# Set your model (any HuggingFace MLX model works)
export MODEL_ID="caiovicentino1/Qwopus3.5-9B-v3-PolarQuant-MLX-4bit"
export BUNDLE_DIR="$HOME/models/$(basename $MODEL_ID)"

# Download the model (if not already present)
huggingface-cli download $MODEL_ID --local-dir $BUNDLE_DIR

# Run the evaluation
export SMAQ_MLX_REPO="$HOME/smaq-mlx"
export BACKENDS="folded_turbo_smaq"
export OUTPUT_JSON="results.json"
export PYTHONPATH="."
bash scripts/run_folded_turbo_eval.sh
```

### What this does:

1. **Loads** the model and tokenizer
2. **Runs** a 12-prompt deterministic test suite (exact-match scoring)
3. **Measures** perplexity on WikiText-2 (teacher-forced)
4. **Extracts** offline `Σ_q` calibration matrices from the model
5. **Compresses** the KV cache using Folded Turbo SMAQ
6. **Reports** PPL, exact-match, cache size, and compression ratio

### Tuning the Evaluation:

```bash
# Shorter/faster eval (for quick sanity check)
export DET_MAX_TOKENS=16       # fewer generation tokens
export PPL_MAX_TEXTS=1         # fewer perplexity samples
export LONG_MAX_TOKENS=8       # shorter long-context test

# Full eval (for paper-quality numbers)
export DET_MAX_TOKENS=48
export PPL_MAX_TEXTS=16
export LONG_MAX_TOKENS=32
```

---

## 3. Use in Your Own Code (Python API)

### Minimal Example — Generate with Compressed KV Cache

```python
import mlx_lm
from smaq_mlx.api import MLXRuntimeConfig, backend_enabled, make_prompt_cache

# Load any MLX model
model, tokenizer = mlx_lm.load("path/to/your/model")

# Configure SMAQ
config = MLXRuntimeConfig(
    backend="folded_turbo_smaq",
    enabled=True,
)

# Generate with compressed KV cache
with backend_enabled(config):
    cache = make_prompt_cache(model, backend="folded_turbo_smaq")
    result = mlx_lm.generate(
        model, tokenizer,
        prompt="Explain quantum computing in simple terms.",
        max_tokens=256,
        prompt_cache=cache,
    )
    print(result)
```

### Seamless Server Deployment (with Offline Σ_q Caching)

Deploying SMAQ behind an OpenAI-compatible HTTP server (`mlx_lm.server`) requires ensuring that `make_prompt_cache` receives the optimal `Sigma_q`. Instead of a messy `.venv`, we recommend standalone `uv run` scripts that handle the caching naturally.

Create `serve.py`:

```python
# /// script
# requires-python = ">=3.9"
# dependencies = ["mlx", "mlx-lm", "datasets", "huggingface_hub"]
# ///

import os, sys, argparse
import mlx.core as mx
import mlx_lm, mlx_lm.server
from smaq_mlx.api import MLXRuntimeConfig, enable_backend
import smaq_mlx.patch
from scripts.experiments.progressive_eval_common import extract_offline_sigma_q, load_wikitext_texts

def setup_smaq(model_path):
    model, tokenizer = mlx_lm.load(model_path)
    sigma_path = os.path.join(model_path, "sigma_q.npz")
    
    if os.path.exists(sigma_path):
        data = mx.load(sigma_path)
        sigma_q = {int(k.split("_")[1]): v for k, v in data.items()}
    else:
        print("Extracting Σ_q calibration...")
        texts = load_wikitext_texts(split="test[:8]", max_texts=4)
        sigma_q = extract_offline_sigma_q(model, tokenizer, texts, max_tokens=2048)
        mx.savez(sigma_path, **{f"layer_{k}": v for k, v in sigma_q.items()})

    # Globally inject the offline metrics so mlx_lm.server uses them automatically
    original_make = smaq_mlx.patch.make_prompt_cache
    def _injected_make(model_inst, **kwargs):
        kwargs.setdefault("Sigma_q", sigma_q)
        kwargs["backend"] = "folded_turbo_smaq"
        return original_make(model_inst, **kwargs)
    
    smaq_mlx.patch.make_prompt_cache = _injected_make
    enable_backend(MLXRuntimeConfig(backend="folded_turbo_smaq", enabled=True))

if __name__ == "__main__":
    setup_smaq(sys.argv[2] if len(sys.argv) > 2 else "path/to/model")
    mlx_lm.server.main()
```

Launch the server with the cached metrics seamlessly:
```bash
uv run serve.py --model path/to/model --port 8080
```
---

## 4. Supported Architectures

SMAQ works with any model that `mlx-lm` can load. Tested configurations:

| Model | Arch | Weight Quant | Layers Compressed | Result |
|-------|------|-------------|-------------------|--------|
| Qwen3.5-9B TQ3 | Qwen3.5 hybrid | TurboQuant 3-bit | 8 of 32 (full attn) | PPL +3.1%, 4.83× |
| Qwopus3.5-9B PolarQuant | Qwen3.5 hybrid | HLWQ 4-bit | 8 of 32 (full attn) | PPL +3.1%, 4.83× |
| Gemma-4-E4B | Gemma4 hybrid | 4-bit | 24 of 42 (cacheable) | 4.84× compression |
| Carnice-27b Q6 | Qwen3.5 hybrid | Q6 6-bit | 16 of 64 (full attn) | PPL +1.8%, 4.83× |

### Architecture Notes

- **Hybrid attention** (Qwen3.5): Has both `linear_attention` and `full_attention` layers. SMAQ compresses only `full_attention` layers (the ones that actually build a KV cache).
- **Sliding + full** (Gemma4): Has sliding window layers + full attention layers + KV-shared layers. SMAQ compresses all cacheable layers.
- **Standard transformers**: All layers compressed by default.

---

## 5. What's Happening Under The Hood

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Your Prompt │ ──▶ │  Offline Σ_q     │ ──▶ │  Log-Compressed   │
│              │     │  Calibration     │     │  Spectral Shaping │
└──────────────┘     │  (query covar.)  │     │  E_f = V·diag(f)  │
                     └──────────────────┘     └─────────┬─────────┘
                                                        │
                     ┌──────────────────┐               ▼
                     │  Hadamard Basis  │ ◀── Rotate Σ_q into
                     │  Rotation        │     orthogonal frame
                     └────────┬─────────┘
                              │
                              ▼ Extract diagonal
                     ┌──────────────────┐
                     │  Per-Head Scalar │     ← "Folded" metric
                     │  Boundary Scale  │     (no dense matmul!)
                     └────────┬─────────┘
                              │
                              ▼ Apply during inference
                     ┌──────────────────┐
                     │  3-bit Scalar    │     TurboQuant codec
                     │  Quantization    │     with SMAQ-aware
                     │  + Dequant       │     boundaries
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  4.83× Smaller   │     17 MB instead of
                     │  KV Cache!       │     83 MB
                     └──────────────────┘
```

**Key insight:** Instead of naively rotating vectors (TurboQuant) or using dense metric transforms (which collapse), SMAQ:
1. Learns which directions matter via query covariance (`Σ_q`)
2. Compresses the metric into a lightweight diagonal scaling
3. Folds this scaling into the quantizer boundaries
4. Result: same compression ratio as TurboQuant, but with query-aware precision allocation

---

## 6. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `alexcovo/qwen35-9b-mlx-turboquant-tq3` | HuggingFace model ID |
| `BUNDLE_DIR` | `~/models/<basename>` | Local model directory |
| `SMAQ_MLX_REPO` | `../smaq-mlx` | Path to smaq-mlx repo |
| `BACKENDS` | `smaq,turboquant,folded_turbo_smaq` | Comma-separated backend list |
| `DET_MAX_TOKENS` | `48` | Max tokens for deterministic suite |
| `PPL_MAX_TEXTS` | `16` | Number of WikiText samples for PPL |
| `PPL_CHUNK_TOKENS` | `256` | Token chunk size for PPL eval |
| `LONG_MAX_TOKENS` | `32` | Max tokens for long-context test |
| `TURBOQUANT_BITS` | `3` | KV quantization bit width |
| `OUTPUT_JSON` | _(none)_ | Path to save JSON results |

---

## 7. Troubleshooting

**Q: "The model is too large for my Mac"**
> Check `mlx_lm.load()` peak memory. The model weights + KV cache need to fit in unified memory. SMAQ reduces KV by ~5×, but model weights are the main bottleneck.

**Q: "Some layers show `ArraysCache` instead of `FoldedTurboSMAQKVCache`"**
> This is expected for hybrid architectures. `linear_attention` layers don't build traditional KV caches, so they use standard `ArraysCache`. Only `full_attention` layers get compressed.

**Q: "PPL is much higher than expected"**
> Make sure offline `Σ_q` calibration is properly injected. Without passing `Sigma_q` explicitly (via `_injected_make_prompt_cache` for servers or `make_prompt_cache` directly for standard inference), the system falls back to dynamic prompt-time calibration which is significantly less stable.

---

## License

SMAQ is released under the MIT License. See [LICENSE](LICENSE) for details.

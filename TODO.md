# SMAQ-MLX Implementation Plan

Port of [smaq](https://github.com/gauravsaini/smaq) to MLX for Apple Silicon, following [turboquant-mlx](https://github.com/sharpner/turboquant-mlx) architecture patterns.

## Architecture Mapping

| Original (PyTorch/CUDA) | MLX Port (Apple Silicon) |
|---|---|
| `torch` ops | `mlx.core` (mx) ops |
| Triton kernels | MLX native ops + `mx.quantized_matmul` |
| CUDA device | Apple Metal (unified memory) |
| vLLM integration | mlx-lm integration |

## Core Concept

SMAQ replaces TurboQuant's random rotation with **log-compressed spectral metric shaping** derived from query covariance. Keys are compressed using query-aware metrics instead of rotation-invariant quantization.

## Implementation Checklist

### Phase 1: Core Math & Quantizers ✅
- [x] `smaq/ssf.py` — Log-compressed spectral shaping functions
- [x] `smaq/block_vq.py` — Block VQ quantizer (k-means in shaped space)
- [x] `smaq/quantizer.py` — Scalar quantizer (faster deployment path)

### Phase 2: KV Cache Infrastructure 🔄
- [x] `smaq/kv_cache.py` — SMAQ KV cache with prefill/append/attend
- [ ] `smaq/capture.py` — Ring buffer + streaming capture engine
- [ ] `smaq/store.py` — Chunked compressed KV store with lazy flatten
- [ ] `smaq/score.py` — Hybrid attention: compressed history + exact recent

### Phase 3: mlx-lm Integration
- [ ] `smaq/patch.py` — Monkey-patch for mlx-lm SDPA dispatch
- [ ] `smaq/attention_smaq.py` — SMAQ SDPA with MLX native ops

### Phase 4: Scripts & Benchmarks
- [ ] `benchmark.py` — Speed + quality benchmark
- [ ] `run_llm.py` — Interactive demo with SMAQ KV cache
- [ ] `requirements.txt` — Dependencies
- [ ] `README.md` — Documentation

### Phase 5: Tests
- [ ] `tests/test_smaq.py` — Unit tests for all components
- [ ] `tests/test_integration.py` — Integration with mlx-lm

## File Structure

```
smaq-mlx/
├── smaq/
│   ├── __init__.py
│   ├── ssf.py                 # Spectral shaping (core math)
│   ├── block_vq.py            # Block VQ quantizer
│   ├── quantizer.py           # Scalar quantizer
│   ├── kv_cache.py            # SMAQ KV cache
│   ├── capture.py             # Ring buffer + capture engine
│   ├── store.py               # Compressed KV store
│   ├── score.py               # Hybrid attention
│   ├── patch.py               # mlx-lm SDPA monkey-patch
│   └── attention_smaq.py      # SMAQ SDPA implementation
├── tests/
│   ├── test_smaq.py
│   └── test_integration.py
├── benchmark.py
├── run_llm.py
├── requirements.txt
├── README.md
└── TODO.md
```

## Progress Log

### 2026-03-31 — Initial implementation
- **Phase 1-3 complete**: All core components implemented
- **Phase 4 complete**: benchmark.py, run_llm.py, requirements.txt, README.md
- **Phase 5 complete**: 25/25 unit tests passing
- **MLX compatibility fixes**: 
  - `mx.linalg.eigh` → CPU stream
  - `scatter_update` → `.at[].add()` / slice assignment
  - `mx.random.choice` → `mx.random.categorical`
  - `.copy()` → `mx.array()`

### 2026-03-31 — Model testing
- **Tested models on 8GB Mac**:
  - Qwen3.5-2B-4bit: VLM, requires PyTorch/torchvision — not suitable
  - Qwen3.5-2B-OptiQ-4bit: text-gen, 0.48GB RSS, but hybrid architecture (18 linear_attn + 6 self_attn layers) — SMAQ only applies to 6/24 layers
  - **Llama-3.2-1B-Instruct-4bit**: text-gen, 0.72GB RSS, pure attention (16 layers, head_dim=64) — best fit
- **SMAQ integration status**:
  - Core math verified: SMAQ shaping reduces attention score MSE vs identity metric
  - KVCache implements mlx-lm interface (update_and_fetch, offset, state)
  - Drop-in replacement works but quantization quality needs proper calibration data
  - Block VQ with 256 centroids/8D blocks shows good reconstruction quality
- **Known limitations**:
  - Random calibration data produces poor quantization — needs domain-specific calibration
  - Scalar quantizer at 3-4 bits has high score MSE (~6-8) vs exact attention
  - Full SDPA interception (computing attention directly against quantized keys) needs deeper mlx-lm integration

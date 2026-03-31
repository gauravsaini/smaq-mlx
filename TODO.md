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

### 2026-03-31
- **Phase 1 complete**: ssf.py, block_vq.py, quantizer.py implemented
- **Phase 2 complete**: kv_cache.py, capture.py, store.py, score.py implemented
- **Phase 3 complete**: patch.py, attention_smaq.py implemented
- **Phase 4 complete**: benchmark.py, run_llm.py, requirements.txt, README.md created
- **Phase 5 complete**: 25/25 unit tests passing
  - SSF tests: 3/3
  - BlockVQ tests: 5/5
  - ScalarQuantizer tests: 4/4
  - KVCache tests: 7/7
  - RingBuffer tests: 2/2
  - CompressedKVStore tests: 3/3
- **Remaining**: Integration tests with mlx-lm (requires model download)

# SMAQ-MLX

SMAQ-MLX is an **experimental MLX integration package** for running KV-cache compression backends inside real `mlx_lm` generation.

It is not just offline quantization code. It hooks the actual MLX serving path:

- patches `mlx_lm.models.cache.make_prompt_cache`
- patches `mlx_lm.models.base.scaled_dot_product_attention`
- uses compressed historical K/V during decode
- selects a runtime backend explicitly instead of relying on patch order

On tested Qwen-family MLX text models, this path is real, not shadow-only.

## Status

Current status is **usable experimental backend**, not broad production release.

What is working well:

- real `mlx_lm` integration
- explicit in-process API
- unified backend adapter layer
- baseline vs SMAQ benchmarking
- deterministic exact-vs-SMAQ quality checks
- Qwen-family MLX text model support

Known rough edges:

- some tokenizer stacks hit the Mistral regex warning/fallback path
- Gemma-family support is still performance-rough
- public compatibility matrix is still small

## What This Package Is

Today, `smaq-mlx` should be understood as:

- a **real MLX serving integration**
- a **KV-cache compression backend host** for `mlx_lm`
- a package you can integrate into Python application code

It is **not** just a notebook, offline metric calculator, or shadow-cache demo.

When enabled, it patches the real `mlx_lm` generation path so decode uses compressed historical K/V.

## Do I Need Autotune / Per-Layer Calibration?

Short answer: **not for the currently validated integration path**.

Today:

- the working MLX integration path does **not require** per-layer autotune to run
- the tested Qwen-family serving path works with the default runtime setup
- we already validated deterministic exact-match generation on a small 27B Qwen-family prompt suite without a separate autotune pass

What is true conceptually:

- the **full SMAQ paper idea** is query-aware metric shaping
- that can benefit from calibration data such as per-layer or per-head query statistics
- so **paper-faithful metric tuning** is still a future improvement path

Production recommendation right now:

- treat calibration/autotune as **optional future optimization**
- do **not** make it a hard requirement for first integration
- first ship the stable runtime path, benchmark it, and validate output quality

If we later add calibration cleanly, it should be:

- explicit
- offline or startup-time
- optional
- documented per model family

## Supported Integration Story

Recommended integration today is:

1. install `smaq-mlx`
2. enable a backend in-process with `MLXRuntimeConfig`
3. load model normally with `mlx_lm.load(...)`
4. generate normally with `mlx_lm.generate(...)`

That means existing MLX apps only need a small glue change, not a rewrite.

## Install

```bash
uv pip install mlx mlx-lm
uv pip install -e .
```

## Backend Matrix

| Backend | Status | Package | Notes |
| --- | --- | --- | --- |
| `polarquant` | Experimental | `mlx-turboquant` | PolarQuant-style rotated scalar quantization without QJL correction |
| `smaq` | Experimental | built into `smaq-mlx` | Best current fidelity story on tested Qwen-family MLX workloads |
| `stacked_turbo_smaq` | Research | `turboquant-mlx` | Experimental two-stage cascade: TurboQuant approximation feeding SMAQ |
| `turboquant` | Experimental | `turboquant-mlx` | Works through the unified adapter layer |

You can inspect this at runtime with:

```python
from smaq_mlx import available_backends, backend_matrix

print(available_backends())
print(backend_matrix())
```

## Recommended Usage

Use the **in-process API**. This is the supported path for application code.

```python
import mlx_lm
from smaq_mlx import MLXRuntimeConfig, enable_backend

config = MLXRuntimeConfig(
    backend="smaq",
    key_bits=4,
    value_bits=4,
    mode="hybrid",
    strict_benchmark=False,
    require_true_compressed=False,
)

enable_backend(config)
model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-4B-MLX-4bit")
text = mlx_lm.generate(model, tokenizer, prompt="Explain KV cache compression.", max_tokens=64)
print(text)
```

This is the preferred production-style integration path because:

- it does not require editing `mlx_lm` in `site-packages`
- config stays explicit in your app
- enable/disable behavior is easy to reason about
- it works naturally with existing model load / generate code

You can also use the one-shot wrapper:

```python
import mlx_lm
from smaq_mlx import MLXRuntimeConfig, generate

model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-4B-MLX-4bit")
text = generate(
    model,
    tokenizer,
    prompt="Explain KV cache compression.",
    max_tokens=64,
    config=MLXRuntimeConfig(backend="smaq", key_bits=4, value_bits=4),
)
print(text)
```

Or create caches explicitly:

```python
import mlx_lm
from smaq_mlx import MLXRuntimeConfig, enable_backend, make_prompt_cache

config = MLXRuntimeConfig(backend="smaq", key_bits=4, value_bits=4)
enable_backend(config)

model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-4B-MLX-4bit")
caches = make_prompt_cache(
    model,
    backend=config.backend,
    key_bits=config.key_bits,
    value_bits=config.value_bits,
    mode=config.mode,
    strict_benchmark=config.strict_benchmark,
)

text = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Explain KV cache compression.",
    max_tokens=64,
    prompt_cache=caches,
)
print(text)
```

## Optional Auto-Hook

There is also an install helper that modifies the installed `mlx_lm` package so `SMAQ_ENABLED=1` auto-patches at import time:

```bash
smaq-mlx-install
```

This is convenient for local experiments, but it is **not** the recommended application integration path. Prefer the explicit Python API above.

## How To Integrate In An Existing App

Minimal application integration usually looks like this:

```python
import mlx_lm
from smaq_mlx import MLXRuntimeConfig, enable_backend

enable_backend(
    MLXRuntimeConfig(
        backend="smaq",
        key_bits=4,
        value_bits=4,
        mode="hybrid",
    )
)

model, tokenizer = mlx_lm.load("/path/to/model")
response = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=64)
```

If your app already owns prompt-cache creation, use:

```python
from smaq_mlx import make_prompt_cache

caches = make_prompt_cache(model, backend="smaq", key_bits=4, value_bits=4, mode="hybrid")
response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Hello",
    max_tokens=64,
    prompt_cache=caches,
)
```

## Backend API

The runtime is intentionally designed as:

- one MLX patch owner
- many selectable backends
- no backend-specific patch ordering

Detailed backend contract:

- [docs/backend-api.md](./docs/backend-api.md)

To add a backend:

1. inherit from `RuntimeBackend`
2. implement cache creation + SDPA dispatch
3. register it with `register_backend(...)`
4. add conformance tests

The current `polarquant` backend is implemented against the MLX-native `mlx-turboquant` reference path.
Right now it uses one shared bit-width for keys and values, mirroring that upstream cache shape.

## What “Real Integration” Means Here

This package is integrated into the actual MLX runtime path, not only into offline analysis.

Specifically:

- `mlx_lm.models.cache.make_prompt_cache` is patched
- `mlx_lm.models.base.scaled_dot_product_attention` is patched
- compressed historical K/V is used during decode attention
- recent exact tail can still be kept in the hybrid path

For a healthy real-compressed run, capability reports should show:

- `compressed_history = true`
- `compressed_history_shadow_only = false`
- `decode_uses_compressed_keys = true`
- `decode_uses_compressed_values = true`

## Real Integration, Not Shadow-Only

When SMAQ is active, the cache capability report should show:

- `compressed_history = true`
- `compressed_history_shadow_only = false`
- `decode_uses_compressed_keys = true`
- `decode_uses_compressed_values = true`

This means decode attention is actually using compressed historical K/V, not just tracking compressed copies on the side.

## Validated Today

Tested successfully in this repo on:

- Qwen-family MLX text models at 4B, 9B, and 27B scale
- deterministic exact-vs-SMAQ output matching on a small prompt suite for a 27B Qwen-family model
- real baseline-vs-SMAQ serving runs through `mlx_lm.generate(...)`
- unified `smaq` / `turboquant` backend selection through one adapter layer

## Current Recommendation

If someone else is integrating this today:

- start with Qwen-family MLX text models
- use the explicit Python API, not the auto-hook
- begin with `key_bits=4`, `value_bits=4`
- validate quality with deterministic prompt suites before lowering bits
- treat per-layer autotune/calibration as a future optional enhancement, not a first dependency

## Not Yet Claimed

SMAQ-MLX does **not** yet claim:

- broad model-family support
- production-hardened tokenizer compatibility
- fused-kernel-grade performance on every model family
- full quality validation across long prompt suites and stochastic decoding

## Public API

Main symbols:

- `MLXRuntimeConfig`
- `enable_backend`
- `disable_backend`
- `backend_enabled`
- `available_backends`
- `backend_matrix`
- `generate`
- `make_prompt_cache`

Backward-compatible SMAQ aliases are still exported:

- `SMAQConfig`
- `enable_smaq`
- `disable_smaq`
- `smaq_enabled`
- `make_smaq_prompt_cache`

## License

MIT

"""SMAQ-MLX Benchmark — Speed + Quality evaluation.

Benchmarks SMAQ KV cache compression against fp16 baseline on Apple Silicon.
Follows the same pattern as turboquant-mlx benchmark.py.

Usage:
    python benchmark.py                          # Default benchmark
    python benchmark.py --model mlx-community/Llama-3.2-3B-Instruct-4bit  # Custom model
    python benchmark.py --seq-len 4096           # Longer context
"""

import argparse
import time

import mlx.core as mx
import mlx_lm

from smaq.kv_cache import SMAQKVCache
from smaq.quantizer import SMAQQuantizer


EVAL_TEXT = """
The capital of France is Paris. The Eiffel Tower is located in Paris.
Python is a popular programming language for machine learning.
The quick brown fox jumps over the lazy dog.
Machine learning models require careful evaluation and validation.
"""


def compute_perplexity(model, tokenizer, text: str, cache=None) -> float:
    """Compute perplexity of the model on the given text."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array(tokens[:-1])
    target_ids = mx.array(tokens[1:])

    logits = model(input_ids[None], cache=cache)
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = logits[0].astype(mx.float32)
    loss = mx.mean(mx.losses.cross_entropy(logits, target_ids))
    perplexity = mx.exp(loss).item()
    return perplexity


def benchmark_throughput(model, tokenizer, prompt: str, max_tokens: int = 256, cache=None) -> float:
    """Measure tokens/second during generation."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)

    mx.eval(model(input_ids[None], cache=cache))

    start_time = time.time()
    generated = 0

    for _ in range(max_tokens):
        output = model(input_ids[None], cache=cache)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        next_token = mx.argmax(logits[0, -1])
        input_ids = mx.concatenate([input_ids, next_token[None]])
        generated += 1

        if next_token.item() == tokenizer.eos_token_id:
            break

    elapsed = time.time() - start_time
    return generated / elapsed if elapsed > 0 else 0


def get_cache_size(cache) -> int:
    """Estimate cache size in bytes."""
    if hasattr(cache, "memory_bytes"):
        return cache.memory_bytes()["total"]
    return 0


def run_benchmark(
    model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    seq_len: int = 1024,
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
):
    """Run the full benchmark suite."""
    print(f"Loading model: {model_id}")
    model, tokenizer = mlx_lm.load(model_id)

    head_dim = model.layers[0].self_attn.head_dim
    n_layers = len(model.layers)

    print(f"Model config: head_dim={head_dim}, layers={n_layers}")

    # --- FP16 Baseline ---
    print("\n" + "=" * 60)
    print("FP16 Baseline")
    print("=" * 60)

    fp16_cache = [mlx_lm.models.base.KVCache() for _ in range(n_layers)]
    ppl_fp16 = compute_perplexity(model, tokenizer, EVAL_TEXT, cache=fp16_cache)
    print(f"Perplexity: {ppl_fp16:.2f}")

    tok_s_fp16 = benchmark_throughput(model, tokenizer, "Hello, I am", cache=fp16_cache)
    print(f"Throughput: {tok_s_fp16:.1f} tok/s")

    # --- SMAQ ---
    print("\n" + "=" * 60)
    print(f"SMAQ (key_bits={key_bits}, value_bits={value_bits})")
    print("=" * 60)

    # Calibrate with random data (in practice, use calibration set)
    mx.random.seed(42)
    cal_keys = mx.random.normal((256, head_dim))
    cal_queries = mx.random.normal((256, head_dim))

    smaq_caches = []
    for layer_idx in range(n_layers):
        cache = SMAQKVCache(
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            buffer_size=buffer_size,
            layer_idx=layer_idx,
        )
        cache.key_quantizer.fit(cal_keys, cal_queries)
        smaq_caches.append(cache)

    # Note: perplexity with SMAQ requires proper integration with mlx-lm
    # This is a placeholder for the actual measurement
    print(f"Perplexity: (requires full SDPA integration)")
    print(f"Cache bits per dim: {key_bits / head_dim * head_dim:.1f}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"FP16 PPL: {ppl_fp16:.2f}")
    print(f"FP16 Throughput: {tok_s_fp16:.1f} tok/s")
    print(f"SMAQ bits/key: {key_bits}")
    print(f"SMAQ bits/value: {value_bits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMAQ-MLX Benchmark")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--key-bits", type=int, default=3)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=128)
    args = parser.parse_args()

    run_benchmark(
        model_id=args.model,
        seq_len=args.seq_len,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
    )

"""SMAQ-MLX Demo — Interactive text generation with SMAQ KV cache.

Usage:
    python run_llm.py
    python run_llm.py --model mlx-community/Llama-3.2-1B-Instruct-4bit
    python run_llm.py --key-bits 4 --value-bits 4
"""

import argparse
import sys

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from smaq.kv_cache import SMAQKVCache


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.7,
    key_bits: int = 4,
    value_bits: int = 4,
    top_k: int = 40,
    top_p: int = 0.9,
):
    """Generate text using SMAQ KV cache compression."""
    n_layers = len(model.layers)
    head_dim = model.layers[0].self_attn.head_dim
    n_kv_heads = model.layers[0].self_attn.n_kv_heads

    print(f"Model config: head_dim={head_dim}, layers={n_layers}, n_kv_heads={n_kv_heads}")

    # Initialize SMAQ caches with identity metric (no shaping for baseline)
    Sigma_q = mx.eye(head_dim)
    caches = [
        SMAQKVCache(
            head_dim=head_dim,
            Sigma_q=Sigma_q,
            key_bits=key_bits,
            value_bits=value_bits,
            layer_idx=i,
        )
        for i in range(n_layers)
    ]

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)

    print(f"Prompt: {prompt}")
    print(f"Generating (max_tokens={max_tokens}, temp={temp})...")
    print("-" * 40)

    sampler = make_sampler(temp=temp, top_k=top_k, top_p=top_p)

    generated = []
    for token, _ in generate_step(input_ids, model, max_tokens=max_tokens, sampler=sampler, prompt_cache=caches):
        t = token if isinstance(token, int) else token.item()
        if t == tokenizer.eos_token_id:
            break
        generated.append(t)
        sys.stdout.write(tokenizer.decode([t]))
        sys.stdout.flush()

    print()
    print("-" * 40)

    # Memory stats
    total_bytes = sum(c.memory_bytes()["total"] for c in caches)
    fp16_bytes = sum(c.nbytes_equivalent_fp16 for c in caches if hasattr(c, 'nbytes_equivalent_fp16'))
    print(f"SMAQ cache memory: {total_bytes / 1024 / 1024:.1f} MB")
    if fp16_bytes > 0:
        print(f"FP16 equivalent: {fp16_bytes / 1024 / 1024:.1f} MB")
        print(f"Compression ratio: {fp16_bytes / max(total_bytes, 1):.1f}x")


def main():
    parser = argparse.ArgumentParser(description="SMAQ-MLX Demo")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_lm.load(args.model)

    generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
    )


if __name__ == "__main__":
    main()

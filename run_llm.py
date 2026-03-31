"""SMAQ-MLX Demo — Interactive text generation with SMAQ KV cache.

Usage:
    python run_llm.py
    python run_llm.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
    python run_llm.py --key-bits 3 --value-bits 2
"""

import argparse
import sys

import mlx.core as mx
import mlx_lm

from smaq.kv_cache import SMAQKVCache


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.7,
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
):
    """Generate text using SMAQ KV cache compression."""
    n_layers = len(model.layers)
    head_dim = model.layers[0].self_attn.head_dim

    # Initialize SMAQ caches
    mx.random.seed(42)
    cal_keys = mx.random.normal((256, head_dim))
    cal_queries = mx.random.normal((256, head_dim))

    caches = []
    for layer_idx in range(n_layers):
        cache = SMAQKVCache(
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            buffer_size=buffer_size,
            layer_idx=layer_idx,
        )
        # In practice, fit on calibration data from the domain
        cache.key_quantizer.fit(cal_keys, cal_queries)
        caches.append(cache)

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)

    print(f"Prompt: {prompt}")
    print(f"Generating (max_tokens={max_tokens}, temp={temp})...")
    print("-" * 40)

    # Prefill
    logits = model(input_ids[None], cache=caches)
    if isinstance(logits, tuple):
        logits = logits[0]

    generated = []
    for i in range(max_tokens):
        next_logits = logits[0, -1] / temp
        probs = mx.softmax(next_logits)
        next_token = mx.random.categorical(probs)

        token = next_token.item()
        if token == tokenizer.eos_token_id:
            break

        generated.append(token)
        input_ids = mx.concatenate([input_ids, next_token[None]])

        # Forward pass for next token
        logits = model(input_ids[None, -1:], cache=caches)
        if isinstance(logits, tuple):
            logits = logits[0]

        if (i + 1) % 32 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

    print()
    output = tokenizer.decode(generated)
    print(f"\nGenerated:\n{output}")

    # Memory stats
    total_bytes = sum(c.memory_bytes()["total"] for c in caches)
    print(f"\nCache memory: {total_bytes / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="SMAQ-MLX Demo")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--key-bits", type=int, default=3)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=128)
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
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()

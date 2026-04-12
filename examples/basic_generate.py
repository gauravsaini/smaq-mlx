"""Minimal explicit SMAQ-MLX integration example."""

import mlx_lm

from smaq_mlx import SMAQConfig, enable_smaq


def main():
    enable_smaq(SMAQConfig(key_bits=4, value_bits=4, mode="hybrid"))
    model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-4B-MLX-4bit")
    text = mlx_lm.generate(
        model,
        tokenizer,
        prompt="Explain KV cache compression in two short sentences.",
        max_tokens=48,
    )
    print(text)


if __name__ == "__main__":
    main()

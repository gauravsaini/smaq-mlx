"""Validate memory leak fix in SMAQKVCache.

Tests:
1. No shadow duplication — memory should be compressed + buffer only
2. trim() properly reduces stored data
3. Functional correctness — update_and_fetch returns valid KV arrays
"""

import mlx.core as mx
from smaq_mlx.kv_cache import SMAQKVCache

HEAD_DIM = 128
N_KV_HEADS = 4
BUFFER_SIZE = 8


def test_no_shadow_leak():
    """Verify that memory is only compressed + buffer, no FP16 shadow."""
    print("=" * 70)
    print("TEST 1: No shadow duplication after fix")
    print("=" * 70)

    cache = SMAQKVCache(
        head_dim=HEAD_DIM, key_bits=3, value_bits=2,
        buffer_size=BUFFER_SIZE, layer_idx=0,
    )

    # Prefill 20 tokens
    prefill_k = mx.random.normal((1, N_KV_HEADS, 20, HEAD_DIM))
    prefill_v = mx.random.normal((1, N_KV_HEADS, 20, HEAD_DIM))
    mx.eval(prefill_k, prefill_v)
    keys, values = cache.update_and_fetch(prefill_k, prefill_v)
    mx.eval(keys, values)
    print(f"After prefill(20): offset={cache.offset}, returned keys.shape={keys.shape}")

    # Decode 30 tokens
    for i in range(30):
        dk = mx.random.normal((1, N_KV_HEADS, 1, HEAD_DIM))
        dv = mx.random.normal((1, N_KV_HEADS, 1, HEAD_DIM))
        mx.eval(dk, dv)
        keys, values = cache.update_and_fetch(dk, dv)
        mx.eval(keys, values)

    mem = cache.memory_bytes()
    fp16_equiv = cache.nbytes_equivalent_fp16
    actual_total = mem["total"]
    print(f"\nFinal offset: {cache.offset}")
    print(f"Returned keys shape: {keys.shape}")
    print(f"Compressed keys:  {mem['compressed_keys']/1024:.1f} KB")
    print(f"Compressed values: {mem['compressed_values']/1024:.1f} KB")
    print(f"Exact buffer:     {mem['exact_buffer']/1024:.1f} KB")
    print(f"Shadow:           {mem['shadow']/1024:.1f} KB  ← should be 0")
    print(f"ACTUAL total:     {actual_total/1024:.1f} KB")
    print(f"FP16 equivalent:  {fp16_equiv/1024:.1f} KB")
    print(f"Compression ratio: {fp16_equiv / max(actual_total, 1):.2f}x")
    assert mem["shadow"] == 0, "FAIL: Shadow should be 0"
    assert actual_total < fp16_equiv, f"FAIL: {actual_total} should be < {fp16_equiv}"
    print("✅ PASS: No shadow, actual compression achieved!")


def test_trim_works():
    """Verify that trim() actually frees memory."""
    print("\n" + "=" * 70)
    print("TEST 2: trim() properly reduces data")
    print("=" * 70)

    cache = SMAQKVCache(
        head_dim=HEAD_DIM, key_bits=3, value_bits=2,
        buffer_size=BUFFER_SIZE, layer_idx=0,
    )

    k = mx.random.normal((1, N_KV_HEADS, 30, HEAD_DIM))
    v = mx.random.normal((1, N_KV_HEADS, 30, HEAD_DIM))
    mx.eval(k, v)
    cache.update_and_fetch(k, v)

    before_offset = cache.offset
    before_bytes = cache.memory_bytes()["total"]
    print(f"Before trim: offset={before_offset}, total_bytes={before_bytes}")

    trimmed = cache.trim(15)
    after_offset = cache.offset
    after_bytes = cache.memory_bytes()["total"]
    print(f"After trim(15): offset={after_offset}, total_bytes={after_bytes}")
    print(f"Trimmed: {trimmed}")
    print(f"Offset reduced: {before_offset} → {after_offset}")
    print(f"Bytes reduced: {before_bytes} → {after_bytes}")

    assert after_offset == 15, f"FAIL: offset should be 15, got {after_offset}"
    assert after_bytes < before_bytes, f"FAIL: bytes should decrease, {after_bytes} >= {before_bytes}"
    print("✅ PASS: trim() properly frees memory!")


def test_functional_correctness():
    """Verify update_and_fetch returns correct-shape KV arrays."""
    print("\n" + "=" * 70)
    print("TEST 3: Functional correctness")
    print("=" * 70)

    cache = SMAQKVCache(
        head_dim=HEAD_DIM, key_bits=4, value_bits=4,
        buffer_size=BUFFER_SIZE, layer_idx=0,
    )

    # Prefill
    k1 = mx.random.normal((1, N_KV_HEADS, 10, HEAD_DIM))
    v1 = mx.random.normal((1, N_KV_HEADS, 10, HEAD_DIM))
    mx.eval(k1, v1)
    keys, values = cache.update_and_fetch(k1, v1)
    mx.eval(keys, values)
    assert keys.shape == (1, N_KV_HEADS, 10, HEAD_DIM), f"FAIL: prefill keys shape = {keys.shape}"
    print(f"After prefill(10): keys.shape={keys.shape} ✓")

    # Decode
    for step in range(5):
        dk = mx.random.normal((1, N_KV_HEADS, 1, HEAD_DIM))
        dv = mx.random.normal((1, N_KV_HEADS, 1, HEAD_DIM))
        mx.eval(dk, dv)
        keys, values = cache.update_and_fetch(dk, dv)
        mx.eval(keys, values)
        expected_len = 11 + step
        assert keys.shape[-2] == expected_len, f"FAIL step {step}: keys seq_len={keys.shape[-2]}, expected {expected_len}"

    print(f"After decode(5): keys.shape={keys.shape} ✓")
    print(f"Cache offset: {cache.offset} ✓")
    
    # State roundtrip
    state = cache.state
    assert len(state) == 2, f"FAIL: state should have 2 items, got {len(state)}"
    assert state[0].shape[-2] == 15, f"FAIL: state keys seq={state[0].shape[-2]}, expected 15"
    print(f"State keys shape: {state[0].shape} ✓")
    
    # empty()
    assert not cache.empty(), "FAIL: cache should not be empty"
    print("empty() = False ✓")
    
    print("✅ PASS: All functional checks passed!")


if __name__ == "__main__":
    test_no_shadow_leak()
    test_trim_works()
    test_functional_correctness()
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED — Memory leak fixed!")
    print("=" * 70)

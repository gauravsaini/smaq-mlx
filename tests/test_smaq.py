"""Unit tests for SMAQ-MLX components."""

import unittest

import mlx.core as mx
import numpy as np

from smaq.ssf import ssf_log, build_smaq_metric
from smaq.block_vq import SMAQBlockVQ, BlockVQQuantized
from smaq.quantizer import SMAQQuantizer, SMAQQuantized
from smaq.kv_cache import SMAQKVCache, quantize_values, dequantize_values
from smaq.capture import RingBuffer, KVCaptureEngine
from smaq.store import CompressedKVStore


class TestSSF(unittest.TestCase):
    """Tests for spectral shaping functions."""

    def test_ssf_log_positive_eigenvalues(self):
        eigvals = mx.array([0.1, 0.5, 1.0, 2.0])
        result = ssf_log(eigvals, c=5.0)
        self.assertEqual(result.shape, eigvals.shape)
        self.assertTrue(mx.all(result > 0).item())

    def test_ssf_log_negative_eigenvalues_clamped(self):
        eigvals = mx.array([-1.0, 0.0, 0.5])
        result = ssf_log(eigvals, c=5.0)
        self.assertEqual(result.shape, eigvals.shape)

    def test_build_smaq_metric_symmetric(self):
        d = 8
        mx.random.seed(42)
        A = mx.random.normal((d, d))
        Sigma_q = A @ A.T / d

        E, E_inv = build_smaq_metric(Sigma_q)
        self.assertEqual(E.shape, (d, d))
        self.assertEqual(E_inv.shape, (d, d))

        # E @ E_inv should be approximately identity
        product = E @ E_inv
        identity = mx.eye(d)
        error = mx.max(mx.abs(product - identity)).item()
        self.assertLess(error, 0.01)


class TestBlockVQ(unittest.TestCase):
    """Tests for Block VQ quantizer."""

    def test_block_vq_init(self):
        vq = SMAQBlockVQ(head_dim=64, block_dim=8, n_centroids=256)
        self.assertEqual(vq.n_blocks, 8)
        self.assertEqual(vq.block_dim, 8)
        self.assertEqual(vq.n_centroids, 256)

    def test_block_vq_invalid_dim(self):
        with self.assertRaises(ValueError):
            SMAQBlockVQ(head_dim=63, block_dim=8)

    def test_block_vq_fit_and_quantize(self):
        head_dim = 64
        vq = SMAQBlockVQ(head_dim=head_dim, block_dim=8, n_centroids=256)

        mx.random.seed(42)
        cal_keys = mx.random.normal((128, head_dim))
        cal_queries = mx.random.normal((128, head_dim))

        vq.fit(cal_keys, cal_queries)
        self.assertTrue(vq._fitted)

        # Quantize
        q = vq.quantize(cal_keys[:10])
        self.assertIsInstance(q, BlockVQQuantized)
        self.assertEqual(q.indices.shape, (10, vq.n_blocks))

    def test_block_vq_dequantize_shape(self):
        head_dim = 64
        vq = SMAQBlockVQ(head_dim=head_dim, block_dim=8, n_centroids=256)

        mx.random.seed(42)
        cal_keys = mx.random.normal((128, head_dim))
        cal_queries = mx.random.normal((128, head_dim))
        vq.fit(cal_keys, cal_queries)

        q = vq.quantize(cal_keys[:5])
        k_hat = vq.dequantize(q)
        self.assertEqual(k_hat.shape, (5, head_dim))

    def test_block_vq_bits_per_dim(self):
        vq = SMAQBlockVQ(head_dim=64, block_dim=8, n_centroids=256)
        self.assertAlmostEqual(vq.bits_per_dim, 1.0)

    def test_block_vq_logit_mse(self):
        head_dim = 64
        vq = SMAQBlockVQ(head_dim=head_dim, block_dim=8, n_centroids=256)

        mx.random.seed(42)
        cal_keys = mx.random.normal((128, head_dim))
        cal_queries = mx.random.normal((128, head_dim))
        vq.fit(cal_keys, cal_queries)

        test_keys = mx.random.normal((32, head_dim))
        test_queries = mx.random.normal((32, head_dim))
        mse = vq.logit_mse(test_queries, test_keys)
        self.assertGreaterEqual(mse, 0)


class TestScalarQuantizer(unittest.TestCase):
    """Tests for SMAQ scalar quantizer."""

    def test_quantizer_init(self):
        q = SMAQQuantizer(dim=64, bits=3)
        self.assertEqual(q.dim, 64)
        self.assertEqual(q.bits, 3)

    def test_quantizer_quantize_dequantize(self):
        dim = 64
        q = SMAQQuantizer(dim=dim, bits=3)

        mx.random.seed(42)
        keys = mx.random.normal((10, dim))
        quantized = q.quantize(keys)

        self.assertIsInstance(quantized, SMAQQuantized)
        self.assertEqual(quantized.bits, 3)

        reconstructed = q.dequantize(quantized)
        self.assertEqual(reconstructed.shape, keys.shape)

    def test_quantizer_attention_score(self):
        dim = 64
        q = SMAQQuantizer(dim=dim, bits=3)

        mx.random.seed(42)
        keys = mx.random.normal((10, dim))
        queries = mx.random.normal((5, dim))

        quantized = q.quantize(keys)
        scores = q.attention_score(queries, quantized, scale=1.0 / 8.0)
        self.assertEqual(scores.shape, (5, 10))

    def test_quantizer_rotate_query(self):
        dim = 64
        q = SMAQQuantizer(dim=dim, bits=3)

        mx.random.seed(42)
        query = mx.random.normal((5, dim))
        rotated = q.rotate_query(query)
        self.assertEqual(rotated.shape, query.shape)


class TestKVCache(unittest.TestCase):
    """Tests for SMAQ KV cache."""

    def test_kv_cache_init(self):
        cache = SMAQKVCache(head_dim=64, key_bits=3, value_bits=2)
        self.assertEqual(cache.head_dim, 64)
        self.assertEqual(cache.key_bits, 3)
        self.assertEqual(cache.value_bits, 2)

    def test_kv_cache_prefill_small(self):
        cache = SMAQKVCache(head_dim=64, buffer_size=128)

        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 64, 64))
        values = mx.random.normal((1, 4, 64, 64))

        cache.prefill(keys, values)
        self.assertEqual(cache.seq_len, 64)
        self.assertIsNotNone(cache.key_buffer)

    def test_kv_cache_prefill_large(self):
        cache = SMAQKVCache(head_dim=64, buffer_size=32)

        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 128, 64))
        values = mx.random.normal((1, 4, 128, 64))

        cache.prefill(keys, values)
        self.assertEqual(cache.seq_len, 128)
        self.assertIsNotNone(cache.key_quantized)
        self.assertEqual(cache.key_buffer.shape[-2], 32)

    def test_kv_cache_append(self):
        cache = SMAQKVCache(head_dim=64, buffer_size=32)

        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 16, 64))
        values = mx.random.normal((1, 4, 16, 64))

        cache.prefill(keys, values)
        self.assertEqual(cache.seq_len, 16)

        # Append one token
        new_key = mx.random.normal((1, 4, 1, 64))
        new_value = mx.random.normal((1, 4, 1, 64))
        cache.append(new_key, new_value)
        self.assertEqual(cache.seq_len, 17)

    def test_kv_cache_memory_bytes(self):
        cache = SMAQKVCache(head_dim=64, buffer_size=32)

        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 128, 64))
        values = mx.random.normal((1, 4, 128, 64))

        cache.prefill(keys, values)
        mem = cache.memory_bytes()
        self.assertIn("total", mem)
        self.assertGreater(mem["total"], 0)

    def test_quantize_values_2bit(self):
        mx.random.seed(42)
        v = mx.random.normal((1, 4, 64, 64))
        data, scales, zeros, bits = quantize_values(v, bits=2, group_size=32)
        self.assertEqual(bits, 2)

        reconstructed = dequantize_values(data, scales, zeros, bits, group_size=32)
        self.assertEqual(reconstructed.shape, v.shape)

    def test_quantize_values_4bit(self):
        mx.random.seed(42)
        v = mx.random.normal((1, 4, 64, 64))
        data, scales, zeros, bits = quantize_values(v, bits=4, group_size=32)
        self.assertEqual(bits, 4)

        reconstructed = dequantize_values(data, scales, zeros, bits, group_size=32)
        self.assertEqual(reconstructed.shape, v.shape)


class TestRingBuffer(unittest.TestCase):
    """Tests for RingBuffer."""

    def test_ring_buffer_write_drain(self):
        buf = RingBuffer(capacity=16, num_kv_heads=4, head_dim=64)

        mx.random.seed(42)
        key = mx.random.normal((8, 4, 64))
        value = mx.random.normal((8, 4, 64))

        overflow = buf.write(key, value, 8)
        self.assertIsNone(overflow)
        self.assertEqual(buf.size, 8)

        drained = buf.drain()
        self.assertIsNotNone(drained)
        self.assertEqual(drained[0].shape, (8, 4, 64))
        self.assertEqual(buf.size, 0)

    def test_ring_buffer_overflow(self):
        buf = RingBuffer(capacity=8, num_kv_heads=4, head_dim=64)

        mx.random.seed(42)
        key = mx.random.normal((16, 4, 64))
        value = mx.random.normal((16, 4, 64))

        overflow = buf.write(key, value, 16)
        self.assertIsNotNone(overflow)
        self.assertEqual(overflow[0].shape[0], 8)
        self.assertEqual(buf.size, 8)


class TestCompressedKVStore(unittest.TestCase):
    """Tests for CompressedKVStore."""

    def test_store_append(self):
        store = CompressedKVStore(head_dim=64, num_kv_heads=4)

        mx.random.seed(42)
        key = mx.random.normal((32, 4, 64))
        value = mx.random.normal((32, 4, 64))

        store.append_chunk(key, value)
        self.assertEqual(store.num_tokens, 32)
        self.assertEqual(store.num_chunks, 1)

    def test_store_get_flat_cache(self):
        store = CompressedKVStore(head_dim=64, num_kv_heads=4)

        mx.random.seed(42)
        key = mx.random.normal((32, 4, 64))
        value = mx.random.normal((32, 4, 64))

        store.append_chunk(key, value)
        flat = store.get_flat_cache()
        self.assertIsNotNone(flat)
        self.assertEqual(flat.num_tokens, 32)

    def test_store_reset(self):
        store = CompressedKVStore(head_dim=64, num_kv_heads=4)

        mx.random.seed(42)
        key = mx.random.normal((32, 4, 64))
        value = mx.random.normal((32, 4, 64))

        store.append_chunk(key, value)
        store.reset()
        self.assertEqual(store.num_tokens, 0)
        self.assertIsNone(store.get_flat_cache())


if __name__ == "__main__":
    unittest.main()

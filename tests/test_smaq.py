"""Unit tests for SMAQ-MLX components."""

import unittest

import mlx.core as mx
import numpy as np

from smaq_mlx.ssf import ssf_log, build_smaq_metric
from smaq_mlx.backends import PlaceholderBackend, RuntimeBackend, available_backends, backend_matrix, get_backend, validate_backend
from smaq_mlx.block_vq import SMAQBlockVQ, BlockVQQuantized
from smaq_mlx.core import CacheCapabilities
from smaq_mlx.layout import GemmaLayoutAdapter, QwenLayoutAdapter
from smaq_mlx.attention_smaq import smaq_sdpa
from smaq_mlx.api import MLXRuntimeConfig, SMAQConfig
from smaq_mlx.quantizer import SMAQQuantizer, SMAQQuantized
from smaq_mlx.kv_cache import SMAQKVCache, quantize_values, dequantize_values
from smaq_mlx.rotor_cache import RotorQuantKVCache
from smaq_mlx.rotorquant import RotorQuantMSE
from smaq_mlx.capture import RingBuffer, KVCaptureEngine
from smaq_mlx.folded_cache import FoldedTurboSMAQKVCache
from smaq_mlx.folded_turboquant import FoldedTurboQuantizer
from smaq_mlx.store import CompressedKVStore
from smaq_mlx.patch import clear_configuration, configure, current_configuration


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
        self.assertEqual(cache.offset, 0)

    def test_kv_cache_update_and_fetch(self):
        cache = SMAQKVCache(head_dim=64, key_bits=4, value_bits=4)

        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 16, 64))
        values = mx.random.normal((1, 4, 16, 64))

        returned_k, returned_v = cache.update_and_fetch(keys, values)
        self.assertEqual(cache.offset, 16)
        self.assertEqual(returned_k.shape, keys.shape)
        self.assertEqual(returned_v.shape, values.shape)

    def test_kv_cache_multiple_updates(self):
        cache = SMAQKVCache(head_dim=64, key_bits=4, value_bits=4)

        mx.random.seed(42)
        keys1 = mx.random.normal((1, 4, 16, 64))
        values1 = mx.random.normal((1, 4, 16, 64))
        cache.update_and_fetch(keys1, values1)
        self.assertEqual(cache.offset, 16)

        keys2 = mx.random.normal((1, 4, 4, 64))
        values2 = mx.random.normal((1, 4, 4, 64))
        cache.update_and_fetch(keys2, values2)
        self.assertEqual(cache.offset, 20)

        returned_k, returned_v = cache.update_and_fetch(keys2, values2)
        self.assertEqual(cache.offset, 24)
        self.assertEqual(returned_k.shape[-2], 24)

    def test_kv_cache_memory_bytes(self):
        cache = SMAQKVCache(head_dim=64, key_bits=4, value_bits=4)

        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 64, 64))
        values = mx.random.normal((1, 4, 64, 64))

        cache.update_and_fetch(keys, values)
        mem = cache.memory_bytes()
        self.assertIn("total", mem)
        self.assertGreater(mem["total"], 0)

    def test_kv_cache_empty_and_trim(self):
        cache = SMAQKVCache(head_dim=64, key_bits=4, value_bits=4)
        self.assertTrue(cache.empty())

        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 16, 64))
        values = mx.random.normal((1, 4, 16, 64))
        cache.update_and_fetch(keys, values)
        self.assertFalse(cache.empty())

        trimmed = cache.trim(4)
        self.assertEqual(trimmed, 4)
        self.assertEqual(cache.offset, 12)

    def test_quantize_values_2bit(self):
        mx.random.seed(42)
        v = mx.random.normal((1, 4, 64, 64))
        data, scales, zeros, bits = quantize_values(v, bits=2, group_size=32)
        self.assertEqual(bits, 2)

        reconstructed = dequantize_values(data, scales, zeros, bits, group_size=32)
        self.assertEqual(reconstructed.shape, v.shape)


class TestRotorQuant(unittest.TestCase):
    """Tests for RotorQuant-inspired backend pieces."""

    def test_rotor_quant_roundtrip_shape(self):
        quantizer = RotorQuantMSE(d=64, bit_width=3, seed=42)
        vectors = np.random.randn(8, 64).astype(np.float32)
        quantized = quantizer.quantize(vectors)
        reconstructed = quantizer.dequantize(quantized.indices, quantized.norms)
        self.assertEqual(reconstructed.shape, vectors.shape)

    def test_rotor_cache_update_and_fetch(self):
        cache = RotorQuantKVCache(bits=3, head_dim=64)
        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 8, 64))
        values = mx.random.normal((1, 4, 8, 64))
        returned_k, returned_v = cache.update_and_fetch(keys, values)
        self.assertEqual(returned_k.shape, (1, 4, 8, 64))
        self.assertEqual(returned_v.shape, (1, 4, 8, 64))
        self.assertGreater(cache.nbytes, 0)
        self.assertGreater(cache.uncompressed_nbytes, 0)
        self.assertGreater(cache.compression_ratio, 1.0)


class TestFoldedTurboSMAQ(unittest.TestCase):
    """Tests for the folded Turbo+SMAQ single-cache path."""

    def test_folded_quantizer_roundtrip_shape(self):
        quantizer = FoldedTurboQuantizer(dim=64, bits=3, seed=42)
        mx.random.seed(42)
        vectors = mx.random.normal((8, 64))
        packed, norms = quantizer.quantize(vectors)
        reconstructed = quantizer.dequantize(packed, norms)
        self.assertEqual(reconstructed.shape, vectors.shape)

    def test_folded_cache_update_fit_and_fetch(self):
        cache = FoldedTurboSMAQKVCache(bits=3, head_dim=64)
        mx.random.seed(42)
        keys = mx.random.normal((1, 4, 8, 64))
        values = mx.random.normal((1, 4, 8, 64))
        cache.update_and_fetch(keys, values)
        queries = mx.random.normal((1, 4, 8, 64))
        cache.fit_metric_from_queries(queries)
        returned_k, returned_v = cache.materialize(dtype=mx.float32)
        self.assertTrue(cache.metric_fitted)
        self.assertEqual(returned_k.shape, (1, 4, 8, 64))
        self.assertEqual(returned_v.shape, (1, 4, 8, 64))
        self.assertGreater(cache.compression_ratio, 1.0)


class TestPublicApi(unittest.TestCase):
    """Tests for user-facing public API helpers."""

    def tearDown(self):
        clear_configuration()

    def test_runtime_config_defaults(self):
        config = MLXRuntimeConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.backend, "smaq")
        self.assertEqual(config.key_bits, 4)
        self.assertEqual(config.value_bits, 4)

    def test_patch_runtime_configuration(self):
        configure(MLXRuntimeConfig(backend="turboquant", key_bits=3, value_bits=2, strict_benchmark=True))
        current = current_configuration()
        self.assertEqual(current["backend"], "turboquant")
        self.assertEqual(current["key_bits"], 3)
        self.assertEqual(current["value_bits"], 2)
        self.assertTrue(current["strict_benchmark"])

    def test_available_backends(self):
        names = available_backends()
        self.assertIn("folded_turbo_smaq", names)
        self.assertIn("polarquant", names)
        self.assertIn("rotorquant", names)
        self.assertIn("smaq", names)
        self.assertIn("stacked_turbo_smaq", names)
        self.assertIn("turboquant", names)

    def test_get_backend(self):
        self.assertEqual(get_backend("folded_turbo_smaq").name, "folded_turbo_smaq")
        self.assertEqual(get_backend("polarquant").name, "polarquant")
        self.assertEqual(get_backend("rotorquant").name, "rotorquant")
        self.assertEqual(get_backend("smaq").name, "smaq")
        self.assertEqual(get_backend("stacked_turbo_smaq").name, "stacked_turbo_smaq")
        self.assertEqual(get_backend("turboquant").name, "turboquant")

    def test_backend_matrix(self):
        entries = backend_matrix()
        by_name = {entry["name"]: entry for entry in entries}
        self.assertEqual(by_name["folded_turbo_smaq"]["status"], "research")
        self.assertEqual(by_name["polarquant"]["status"], "experimental")
        self.assertEqual(by_name["rotorquant"]["status"], "experimental")
        self.assertEqual(by_name["polarquant"]["requires_package"], "mlx-turboquant")
        self.assertEqual(by_name["smaq"]["status"], "experimental")
        self.assertEqual(by_name["stacked_turbo_smaq"]["status"], "research")
        self.assertEqual(by_name["turboquant"]["requires_package"], "turboquant-mlx")

    def test_validate_backend_rejects_wrong_type(self):
        with self.assertRaises(TypeError):
            validate_backend(object())

    def test_patch_runtime_configuration_supports_polarquant(self):
        configure(
            MLXRuntimeConfig(
                backend="polarquant",
                polarquant_bits=3,
                polarquant_key_seed=11,
                polarquant_value_seed=12,
            )
        )
        current = current_configuration()
        self.assertEqual(current["backend"], "polarquant")
        self.assertEqual(current["polarquant_bits"], 3)
        self.assertEqual(current["polarquant_key_seed"], 11)
        self.assertEqual(current["polarquant_value_seed"], 12)

    def test_patch_runtime_configuration_supports_rotorquant(self):
        configure(
            MLXRuntimeConfig(
                backend="rotorquant",
                rotorquant_bits=3,
                rotorquant_key_seed=21,
                rotorquant_value_seed=22,
            )
        )
        current = current_configuration()
        self.assertEqual(current["backend"], "rotorquant")
        self.assertEqual(current["rotorquant_bits"], 3)
        self.assertEqual(current["rotorquant_key_seed"], 21)
        self.assertEqual(current["rotorquant_value_seed"], 22)

    def test_patch_runtime_configuration_supports_folded_backend(self):
        configure(
            MLXRuntimeConfig(
                backend="folded_turbo_smaq",
                folded_turbo_bits=3,
                folded_turbo_key_seed=31,
                folded_turbo_value_seed=32,
                folded_smaq_c=7.0,
            )
        )
        current = current_configuration()
        self.assertEqual(current["backend"], "folded_turbo_smaq")
        self.assertEqual(current["folded_turbo_bits"], 3)
        self.assertEqual(current["folded_turbo_key_seed"], 31)
        self.assertEqual(current["folded_turbo_value_seed"], 32)
        self.assertEqual(current["folded_smaq_c"], 7.0)

    def test_validate_backend_accepts_custom_backend(self):
        class DummyBackend(RuntimeBackend):
            def __init__(self):
                super().__init__(name="dummy", description="dummy")

            def supports_cache(self, cache):
                return False

            def make_prompt_cache(self, model, *, cache_module, config):
                return []

            def sdpa(self, queries, keys, values, cache, *, scale, mask, sinks=None, config=None, original_sdpa=None, **kwargs):
                return None

        validate_backend(DummyBackend())

    def test_placeholder_backend_raises(self):
        backend = PlaceholderBackend(name="planned", description="planned backend")
        with self.assertRaises(RuntimeError):
            backend.make_prompt_cache(None, cache_module=None, config={})

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


class TestGenericMlxCore(unittest.TestCase):
    """Tests for generic cache metadata and compressed SDPA path."""

    def test_capability_report(self):
        cache = SMAQKVCache(head_dim=64, key_bits=3, value_bits=2, mode="hybrid")
        report = cache.capability_report()
        self.assertTrue(report["compressed_history"])
        self.assertFalse(report["compressed_history_shadow_only"])
        self.assertEqual(report["strategy_name"], "smaq_mlx_cache")

    def test_layout_adapters(self):
        qwen = QwenLayoutAdapter()
        gemma = GemmaLayoutAdapter()
        keys_q = mx.random.normal((1, 4, 8, 8))
        values_q = mx.random.normal((1, 4, 8, 8))
        keys_g = mx.random.normal((1, 4, 8, 16))
        values_g = mx.random.normal((1, 4, 8, 16))

        qwen_info = qwen.normalize_kv(keys_q, values_q, expected_head_dim=8)
        gemma_info = gemma.normalize_kv(keys_g, values_g, expected_head_dim=8)

        self.assertFalse(qwen_info.unified_kv)
        self.assertTrue(gemma_info.unified_kv)
        self.assertEqual(gemma_info.effective_head_dim, 16)

    def test_cache_layout_configures_on_first_update(self):
        cache = SMAQKVCache(head_dim=8, key_bits=3, value_bits=2, layout_adapter=GemmaLayoutAdapter())
        keys = mx.random.normal((1, 4, 8, 16))
        values = mx.random.normal((1, 4, 8, 16))
        cache.update_and_fetch(keys, values)

        self.assertEqual(cache.head_dim, 16)
        self.assertIsNotNone(cache.key_quantizer)
        self.assertEqual(cache.capabilities.decode_uses_compressed_keys, True)

    def test_true_compressed_sdpa_uses_cache_methods(self):
        class DummyCache:
            def __init__(self):
                self.capabilities = CacheCapabilities(
                    strategy_name="dummy",
                    metric_name="metric",
                    quantization_name="quant",
                    compressed_history=True,
                    compressed_history_shadow_only=False,
                    values_compressed=True,
                    decode_uses_compressed_keys=True,
                    decode_uses_compressed_values=True,
                )
                self._scores = None
                self.key_quantized = object()

            def attention_scores(self, query, scale=None):
                self._scores = query
                return mx.ones((query.shape[0], query.shape[1], query.shape[2], 4))

            def attend(self, attn_weights):
                return mx.ones((attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], 8))

            def report(self):
                return {"mode": "compressed"}

        cache = DummyCache()
        queries = mx.random.normal((1, 2, 3, 8))
        out = smaq_sdpa(queries, cache, scale=1.0, require_true_compressed=True)
        self.assertEqual(out.shape, (1, 2, 3, 8))
        self.assertIsNotNone(cache._scores)

    def test_true_compressed_guard_raises_for_shadow_cache(self):
        class ShadowCache:
            def __init__(self):
                self.capabilities = CacheCapabilities(
                    strategy_name="shadow",
                    metric_name="metric",
                    quantization_name="quant",
                    compressed_history=False,
                    compressed_history_shadow_only=True,
                    values_compressed=False,
                    decode_uses_compressed_keys=False,
                    decode_uses_compressed_values=False,
                )

            def attention_scores(self, query, scale=None):
                return mx.ones((query.shape[0], query.shape[1], query.shape[2], 4))

            def attend(self, attn_weights):
                return mx.ones((attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], 8))

            def report(self):
                return {"mode": "shadow"}

        with self.assertRaises(RuntimeError):
            smaq_sdpa(mx.random.normal((1, 2, 3, 8)), ShadowCache(), scale=1.0, require_true_compressed=True)


if __name__ == "__main__":
    unittest.main()

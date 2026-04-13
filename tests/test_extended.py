"""Extended unit tests for SMAQ-MLX components."""

import unittest
import math
import mlx.core as mx
import numpy as np
from unittest.mock import MagicMock
import sys

# Mock turboquant_mlx if not installed
if "turboquant_mlx" not in sys.modules:
    mock_tq_root = MagicMock()
    sys.modules["turboquant_mlx"] = mock_tq_root
    
    mock_tq_cache_mod = MagicMock()
    sys.modules["turboquant_mlx.cache"] = mock_tq_cache_mod
    
    # This is what's imported in stacked_cache.py
    mock_tq_cache_class = MagicMock()
    mock_tq_cache_mod.TurboQuantKVCache = mock_tq_cache_class
    
    mock_tq_fused_mod = MagicMock()
    sys.modules["turboquant_mlx.fused_attention"] = mock_tq_fused_mod

from smaq_mlx.progressive_cache import ProgressiveSMAQCache, progressive_sdpa
from smaq_mlx.stacked_cache import TurboSMAQCascadeCache
from smaq_mlx.score import compute_hybrid_attention
from smaq_mlx.rotor_ops import (
    gp_rotor_mv, rotor_sandwich, make_random_rotor,
    embed_vectors, extract_vectors, optimal_centroids,
    nearest_centroid_indices
)
from smaq_mlx.store import CompressedKVStore
from smaq_mlx.kv_cache import SMAQKVCache

class TestExperimental(unittest.TestCase):
    """Tests for experimental cache components."""

    def setUp(self):
        self.head_dim = 64
        self.key_bits = 4
        self.value_bits = 4
        self.turbo_bits = 3

    def test_progressive_smaq_cache_init(self):
        # We need to mock TurboQuantKVCache behavior for this test if turboquant_mlx is mocked
        import turboquant_mlx.cache as tq_cache
        tq_cache.TurboQuantKVCache = MagicMock()
        
        cache = ProgressiveSMAQCache(
            head_dim=self.head_dim,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            turboquant_bits=self.turbo_bits
        )
        self.assertEqual(cache.head_dim, self.head_dim)
        self.assertEqual(cache.offset, 0)
        self.assertEqual(cache.capabilities.strategy_name, "progressive_smaq")

    def test_turbo_smaq_cascade_cache_init(self):
        import turboquant_mlx.cache as tq_cache
        tq_cache.TurboQuantKVCache = MagicMock()

        cache = TurboSMAQCascadeCache(
            head_dim=self.head_dim,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            turboquant_bits=self.turbo_bits
        )
        self.assertEqual(cache.turboquant_bits, self.turbo_bits)
        self.assertEqual(cache.offset, 0)
        self.assertEqual(cache.capabilities.strategy_name, "stacked_turbo_smaq")

    def test_progressive_sdpa(self):
        import turboquant_mlx.cache as tq_cache
        tq_cache.TurboQuantKVCache = MagicMock()
        
        cache = ProgressiveSMAQCache(
            head_dim=self.head_dim,
            coarse_k=10
        )
        
        # Test prefill path (offset < coarse_k)
        keys = mx.random.normal((1, 4, 8, self.head_dim))
        values = mx.random.normal((1, 4, 8, self.head_dim))
        cache.update_and_fetch(keys, values)
        
        queries = mx.random.normal((1, 4, 1, self.head_dim))
        out = progressive_sdpa(queries, cache)
        self.assertEqual(out.shape, (1, 4, 1, self.head_dim))

    def test_turbo_smaq_cascade_update(self):
        import turboquant_mlx.cache
        mock_tq_cache_instance = MagicMock()
        turboquant_mlx.cache.TurboQuantKVCache.return_value = mock_tq_cache_instance
        
        # Define shapes
        keys = mx.random.normal((1, 4, 8, self.head_dim))
        values = mx.random.normal((1, 4, 8, self.head_dim))
        
        # Mock TQ output (same shape as input)
        mock_tq_cache_instance.update_and_fetch.return_value = (keys, values)

        cache = TurboSMAQCascadeCache(
            head_dim=self.head_dim,
        )
        cache.turbo_cache = mock_tq_cache_instance
        
        returned_k, returned_v = cache.update_and_fetch(keys, values)
        self.assertEqual(cache.offset, 8)
        self.assertEqual(returned_k.shape, (1, 4, 8, self.head_dim))
        mock_tq_cache_instance.update_and_fetch.assert_called_once()

    def test_infer_model_layout_adapter(self):
        from smaq_mlx.layout import infer_model_layout_adapter, QwenLayoutAdapter, GemmaLayoutAdapter, ModelLayoutAdapter
        
        class DummyGemma: pass
        model_g = DummyGemma()
        adapter_g = infer_model_layout_adapter(model_g)
        self.assertIsInstance(adapter_g, GemmaLayoutAdapter)
        
        class DummyQwen: pass
        model_q = DummyQwen()
        adapter_q = infer_model_layout_adapter(model_q)
        self.assertIsInstance(adapter_q, QwenLayoutAdapter)
        
        class Other: pass
        model_o = Other()
        adapter_o = infer_model_layout_adapter(model_o)
        self.assertIsInstance(adapter_o, ModelLayoutAdapter)

class TestScore(unittest.TestCase):
    """Tests for score calculation components."""

    def test_compute_hybrid_attention_empty(self):
        store = CompressedKVStore(head_dim=64, num_kv_heads=4)
        query = mx.random.normal((1, 4, 64))
        
        # Test with empty store and no recent tokens
        out = compute_hybrid_attention(
            query=query,
            store=store,
            recent_k=None,
            recent_v=None,
            num_query_heads=4
        )
        self.assertEqual(out.shape, (1, 4, 64))
        self.assertTrue(mx.all(out == 0).item())

    def test_compute_hybrid_attention_recent_only(self):
        store = CompressedKVStore(head_dim=64, num_kv_heads=4)
        query = mx.random.normal((1, 4, 64))
        recent_k = mx.random.normal((1, 4, 8, 64))
        recent_v = mx.random.normal((1, 4, 8, 64))
        
        out = compute_hybrid_attention(
            query=query,
            store=store,
            recent_k=recent_k,
            recent_v=recent_v,
            num_query_heads=4
        )
        self.assertEqual(out.shape, (1, 4, 64))

class TestRotorOps(unittest.TestCase):
    """Tests for rotor operations math."""

    def test_gp_rotor_mv_identity(self):
        x = np.random.randn(8).astype(np.float32)
        # Identity rotor: s=1, bivectors=0
        r = gp_rotor_mv(1.0, 0.0, 0.0, 0.0, x)
        np.testing.assert_allclose(r, x, atol=1e-6)

    def test_rotor_sandwich_identity(self):
        x = np.random.randn(8).astype(np.float32)
        r = rotor_sandwich(1.0, 0.0, 0.0, 0.0, x)
        np.testing.assert_allclose(r, x, atol=1e-6)

    def test_make_random_rotor(self):
        rng = np.random.default_rng(42)
        rotor = make_random_rotor(rng)
        self.assertEqual(rotor.shape, (8,))
        # Check normalization: R[0]^2 + R[4]^2 + R[5]^2 + R[6]^2 = 1
        norm_sq = rotor[0]**2 + rotor[4]**2 + rotor[5]**2 + rotor[6]**2
        self.assertAlmostEqual(norm_sq, 1.0, places=6)

    def test_embed_extract_vectors(self):
        vectors = np.random.randn(4, 7).astype(np.float32) # Not multiple of 3
        mv, original_dim = embed_vectors(vectors)
        self.assertEqual(original_dim, 7)
        self.assertEqual(mv.shape, (4, 3, 8)) # 7 padded to 9 -> 3 groups of 3
        
        reconstructed = extract_vectors(mv, original_dim)
        np.testing.assert_allclose(reconstructed, vectors, atol=1e-6)

    def test_optimal_centroids(self):
        centroids = optimal_centroids(bit_width=2, d_eff=64)
        self.assertEqual(centroids.shape, (4,))
        self.assertTrue(np.all(np.diff(centroids) > 0)) # Sorted

    def test_nearest_centroid_indices(self):
        centroids = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([-0.8, 0.1, 1.2, -1.5, 0.6], dtype=np.float32)
        indices = nearest_centroid_indices(values, centroids)
        # -0.8 -> 0; 0.1 -> 1; 1.2 -> 2; -1.5 -> 0; 0.6 -> 2 (boundaries at -0.5, 0.5)
        expected = np.array([0, 1, 2, 0, 2], dtype=np.int32)
        np.testing.assert_array_equal(indices, expected)

if __name__ == "__main__":
    unittest.main()

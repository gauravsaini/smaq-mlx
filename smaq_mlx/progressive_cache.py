import math
from typing import Optional

import mlx.core as mx

from smaq_mlx.core import CacheCapabilities
from smaq_mlx.kv_cache import SMAQKVCache

try:
    from turboquant_mlx.cache import TurboQuantKVCache
    from turboquant_mlx.fused_attention import turboquant_attention
except ImportError:
    TurboQuantKVCache = None
    turboquant_attention = None

class ProgressiveSMAQCache:
    """Progressive Resolution KV Cache (TQ Coarse -> SMAQ Fine)."""

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 4,
        value_bits: int = 4,
        turboquant_bits: int = 3,
        coarse_k: int = 512,
        layer_idx: int = 0,
        layout_adapter=None,
        mode: str = "hybrid",
        strict_benchmark: bool = True,
        Sigma_q: Optional[mx.array] = None,
    ):
        if TurboQuantKVCache is None:
            raise RuntimeError("turboquant-mlx must be installed")
            
        self.coarse_k = coarse_k
        self.head_dim = head_dim
        
        self.tq_cache = TurboQuantKVCache(
            bits=turboquant_bits,
            k_bits=turboquant_bits,
            v_bits=turboquant_bits,
            seed=42,
            fused=False,
        )
        self.smaq_cache = SMAQKVCache(
            head_dim=head_dim,
            Sigma_q=Sigma_q,
            key_bits=key_bits,
            value_bits=value_bits,
            layer_idx=layer_idx,
            layout_adapter=layout_adapter,
            mode=mode,
            strict_benchmark=strict_benchmark,
        )
        self._capabilities = CacheCapabilities(
            strategy_name="progressive_smaq",
            metric_name="smaq_progressive",
            quantization_name="tq_coarse_smaq_fine",
            compressed_history=True,
            compressed_history_shadow_only=False,
            values_compressed=True,
            decode_uses_compressed_keys=True,
            decode_uses_compressed_values=True,
        )

    @property
    def offset(self): return self.smaq_cache.offset
    
    @property
    def capabilities(self): return self._capabilities
    
    def report(self):
        return {
            "strategy_name": self._capabilities.strategy_name,
            "coarse_k": self.coarse_k,
            "offset": self.offset,
        }
        
    def __getattr__(self, name):
        # Forward missing attributes directly to smaq_cache as the primary API
        return getattr(self.smaq_cache, name)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        # Update both caches
        self.tq_cache.update_and_fetch(keys, values)
        return self.smaq_cache.update_and_fetch(keys, values)


def progressive_sdpa(queries, cache: ProgressiveSMAQCache, scale=None, mask=None, **kwargs):
    """Custom SDPA for Progressive Resolution."""
    if scale is None:
        scale = 1.0 / math.sqrt(cache.head_dim)
        
    is_decode = queries.shape[2] == 1
    
    if not is_decode or cache.offset < cache.coarse_k:
        # Full exact/SMAQ fallback during prefill or small contexts
        scores = cache.smaq_cache.attention_scores(queries, scale)
        if mask is not None:
            scores = scores + mask
        attn_weights = mx.softmax(scores, axis=-1)
        return cache.smaq_cache.attend(attn_weights)

    # Decode Path: 1. Coarse Selection using TQ
    # For now, do standard TQ attention scores if available, else approximate
    # This requires recovering the keys from TQ cache which we can do by peeking
    if not hasattr(cache.tq_cache, "k_q"):
         # Fallback to SMAQ if TQ isn't fully setup
         scores = cache.smaq_cache.attention_scores(queries, scale)
         if mask is not None:
             scores = scores + mask
         attn_weights = mx.softmax(scores, axis=-1)
         return cache.smaq_cache.attend(attn_weights)
         
    # Dummy implementation for MLX Python script proof-of-concept
    # 1. Coarse TQ Scores
    if turboquant_attention and getattr(cache.tq_cache, "fused", False):
        # We can't easily extract just the scores from fused TQ attention
        pass 
        
    # We will compute SMAQ scores fully, but mask out everything except Top-K TQ.
    # Since extracting full TQ scores in Python MLX is heavy without kernel,
    # we simulate the top-k masking on SMAQ scores to validate the quality + memory hit.
    scores = cache.smaq_cache.attention_scores(queries, scale)
    if mask is not None:
        scores = scores + mask
        
    # Pick Top K based on the scores
    # In real progressive: tq_scores -> topk_indices -> compute smaq on topk 
    topk_k = min(cache.coarse_k, scores.shape[-1])
    topk_vals = mx.topk(scores, k=topk_k, axis=-1)
    thresholds = mx.min(topk_vals, axis=-1, keepdims=True)
    
    # Mask out non-top-K
    mask_topk = mx.where(scores >= thresholds, 0.0, -1e9)
    scores = scores + mask_topk
    
    attn_weights = mx.softmax(scores, axis=-1)
    return cache.smaq_cache.attend(attn_weights)

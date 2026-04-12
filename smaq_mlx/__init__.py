from smaq_mlx.block_vq import BlockVQQuantized, SMAQBlockVQ
from smaq_mlx.api import (
    MLXRuntimeConfig,
    SMAQConfig,
    available_backends,
    backend_matrix,
    backend_enabled,
    disable_backend,
    disable_smaq,
    enable_backend,
    enable_smaq,
    generate,
    get_runtime_configuration,
    get_smaq_configuration,
    smaq_enabled,
)
from smaq_mlx.backends import get_backend, register_backend, validate_backend
from smaq_mlx.core import CacheCapabilities, IdentityCalibrationProvider, LayoutInfo, StaticCalibrationProvider
from smaq_mlx.capture import KVCaptureEngine, RingBuffer
from smaq_mlx.folded_cache import FoldedTurboSMAQKVCache
from smaq_mlx.layout import GemmaLayoutAdapter, ModelLayoutAdapter, QwenLayoutAdapter, infer_model_layout_adapter
from smaq_mlx.kv_cache import SMAQKVCache
from smaq_mlx.quantizer import SMAQQuantized, SMAQQuantizer
from smaq_mlx.rotor_cache import RotorQuantKVCache
from smaq_mlx.score import compute_hybrid_attention
from smaq_mlx.rotorquant import RotorQuantMSE
from smaq_mlx.stacked_cache import TurboSMAQCascadeCache
from smaq_mlx.store import CompressedKVStore
from smaq_mlx.attention_smaq import smaq_sdpa
from smaq_mlx.ssf import build_smaq_metric, ssf_log
from smaq_mlx.patch import apply as apply_patch, clear_configuration, configure, current_configuration, make_prompt_cache, make_smaq_prompt_cache
from smaq_mlx.patch import revert as revert_patch

__version__ = "0.1.0"

__all__ = [
    "BlockVQQuantized",
    "CacheCapabilities",
    "CompressedKVStore",
    "FoldedTurboSMAQKVCache",
    "GemmaLayoutAdapter",
    "IdentityCalibrationProvider",
    "KVCaptureEngine",
    "LayoutInfo",
    "MLXRuntimeConfig",
    "ModelLayoutAdapter",
    "QwenLayoutAdapter",
    "RingBuffer",
    "RotorQuantKVCache",
    "RotorQuantMSE",
    "SMAQBlockVQ",
    "SMAQConfig",
    "SMAQKVCache",
    "SMAQQuantized",
    "SMAQQuantizer",
    "StaticCalibrationProvider",
    "TurboSMAQCascadeCache",
    "apply_patch",
    "available_backends",
    "backend_enabled",
    "backend_matrix",
    "build_smaq_metric",
    "clear_configuration",
    "compute_hybrid_attention",
    "configure",
    "current_configuration",
    "disable_backend",
    "disable_smaq",
    "enable_backend",
    "enable_smaq",
    "generate",
    "get_backend",
    "get_runtime_configuration",
    "get_smaq_configuration",
    "infer_model_layout_adapter",
    "make_prompt_cache",
    "make_smaq_prompt_cache",
    "register_backend",
    "revert_patch",
    "smaq_enabled",
    "smaq_sdpa",
    "ssf_log",
    "validate_backend",
]

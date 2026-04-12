"""Backend registry for MLX KV-cache runtimes.

This module gives us one runtime patch layer with pluggable backends underneath.
That avoids the "last monkey-patch wins" failure mode when multiple KV runtimes
target the same MLX seams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import mlx.core as mx

from smaq_mlx.attention_smaq import smaq_sdpa
from smaq_mlx.kv_cache import SMAQKVCache
from smaq_mlx.layout import infer_model_layout_adapter


@dataclass(frozen=True)
class RuntimeBackend:
    """Abstract runtime backend contract.

    A backend owns two things:
    1. how compatible prompt-cache entries are created
    2. how SDPA dispatch works for those cache entries

    Backends should be self-contained and should not patch MLX directly.
    The patch layer in ``smaq_mlx.patch`` remains the single runtime owner.
    """

    name: str
    description: str = ""
    status: str = "experimental"
    requires_package: str | None = None

    def supports_cache(self, cache: Any) -> bool:
        raise NotImplementedError

    def make_prompt_cache(self, model: Any, *, cache_module: Any, config: Dict[str, Any]) -> list[Any]:
        raise NotImplementedError

    def sdpa(
        self,
        queries: Any,
        keys: Any,
        values: Any,
        cache: Any,
        *,
        scale: float,
        mask: Any,
        sinks: Any = None,
        config: Dict[str, Any],
        original_sdpa: Any = None,
        **kwargs,
    ):
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "requires_package": self.requires_package,
        }


def _get_head_dim(model, layer_idx: int) -> int:
    layer = model.layers[layer_idx]
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        if hasattr(attn, "head_dim"):
            return attn.head_dim
        if hasattr(attn, "hidden_size") and hasattr(attn, "num_heads"):
            return attn.hidden_size // attn.num_heads
    return 128


class SMAQBackend(RuntimeBackend):
    def __init__(self):
        super().__init__(
            name="smaq",
            description="SMAQ compressed-history backend for MLX decode.",
            status="experimental",
        )

    def supports_cache(self, cache: Any) -> bool:
        return isinstance(cache, SMAQKVCache)

    def make_prompt_cache(self, model: Any, *, cache_module: Any, config: Dict[str, Any]) -> list[Any]:
        layout_adapter = config.get("layout_adapter") or infer_model_layout_adapter(model)
        key_bits = int(config.get("key_bits", 4))
        value_bits = int(config.get("value_bits", 4))
        Sigma_q = config.get("Sigma_q")
        mode = str(config.get("mode", "hybrid"))
        strict_benchmark = bool(config.get("strict_benchmark", False))

        if hasattr(model, "make_cache") and cache_module is not None:
            original_caches = model.make_cache()
            result = []
            for i, c in enumerate(original_caches):
                if isinstance(c, (cache_module.KVCache, cache_module.RotatingKVCache)):
                    head_dim = _get_head_dim(model, i)
                    result.append(
                        SMAQKVCache(
                            head_dim=head_dim,
                            Sigma_q=Sigma_q,
                            key_bits=key_bits,
                            value_bits=value_bits,
                            layer_idx=i,
                            layout_adapter=layout_adapter,
                            mode=mode,
                            strict_benchmark=strict_benchmark,
                        )
                    )
                else:
                    result.append(c)
            return result

        num_layers = len(model.layers)
        head_dim = _get_head_dim(model, 0)
        return [
            SMAQKVCache(
                head_dim=head_dim,
                Sigma_q=Sigma_q,
                key_bits=key_bits,
                value_bits=value_bits,
                layer_idx=i,
                layout_adapter=layout_adapter,
                mode=mode,
                strict_benchmark=strict_benchmark,
            )
            for i in range(num_layers)
        ]

    def sdpa(
        self,
        queries: Any,
        keys: Any,
        values: Any,
        cache: Any,
        *,
        scale: float,
        mask: Any,
        sinks: Any = None,
        config: Dict[str, Any],
        original_sdpa: Any = None,
        **kwargs,
    ):
        require_true = bool(config.get("require_true_compressed", False))
        return smaq_sdpa(
            queries,
            cache,
            scale=scale,
            mask=mask,
            require_true_compressed=require_true,
        )


class TurboQuantBackend(RuntimeBackend):
    def __init__(self):
        super().__init__(
            name="turboquant",
            description="TurboQuant / PolarQuant-style bit-packed KV backend for MLX.",
            status="experimental",
            requires_package="turboquant-mlx",
        )

    @staticmethod
    def _imports():
        try:
            from turboquant_mlx.cache import TurboQuantKVCache
            from turboquant_mlx.fused_attention import turboquant_attention
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TurboQuant backend requires the `turboquant-mlx` package to be installed."
            ) from exc
        return TurboQuantKVCache, turboquant_attention

    def supports_cache(self, cache: Any) -> bool:
        try:
            TurboQuantKVCache, _ = self._imports()
        except RuntimeError:
            return False
        return isinstance(cache, TurboQuantKVCache)

    def make_prompt_cache(self, model: Any, *, cache_module: Any, config: Dict[str, Any]) -> list[Any]:
        TurboQuantKVCache, _ = self._imports()
        bits = int(config.get("turboquant_bits") or config.get("key_bits", 3) or 3)
        k_bits = config.get("key_bits")
        v_bits = config.get("value_bits")
        seed = int(config.get("turboquant_seed", 42))
        fused = bool(config.get("turboquant_fused", True))

        if hasattr(model, "make_cache") and cache_module is not None:
            original_caches = model.make_cache()
            result = []
            for c in original_caches:
                if isinstance(c, cache_module.KVCache):
                    result.append(
                        TurboQuantKVCache(
                            bits=bits,
                            k_bits=k_bits,
                            v_bits=v_bits,
                            seed=seed,
                            fused=fused,
                        )
                    )
                else:
                    result.append(c)
            return result

        num_layers = len(model.layers)
        return [
            TurboQuantKVCache(
                bits=bits,
                k_bits=k_bits,
                v_bits=v_bits,
                seed=seed,
                fused=fused,
            )
            for _ in range(num_layers)
        ]

    def sdpa(
        self,
        queries: Any,
        keys: Any,
        values: Any,
        cache: Any,
        *,
        scale: float,
        mask: Any,
        sinks: Any = None,
        config: Dict[str, Any],
        original_sdpa: Any = None,
        **kwargs,
    ):
        TurboQuantKVCache, turboquant_attention = self._imports()
        if not isinstance(cache, TurboQuantKVCache):  # pragma: no cover - defensive
            raise TypeError("TurboQuant backend received unexpected cache type")

        is_decode = queries.shape[2] == 1
        if is_decode and cache.offset > 0 and getattr(cache, "fused", False):
            return turboquant_attention(queries, cache, scale, mask, v_buffer=values)

        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )


class PolarQuantBackend(RuntimeBackend):
    def __init__(self):
        super().__init__(
            name="polarquant",
            description="PolarQuant-style MLX backend using rotated scalar quantization without QJL correction.",
            status="experimental",
            requires_package="mlx-turboquant",
        )

    @staticmethod
    def _imports():
        try:
            from mlx_turboquant.cache import TurboQuantKVCache as PolarQuantKVCache
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PolarQuant backend requires the `mlx-turboquant` package to be installed."
            ) from exc
        return PolarQuantKVCache

    @staticmethod
    def _resolve_bits(config: Dict[str, Any]) -> float:
        bits = config.get("polarquant_bits")
        key_bits = config.get("key_bits")
        value_bits = config.get("value_bits")
        if bits is not None:
            return float(bits)
        if key_bits is not None and value_bits is not None and float(key_bits) != float(value_bits):
            raise ValueError(
                "PolarQuant backend currently requires a shared bit-width for keys and values. "
                "Set `polarquant_bits`, or keep `key_bits == value_bits`."
            )
        if key_bits is not None:
            return float(key_bits)
        if value_bits is not None:
            return float(value_bits)
        return 3.0

    def supports_cache(self, cache: Any) -> bool:
        try:
            PolarQuantKVCache = self._imports()
        except RuntimeError:
            return False
        return isinstance(cache, PolarQuantKVCache)

    def make_prompt_cache(self, model: Any, *, cache_module: Any, config: Dict[str, Any]) -> list[Any]:
        PolarQuantKVCache = self._imports()
        bits = self._resolve_bits(config)
        key_seed = int(config.get("polarquant_key_seed", 42))
        value_seed = int(config.get("polarquant_value_seed", 43))

        if hasattr(model, "make_cache") and cache_module is not None:
            original_caches = model.make_cache()
            result = []
            for i, c in enumerate(original_caches):
                if isinstance(c, cache_module.KVCache):
                    result.append(
                        PolarQuantKVCache(
                            bits=bits,
                            head_dim=_get_head_dim(model, i),
                            key_seed=key_seed,
                            value_seed=value_seed,
                        )
                    )
                else:
                    result.append(c)
            return result

        num_layers = len(model.layers)
        return [
            PolarQuantKVCache(
                bits=bits,
                head_dim=_get_head_dim(model, i),
                key_seed=key_seed,
                value_seed=value_seed,
            )
            for i in range(num_layers)
        ]

    def sdpa(
        self,
        queries: Any,
        keys: Any,
        values: Any,
        cache: Any,
        *,
        scale: float,
        mask: Any,
        sinks: Any = None,
        config: Dict[str, Any],
        original_sdpa: Any = None,
        **kwargs,
    ):
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )


class StackedTurboSMAQBackend(RuntimeBackend):
    def __init__(self):
        super().__init__(
            name="stacked_turbo_smaq",
            description="Experimental cascade: TurboQuant approximation feeding SMAQ compressed decode.",
            status="research",
            requires_package="turboquant-mlx",
        )

    @staticmethod
    def _imports():
        try:
            from smaq_mlx.stacked_cache import TurboSMAQCascadeCache
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Stacked TurboQuant+SMAQ backend requires `turboquant-mlx` to be installed."
            ) from exc
        return TurboSMAQCascadeCache

    def supports_cache(self, cache: Any) -> bool:
        try:
            cache_type = self._imports()
        except RuntimeError:
            return False
        return isinstance(cache, cache_type)

    def make_prompt_cache(self, model: Any, *, cache_module: Any, config: Dict[str, Any]) -> list[Any]:
        CascadeCache = self._imports()
        key_bits = int(config.get("key_bits", 4))
        value_bits = int(config.get("value_bits", 4))
        turboquant_bits = int(config.get("turboquant_bits") or 3)
        turboquant_seed = int(config.get("turboquant_seed", 42))
        layout_adapter = config.get("layout_adapter") or infer_model_layout_adapter(model)
        Sigma_q = config.get("Sigma_q")
        mode = str(config.get("mode", "hybrid"))
        strict_benchmark = bool(config.get("strict_benchmark", True))

        if hasattr(model, "make_cache") and cache_module is not None:
            original_caches = model.make_cache()
            result = []
            for i, c in enumerate(original_caches):
                if isinstance(c, (cache_module.KVCache, cache_module.RotatingKVCache)):
                    head_dim = _get_head_dim(model, i)
                    result.append(
                        CascadeCache(
                            head_dim=head_dim,
                            key_bits=key_bits,
                            value_bits=value_bits,
                            turboquant_bits=turboquant_bits,
                            turboquant_seed=turboquant_seed,
                            layer_idx=i,
                            layout_adapter=layout_adapter,
                            mode=mode,
                            strict_benchmark=strict_benchmark,
                            Sigma_q=Sigma_q,
                        )
                    )
                else:
                    result.append(c)
            return result

        num_layers = len(model.layers)
        head_dim = _get_head_dim(model, 0)
        return [
            CascadeCache(
                head_dim=head_dim,
                key_bits=key_bits,
                value_bits=value_bits,
                turboquant_bits=turboquant_bits,
                turboquant_seed=turboquant_seed,
                layer_idx=i,
                layout_adapter=layout_adapter,
                mode=mode,
                strict_benchmark=strict_benchmark,
                Sigma_q=Sigma_q,
            )
            for i in range(num_layers)
        ]

    def sdpa(
        self,
        queries: Any,
        keys: Any,
        values: Any,
        cache: Any,
        *,
        scale: float,
        mask: Any,
        sinks: Any = None,
        config: Dict[str, Any],
        original_sdpa: Any = None,
        **kwargs,
    ):
        require_true = bool(config.get("require_true_compressed", False))
        return smaq_sdpa(
            queries,
            cache,
            scale=scale,
            mask=mask,
            require_true_compressed=require_true,
        )


class PlaceholderBackend(RuntimeBackend):
    """Registered placeholder for a future backend.

    This keeps the runtime API extensible without pretending the backend is
    implemented today.
    """

    def __init__(self, name: str, description: str, requires_package: str | None = None):
        super().__init__(
            name=name,
            description=description,
            status="planned",
            requires_package=requires_package,
        )

    def supports_cache(self, cache: Any) -> bool:
        return False

    def make_prompt_cache(self, model: Any, *, cache_module: Any, config: Dict[str, Any]) -> list[Any]:
        raise RuntimeError(
            f"The `{self.name}` backend is registered as a placeholder but is not implemented yet."
        )

    def sdpa(
        self,
        queries: Any,
        keys: Any,
        values: Any,
        cache: Any,
        *,
        scale: float,
        mask: Any,
        sinks: Any = None,
        config: Dict[str, Any],
        original_sdpa: Any = None,
        **kwargs,
    ):
        raise RuntimeError(
            f"The `{self.name}` backend is registered as a placeholder but is not implemented yet."
        )


_BACKENDS: Dict[str, RuntimeBackend] = {
    "polarquant": PolarQuantBackend(),
    "smaq": SMAQBackend(),
    "stacked_turbo_smaq": StackedTurboSMAQBackend(),
    "turboquant": TurboQuantBackend(),
}


def validate_backend(backend: RuntimeBackend):
    if not isinstance(backend, RuntimeBackend):
        raise TypeError("Backend must inherit from RuntimeBackend")
    if not backend.name or not isinstance(backend.name, str):
        raise ValueError("Backend must define a non-empty string name")
    for method_name in ("supports_cache", "make_prompt_cache", "sdpa"):
        method = getattr(backend, method_name, None)
        if method is None or not callable(method):
            raise TypeError(f"Backend `{backend.name}` is missing callable `{method_name}`")


def register_backend(backend: RuntimeBackend):
    validate_backend(backend)
    _BACKENDS[backend.name] = backend


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def backend_matrix() -> list[dict[str, Any]]:
    return [get_backend(name).metadata() for name in available_backends()]


def get_backend(name: str) -> RuntimeBackend:
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown MLX runtime backend: {name}") from exc


def make_prompt_cache_for_backend(
    backend_name: str,
    model: Any,
    *,
    cache_module: Any,
    config: Dict[str, Any],
) -> list[Any]:
    backend = get_backend(backend_name)
    return backend.make_prompt_cache(model, cache_module=cache_module, config=config)


def iter_backends() -> Iterable[RuntimeBackend]:
    return _BACKENDS.values()


def dispatch_sdpa(
    queries: Any,
    keys: Any,
    values: Any,
    cache: Any,
    *,
    scale: float,
    mask: Any,
    sinks: Any = None,
    config: Dict[str, Any],
    original_sdpa: Any = None,
    **kwargs,
):
    for backend in iter_backends():
        if backend.supports_cache(cache):
            return backend.sdpa(
                queries,
                keys,
                values,
                cache,
                scale=scale,
                mask=mask,
                sinks=sinks,
                config=config,
                original_sdpa=original_sdpa,
                **kwargs,
            )
    return None

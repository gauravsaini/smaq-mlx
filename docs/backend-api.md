# Backend API

`smaq-mlx` uses a **single MLX runtime patch** with pluggable backend implementations underneath it.

That means:

- `smaq_mlx.patch` remains the only runtime patch owner
- backends do **not** monkey-patch `mlx_lm` themselves
- backends only implement cache creation and SDPA dispatch for their own cache type

## Contract

Backends must inherit from `smaq_mlx.backends.RuntimeBackend`.

Required surface:

- `name`
- `supports_cache(cache) -> bool`
- `make_prompt_cache(model, *, cache_module, config) -> list`
- `sdpa(queries, keys, values, cache, *, scale, mask, sinks=None, config, original_sdpa=None, **kwargs)`

Optional metadata:

- `description`
- `status`
- `requires_package`

Backends are validated by `validate_backend(...)` before registration.

## Design Rules

1. The patch layer owns `mlx_lm.models.cache.make_prompt_cache`.
2. The patch layer owns `mlx_lm.models.base.scaled_dot_product_attention`.
3. A backend must be selectable by config, not by patch order.
4. A backend should expose enough metadata for support matrices and diagnostics.
5. A backend should degrade clearly when its optional dependency is missing.

## Registering A Backend

```python
from smaq_mlx import register_backend
from smaq_mlx.backends import RuntimeBackend


class MyBackend(RuntimeBackend):
    def __init__(self):
        super().__init__(
            name="mybackend",
            description="Example backend.",
            status="experimental",
            requires_package="mybackend-mlx",
        )

    def supports_cache(self, cache):
        return False

    def make_prompt_cache(self, model, *, cache_module, config):
        raise NotImplementedError

    def sdpa(self, queries, keys, values, cache, *, scale, mask, sinks=None, config, original_sdpa=None, **kwargs):
        raise NotImplementedError


register_backend(MyBackend())
```

## Current Backends

- `polarquant`: implemented through the MLX-native `mlx-turboquant` reference cache path
- `smaq`: implemented
- `turboquant`: implemented through the unified adapter layer

## Placeholder Backends

Placeholder registrations are allowed when:

- we want public API stability for a planned backend
- we do **not** want to pretend the backend is already implemented

We still keep placeholder registrations available for future backends, but `polarquant` is no longer one of them.

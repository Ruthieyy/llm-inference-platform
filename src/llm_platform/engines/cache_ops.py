"""KV-cache row select and concat.

Current transformers builds store one ``CacheLayer`` per block on ``cache.layers``
with ``.keys`` and ``.values`` shaped ``[batch, heads, seq, dim]``. ``batch_select_indices``
is the supported way to drop finished rows. Older caches are tuples of ``(key, value)``.
"""

from __future__ import annotations

from typing import Any

import torch


def select_rows(cache: Any, index: torch.Tensor) -> Any:
    if hasattr(cache, "batch_select_indices") and getattr(cache, "layers", None) is not None:
        cache.batch_select_indices(index)
        return cache
    pairs = [(key.index_select(0, index), value.index_select(0, index)) for key, value in _pairs(cache)]
    return _rebuild(cache, pairs)


def concat_rows(left: Any, right: Any) -> Any:
    if getattr(left, "layers", None) and getattr(right, "layers", None):
        if len(left.layers) != len(right.layers):
            raise ValueError("cache depth mismatch")
        cloned = left.__class__()
        cloned.layers = [_concat_layer(a, b) for a, b in zip(left.layers, right.layers, strict=True)]
        return cloned
    left_pairs = _pairs(left)
    right_pairs = _pairs(right)
    if len(left_pairs) != len(right_pairs):
        raise ValueError("cache depth mismatch")
    merged = [
        (torch.cat([lk, rk], dim=0), torch.cat([lv, rv], dim=0))
        for (lk, lv), (rk, rv) in zip(left_pairs, right_pairs, strict=True)
    ]
    return _rebuild(left, merged)


def _concat_layer(left: Any, right: Any) -> Any:
    kwargs = {}
    if getattr(left, "sliding_window", None) is not None:
        kwargs["sliding_window"] = left.sliding_window
    layer = left.__class__(**kwargs)
    if left.keys is None or right.keys is None:
        return layer
    layer.keys = torch.cat([left.keys, right.keys], dim=0)
    layer.values = torch.cat([left.values, right.values], dim=0)
    layer.is_initialized = True
    if hasattr(left, "dtype"):
        layer.dtype = left.dtype
        layer.device = left.device
    return layer


def _pairs(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layers = getattr(cache, "layers", None)
    if layers:
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                continue
            pairs.append((key, value))
        if pairs:
            return pairs
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache, strict=True))
    if isinstance(cache, tuple):
        return [(layer[0], layer[1]) for layer in cache]
    raise TypeError(f"unsupported cache type {type(cache)!r}")


def _rebuild(cache: Any, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> Any:
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cloned = cache.__class__()
        cloned.key_cache = [key for key, _ in pairs]
        cloned.value_cache = [value for _, value in pairs]
        return cloned
    return tuple((key, value) for key, value in pairs)

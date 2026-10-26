"""Group-wise symmetric int4 quantization, pure Python.

This is the specification the torch path mirrors. Range per group is [-7, 7]
with scale = max(abs(group)) / 7 so zero is exact. Values are stored as
nibbles in ``[0, 14]`` (quantized value plus 7).
"""

from __future__ import annotations


def quantize_int4(values: list[float], group_size: int) -> tuple[list[int], list[float]]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    data = list(values)
    remainder = len(data) % group_size
    if remainder:
        data.extend([0.0] * (group_size - remainder))
    nibbles: list[int] = []
    scales: list[float] = []
    for start in range(0, len(data), group_size):
        group = data[start : start + group_size]
        max_abs = max(abs(value) for value in group)
        scale = (max_abs / 7.0) if max_abs > 0 else 1.0
        scales.append(scale)
        for value in group:
            quantized = int(round(value / scale))
            if quantized > 7:
                quantized = 7
            elif quantized < -7:
                quantized = -7
            nibbles.append(quantized + 7)
    return nibbles, scales


def dequantize_int4(
    nibbles: list[int], scales: list[float], group_size: int, length: int
) -> list[float]:
    restored: list[float] = []
    for index, nibble in enumerate(nibbles):
        scale = scales[index // group_size]
        restored.append((nibble - 7) * scale)
    return restored[:length]

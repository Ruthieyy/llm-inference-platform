"""CPU weight quantization used by the transformers backend.

INT8 uses PyTorch dynamic quantization (FBGEMM-backed on x86). INT4 is a
group-wise symmetric weight-only scheme: weights stay packed, and each forward
dequantizes the group back to float32. Without an INT4 matmul kernel that is a
memory win and usually a latency loss. The practical INT4 path on CPU is a
GGUF model served by llama.cpp; this module is the in-process comparison point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def resident_nbytes(model: nn.Module) -> int:
    total = 0
    seen: set[int] = set()
    tensors = list(model.parameters()) + list(model.buffers())
    for tensor in tensors:
        if tensor is None:
            continue
        pointer = tensor.data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += tensor.nbytes
    return total


def apply_dynamic_int8(model: nn.Module) -> nn.Module:
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized


class Int4Linear(nn.Module):
    """Weight-only symmetric INT4 linear. Bias stays float32."""

    def __init__(self, linear: nn.Linear, group_size: int = 64):
        super().__init__()
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        weight = linear.weight.detach().float().contiguous()
        out_features, in_features = weight.shape
        flat = weight.reshape(-1)
        remainder = flat.numel() % group_size
        pad = (group_size - remainder) % group_size
        if pad:
            flat = F.pad(flat, (0, pad))
        grouped = flat.view(-1, group_size)
        max_abs = grouped.abs().amax(dim=1).clamp(min=1e-8)
        scales = max_abs / 7.0
        quantized = torch.round(grouped / scales.unsqueeze(1)).clamp(-7, 7)
        nibbles = (quantized + 7).to(torch.uint8).reshape(-1)
        packed_pad = nibbles.numel() % 2
        if packed_pad:
            nibbles = F.pad(nibbles, (0, 1))
        pairs = nibbles.view(-1, 2)
        packed = (pairs[:, 0] | (pairs[:, 1] << 4)).to(torch.uint8).contiguous()

        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.grouped_numel = int(flat.numel())
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales.to(torch.float16).contiguous())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().float().contiguous())
        else:
            self.bias = None

    def _weight(self) -> torch.Tensor:
        low = (self.packed & 0x0F).to(torch.int16)
        high = ((self.packed >> 4) & 0x0F).to(torch.int16)
        nibbles = torch.stack((low, high), dim=1).reshape(-1)[: self.grouped_numel]
        values = nibbles.to(torch.float32) - 7.0
        scaled = values.view(-1, self.group_size) * self.scales.float().unsqueeze(1)
        flat = scaled.reshape(-1)[: self.out_features * self.in_features]
        return flat.view(self.out_features, self.in_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self._weight(), self.bias)


def apply_int4_weight_only(model: nn.Module, group_size: int = 64) -> nn.Module:
    """Replace every Linear except ``lm_head`` with packed INT4 weights."""
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name != "lm_head":
                replacements.append((module, name, child))
    for module, name, child in replacements:
        setattr(module, name, Int4Linear(child, group_size=group_size))
    return model


def cast_weights(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    return model.to(dtype=dtype)

from llm_platform.quant_math import dequantize_int4, quantize_int4


def test_int4_roundtrip_error_is_bounded():
    values = [0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.3, 0.7, 3.5, -2.2]
    group_size = 4
    nibbles, scales = quantize_int4(values, group_size)
    restored = dequantize_int4(nibbles, scales, group_size, len(values))
    assert len(restored) == len(values)
    for original, approx in zip(values, restored, strict=True):
        max_abs = max(abs(value) for value in values)
        assert abs(original - approx) <= (max_abs / 7.0) + 1e-9


def test_int4_represents_zero_exactly():
    nibbles, scales = quantize_int4([0.0, 0.0, 0.0, 0.0], group_size=4)
    restored = dequantize_int4(nibbles, scales, 4, 4)
    assert restored == [0.0, 0.0, 0.0, 0.0]


def test_int4_pads_partial_groups():
    nibbles, scales = quantize_int4([1.0, -1.0], group_size=4)
    assert len(nibbles) == 4
    assert len(scales) == 1
    restored = dequantize_int4(nibbles, scales, 4, 2)
    assert abs(restored[0] - 1.0) < 1e-9
    assert abs(restored[1] + 1.0) < 1e-9

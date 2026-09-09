import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.mega_moe import (
    _native_mxfp4_to_megamoe_layout,
    _shuffle_mxfp4_scale,
    _shuffle_mxfp4_weight,
    build_flydsl_mega_moe_experts_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def _reference_shuffle_weight(
    src: torch.Tensor, *, experts: int, gate_up: bool
) -> torch.Tensor:
    rows = src.shape[1] // 2 if gate_up else src.shape[1]
    n_outer = rows // 16
    k_outer = src.shape[2] // 64
    if gate_up:
        return (
            src.view(experts, 2, n_outer, 16, k_outer, 4, 16)
            .permute(0, 2, 1, 4, 5, 3, 6)
            .contiguous()
            .view_as(src)
        )
    return (
        src.view(experts, n_outer, 16, k_outer, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view_as(src)
    )


def _reference_shuffle_scale(
    src: torch.Tensor, *, experts: int, gate_up: bool
) -> torch.Tensor:
    n_outer = src.shape[1] // 32
    k_outer = src.shape[2] // 8
    if gate_up:
        return (
            src.view(experts, 2, n_outer, 16, k_outer, 2, 4)
            .permute(0, 2, 4, 6, 3, 5, 1)
            .contiguous()
            .view_as(src)
        )
    return (
        src.view(experts, n_outer, 2, 16, k_outer, 2, 4)
        .permute(0, 1, 4, 6, 3, 5, 2)
        .contiguous()
        .view_as(src)
    )


@pytest.mark.parametrize("gate_up,rows", [(True, 64), (False, 32)])
def test_flydsl_megamoe_weight_shuffle_matches_kernel_layout(gate_up, rows):
    src = torch.arange(2 * rows * 64, dtype=torch.int64).to(torch.uint8)
    src = src.view(2, rows, 64)

    actual = _shuffle_mxfp4_weight(src, experts=2, gate_up=gate_up)
    expected = _reference_shuffle_weight(src, experts=2, gate_up=gate_up)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("gate_up,rows", [(True, 64), (False, 32)])
def test_flydsl_megamoe_scale_shuffle_matches_kernel_layout(gate_up, rows):
    src = torch.arange(2 * rows * 8, dtype=torch.int64).to(torch.uint8)
    src = src.view(2, rows, 8)

    actual = _shuffle_mxfp4_scale(src, experts=2, gate_up=gate_up)
    expected = _reference_shuffle_scale(src, experts=2, gate_up=gate_up)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_flydsl_megamoe_scale_shuffle_rejects_unaligned_rows():
    with pytest.raises(ValueError, match="rows=48"):
        _shuffle_mxfp4_scale(
            torch.empty((2, 48, 8), dtype=torch.uint8),
            experts=2,
            gate_up=True,
        )


def test_native_mxfp4_layout_preserves_packed_bytes():
    weight_u8 = torch.arange(2 * 64 * 128, dtype=torch.int64).to(torch.uint8)
    weight_u8 = weight_u8.view(2, 64, 128)
    scale_u8 = torch.arange(2 * 64 * 8, dtype=torch.int64).to(torch.uint8)
    scale_u8 = scale_u8.view(2, 64, 8)

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    weight = weight_u8.view(fp4_dtype) if fp4_dtype is not None else weight_u8
    scale = scale_u8.view(e8m0_dtype) if e8m0_dtype is not None else scale_u8

    actual_weight, actual_scale = _native_mxfp4_to_megamoe_layout(
        weight, scale, gate_up=True
    )

    assert actual_weight.dtype == torch.uint8
    assert actual_scale.dtype == torch.uint8
    torch.testing.assert_close(
        actual_weight,
        _shuffle_mxfp4_weight(weight_u8, experts=2, gate_up=True),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_scale,
        _shuffle_mxfp4_scale(scale_u8, experts=2, gate_up=True),
        rtol=0,
        atol=0,
    )


def test_native_mxfp4_layout_rejects_invalid_scale_shape():
    with pytest.raises(ValueError, match="Unexpected native MXFP4"):
        _native_mxfp4_to_megamoe_layout(
            torch.empty((1, 32, 128), dtype=torch.uint8),
            torch.empty((1, 32, 7), dtype=torch.uint8),
            gate_up=False,
        )


def test_native_mxfp4_layout_rejects_non_e8m0_scale_dtype():
    with pytest.raises(ValueError, match="packed byte dtypes"):
        _native_mxfp4_to_megamoe_layout(
            torch.empty((1, 32, 128), dtype=torch.uint8),
            torch.empty((1, 32, 8), dtype=torch.float32),
            gate_up=False,
        )


def test_build_flydsl_megamoe_skips_requantization_for_native_mxfp4():
    w13 = torch.arange(2 * 64 * 128, dtype=torch.int64).to(torch.uint8)
    w13 = w13.view(2, 64, 128)
    s13 = torch.arange(2 * 64 * 8, dtype=torch.int64).to(torch.uint8)
    s13 = s13.view(2, 64, 8)
    w2 = torch.arange(2 * 32 * 128, dtype=torch.int64).to(torch.uint8)
    w2 = w2.view(2, 32, 128)
    s2 = torch.arange(2 * 32 * 8, dtype=torch.int64).to(torch.uint8)
    s2 = s2.view(2, 32, 8)
    experts = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13, requires_grad=False),
        w13_weight_scale_inv=torch.nn.Parameter(s13, requires_grad=False),
        w2_weight=torch.nn.Parameter(w2, requires_grad=False),
        w2_weight_scale_inv=torch.nn.Parameter(s2, requires_grad=False),
        quant_config=SimpleNamespace(),
        layer_id=1,
    )

    with patch(
        "sglang.srt.layers.moe.mega_moe.is_gfx95_supported", return_value=True
    ), patch(
        "sglang.srt.distributed.get_moe_expert_parallel_rank", return_value=1
    ), patch(
        "sglang.srt.layers.moe.mega_moe._fp8_blockwise_to_mxfp4"
    ) as requantize:
        build_flydsl_mega_moe_experts_weights(experts)

    requantize.assert_not_called()
    assert experts._flydsl_mega_moe_weights_built
    torch.testing.assert_close(
        experts.w13_weight,
        _shuffle_mxfp4_weight(w13, experts=2, gate_up=True),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        experts.w13_weight_scale_inv,
        _shuffle_mxfp4_scale(s13, experts=2, gate_up=True),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        experts.w2_weight,
        _shuffle_mxfp4_weight(w2, experts=2, gate_up=False),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        experts.w2_weight_scale_inv,
        _shuffle_mxfp4_scale(s2, experts=2, gate_up=False),
        rtol=0,
        atol=0,
    )


def test_build_flydsl_megamoe_still_requantizes_fp8_weights():
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("PyTorch build has no float8_e4m3fn dtype")

    experts = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.empty((1, 64, 256), dtype=fp8_dtype), requires_grad=False
        ),
        w13_weight_scale_inv=torch.nn.Parameter(
            torch.empty((1, 2, 2), dtype=torch.float32), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.empty((1, 32, 256), dtype=fp8_dtype), requires_grad=False
        ),
        w2_weight_scale_inv=torch.nn.Parameter(
            torch.empty((1, 1, 2), dtype=torch.float32), requires_grad=False
        ),
        quant_config=SimpleNamespace(weight_block_size=[128, 128]),
        layer_id=1,
    )

    def fake_requantize(weight, scale, block_size, *, gate_up):
        del scale, block_size
        rows = weight.shape[1]
        return (
            torch.empty((1, rows, 128), dtype=torch.uint8),
            torch.empty((1, rows, 8), dtype=torch.uint8),
        )

    with patch(
        "sglang.srt.layers.moe.mega_moe.is_gfx95_supported", return_value=True
    ), patch(
        "sglang.srt.distributed.get_moe_expert_parallel_rank", return_value=1
    ), patch(
        "sglang.srt.layers.moe.mega_moe._fp8_blockwise_to_mxfp4",
        side_effect=fake_requantize,
    ) as requantize:
        build_flydsl_mega_moe_experts_weights(experts)

    assert requantize.call_count == 2
    assert experts._flydsl_mega_moe_weights_built


def test_build_flydsl_megamoe_rejects_mixed_weight_formats():
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("PyTorch build has no float8_e4m3fn dtype")

    experts = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.empty((1, 64, 128), dtype=torch.uint8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.empty((1, 32, 256), dtype=fp8_dtype), requires_grad=False
        ),
    )

    with patch(
        "sglang.srt.layers.moe.mega_moe.is_gfx95_supported", return_value=True
    ), pytest.raises(ValueError, match="same source format"):
        build_flydsl_mega_moe_experts_weights(experts)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

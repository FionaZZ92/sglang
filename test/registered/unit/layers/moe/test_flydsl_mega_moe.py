import sys

import pytest
import torch

from sglang.srt.layers.moe.mega_moe import (
    _shuffle_mxfp4_scale,
    _shuffle_mxfp4_weight,
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

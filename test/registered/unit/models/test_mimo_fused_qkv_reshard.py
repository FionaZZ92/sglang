from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.model_config import (
    can_mimo_v2_fused_qkv_reshard,
    get_mimo_v2_fused_qkv_expected_tp_size,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.mimo_v2 import (
    MiMoV2FusedQKVParallelLinear,
    _mimo_fused_qkv_source_shard_sizes,
    _select_mimo_fused_qkv_source_shards,
    load_mimo_v2_qkv_proj_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def test_mimo_v2_pro_source_qkv_layout():
    assert _mimo_fused_qkv_source_shard_sizes(
        total_num_heads=128,
        total_num_kv_heads=8,
        head_size=192,
        v_head_size=128,
        source_tp_size=8,
    ) == (3072, 192, 128)


@pytest.mark.parametrize("target_tp_size", [1, 2, 4, 8])
def test_mimo_fused_qkv_source_tp_accepts_coarsening_divisors(target_tp_size):
    config = SimpleNamespace(
        attention_projection_layout="fused_qkv", num_key_value_heads=8
    )
    source_tp_size = get_mimo_v2_fused_qkv_expected_tp_size(config)
    assert can_mimo_v2_fused_qkv_reshard(source_tp_size, target_tp_size)


@pytest.mark.parametrize("target_tp_size", [0, 3, 16])
def test_mimo_fused_qkv_source_tp_rejects_non_coarsening_topologies(
    target_tp_size,
):
    assert not can_mimo_v2_fused_qkv_reshard(8, target_tp_size)


def test_select_mimo_fused_qkv_source_shards_for_target_rank():
    loaded = torch.arange(20, dtype=torch.float32).view(20, 1)
    selected = _select_mimo_fused_qkv_source_shards(
        loaded,
        source_tp_size=4,
        target_tp_size=2,
        target_tp_rank=1,
        source_shard_size=5,
    )
    assert selected.shape == (2, 5, 1)
    assert selected[:, :, 0].tolist() == [
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ]


def test_unquantized_coarsened_qkv_matches_canonical_local_projection():
    layer = MiMoV2FusedQKVParallelLinear(
        hidden_size=3,
        head_size=2,
        total_num_heads=16,
        total_num_kv_heads=8,
        source_tp_size=8,
        v_head_size=1,
        quant_config=None,
        tp_rank=0,
        tp_size=1,
    )
    loaded = torch.arange(56 * 3, dtype=torch.float32).view(56, 3)
    load_mimo_v2_qkv_proj_weight(
        "model.layers.0.self_attn.qkv_proj.weight",
        layer.weight,
        loaded,
        expected_fused_tp_size=8,
    )

    source = loaded.view(8, 7, 3)
    canonical_local_weight = torch.cat(
        (
            source[:, :4].flatten(0, 1),
            source[:, 4:6].flatten(0, 1),
            source[:, 6:7].flatten(0, 1),
        ),
        dim=0,
    ).to(torch.bfloat16)
    input_ = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.bfloat16)

    actual, bias = layer(input_)
    expected = torch.nn.functional.linear(input_, canonical_local_weight)
    assert bias is None
    assert layer.weight.dtype == torch.bfloat16
    assert torch.equal(layer.weight, canonical_local_weight)
    assert torch.equal(actual, expected)


def _block_fp8_layer(*, tp_rank=0, tp_size=1):
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    return MiMoV2FusedQKVParallelLinear(
        hidden_size=128,
        head_size=2,
        total_num_heads=8,
        total_num_kv_heads=4,
        source_tp_size=4,
        v_head_size=1,
        quant_config=quant_config,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def _block_fp8_checkpoint_tensors():
    weight = torch.zeros((28, 128), dtype=torch.float8_e4m3fn)
    for source_rank in range(4):
        weight[source_rank * 7 : (source_rank + 1) * 7].fill_(source_rank + 1)
    scale = torch.tensor([[0.5], [1.0], [2.0], [4.0]])
    return weight, scale


def test_block_fp8_coarsened_qkv_dequantizes_and_regroups_after_pair_load():
    layer = _block_fp8_layer()
    loaded_weight, loaded_scale = _block_fp8_checkpoint_tensors()

    load_mimo_v2_qkv_proj_weight("qkv_proj.weight", layer.weight, loaded_weight, 4)
    assert hasattr(layer.weight, "mimo_fused_qkv_pending_weight")
    load_mimo_v2_qkv_proj_weight(
        "qkv_proj.weight_scale_inv", layer.weight, loaded_scale, 4
    )

    assert layer.weight.shape == (28, 128)
    assert layer.weight.dtype == torch.bfloat16
    assert layer.quant_method.__class__.__name__ == "UnquantizedLinearMethod"
    assert not hasattr(layer, "weight_scale_inv")
    assert not hasattr(layer.weight, "mimo_fused_qkv_pending_weight")
    assert not hasattr(layer.weight, "mimo_fused_qkv_pending_scale")

    expected_source_values = [0.5, 2.0, 6.0, 16.0]
    q, k, v = layer.weight.split((16, 8, 4), dim=0)
    assert q[:, 0].tolist() == sum(
        ([value] * 4 for value in expected_source_values), []
    )
    assert k[:, 0].tolist() == sum(
        ([value] * 2 for value in expected_source_values), []
    )
    assert v[:, 0].tolist() == expected_source_values

    projected_q, projected_k, projected_v = layer.forward_qkv(
        torch.ones((1, 128), dtype=torch.bfloat16)
    )
    assert projected_q.shape == (1, 16)
    assert projected_k.shape == (1, 8)
    assert projected_v.shape == (1, 4)


def test_block_fp8_qkv_pairing_is_safe_when_scale_arrives_first():
    layer = _block_fp8_layer(tp_rank=1, tp_size=2)
    loaded_weight, loaded_scale = _block_fp8_checkpoint_tensors()

    load_mimo_v2_qkv_proj_weight(
        "qkv_proj.weight_scale_inv", layer.weight, loaded_scale, 4
    )
    assert hasattr(layer.weight, "mimo_fused_qkv_pending_scale")
    load_mimo_v2_qkv_proj_weight("qkv_proj.weight", layer.weight, loaded_weight, 4)

    q, k, v = layer.weight.split((8, 4, 2), dim=0)
    assert q[:, 0].tolist() == [6.0] * 4 + [16.0] * 4
    assert k[:, 0].tolist() == [6.0] * 2 + [16.0] * 2
    assert v[:, 0].tolist() == [6.0, 16.0]
    assert not hasattr(layer.weight, "mimo_fused_qkv_pending_weight")
    assert not hasattr(layer.weight, "mimo_fused_qkv_pending_scale")


def test_tp8_existing_sharded_qkv_load_path_is_unchanged():
    loaded = torch.arange(12, dtype=torch.float32).view(3, 4)
    param = torch.nn.Parameter(torch.zeros_like(loaded), requires_grad=False)
    load_mimo_v2_qkv_proj_weight(
        "model.layers.0.self_attn.qkv_proj.weight",
        param,
        loaded,
        expected_fused_tp_size=8,
    )
    assert torch.equal(param, loaded)


def test_select_mimo_fused_qkv_rejects_non_divisor_target_tp():
    with pytest.raises(ValueError, match="Cannot coarsen"):
        _select_mimo_fused_qkv_source_shards(
            torch.zeros(40, 1),
            source_tp_size=8,
            target_tp_size=3,
            target_tp_rank=0,
            source_shard_size=5,
        )

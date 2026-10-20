from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers import communicator
from sglang.srt.layers.moe import topk as moe_topk
from sglang.srt.layers.moe.moe_runner import aiter as aiter_runner
from sglang.srt.layers.moe.token_dispatcher import standard
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import mimo_v2


def _decoder_stub(weight_dtype: torch.dtype):
    layer = mimo_v2.MiMoV2DecoderLayer.__new__(mimo_v2.MiMoV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.mlp = SimpleNamespace(
        _enable_a2a_moe=False,
        experts=SimpleNamespace(w13_weight=torch.empty(1, dtype=weight_dtype)),
    )
    return layer


@pytest.mark.parametrize(
    (
        "enabled",
        "gfx95",
        "gfx942",
        "backend",
        "ep_size",
        "weight_dtype",
        "expected",
    ),
    [
        (False, True, False, "aiter", 1, torch.float8_e4m3fn, ""),
        (True, False, False, "aiter", 1, torch.float8_e4m3fn, ""),
        (True, False, True, "aiter", 1, torch.float8_e4m3fn, "fp8_moe"),
        (True, True, False, "triton", 1, torch.float8_e4m3fn, ""),
        (True, True, False, "aiter", 2, torch.float8_e4m3fn, ""),
        (True, True, False, "aiter", 1, torch.bfloat16, ""),
        (True, True, False, "auto", 1, torch.float8_e4m3fn, "fp8_moe"),
        (True, True, False, "aiter", 1, torch.float8_e4m3fn, "fp8_moe"),
    ],
)
def test_mimo_fused_rms_moe_quant_selector(
    monkeypatch, enabled, gfx95, gfx942, backend, ep_size, weight_dtype, expected
):
    monkeypatch.setattr(mimo_v2, "is_gfx95_supported", lambda: gfx95)
    monkeypatch.setattr(mimo_v2, "is_gfx942_supported", lambda: gfx942)
    monkeypatch.setattr(
        mimo_v2,
        "get_moe_runner_backend",
        lambda: SimpleNamespace(
            is_aiter=lambda: backend == "aiter",
            is_auto=lambda: backend == "auto",
        ),
    )
    monkeypatch.setattr(mimo_v2, "get_moe_expert_parallel_world_size", lambda: ep_size)
    layer = _decoder_stub(weight_dtype)

    with mimo_v2.envs.SGLANG_MIMO_FUSED_RMS_MOE_QUANT.override(enabled):
        assert layer._detect_fused_rms_moe_quant_format() == expected


def test_mimo_moe_inputs_preserve_router_bf16_and_prequantized_expert_input():
    normalized = torch.empty(7, 6144, dtype=torch.bfloat16)
    quantized = torch.empty(7, 6144, dtype=torch.float8_e4m3fn)
    scale = torch.empty(7, 48, dtype=torch.float32)

    assert mimo_v2._mimo_moe_inputs(normalized) == (normalized, normalized)
    router_input, expert_input = mimo_v2._mimo_moe_inputs(
        (normalized, quantized, scale)
    )
    assert router_input is normalized
    assert expert_input == (quantized, scale)


@pytest.mark.parametrize("convert", [False, True])
def test_mimo_packed_experts_select_fused_quant_before_conversion(monkeypatch, convert):
    monkeypatch.setattr(mimo_v2, "is_gfx942_supported", lambda: True)
    monkeypatch.setattr(mimo_v2, "get_moe_expert_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        mimo_v2,
        "get_moe_runner_backend",
        lambda: SimpleNamespace(
            is_aiter=lambda: True,
            is_auto=lambda: False,
        ),
    )
    layer = _decoder_stub(torch.uint8)
    layer.mlp.experts.quant_method = SimpleNamespace(
        convert_mxfp4_to_fp8=convert,
    )
    with (
        mimo_v2.envs.SGLANG_USE_AITER.override(True),
        mimo_v2.envs.SGLANG_MIMO_FUSED_RMS_MOE_QUANT.override(True),
        mimo_v2.envs.SGLANG_MXFP4_TO_FP8_128BLOCK.override(convert),
    ):
        assert layer._detect_fused_rms_moe_quant_format() == (
            "fp8_moe" if convert else ""
        )


def test_mimo_noaux_topk_reweights_from_logsigmoid_without_underflow():
    router_logits = torch.tensor([[-1000.0, -1001.0, -1002.0, -1003.0]])
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)

    actual = moe_topk._stable_noaux_sigmoid_topk_weights(
        router_logits,
        topk_ids,
        renormalize=True,
    )
    expected = torch.softmax(
        torch.nn.functional.logsigmoid(router_logits.gather(1, topk_ids.long())),
        dim=-1,
    )

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.sum(dim=-1), torch.ones(1))
    assert torch.allclose(actual, expected)


def test_mimo_noaux_stable_reweighting_is_opt_in():
    topk = moe_topk.TopK(
        top_k=2,
        use_grouped_topk=True,
        num_expert_group=2,
        topk_group=1,
        correction_bias=torch.zeros(4),
    )

    assert topk.topk_config.stable_noaux_sigmoid_weights is False


@pytest.mark.parametrize(
    ("forward_mode", "expected"),
    [
        (ForwardMode.EXTEND, "fp8_moe"),
        (ForwardMode.MIXED, "fp8_moe"),
        (ForwardMode.DRAFT_EXTEND_V2, "fp8_moe"),
        (ForwardMode.DECODE, ""),
        (ForwardMode.TARGET_VERIFY, ""),
    ],
)
def test_mimo_moe_quant_is_context_prefill_only(forward_mode, expected):
    layer = _decoder_stub(torch.float8_e4m3fn)
    layer._fused_rms_moe_quant_format = "fp8_moe"
    forward_batch = SimpleNamespace(forward_mode=forward_mode)
    assert layer._prefill_moe_quant_format(forward_batch) == expected


def test_fused_moe_norm_adapter_returns_bf16_fp8_scale_and_residual(monkeypatch):
    hidden = torch.randn(3, 6144)
    residual = torch.randn(3, 6144)
    calls = []

    def fake_fused(*args):
        calls.append(args)

    monkeypatch.setattr(
        communicator, "_aiter_mimo_add_rmsnorm_fp8_group_quant", fake_fused
    )
    layernorm = SimpleNamespace(weight=torch.ones(6144), variance_epsilon=1e-6)
    adapter = communicator._FusedRMSNormFP8GroupQuantForMoe(layernorm)

    output, actual_residual = adapter(hidden, residual)
    normalized, quantized, scale = output
    assert normalized.shape == hidden.shape
    assert normalized.dtype == hidden.dtype
    assert quantized.shape == hidden.shape
    assert quantized.dtype == communicator._aiter_fp8_dtype
    assert scale.shape == (3, 48)
    assert scale.dtype == torch.float32
    assert actual_residual.shape == hidden.shape
    assert calls[0][0] is quantized
    assert calls[0][1] is normalized
    assert calls[0][2] is scale
    assert calls[0][3] is hidden
    assert calls[0][4] is residual
    assert calls[0][5] is actual_residual
    assert calls[0][6] is layernorm.weight
    assert calls[0][7] == layernorm.variance_epsilon
    assert scale._aiter_moe_scale_is_transposed is True


def test_prepare_mlp_wraps_layernorm_only_for_supported_contract(monkeypatch):
    captured = []
    layer = communicator.LayerCommunicator.__new__(communicator.LayerCommunicator)
    layer._context = SimpleNamespace(attn_dp_size=1)
    layer.post_attention_layernorm = object()
    layer._communicate_with_all_reduce_and_layer_norm_fn = lambda **kwargs: (
        captured.append(kwargs["layernorm"]) or ("hidden", "residual")
    )
    monkeypatch.setattr(communicator, "_use_aiter", True)
    monkeypatch.setattr(communicator, "_is_gfx95_supported", True)
    monkeypatch.setattr(communicator, "get_moe_cp_size", lambda: 1)

    layer.prepare_mlp("hidden", "residual", object(), quant_format="fp8_moe")
    assert isinstance(captured[-1], communicator._FusedRMSNormFP8GroupQuantForMoe)

    layer._context.attn_dp_size = 2
    layer.prepare_mlp("hidden", "residual", object(), quant_format="fp8_moe")
    assert captured[-1] is layer.post_attention_layernorm


def test_standard_dispatch_preserves_prequantized_activation_scale(monkeypatch):
    monkeypatch.setattr(
        standard, "should_use_flashinfer_cutlass_moe_fp4_allgather", lambda: False
    )
    dispatcher = standard.StandardDispatcher.__new__(standard.StandardDispatcher)
    dispatcher.moe_ep_size = 1
    dispatcher.skip_local_expert_mapping = False
    dispatcher.local_expert_mapping = None
    quantized = torch.empty(4, 16, dtype=torch.float8_e4m3fn)
    scale = torch.empty(4, 1)
    scale._aiter_moe_scale_is_transposed = True
    topk_output = object()

    output = dispatcher.dispatch((quantized, scale), topk_output)
    assert output.hidden_states is quantized
    assert output.hidden_states_scale is scale
    assert output.hidden_states_scale._aiter_moe_scale_is_transposed is True
    assert output.topk_output is topk_output


def test_aiter_standard_pre_permute_forwards_activation_scale():
    quantized = torch.empty(4, 16, dtype=torch.float8_e4m3fn)
    scale = torch.empty(4, 1)
    scale._aiter_moe_scale_is_transposed = True
    topk_weights = torch.ones(4, 1)
    topk_ids = torch.zeros(4, 1, dtype=torch.int64)
    dispatch_output = standard.StandardDispatchOutput(
        hidden_states=quantized,
        hidden_states_scale=scale,
        topk_output=(topk_weights, topk_ids, None),
    )
    quant_info = SimpleNamespace(
        doweight_stage1=False,
        quant_type=aiter_runner.AiterQuantType.PER_128X128,
    )
    runner_config = SimpleNamespace(apply_router_weight_on_input=False)

    output = aiter_runner.pre_permute_standard_to_aiter(
        dispatch_output, quant_info, runner_config, {}
    )
    assert output.hidden_states is quantized
    assert output.a1_scale is scale
    assert output.a1_scale_is_transposed is True
    assert output.topk_ids.dtype == torch.int32
    assert output.output_dtype == torch.bfloat16

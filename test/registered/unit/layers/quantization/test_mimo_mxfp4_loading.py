import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.configs.model_config import is_mimo_v2_mxfp4
from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.aiter import AiterMoeQuantInfo
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMiMoMXFP4Loading(CustomTestCase):
    def test_fp8_conversion_updates_runtime_dispatch_only(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            is_fp4_experts=True,
            store_dtype="mxfp4",
        )
        layer = torch.nn.Module()
        layer.dispatcher = SimpleNamespace(expert_mask_gpu=None)
        for name in ("w13", "w2"):
            layer.register_parameter(
                name + "_weight",
                torch.nn.Parameter(
                    torch.zeros((2, 256, 128), dtype=torch.uint8), requires_grad=False
                ),
            )
            layer.register_parameter(
                name + "_weight_scale_inv",
                torch.nn.Parameter(
                    torch.full((2, 256, 8), 127, dtype=torch.uint8), requires_grad=False
                ),
            )
        q = torch.ones((2, 256, 256)).to(torch.float8_e4m3fnuz)
        s = torch.full((2, 2, 2), 0.03125)
        convert = Mock(return_value=(q, s))
        original_params = dict(layer.named_parameters())
        for param in original_params.values():
            param.weight_loader = Mock()
            param.quant_method = "block"
        module = "sglang.srt.layers.quantization.fp8"
        with (
            patch.dict(os.environ, {"SGLANG_MXFP4_TO_FP8_128BLOCK": "1"}),
            patch(module + "._is_hip", True),
            patch(module + "._use_aiter", True),
            patch(module + "._use_hip_int4", False),
            patch.dict(
                sys.modules,
                {"aiter.ops.flydsl": SimpleNamespace(mxfp4_to_fp8_128block=convert)},
            ),
            patch(module + ".shuffle_weight", side_effect=lambda w, _: w, create=True),
            patch(module + ".normalize_e4m3fn_to_e4m3fnuz") as normalize,
        ):
            method = Fp8MoEMethod(config)
            method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.AITER)
            method.moe_runner_config = MoeRunnerConfig()
            method.process_weights_after_loading_block_quant(layer)
            method.process_weights_after_loading_block_quant(layer)
            info = method.maybe_get_hip_aiter_quant_info(layer)
        self.assertEqual(convert.call_count, 2)
        normalize.assert_not_called()
        self.assertFalse(method.is_fp4_expert)
        self.assertTrue(config.is_fp4_experts)
        self.assertEqual(config.store_dtype, "mxfp4")
        self.assertFalse(info.is_fp4_experts)
        self.assertIsInstance(info, AiterMoeQuantInfo)
        self.assertEqual(info.quant_type.value, "per_128x128")
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fnuz)
        self.assertTrue(layer.w13_weight.is_shuffled)
        self.assertFalse(layer.w13_weight_scale_inv.format_ue8m0)
        for name, param in layer.named_parameters():
            self.assertIs(param, original_params[name])
            self.assertTrue(hasattr(param, "weight_loader"))
            self.assertEqual(param.quant_method, "block")
        torch.testing.assert_close(layer.w13_weight_scale_inv, s, rtol=0, atol=0)

    def test_conversion_switch_defaults_off(self):
        from sglang.srt.environ import envs

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(envs.SGLANG_MXFP4_TO_FP8_128BLOCK.get())

    def test_detects_native_mxfp4_checkpoint(self):
        config = {
            "architectures": ["MiMoV2ForCausalLM"],
            "quantization_config": {
                "quant_method": "fp8",
                "store_dtype": "mxfp4",
            },
        }
        self.assertTrue(is_mimo_v2_mxfp4(config))

        config["quantization_config"]["store_dtype"] = "fp8"
        self.assertFalse(is_mimo_v2_mxfp4(config))

    def test_mxfp4_checkpoint_uses_raw_byte_storage(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            is_fp4_experts=True,
            store_dtype="mxfp4",
        )
        method = Fp8MoEMethod(config)
        layer = torch.nn.Module()
        layer.moe_runner_config = SimpleNamespace(is_gated=True)

        with patch(
            "sglang.srt.layers.quantization.fp8.get_tensor_model_parallel_world_size",
            return_value=1,
        ):
            method.create_weights(
                layer=layer,
                num_experts=2,
                hidden_size=128,
                intermediate_size_per_partition=128,
                params_dtype=torch.bfloat16,
            )

        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale_inv.dtype, torch.uint8)


@pytest.fixture
def conversion_method(monkeypatch):
    from sglang.srt.layers.quantization import fp8

    monkeypatch.setenv("SGLANG_MXFP4_TO_FP8_128BLOCK", "1")
    monkeypatch.setattr(fp8, "_is_hip", True)
    monkeypatch.setattr(fp8, "_use_aiter", True)
    monkeypatch.setattr(fp8, "_use_hip_int4", False)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
        is_fp4_experts=True,
        store_dtype="mxfp4",
    )
    method = Fp8MoEMethod(config)
    method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.AITER)
    method.moe_runner_config = MoeRunnerConfig()
    return method


@pytest.mark.parametrize("hidden,inter", [(6144, 64), (160, 256)])
def test_conversion_rejects_partial_moe_tiles(conversion_method, hidden, inter):
    layer = torch.nn.Module()
    layer.moe_runner_config = MoeRunnerConfig()
    with pytest.raises(ValueError, match="divisible by 128"):
        conversion_method.create_weights(layer, 2, hidden, inter, torch.bfloat16)
    assert not list(layer.parameters())


@pytest.mark.parametrize("with_bias,gated", [(True, True), (False, False)])
def test_conversion_rejects_unsupported_moe_contract(
    conversion_method, with_bias, gated
):
    layer = torch.nn.Module()
    layer.moe_runner_config = MoeRunnerConfig(is_gated=gated)
    with pytest.raises(ValueError, match="bias-free gated"):
        conversion_method.create_weights(
            layer, 2, 256, 128, torch.bfloat16, with_bias=with_bias
        )


@pytest.mark.parametrize(
    "backend", [MoeRunnerBackend.TRITON, MoeRunnerBackend.DEEP_GEMM]
)
def test_conversion_rejects_non_aiter_runner(conversion_method, monkeypatch, backend):
    from sglang.srt.layers.quantization import fp8

    monkeypatch.setattr(fp8, "get_moe_runner_backend", lambda: backend)
    with pytest.raises(ValueError, match="AITER MoE runner"):
        conversion_method.create_moe_runner(torch.nn.Module(), MoeRunnerConfig())


def test_conversion_rejects_auto_fallback(conversion_method, monkeypatch):
    from sglang.srt.layers.moe.utils import MoeA2ABackend
    from sglang.srt.layers.quantization import fp8

    monkeypatch.setattr(fp8, "get_moe_runner_backend", lambda: MoeRunnerBackend.AUTO)
    monkeypatch.setattr(fp8, "get_moe_a2a_backend", lambda: MoeA2ABackend.MEGAMOE)
    monkeypatch.setattr(
        conversion_method, "is_deepgemm_moe_runner_backend_enabled", lambda: False
    )
    with pytest.raises(ValueError, match="AITER MoE runner"):
        conversion_method.create_moe_runner(torch.nn.Module(), MoeRunnerConfig())


def test_conversion_rejects_clamped_runner(conversion_method, monkeypatch):
    from sglang.srt.layers.quantization import fp8

    monkeypatch.setattr(fp8, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER)
    with pytest.raises(ValueError, match="interleaved SwiGLU"):
        conversion_method.create_moe_runner(
            torch.nn.Module(), MoeRunnerConfig(swiglu_limit=7.0)
        )


@pytest.mark.parametrize(
    "hip,aiter,int4", [(False, True, False), (True, False, False), (True, True, True)]
)
def test_conversion_rejects_incompatible_environment(
    conversion_method, monkeypatch, hip, aiter, int4
):
    from sglang.srt.layers.quantization import fp8

    monkeypatch.setattr(fp8, "_is_hip", hip)
    monkeypatch.setattr(fp8, "_use_aiter", aiter)
    monkeypatch.setattr(fp8, "_use_hip_int4", int4)
    with pytest.raises(ValueError, match="SGLANG_INT4_WEIGHT=0"):
        Fp8MoEMethod(conversion_method.quant_config)


def test_conversion_rejects_incompatible_blocks(conversion_method):
    conversion_method.quant_config.weight_block_size = [64, 128]
    with pytest.raises(ValueError, match="weight_block_size"):
        Fp8MoEMethod(conversion_method.quant_config)


@pytest.mark.parametrize("fused", [False, True])
def test_converted_weights_reject_raw_checkpoint_reload(fused):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    layer = SimpleNamespace(_mxfp4_to_fp8_converted=True)
    args = (torch.empty(0), torch.empty(0), "w13_weight", "w1")
    with pytest.raises(RuntimeError, match="recreate the model"):
        if fused:
            FusedMoE.weight_loader_fused(layer, *args)
        else:
            FusedMoE.weight_loader(layer, *args, 0)


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="Requires ROCm GPU"
)
@pytest.mark.parametrize(
    "tokens,hidden,inter", [(1, 256, 128), (1, 6144, 256), (8192, 6144, 256)]
)
def test_conversion_through_aiter_runner(
    conversion_method, monkeypatch, tokens, hidden, inter
):
    from aiter import QuantType
    from aiter.fused_moe import get_quant, torch_moe
    from sglang.srt.layers.moe.moe_runner.aiter import (
        AiterRunnerCore,
        pre_permute_standard_to_aiter,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.quantization import fp8

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("Requires gfx942 or gfx950")
    # CPU CI imports fp8.py without SGLANG_USE_AITER; bind the real shuffle here.
    from aiter.ops.shuffle import shuffle_weight

    monkeypatch.setattr(fp8, "shuffle_weight", shuffle_weight, raising=False)
    monkeypatch.setattr(fp8, "get_tensor_model_parallel_world_size", lambda: 8)
    torch.manual_seed(719)
    experts, topk = 4, 2
    config = MoeRunnerConfig(top_k=topk, activation="silu")
    layer = torch.nn.Module()
    layer.moe_runner_config = config
    layer.dispatcher = SimpleNamespace(expert_mask_gpu=None, set_quant_config=Mock())
    conversion_method.create_weights(layer, experts, hidden, inter, torch.bfloat16)
    layer.cuda()
    conversion_method.moe_runner_config = config
    reference = []
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    for prefix in ("w13", "w2"):
        packed = getattr(layer, prefix + "_weight")
        codes = getattr(layer, prefix + "_weight_scale_inv")
        packed.random_(0, 256)
        codes.random_(118, 123)
        nibbles = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
        reference.append(
            lut[nibbles] * torch.exp2(codes.float() - 127).repeat_interleave(32, -1)
        )
    conversion_method.process_weights_after_loading(layer)
    layer.dispatcher.set_quant_config.assert_called_once_with(
        {"weight_dtype": layer.w13_weight.dtype}
    )
    info = conversion_method.maybe_get_hip_aiter_quant_info(layer)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    weights, ids = (
        torch.randn(tokens, experts, device="cuda").softmax(-1).topk(topk, -1)
    )
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    ids = ids.int()
    core = AiterRunnerCore(config)

    def run(activation, scale=None):
        dispatch = StandardDispatchOutput(
            hidden_states=activation,
            hidden_states_scale=scale,
            topk_output=(weights, ids, None),
        )
        runner_input = pre_permute_standard_to_aiter(dispatch, info, config, {})
        return core.run(runner_input, info, {}).hidden_states

    actual = run(x)
    expected = torch_moe(x, *reference, weights, ids)
    relative_rmse = (
        actual.float() - expected.float()
    ).square().mean().sqrt() / expected.float().square().mean().sqrt()
    assert torch.isfinite(actual).all()
    assert relative_rmse < 0.06, relative_rmse.item()
    xq, scale = get_quant(QuantType.per_1x128)(
        x, quant_dtype=layer.w13_weight.dtype, transpose_scale=True
    )
    scale._aiter_moe_scale_is_transposed = True
    torch.testing.assert_close(run(xq, scale), actual, rtol=0.02, atol=0.01)
    if tokens == 1:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(x)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0.02, atol=0.01)


@pytest.mark.parametrize("tp_rank", [0, 7])
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_raw_checkpoint_tp_ep_shards(conversion_method, monkeypatch, tp_rank, ep_rank):
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe_layer
    from sglang.srt.layers.quantization import fp8

    monkeypatch.setattr(fp8, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(moe_layer, "get_global_expert_location_metadata", lambda: None)
    layer = moe_layer.FusedMoE.__new__(moe_layer.FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_runner_config = MoeRunnerConfig()
    layer.quant_method = conversion_method
    layer.quant_config = conversion_method.quant_config
    layer.moe_tp_rank, layer.moe_ep_rank = tp_rank, ep_rank
    layer._num_local_routed, layer._num_global_routed = 2, 4
    layer._has_fused_shared = False
    layer.use_triton_kernels = False
    layer.use_flashinfer_trtllm_moe = False
    layer.use_padded_loading = False
    layer.use_presharded_weights = False
    conversion_method.create_weights(
        layer, 2, 128, 128, torch.bfloat16, weight_loader=layer.weight_loader
    )
    generator = torch.Generator().manual_seed(719)
    expected = {
        name: torch.empty_like(param) for name, param in layer.named_parameters()
    }
    for expert in range(4):
        for shard, prefix, full_shape in (
            ("w1", "w13", (1024, 128)),
            ("w3", "w13", (1024, 128)),
            ("w2", "w2", (128, 1024)),
        ):
            for suffix, divisor in (("_weight", 2), ("_weight_scale_inv", 32)):
                name = prefix + suffix
                loaded = torch.randint(
                    0,
                    255,
                    (full_shape[0], full_shape[1] // divisor),
                    dtype=torch.uint8,
                    generator=generator,
                )
                param = getattr(layer, name)
                param.weight_loader(param, loaded, name, shard, expert)
                if expert // 2 == ep_rank:
                    local_expert = expert % 2
                    if shard == "w2":
                        expected[name][local_expert] = loaded[
                            :,
                            tp_rank
                            * (128 // divisor) : (tp_rank + 1)
                            * (128 // divisor),
                        ]
                    else:
                        row = 0 if shard == "w1" else 128
                        expected[name][local_expert, row : row + 128] = loaded[
                            tp_rank * 128 : (tp_rank + 1) * 128
                        ]
    for name, param in layer.named_parameters():
        torch.testing.assert_close(param, expected[name], atol=0, rtol=0)


def test_conversion_missing_api_is_actionable(conversion_method, monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", SimpleNamespace())
    with pytest.raises(
        ImportError, match="aiter.ops.flydsl.mxfp4_to_fp8_128block"
    ) as error:
        conversion_method._convert_mxfp4_to_fp8_128block(torch.nn.Module())
    assert isinstance(error.value.__cause__, ImportError)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

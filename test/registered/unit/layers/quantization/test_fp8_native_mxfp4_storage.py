from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFp8NativeMxfp4Storage(CustomTestCase):
    def test_config_preserves_checkpoint_storage_dtype(self):
        config = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "store_dtype": "mxfp4",
            }
        )

        self.assertEqual(config.store_dtype, "mxfp4")

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
        self.assertEqual(layer.w13_weight.shape, (2, 256, 64))
        self.assertEqual(layer.w2_weight.shape, (2, 128, 64))
        self.assertEqual(layer.w13_weight_scale_inv.shape, (2, 256, 4))
        self.assertEqual(layer.w2_weight_scale_inv.shape, (2, 128, 4))

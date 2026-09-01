# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Platform-specific fused MegaMoE forward paths and expert-weight prep."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    get_dp_global_num_tokens,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.utils import is_gfx95_supported, is_hip

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


_MEGA_MOE_SYMM_BUFFER: dict = {}
_MEGA_MOE_DG_ENV_APPLIED = False
_FLYDSL_MEGA_MOE_OPS: dict = {}
_FLYDSL_MORI_INITIALIZED = False

logger = logging.getLogger(__name__)


def _using_flydsl_mega_moe(moe=None) -> bool:
    return is_hip() and (
        moe is None or getattr(moe.experts, "_use_flydsl_mega_moe", False)
    )


def _import_flydsl_mega_moe():
    try:
        from kernels.mega_moe import MegaMoEV2
        from kernels.mega_moe.quant import per_1x32_mx_quant
    except ImportError as exc:
        raise ImportError(
            "FlyDSL MegaMoE kernels are not importable. Install/export the FlyDSL "
            "kernel package, or add the FlyDSL repository root to PYTHONPATH "
            "(for this container: PYTHONPATH=/root/workspace/xiaomi/FlyDSL)."
        ) from exc
    return MegaMoEV2, per_1x32_mx_quant


def _shuffle_mxfp4_weight(
    src: torch.Tensor, *, experts: int, gate_up: bool
) -> torch.Tensor:
    """Convert row-major packed MXFP4 into MegaMoE's N-major layout."""
    src_type = src.dtype
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None and src_type == fp4_dtype:
        src = src.view(torch.uint8)

    if src.ndim != 3 or src.shape[0] != experts:
        raise ValueError(
            f"Expected expert-major MXFP4 weights with shape [E, N, K/2], got {tuple(src.shape)}"
        )
    _, rows, packed_cols = src.shape
    logical_rows = rows // 2 if gate_up else rows
    if gate_up and rows % 2:
        raise ValueError(f"Gate/up row count must be even, got {rows}")

    n_lane = 16
    k_pack = 16
    k_lane = 64 // n_lane
    if logical_rows % n_lane or packed_cols % (k_lane * k_pack):
        raise ValueError(
            "FlyDSL MegaMoE MXFP4 weights require N divisible by 16 and "
            f"packed K divisible by 64, got N={logical_rows}, packed_K={packed_cols}"
        )

    n_outer = logical_rows // n_lane
    k_outer = packed_cols // (k_lane * k_pack)
    if gate_up:
        shuffled = (
            src.view(experts, 2, n_outer, n_lane, k_outer, k_lane, k_pack)
            .permute(0, 2, 1, 4, 5, 3, 6)
            .contiguous()
        )
    else:
        shuffled = (
            src.view(experts, n_outer, n_lane, k_outer, k_lane, k_pack)
            .permute(0, 1, 3, 4, 2, 5)
            .contiguous()
        )
    return shuffled.view_as(src).view(src_type)


def _shuffle_mxfp4_scale(
    src: torch.Tensor, *, experts: int, gate_up: bool
) -> torch.Tensor:
    """Convert row-major E8M0 scales into MegaMoE's preshuffled layout."""
    if src.ndim != 3 or src.shape[0] != experts:
        raise ValueError(
            f"Expected expert-major MXFP4 scales with shape [E, N, K/32], got {tuple(src.shape)}"
        )
    _, rows, scale_cols = src.shape

    k_pack = 2
    n_pack = 2
    n_lane = 16
    k_lane = 64 // n_lane
    if scale_cols % (k_pack * k_lane) or rows % (n_lane * n_pack):
        raise ValueError(
            "FlyDSL MegaMoE scales require N divisible by 32 and K/32 "
            f"divisible by 8, got rows={rows}, K/32={scale_cols}"
        )

    k_outer = scale_cols // k_pack // k_lane
    n_outer = rows // n_lane // n_pack
    if gate_up:
        shuffled = (
            src.view(experts, n_pack, n_outer, n_lane, k_outer, k_pack, k_lane)
            .permute(0, 2, 4, 6, 3, 5, 1)
            .contiguous()
        )
    else:
        shuffled = (
            src.view(experts, n_outer, n_pack, n_lane, k_outer, k_pack, k_lane)
            .permute(0, 1, 4, 6, 3, 5, 2)
            .contiguous()
        )
    return shuffled.view_as(src)


def _fp8_blockwise_to_mxfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: list[int],
    *,
    gate_up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize one local expert stack from block FP8 to packed MXFP4."""
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    _, per_1x32_mx_quant = _import_flydsl_mega_moe()

    if list(block_size) != [128, 128]:
        raise ValueError(
            "FlyDSL MegaMoE online conversion currently requires FP8 "
            f"weight_block_size=[128, 128], got {block_size}"
        )
    if weight.ndim != 3 or scale.ndim != 3:
        raise ValueError(
            "FlyDSL MegaMoE expects 3-D expert weights/scales, got "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
        )
    if not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("FlyDSL MegaMoE source weights and scales must be contiguous")
    if weight.device != scale.device:
        raise ValueError(
            "FlyDSL MegaMoE source weights and scales must be on the same device"
        )

    experts, rows, cols = weight.shape
    block_n, block_k = block_size
    scale_rows = (
        2 * ((rows // 2 + block_n - 1) // block_n)
        if gate_up
        else (rows + block_n - 1) // block_n
    )
    expected_scale_shape = (
        experts,
        scale_rows,
        (cols + block_k - 1) // block_k,
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "Unexpected FP8 block-scale shape: "
            f"got {tuple(scale.shape)}, expected {expected_scale_shape}"
        )
    if rows % 32 or cols % 256:
        raise ValueError(
            "FlyDSL MegaMoE requires expert rows divisible by 32 and K divisible "
            f"by 256, got rows={rows}, K={cols}"
        )

    qweight = torch.empty(
        (experts, rows, cols // 2), dtype=torch.uint8, device=weight.device
    )
    qscale = torch.empty(
        (experts, rows, cols // 32), dtype=torch.uint8, device=weight.device
    )
    for expert in range(experts):
        dequant = block_quant_dequant(
            weight[expert], scale[expert], block_size, torch.bfloat16
        ).contiguous()
        packed, e8m0 = per_1x32_mx_quant(dequant, quant_mode="fp4")
        qweight[expert].copy_(packed.view(torch.uint8))
        qscale[expert].copy_(e8m0.view(torch.uint8))

    return (
        _shuffle_mxfp4_weight(qweight, experts=experts, gate_up=gate_up),
        _shuffle_mxfp4_scale(qscale, experts=experts, gate_up=gate_up),
    )


def _init_flydsl_mori() -> None:
    global _FLYDSL_MORI_INITIALIZED
    if _FLYDSL_MORI_INITIALIZED:
        return

    import mori.shmem as ms
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    group = get_moe_ep_group()
    uid = ms.shmem_get_unique_id() if group.rank_in_group == 0 else None
    uid_list = [uid]
    dist.broadcast_object_list(uid_list, src=group.ranks[0], group=group.cpu_group)
    status = ms.shmem_init_attr(
        ms.MORI_SHMEM_INIT_WITH_UNIQUEID,
        group.rank_in_group,
        group.world_size,
        uid_list[0],
    )
    if status not in (None, 0):
        raise RuntimeError(f"MORI SHMEM initialization failed with status {status}")
    _FLYDSL_MORI_INITIALIZED = True


def _apply_mega_moe_dg_env() -> None:
    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.

    DeepGEMM reads `DG_USE_FP4_ACTS` (and `DG_USE_MXF4_KIND`) at host-function
    call time — both `get_symm_buffer_for_mega_moe` and `fp8_fp4_mega_moe`.
    Forwarding once at first use is sufficient (these are static config
    flags, not per-request state) and matches the `setdefault` pattern so
    explicit `DG_USE_*` overrides from outside still win.
    """
    global _MEGA_MOE_DG_ENV_APPLIED
    if _MEGA_MOE_DG_ENV_APPLIED:
        return
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        os.environ.setdefault("DG_USE_FP4_ACTS", "1")
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND.get():
        os.environ.setdefault("DG_USE_MXF4_KIND", "1")
    _MEGA_MOE_DG_ENV_APPLIED = True


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> SymmBuffer:
    import deep_gemm

    _apply_mega_moe_dg_env()

    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    weights_attr = (
        "_flydsl_mega_moe_weights_built"
        if _using_flydsl_mega_moe(moe)
        else "_mega_moe_weights_built"
    )
    if not getattr(moe.experts, weights_attr, False):
        if _using_flydsl_mega_moe(moe):
            raise RuntimeError(
                "FlyDSL MegaMoE was selected, but this MoE layer did not build "
                "compatible MXFP4 expert weights. The initial integration supports "
                "block-quantized FP8 routed experts only."
            )
        return False
    if get_is_capture_mode():
        return True

    if _using_flydsl_mega_moe(moe):
        # There is no valid local-expert fallback after conversion to MegaMoE's
        # private weight layout. The forward path reports capacity errors.
        return True

    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens:
        max_tokens_per_rank = max(global_num_tokens)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = (
        envs.SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
        if _using_flydsl_mega_moe(moe)
        else envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    return max_tokens_per_rank <= cap


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _using_flydsl_mega_moe(moe):
        return _forward_flydsl_mega_moe(moe, hidden_states, forward_batch)

    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _get_flydsl_mega_moe_op(moe, hidden_states: torch.Tensor):
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    MegaMoEV2, _ = _import_flydsl_mega_moe()
    group = get_moe_ep_group()
    experts = moe.experts
    max_tokens = envs.SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    if max_tokens <= 0 or max_tokens & (max_tokens - 1):
        raise ValueError(
            "SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK must be a positive "
            f"power of two, got {max_tokens}"
        )
    stage2_p2p_quant = envs.SGLANG_FLYDSL_MEGA_MOE_STAGE2_P2P_QUANT.get()
    if stage2_p2p_quant not in ("auto", "none", "fp8_blockwise_1x32"):
        raise ValueError(
            "SGLANG_FLYDSL_MEGA_MOE_STAGE2_P2P_QUANT must be one of "
            f"auto, none, fp8_blockwise_1x32; got {stage2_p2p_quant!r}"
        )

    key = (
        hidden_states.device.index,
        tuple(group.ranks),
        moe.config.hidden_size,
        moe.config.moe_intermediate_size,
        experts.num_experts,
        moe.config.num_experts_per_tok,
        max_tokens,
        stage2_p2p_quant,
        float(getattr(moe.config, "swiglu_limit", 0.0) or 0.0),
    )
    op = _FLYDSL_MEGA_MOE_OPS.get(key)
    if op is None:
        _init_flydsl_mori()
        op = MegaMoEV2(
            rank=group.rank_in_group,
            world_size=group.world_size,
            model_dim=moe.config.hidden_size,
            inter_dim=moe.config.moe_intermediate_size,
            experts=experts.num_experts,
            topk=moe.config.num_experts_per_tok,
            quant="a8w4",
            w1=experts.w13_weight,
            w1_scale=experts.w13_weight_scale_inv,
            w2=experts.w2_weight,
            w2_scale=experts.w2_weight_scale_inv,
            max_tok_per_rank=max_tokens,
            stage2_p2p_quant=stage2_p2p_quant,
            swiglu_limit=float(getattr(moe.config, "swiglu_limit", 0.0) or 0.0),
        )
        _FLYDSL_MEGA_MOE_OPS[key] = op

    # MegaMoEV2 owns the communication workspace. Rebind only the immutable
    # per-layer weight tensors so all compatible layers share that workspace.
    op._s1_w1 = experts.w13_weight
    op._s1_w1_scale = experts.w13_weight_scale_inv
    op.w2 = experts.w2_weight
    op.w2_scale = experts.w2_weight_scale_inv
    return op


def _forward_flydsl_mega_moe(
    moe,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
) -> torch.Tensor:
    if not is_gfx95_supported():
        raise RuntimeError("FlyDSL MegaMoE currently requires a gfx95x GPU")
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    logical_tokens = hidden_states.shape[0]
    max_tokens = envs.SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens:
        attention_tp_size = get_attention_tp_size()
        required_tokens = max(
            logical_tokens,
            (max(global_num_tokens) + attention_tp_size - 1) // attention_tp_size,
        )
    else:
        import torch.distributed as dist

        from sglang.srt.distributed.parallel_state import get_moe_ep_group

        required_tokens_tensor = torch.tensor(
            logical_tokens, dtype=torch.int32, device=hidden_states.device
        )
        dist.all_reduce(
            required_tokens_tensor,
            op=dist.ReduceOp.MAX,
            group=get_moe_ep_group().device_group,
        )
        required_tokens = int(required_tokens_tensor.item())
    if required_tokens > max_tokens:
        raise ValueError(
            f"FlyDSL MegaMoE requires capacity for {required_tokens} tokens/rank, exceeding "
            "SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
            f"{max_tokens}. Increase the power-of-two capacity or lower "
            "--chunked-prefill-size/--max-prefill-tokens."
        )

    # Every EP rank must enter the fused collective. Give idle ranks a single
    # zero-weight route so the FlyDSL quantizer never launches a zero-sized grid.
    if logical_tokens == 0:
        run_hidden = torch.zeros(
            (1, moe.config.hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        topk_ids = torch.zeros(
            (1, moe.config.num_experts_per_tok),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        topk_weights = torch.zeros(
            (1, moe.config.num_experts_per_tok),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    else:
        run_hidden = hidden_states.to(torch.bfloat16).contiguous()
        router_logits = moe.gate(run_hidden)
        topk_output = moe.topk(
            run_hidden,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
        )
        topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()

    op = _get_flydsl_mega_moe_op(moe, run_hidden)
    output = op.forward(run_hidden, topk_weights, topk_ids)
    return output[:logical_tokens]


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    hidden_size = moe.config.hidden_size

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    ep_group = get_moe_ep_group().device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or shrink "
        f"cuda_graph_max_bs / chunked_prefill_size accordingly"
    )

    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
    )

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    use_fp4_acts = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get()
    if use_fp4_acts:
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            use_fp4_acts=True,
        )
    else:
        from sglang.jit_kernel.dsv4 import mega_moe_pre_dispatch

        mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=32,
        )

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    deep_gemm.fp8_fp4_mega_moe(
        y,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32),
        activation="swiglu",
        activation_clamp=swiglu_limit,
        fast_math=True,
    )
    y = y[:num_tokens]

    if not moe.experts.should_fuse_routed_scaling_factor_in_topk:
        y.mul_(moe.routed_scaling_factor)
    return y


def build_mega_moe_experts_weights(experts) -> None:
    if is_hip() and getattr(experts, "_use_flydsl_mega_moe", False):
        return build_flydsl_mega_moe_experts_weights(experts)

    from deep_gemm import (
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
    )
    from deep_gemm.mega import _interleave_l1_weights, _transpose_sf_for_utccp

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    _, n2, half_k2 = w2.shape
    k2 = half_k2 * 2

    w13_sf = transform_sf_into_required_layout(
        w13_sf_fp32,
        mn=n1,
        k=k1,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    w2_sf = transform_sf_into_required_layout(
        w2_sf_fp32,
        mn=n2,
        k=k2,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        # Build the interleaved L1 weight + scale once; share the weight buffer
        # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
        # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
        # the deep-ep path consumes the non-transposed interleaved scale and a
        # swizzle-aware activation kernel. L2 weight is untouched by the mega
        # transform, so the existing `w2_weight.data` is shared directly.
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
        w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)

        experts.w13_weight.data = w13_interleaved
        experts.w13_weight_scale_inv.data = w13_sf_interleaved
        experts.w2_weight_scale_inv.data = w2_sf
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True

        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
        experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)
    else:
        l1_pair, l2_pair = transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))

        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_weights_built = True


def build_flydsl_mega_moe_experts_weights(experts) -> None:
    """Convert local FP8 block-scale experts to FlyDSL MegaMoE MXFP4."""
    if getattr(experts, "_flydsl_mega_moe_weights_built", False):
        return
    if not is_gfx95_supported():
        raise RuntimeError("FlyDSL MegaMoE currently requires a gfx95x GPU")

    block_size = getattr(experts.quant_config, "weight_block_size", None)
    if block_size is None:
        raise ValueError(
            "FlyDSL MegaMoE currently requires block-quantized FP8 source weights"
        )
    fp8_dtypes = {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
        )
        if dtype is not None
    }
    if (
        experts.w13_weight.dtype not in fp8_dtypes
        or experts.w2_weight.dtype not in fp8_dtypes
    ):
        raise ValueError(
            "FlyDSL MegaMoE online conversion expects FP8 expert weights, got "
            f"w13={experts.w13_weight.dtype}, w2={experts.w2_weight.dtype}"
        )

    from sglang.srt.distributed import get_moe_expert_parallel_rank

    if get_moe_expert_parallel_rank() == 0:
        logger.warning(
            "Converting layer %s EP-local expert weights from FP8 blockscale to "
            "MXFP4 for FlyDSL MegaMoE; validate end-to-end model accuracy after "
            "this lossy conversion.",
            experts.layer_id,
        )
    w13, s13 = _fp8_blockwise_to_mxfp4(
        experts.w13_weight.data,
        experts.w13_weight_scale_inv.data,
        block_size,
        gate_up=True,
    )
    w2, s2 = _fp8_blockwise_to_mxfp4(
        experts.w2_weight.data,
        experts.w2_weight_scale_inv.data,
        block_size,
        gate_up=False,
    )

    experts.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    experts.w13_weight_scale_inv = torch.nn.Parameter(s13, requires_grad=False)
    experts.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    experts.w2_weight_scale_inv = torch.nn.Parameter(s2, requires_grad=False)
    experts._flydsl_mega_moe_weights_built = True

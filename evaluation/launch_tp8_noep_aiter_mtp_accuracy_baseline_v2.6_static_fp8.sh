#!/usr/bin/env bash
set -euo pipefail

# MiMo-V2.6: convert checkpoint MXFP4 experts once at load time to FP8
# 128x128 blockscale weights. Activations remain dynamically quantized.
# Requires AITER's aiter.ops.flydsl.mxfp4_to_fp8_128block (PR #17),
# FlyDSL 0.3.2, and the separate CK MiMo page64 QK192/V128 prefill extension.
# This launcher never applies patches or changes the installed packages.
#
# MODEL_PATH=/models/MiMo-V2.6-Pro-RL bash evaluation/launch_tp8_noep_aiter_mtp_accuracy_baseline_v2.6_static_fp8.sh
# DRY_RUN=1 prints the command; CHECK_ONLY=1 checks dependencies without serving.
# AITER_ROOT may select a source checkout; otherwise use the installed AITER.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Pro-RL}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-30001}"
if [[ -n "${AITER_ROOT:-}" ]]; then
  export PYTHONPATH="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

export SGLANG_MXFP4_TO_FP8_128BLOCK=1
export SGLANG_INT4_WEIGHT=0
# MiMo's encoder otherwise defaults to a CUDA-only attention implementation.
export MM_ATTENTION_BACKEND=aiter_attn
# gfx942 CK LINEAR prefill has no single-token generation-health kernel.
# Use /health for process readiness; validate generation with real prompts.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_ENABLE_ASYNC_ASSERT=1

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX=3
export MC_TE_METRIC=1
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1
export SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY="${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY:-1}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1
# Baseline contract: never inherit the lossy Quick Reduce path or mixed-router
# optimization from the caller. Ablations must use a separate launcher.
unset ROCM_QUICK_REDUCE_QUANTIZATION
export SGLANG_MIMO_MIXED_ROUTER=0

# The local FlyPA prefill path is incompatible with FlyDSL 0.3.2.
export SGLANG_FLYPA_MIMO_PREFILL=0
export SGLANG_FLYDSL_MIMO_PREFILL=0

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_AITER_UNIFIED_VERIFY=1
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
# Optional reproducibility control; unset preserves SGLang's seed selection.
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.03}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export KV_CACHE_DTYPE=bf16
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
export ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
export SGLANG_TBO_MIM_SEQ_LEN="${SGLANG_TBO_MIM_SEQ_LEN:-2000}"

if ! [[ "${SGLANG_TBO_MIM_SEQ_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_TBO_MIM_SEQ_LEN must be a positive integer, observed '${SGLANG_TBO_MIM_SEQ_LEN}'" >&2
  exit 2
fi
if ! [[ "${TOKENIZER_WORKER_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOKENIZER_WORKER_NUM must be a positive integer" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi

export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=1
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT=0
export SGLANG_AITER_MIMO_FRESH_BF16_ASM=0
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN=0
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN=0

# Real MTP acceptance for accuracy validation.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-${SGLANG_ROOT}/logs/mimo_v2.6_static_fp8_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server_tp8_mimo_v2.6_static_fp8_accuracy.log}"

echo "MiMo-V2.6: static FP8 blockscale MoE weights converted from MXFP4 at load time"
echo "Attention: CK prefill; target-verify=${SGLANG_AITER_PA_DECODE_IMPL}; SWA/sink and ordinary decode=AITER/Gluon"

echo "Configuration: max-running=${MAX_RUNNING_REQUESTS}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, ep=1, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=${ROCM_QUICK_REDUCE_QUANTIZATION:-unset}, nccl-min-channels=${NCCL_MIN_NCHANNELS}, mixed-router=${SGLANG_MIMO_MIXED_ROUTER}, fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}, fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}, fresh-bf16-asm=${SGLANG_AITER_MIMO_FRESH_BF16_ASM}, fresh-bf16-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN}, fresh-bf16-swa-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN}, mtp=1, aiter-ar-fusion=0, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, reasoning-parser=${REASONING_PARSER}, tbo=${ENABLE_TWO_BATCH_OVERLAP}, tbo-min-isl=${SGLANG_TBO_MIM_SEQ_LEN}, overlap=enabled"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"
echo "MTP: EAGLE, steps=3, top-k=1, draft-tokens=4, multi-layer=enabled"


cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi
if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a cuda_graph_bs <<< "${CUDA_GRAPH_BS_DECODE}"
  cuda_graph_args+=(--cuda-graph-bs-decode "${cuda_graph_bs[@]}")
fi

custom_all_reduce_args=()
custom_all_reduce_status=enabled
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
  custom_all_reduce_args+=(--disable-custom-all-reduce)
  custom_all_reduce_status=disabled
fi

echo "Custom all-reduce: ${custom_all_reduce_status}"

tbo_args=()
if [[ "${ENABLE_TWO_BATCH_OVERLAP}" == "1" ]]; then
  tbo_args+=(--enable-two-batch-overlap)
fi

kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  kv_cache_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

radix_cache_args=()
radix_cache_status=enabled
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_cache_args+=(--disable-radix-cache)
  radix_cache_status=disabled
fi

echo "Radix cache: ${radix_cache_status}"
echo "HIP non-greedy EAGLE verifier: ${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY}"

mm_attention_args=()
if [[ -n "${MM_ATTENTION_BACKEND:-}" ]]; then
  mm_attention_args+=(--mm-attention-backend "${MM_ATTENTION_BACKEND}")
fi

server_cmd=("${PYTHON_BIN}" -u -m sglang.launch_server \
  "${mm_attention_args[@]}" \
  --model-path "${MODEL_PATH}" \
  --tp-size 8 \
  --ep-size 1 \
  --moe-a2a-backend none \
  --moe-runner-backend aiter \
  --speculative-moe-a2a-backend none \
  --speculative-moe-runner-backend aiter \
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}" \
  "${seed_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens 1048576 \
  --attention-backend aiter \
  "${kv_cache_args[@]}" \
  --page-size 64 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  "${tbo_args[@]}" \
  "${radix_cache_args[@]}" \
  "${cuda_graph_args[@]}" \
  "${custom_all_reduce_args[@]}" \
)

printf 'Command:'
printf ' %q' "${server_cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

# Resolve the MiMo GEMM table from the selected installation, not a machine-
# specific checkout. Avoid merging unrelated model tables with duplicate rows.
if [[ -z "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:-}" ]]; then
  AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="$("${PYTHON_BIN}" - <<'PY_CONFIG'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("aiter")
if spec is None or spec.origin is None:
    raise SystemExit("AITER is not installed in this Python environment")
print(Path(spec.origin).parent / "configs/model_configs/a8w8_blockscale_tuned_gemm_mimo_v2_5_pro.csv")
PY_CONFIG
)"
  export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE
fi

"${PYTHON_BIN}" - <<'PY_CHECK'
import os
from pathlib import Path
import torch
import aiter
import flydsl
from aiter.jit.core import CK_DIR
try:
    from aiter.ops.flydsl import mxfp4_to_fp8_128block
except ImportError as exc:
    raise SystemExit(
        f"AITER at {aiter.__file__} does not expose the required conversion API. "
        "Install the MXFP4-to-FP8 AITER PR and FlyDSL 0.3.2; check PYTHONPATH/AITER_ROOT."
    ) from exc
print(f"AITER: {aiter.__file__}; FlyDSL: {flydsl.__version__}")
if not torch.version.hip or torch.cuda.device_count() < 8:
    raise SystemExit("This TP8 launcher requires eight visible ROCm GPUs")
arches = [torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] for i in range(8)]
if any(arch not in ("gfx942", "gfx950") for arch in arches):
    raise SystemExit(f"MXFP4-to-FP8 supports gfx942/gfx950, found {arches}")
table = Path(os.environ["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"])
if not table.is_file():
    raise SystemExit(f"Missing MiMo blockscale GEMM table: {table}")
print(f"GEMM table: {table}")
# Check the known CK extension used by this baseline. The conversion PR does
# not include it; cached/chunked full attention can otherwise fail only after
# loading the entire model. CK_DIR is AITER's supported source override.
codegen = Path(CK_DIR) / "example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py"
if not codegen.is_file() or "def _get_mimo_d192_tiles(" not in codegen.read_text():
    raise SystemExit(
        f"Missing CK MiMo page64 QK192/V128 prefill extension in {CK_DIR}. "
        "This attention baseline requires the separate "
        "patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch "
        "from AITER (or its integrated equivalent). Prepare that CK dependency "
        "before launch; the script does not apply it."
    )
print(f"CK MiMo prefill extension: {codegen}")
PY_CHECK

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  exit 0
fi
mkdir -p -- "${LOG_DIR}"
"${server_cmd[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_FILE}"

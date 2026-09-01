#!/usr/bin/env bash
set -euo pipefail

# MiMo-V2.5 prefill server using the fused FlyDSL MegaMoE path (EP8 by default).
# Requires MORI's FlyDSL extern-link fix (upstream commit 5fa9e40c or newer).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.5/}"
FLYDSL_ROOT="${FLYDSL_ROOT:-/root/workspace/xiaomi/FlyDSL}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30001}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-}"
EP_SIZE="${EP_SIZE:-${TP_SIZE}}"

if ! [[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${EP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE and EP_SIZE must be positive integers" >&2
  exit 2
fi
if [[ -z "${DP_SIZE}" ]]; then
  if (( TP_SIZE % 4 != 0 )); then
    echo "Cannot infer DP_SIZE: MiMo-V2.5 requires attention TP=4" >&2
    exit 2
  fi
  DP_SIZE=$((TP_SIZE / 4))
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${FLYDSL_ROOT}/kernels/mega_moe/mega_moe.py" ]]; then
  echo "FlyDSL MegaMoE sources were not found under: ${FLYDSL_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${FLYDSL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ARCH="${ARCH:-gfx950}"
export FLYDSL_GPU_ARCH="${FLYDSL_GPU_ARCH:-gfx950}"
export FLYDSL_RUNTIME_CACHE_DIR="${FLYDSL_RUNTIME_CACHE_DIR:-/root/workspace/flydsl_cache/sglang_megamoe_gfx950}"
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-8G}"
export MORI_SOCKET_IFNAME="${MORI_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export USE_ROCM_AITER_ROPE_BACKEND=0

# The current AITER checkout targets an older FlyDSL API. MegaMoE is imported
# directly from FlyDSL, while attention uses Triton for this bring-up script.
export SGLANG_USE_AITER=0
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=0
# With TP8 + attention-DP2, the 16K scheduler chunk is scattered across the
# four attention-TP ranks before MoE, so each EP rank sees at most 4096 rows.
export SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="${SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK:-4096}"
export SGLANG_FLYDSL_MEGA_MOE_STAGE2_P2P_QUANT="${SGLANG_FLYDSL_MEGA_MOE_STAGE2_P2P_QUANT:-auto}"

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-1048576}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/mimo_flydsl_megamoe_tp${TP_SIZE}_ep${EP_SIZE}_${RUN_ID}}"
LOG_FILE="${LOG_FILE:-server.log}"

mkdir -p "${LOG_DIR}" "${FLYDSL_RUNTIME_CACHE_DIR}"

if ! [[ "${SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK must be a positive integer" >&2
  exit 2
fi
if ! [[ "${CHUNKED_PREFILL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHUNKED_PREFILL_SIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "${MAX_PREFILL_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PREFILL_TOKENS must be a positive integer" >&2
  exit 2
fi
for parallel_size in "${DP_SIZE}"; do
  if ! [[ "${parallel_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DP_SIZE must be a positive integer" >&2
    exit 2
  fi
done
if (( TP_SIZE % DP_SIZE != 0 )); then
  echo "TP_SIZE must be divisible by DP_SIZE" >&2
  exit 2
fi
attention_tp_size=$((TP_SIZE / DP_SIZE))
if (( attention_tp_size != 4 )); then
  echo "MiMo-V2.5 requires TP_SIZE / DP_SIZE = 4, got ${attention_tp_size}" >&2
  exit 2
fi
if (( EP_SIZE != TP_SIZE )); then
  echo "The current single-node MegaMoE backend requires EP_SIZE = TP_SIZE" >&2
  exit 2
fi
if (( EP_SIZE > 8 )); then
  echo "FlyDSL MegaMoE currently supports at most 8 local GPUs" >&2
  exit 2
fi
mega_capacity="${SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK}"
if (( mega_capacity & (mega_capacity - 1) )); then
  echo "FlyDSL MegaMoE token capacity must be a power of two: ${mega_capacity}" >&2
  exit 2
fi
required_capacity=$(((CHUNKED_PREFILL_SIZE + attention_tp_size - 1) / attention_tp_size))
if (( mega_capacity < required_capacity )); then
  echo "FlyDSL MegaMoE capacity ${mega_capacity} is smaller than the TP4-scattered chunk requirement ${required_capacity}" >&2
  exit 2
fi

python3 - <<'PY'
from flydsl.compiler.extern_link import link_extern  # noqa: F401
from kernels.mega_moe import MegaMoEV2  # noqa: F401
from mori.ir.flydsl.runtime import shmem_module_init  # noqa: F401
PY

echo "Model: ${MODEL_PATH}"
echo "Configuration: TP${TP_SIZE} / DP${DP_SIZE} / EP${EP_SIZE} / attention-TP${attention_tp_size} / FlyDSL MegaMoE"
echo "MegaMoE capacity: ${SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK} tokens/rank"
echo "Context/chunk/max-prefill: ${CONTEXT_LENGTH}/${CHUNKED_PREFILL_SIZE}/${MAX_PREFILL_TOKENS}; SWA ratio: ${SWA_FULL_TOKENS_RATIO}"
echo "MORI heap: ${MORI_SHMEM_HEAP_SIZE}; FlyDSL cache: ${FLYDSL_RUNTIME_CACHE_DIR}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"

extra_server_args=()
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
  read -r -a extra_server_args <<< "${EXTRA_SERVER_ARGS}"
fi

dp_attention_args=()
if (( DP_SIZE > 1 )); then
  dp_attention_args+=(--enable-dp-attention)
fi

python3 -u -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --no-enable-multimodal \
  --tp-size "${TP_SIZE}" \
  --dp-size "${DP_SIZE}" \
  "${dp_attention_args[@]}" \
  --ep-size "${EP_SIZE}" \
  --moe-a2a-backend megamoe \
  --moe-runner-backend triton \
  --attention-backend triton \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --context-length "${CONTEXT_LENGTH}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --disable-custom-all-reduce \
  --disable-overlap-schedule \
  "${extra_server_args[@]}" \
  2>&1 | tee "${LOG_DIR}/${LOG_FILE}"

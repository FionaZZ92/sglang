#!/usr/bin/env bash
set -euo pipefail

# One-shot launcher and Torch Profiler capture for
# XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash's target backbone on the FlyDSL
# MegaMoE path. DFlash is opt-in because this checkout must also contain the
# MiMo-specific DFlash target-hidden-state integration before it can be enabled.
#
# The profiler records both CPU and GPU activities with Python call stacks and
# input shapes. CUDA graphs are disabled so GPU launches remain attributable to
# their Python call sites in Perfetto/chrome://tracing.
#
# Example:
#   ./evaluation/launch_tp8_ep8_flydsl_megamoe_fp4_dflash_profile.sh
#
# A smaller smoke capture:
#   PROFILE_INPUT_TOKENS=2048 CHUNKED_PREFILL_SIZE=4096 \
#     ./evaluation/launch_tp8_ep8_flydsl_megamoe_fp4_dflash_profile.sh
#
# After MiMo-specific DFlash support is present in this checkout:
#   ENABLE_DFLASH=1 PROFILE_OUTPUT_TOKENS=16 \
#     ./evaluation/launch_tp8_ep8_flydsl_megamoe_fp4_dflash_profile.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.5-Pro-FP4-DFlash}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}/dflash}"
FLYDSL_ROOT="${FLYDSL_ROOT:-/root/workspace/xiaomi/FlyDSL}"

HOST="${HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
SERVER_URL="${SERVER_URL:-http://${CLIENT_HOST}:${PORT}}"

TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-}"
EP_SIZE="${EP_SIZE:-${TP_SIZE}}"
MOE_DENSE_TP_SIZE="${MOE_DENSE_TP_SIZE:-1}"

ENABLE_DFLASH="${ENABLE_DFLASH:-0}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-${CHUNKED_PREFILL_SIZE}}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.65}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
PAGE_SIZE="${PAGE_SIZE:-1}"

PROFILE_INPUT_TOKENS="${PROFILE_INPUT_TOKENS:-16384}"
PROFILE_OUTPUT_TOKENS="${PROFILE_OUTPUT_TOKENS:-1}"
PROFILE_NUM_PROMPTS="${PROFILE_NUM_PROMPTS:-1}"
PROFILE_CONCURRENCY="${PROFILE_CONCURRENCY:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
MERGE_PROFILES="${MERGE_PROFILES:-1}"
DRY_RUN="${DRY_RUN:-0}"

SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-7200}"
PROFILE_STOP_TIMEOUT="${PROFILE_STOP_TIMEOUT:-1800}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/mimo_v25_pro_fp4_dflash_megamoe_profile_${RUN_ID}}"
TRACE_DIR="${TRACE_DIR:-${LOG_DIR}/traces}"
# Leave the filename prefix empty when merging: this checkout's merger discovers
# rank traces by profile id, while the unique run directory already labels them.
PROFILE_PREFIX="${PROFILE_PREFIX:-}"
MODEL_LOADER_EXTRA_CONFIG="${MODEL_LOADER_EXTRA_CONFIG:-{\"enable_multithread_load\":true,\"num_threads\":64}}"

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got ${value@Q}" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got ${value@Q}" >&2
    exit 2
  fi
}

is_true() {
  case "${1,,}" in
    1 | true | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

for pair in \
  "TP_SIZE:${TP_SIZE}" \
  "EP_SIZE:${EP_SIZE}" \
  "MOE_DENSE_TP_SIZE:${MOE_DENSE_TP_SIZE}" \
  "CONTEXT_LENGTH:${CONTEXT_LENGTH}" \
  "CHUNKED_PREFILL_SIZE:${CHUNKED_PREFILL_SIZE}" \
  "MAX_PREFILL_TOKENS:${MAX_PREFILL_TOKENS}" \
  "MAX_RUNNING_REQUESTS:${MAX_RUNNING_REQUESTS}" \
  "PAGE_SIZE:${PAGE_SIZE}" \
  "PROFILE_INPUT_TOKENS:${PROFILE_INPUT_TOKENS}" \
  "PROFILE_OUTPUT_TOKENS:${PROFILE_OUTPUT_TOKENS}" \
  "PROFILE_NUM_PROMPTS:${PROFILE_NUM_PROMPTS}" \
  "PROFILE_CONCURRENCY:${PROFILE_CONCURRENCY}" \
  "SPECULATIVE_NUM_DRAFT_TOKENS:${SPECULATIVE_NUM_DRAFT_TOKENS}" \
  "SERVER_STARTUP_TIMEOUT:${SERVER_STARTUP_TIMEOUT}" \
  "PROFILE_STOP_TIMEOUT:${PROFILE_STOP_TIMEOUT}"; do
  require_positive_int "${pair%%:*}" "${pair#*:}"
done
require_nonnegative_int "WARMUP_REQUESTS" "${WARMUP_REQUESTS}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config does not exist: ${MODEL_PATH}/config.json" >&2
  exit 2
fi
if is_true "${ENABLE_DFLASH}" && [[ ! -f "${DRAFT_MODEL_PATH}/config.json" ]]; then
  echo "DFlash draft config does not exist: ${DRAFT_MODEL_PATH}/config.json" >&2
  exit 2
fi
if is_true "${ENABLE_DFLASH}" && ! grep -q \
  "def set_dflash_layers_to_capture" \
  "${SGLANG_ROOT}/python/sglang/srt/models/mimo_v2.py"; then
  echo "This checkout lacks MiMo-V2 DFlash target-layer capture support." >&2
  echo "Run with ENABLE_DFLASH=0, or integrate MiMo DFlash support first." >&2
  exit 2
fi
if [[ ! -f "${FLYDSL_ROOT}/kernels/mega_moe/mega_moe.py" ]]; then
  echo "FlyDSL MegaMoE sources were not found under: ${FLYDSL_ROOT}" >&2
  exit 2
fi

MODEL_PATH="$(cd -- "${MODEL_PATH}" && pwd)"
if is_true "${ENABLE_DFLASH}"; then
  DRAFT_MODEL_PATH="$(cd -- "${DRAFT_MODEL_PATH}" && pwd)"
fi
FLYDSL_ROOT="$(cd -- "${FLYDSL_ROOT}" && pwd)"
mkdir -p "${LOG_DIR}" "${TRACE_DIR}"
LOG_DIR="$(cd -- "${LOG_DIR}" && pwd)"
TRACE_DIR="$(cd -- "${TRACE_DIR}" && pwd)"

read -r MODEL_ARCH EXPECTED_ATTENTION_TP NUM_EXPERTS QUANT_METHOD STORE_DTYPE < <(
  python3 - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = json.load(config_file)

text_config = config.get("text_config") or config
architectures = config.get("architectures") or text_config.get("architectures") or []
quant_config = config.get("quantization_config") or {}
print(
    architectures[0] if architectures else "unknown",
    text_config.get("num_key_value_heads", 0),
    text_config.get("n_routed_experts", 0),
    quant_config.get("quant_method", "unknown"),
    quant_config.get("store_dtype", "unknown"),
)
PY
)

if [[ "${MODEL_ARCH}" != "MiMoV2ForCausalLM" && "${MODEL_ARCH}" != "MiMoV2FlashForCausalLM" ]]; then
  echo "Expected a MiMo-V2 model, got architecture ${MODEL_ARCH@Q}" >&2
  exit 2
fi
if [[ "${QUANT_METHOD}" != "fp8" || "${STORE_DTYPE}" != "mxfp4" ]]; then
  echo "Expected quant_method=fp8 and store_dtype=mxfp4, got quant_method=${QUANT_METHOD@Q}, store_dtype=${STORE_DTYPE@Q}" >&2
  exit 2
fi
require_positive_int "num_key_value_heads" "${EXPECTED_ATTENTION_TP}"
require_positive_int "n_routed_experts" "${NUM_EXPERTS}"

if [[ -z "${DP_SIZE}" ]]; then
  if (( TP_SIZE % EXPECTED_ATTENTION_TP != 0 )); then
    echo "Cannot infer DP_SIZE: TP_SIZE=${TP_SIZE} is not divisible by the model's required attention TP=${EXPECTED_ATTENTION_TP}" >&2
    exit 2
  fi
  DP_SIZE=$((TP_SIZE / EXPECTED_ATTENTION_TP))
fi
require_positive_int "DP_SIZE" "${DP_SIZE}"

if (( TP_SIZE % DP_SIZE != 0 )); then
  echo "TP_SIZE must be divisible by DP_SIZE" >&2
  exit 2
fi
ATTENTION_TP_SIZE=$((TP_SIZE / DP_SIZE))
if (( ATTENTION_TP_SIZE != EXPECTED_ATTENTION_TP )); then
  echo "${MODEL_ARCH} requires effective attention TP=${EXPECTED_ATTENTION_TP}, got ${ATTENTION_TP_SIZE} (TP=${TP_SIZE}, DP=${DP_SIZE})" >&2
  exit 2
fi
if (( EP_SIZE != TP_SIZE )); then
  echo "The current single-node FlyDSL MegaMoE path requires EP_SIZE=TP_SIZE" >&2
  exit 2
fi
if (( EP_SIZE > 8 )); then
  echo "The current single-node FlyDSL MegaMoE path supports at most 8 ranks" >&2
  exit 2
fi
if (( NUM_EXPERTS % EP_SIZE != 0 )); then
  echo "n_routed_experts=${NUM_EXPERTS} must be divisible by EP_SIZE=${EP_SIZE}" >&2
  exit 2
fi
if (( PROFILE_INPUT_TOKENS + PROFILE_OUTPUT_TOKENS > CONTEXT_LENGTH )); then
  echo "PROFILE_INPUT_TOKENS + PROFILE_OUTPUT_TOKENS exceeds CONTEXT_LENGTH" >&2
  exit 2
fi
if is_true "${MERGE_PROFILES}" && [[ -n "${PROFILE_PREFIX}" ]]; then
  echo "MERGE_PROFILES=1 requires an empty PROFILE_PREFIX on this checkout." >&2
  echo "The run-specific TRACE_DIR already keeps trace names unambiguous." >&2
  exit 2
fi

export PYTHONPATH="${SGLANG_ROOT}/python:${FLYDSL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export ARCH="${ARCH:-gfx950}"
export FLYDSL_GPU_ARCH="${FLYDSL_GPU_ARCH:-gfx950}"
export FLYDSL_RUNTIME_CACHE_DIR="${FLYDSL_RUNTIME_CACHE_DIR:-/root/workspace/flydsl_cache/sglang_megamoe_gfx950}"
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-8G}"
export MORI_SOCKET_IFNAME="${MORI_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export USE_ROCM_AITER_ROPE_BACKEND=0

# MegaMoE imports FlyDSL directly. Keep the incompatible AITER checkout out of
# this path, matching the existing MiMo-V2.5 MegaMoE launcher.
export SGLANG_USE_AITER=0
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=0
export SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="${SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK:-4096}"
export SGLANG_FLYDSL_MEGA_MOE_STAGE2_P2P_QUANT="${SGLANG_FLYDSL_MEGA_MOE_STAGE2_P2P_QUANT:-auto}"

# Explicitly retain Python source stacks and operator shapes in the Chrome
# traces. "GPU" is mapped to ProfilerActivity.CUDA (HIP on ROCm) by SGLang.
export SGLANG_TORCH_PROFILER_DIR="${TRACE_DIR}"
export SGLANG_PROFILE_WITH_STACK=true
export SGLANG_PROFILE_RECORD_SHAPES=true

mkdir -p "${FLYDSL_RUNTIME_CACHE_DIR}"
require_positive_int \
  "SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK" \
  "${SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK}"
MEGA_CAPACITY="${SGLANG_FLYDSL_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK}"
if (( MEGA_CAPACITY & (MEGA_CAPACITY - 1) )); then
  echo "FlyDSL MegaMoE token capacity must be a power of two: ${MEGA_CAPACITY}" >&2
  exit 2
fi
REQUIRED_CAPACITY=$(((CHUNKED_PREFILL_SIZE + ATTENTION_TP_SIZE - 1) / ATTENTION_TP_SIZE))
if (( MEGA_CAPACITY < REQUIRED_CAPACITY )); then
  echo "MegaMoE capacity ${MEGA_CAPACITY} is smaller than the per-rank chunk requirement ${REQUIRED_CAPACITY}" >&2
  exit 2
fi

SERVER_LOG="${LOG_DIR}/server.log"
WARMUP_LOG="${LOG_DIR}/warmup.log"
PROFILE_BENCH_LOG="${LOG_DIR}/profile_benchmark.log"
PROFILE_RESULT="${LOG_DIR}/profile_benchmark.jsonl"
PROFILE_START_RESPONSE="${LOG_DIR}/start_profile.response.txt"
PROFILE_STOP_RESPONSE="${LOG_DIR}/stop_profile.response.txt"

extra_server_args=()
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
  read -r -a extra_server_args <<< "${EXTRA_SERVER_ARGS}"
fi

server_cmd=(
  python3 -u -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --no-enable-multimodal
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --ep-size "${EP_SIZE}"
  --moe-dense-tp-size "${MOE_DENSE_TP_SIZE}"
  --moe-a2a-backend megamoe
  --moe-runner-backend triton
  --attention-backend triton
  --quantization fp8
  --dtype bfloat16
  --reasoning-parser mimo
  --tool-call-parser mimo
  --context-length "${CONTEXT_LENGTH}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}"
  --page-size "${PAGE_SIZE}"
  --disable-prefill-cuda-graph
  --disable-decode-cuda-graph
  --disable-custom-all-reduce
  --disable-overlap-schedule
  --skip-server-warmup
  --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}"
)

if (( DP_SIZE > 1 )); then
  server_cmd+=(--enable-dp-attention --enable-dp-lm-head)
fi
if is_true "${ENABLE_DFLASH}"; then
  server_cmd+=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-draft-model-quantization unquant
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    --speculative-draft-window-size 1024
    --disable-chunked-prefix-cache
  )
fi
server_cmd+=("${extra_server_args[@]}")

{
  echo "run_id=${RUN_ID}"
  echo "model=${MODEL_PATH}"
  echo "draft_model=${DRAFT_MODEL_PATH}"
  echo "enable_dflash=${ENABLE_DFLASH}"
  echo "topology=TP${TP_SIZE}/DP${DP_SIZE}/EP${EP_SIZE}/attention-TP${ATTENTION_TP_SIZE}"
  echo "profile=input:${PROFILE_INPUT_TOKENS},output:${PROFILE_OUTPUT_TOKENS},prompts:${PROFILE_NUM_PROMPTS},concurrency:${PROFILE_CONCURRENCY}"
  echo "trace_dir=${TRACE_DIR}"
  printf "server_command="
  printf "%q " "${server_cmd[@]}"
  printf "\n"
} | tee "${LOG_DIR}/run_config.txt"

if is_true "${DRY_RUN}"; then
  echo "DRY_RUN=1: validation completed; server was not started."
  exit 0
fi

python3 - "${TP_SIZE}" <<'PY'
import sys

import torch
from flydsl.compiler.extern_link import link_extern  # noqa: F401
from kernels.mega_moe import MegaMoEV2  # noqa: F401
from mori.ir.flydsl.runtime import shmem_module_init  # noqa: F401

required = int(sys.argv[1])
available = torch.cuda.device_count()
if available < required:
    raise SystemExit(f"Need at least {required} visible GPUs, found {available}")
if not hasattr(torch.profiler.ProfilerActivity, "CUDA"):
    raise SystemExit("This PyTorch build does not expose ProfilerActivity.CUDA")
print(f"Profiler preflight passed: torch={torch.__version__}, visible_gpus={available}")
PY

SERVER_PID=""
SERVER_OWNS_PROCESS_GROUP=0
PROFILE_ACTIVE=0

stop_profile_if_active() {
  if (( PROFILE_ACTIVE )); then
    echo "Stopping active profiler before shutdown..."
    curl --fail --silent --show-error \
      --max-time "${PROFILE_STOP_TIMEOUT}" \
      -X POST "${SERVER_URL}/stop_profile" \
      >"${PROFILE_STOP_RESPONSE}" 2>&1 || true
    PROFILE_ACTIVE=0
  fi
}

stop_server() {
  if [[ -z "${SERVER_PID}" ]] || ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    return
  fi
  echo "Stopping SGLang server (pid=${SERVER_PID})..."
  if (( SERVER_OWNS_PROCESS_GROUP )); then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
  else
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" 2>/dev/null || true
      return
    fi
    sleep 1
  done
  if (( SERVER_OWNS_PROCESS_GROUP )); then
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  else
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  stop_profile_if_active
  stop_server
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

cd "${SGLANG_ROOT}"
if curl --fail --silent --max-time 2 "${SERVER_URL}/health" >/dev/null 2>&1; then
  echo "A server is already responding at ${SERVER_URL}; choose another PORT." >&2
  exit 2
fi
echo "Starting ${MODEL_ARCH} with native MXFP4 experts and FlyDSL MegaMoE..."
if ! is_true "${ENABLE_DFLASH}"; then
  echo "DFlash draft is disabled; profiling the target backbone only."
fi
echo "Profiler traces will be written to ${TRACE_DIR}"
if command -v setsid >/dev/null 2>&1; then
  setsid "${server_cmd[@]}" > >(tee "${SERVER_LOG}") 2>&1 &
  SERVER_PID=$!
  SERVER_OWNS_PROCESS_GROUP=1
else
  "${server_cmd[@]}" > >(tee "${SERVER_LOG}") 2>&1 &
  SERVER_PID=$!
fi

echo "Waiting up to ${SERVER_STARTUP_TIMEOUT}s for ${SERVER_URL}/health_generate ..."
deadline=$((SECONDS + SERVER_STARTUP_TIMEOUT))
next_status=$((SECONDS + 60))
while ! curl --fail --silent --show-error --max-time 10 \
  "${SERVER_URL}/health_generate" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "SGLang exited before becoming ready. Last server log lines:" >&2
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for SGLang. Last server log lines:" >&2
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= next_status )); then
    echo "Still loading the model (${SECONDS}s elapsed)..."
    next_status=$((SECONDS + 60))
  fi
  sleep 5
done

echo "Server is ready. Saving server metadata..."
curl --fail --silent --show-error --max-time 30 \
  "${SERVER_URL}/server_info" >"${LOG_DIR}/server_info.json"

benchmark_cmd=(
  python3 -m sglang.bench_serving
  --backend sglang
  --model "${MODEL_PATH}"
  --host "${CLIENT_HOST}"
  --port "${PORT}"
  --dataset-name random
  --random-input-len "${PROFILE_INPUT_TOKENS}"
  --random-output-len "${PROFILE_OUTPUT_TOKENS}"
  --random-range-ratio 1.0
  --max-concurrency "${PROFILE_CONCURRENCY}"
  --tokenize-prompt
  --disable-tqdm
)

if (( WARMUP_REQUESTS > 0 )); then
  echo "Running ${WARMUP_REQUESTS} warmup request(s) outside the trace..."
  "${benchmark_cmd[@]}" \
    --num-prompts "${WARMUP_REQUESTS}" \
    --warmup-requests 0 \
    --output-file "${LOG_DIR}/warmup.jsonl" \
    2>&1 | tee "${WARMUP_LOG}"
fi

# Clear the radix cache before profiling so the measured prefill is not a cache hit.
curl --fail --silent --show-error --max-time 60 \
  -X POST "${SERVER_URL}/flush_cache" >/dev/null

if is_true "${MERGE_PROFILES}"; then
  MERGE_PROFILES_JSON=true
else
  MERGE_PROFILES_JSON=false
fi
PROFILE_PAYLOAD="$(
  python3 - "${TRACE_DIR}" "${PROFILE_PREFIX}" "${MERGE_PROFILES_JSON}" <<'PY'
import json
import sys

print(
    json.dumps(
        {
            "output_dir": sys.argv[1],
            "activities": ["CPU", "GPU"],
            "with_stack": True,
            "record_shapes": True,
            "profile_by_stage": False,
            "merge_profiles": sys.argv[3] == "true",
            "profile_prefix": sys.argv[2],
        }
    )
)
PY
)"

echo "Starting Torch Profiler (CPU + GPU, with_stack=true, record_shapes=true)..."
curl --fail --silent --show-error --max-time 120 \
  -X POST \
  -H "Content-Type: application/json" \
  --data-binary "${PROFILE_PAYLOAD}" \
  "${SERVER_URL}/start_profile" | tee "${PROFILE_START_RESPONSE}"
PROFILE_ACTIVE=1

echo "Running profiled workload..."
"${benchmark_cmd[@]}" \
  --num-prompts "${PROFILE_NUM_PROMPTS}" \
  --warmup-requests 0 \
  --output-file "${PROFILE_RESULT}" \
  2>&1 | tee "${PROFILE_BENCH_LOG}"

echo "Stopping Torch Profiler and exporting Chrome traces..."
curl --fail --silent --show-error \
  --max-time "${PROFILE_STOP_TIMEOUT}" \
  -X POST "${SERVER_URL}/stop_profile" | tee "${PROFILE_STOP_RESPONSE}"
PROFILE_ACTIVE=0

mapfile -t trace_files < <(
  find "${TRACE_DIR}" -maxdepth 1 -type f -name '*.trace.json.gz' -size +0c | sort
)
if (( ${#trace_files[@]} == 0 )); then
  echo "Profiler stopped, but no non-empty trace was found in ${TRACE_DIR}" >&2
  exit 1
fi

echo "Torch Profiler capture completed. Trace files:"
for trace_file in "${trace_files[@]}"; do
  gzip -t "${trace_file}"
  du -h "${trace_file}"
done
echo "Open the .trace.json.gz files with https://ui.perfetto.dev/ or chrome://tracing."

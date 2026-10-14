#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_DIR="$(dirname -- "${SCRIPT_DIR}")"
SGLANG_ROOT="${SGLANG_ROOT:-$(dirname -- "${EVALUATION_DIR}")}"
AITER_ROOT="${AITER_ROOT:-/sgl-workspace/aiter}"
export PYTHONPATH="${SGLANG_ROOT}/python:${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:-${AITER_ROOT}/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_mimo_v2_5_pro.csv}"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE:-${AITER_ROOT}/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_mimo_v2_5_pro.csv}"

TARGET_MODEL_VARIANT="${TARGET_MODEL_VARIANT:-mxfp4}"
case "${TARGET_MODEL_VARIANT}" in
  fp8)
    export MODEL="${MODEL:-/models/MiMo-V2.5-Pro}"
    export MIMO_TARGET_WEIGHT_FORMAT=fp8
    default_fmoe_config="${AITER_ROOT}/aiter/configs/model_configs/mimo_v2_5_pro_b16_tuned_fmoe.csv"
    ;;
  mxfp4)
    export MODEL="${MODEL:-/models/MiMo-V2.5-Pro-FP4-DFlash}"
    export MIMO_TARGET_WEIGHT_FORMAT=mxfp4
    default_fmoe_config="${AITER_ROOT}/aiter/configs/model_configs/mimo_v2_5_pro_tuned_fmoe.csv"
    ;;
  *)
    echo "TARGET_MODEL_VARIANT must be fp8 or mxfp4, observed '${TARGET_MODEL_VARIANT}'" >&2
    exit 2
    ;;
esac

export AITER_CONFIG_FMOE="${AITER_CONFIG_FMOE:-${default_fmoe_config}}"
export AITER_MXFP4_STAGE2_OUTPUT_DTYPE="${AITER_MXFP4_STAGE2_OUTPUT_DTYPE:-bf16}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30001}"
export MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-aiter}"
export MM_ATTENTION_BACKEND="${MM_ATTENTION_BACKEND:-}"

export SPECULATIVE_ALGORITHM=DFLASH
export SPECULATIVE_DRAFT_MODEL="${SPECULATIVE_DRAFT_MODEL:-/models/MiMo-V2.5-Pro-FP4-DFlash/dflash}"
export SPECULATIVE_NUM_DRAFT_TOKENS=8
export SPECULATIVE_DRAFT_ATTENTION_BACKEND=aiter
export SPECULATIVE_DRAFT_KV_CACHE_DTYPE=bf16
export SPECULATIVE_DRAFT_WINDOW_SIZE=1024

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

# Accuracy baseline contract: BF16 KV, no lossy Quick Reduce, and no mixed router.
export KV_CACHE_DTYPE=bf16
unset ROCM_QUICK_REDUCE_QUANTIZATION
export ENABLE_PREFILL_QUICK_REDUCE=0
export SGLANG_MIMO_MIXED_ROUTER=0

# Full prompt attention stays on the gfx950 BF16 ASM path. Cached target SWA
# prefill and DFlash verification use the qualified AITER-hosted FlyPA paths.
export SGLANG_FLYDSL_MIMO_PREFILL=0
export SGLANG_FLYPA_MIMO_PREFILL=1
export SGLANG_AITER_PA_DECODE_IMPL=flydsl
export SGLANG_AITER_TARGET_VERIFY_SWA_IMPL=flydsl
export SGLANG_AITER_DFLASH_SWA_IMPL=flydsl
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-1048576}"
export PAGE_SIZE="${PAGE_SIZE:-64}"
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
export ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
export SGLANG_TBO_MIM_SEQ_LEN="${SGLANG_TBO_MIM_SEQ_LEN:-2000}"

export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"

# Accuracy uses real DFlash acceptance by default. Performance runs must opt in
# explicitly so an accuracy invocation can never inherit simulated acceptance.
SGLANG_DFLASH_PERFORMANCE_MODE="${SGLANG_DFLASH_PERFORMANCE_MODE:-0}"
if [[ "${SGLANG_DFLASH_PERFORMANCE_MODE}" == "1" ]]; then
  export SGLANG_SIMULATE_ACC_LEN=4
  export SGLANG_SIMULATE_ACC_METHOD=match-expected
elif [[ "${SGLANG_DFLASH_PERFORMANCE_MODE}" == "0" ]]; then
  unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
else
  echo "SGLANG_DFLASH_PERFORMANCE_MODE must be 0 or 1" >&2
  exit 2
fi

if [[ "${PAGE_SIZE}" != "64" ]]; then
  echo "DFlash FlyPA SWA requires PAGE_SIZE=64" >&2
  exit 2
fi
if [[ "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}" != "aiter" ]]; then
  echo "DFlash FlyPA SWA requires SPECULATIVE_DRAFT_ATTENTION_BACKEND=aiter" >&2
  exit 2
fi
if [[ "${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}" != "bf16" ]]; then
  echo "DFlash FlyPA SWA requires SPECULATIVE_DRAFT_KV_CACHE_DTYPE=bf16" >&2
  exit 2
fi
if ! [[ "${SPECULATIVE_NUM_DRAFT_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPECULATIVE_NUM_DRAFT_TOKENS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${MAX_RUNNING_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNNING_REQUESTS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${TOKENIZER_WORKER_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOKENIZER_WORKER_NUM must be a positive integer" >&2
  exit 2
fi
if ! [[ "${SGLANG_TBO_MIM_SEQ_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_TBO_MIM_SEQ_LEN must be a positive integer" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-${EVALUATION_DIR}/logs/dflash_accuracy_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server_tp8_aiter_dflash_accuracy.log}"
mkdir -p "${LOG_DIR}"

seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi

cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a cuda_graph_bs <<<"${CUDA_GRAPH_BS_DECODE}"
  cuda_graph_args+=(--cuda-graph-bs-decode "${cuda_graph_bs[@]}")
fi

decode_attention_args=()
if [[ -n "${DECODE_ATTENTION_BACKEND}" ]]; then
  decode_attention_args+=(--decode-attention-backend "${DECODE_ATTENTION_BACKEND}")
fi

mm_attention_args=()
if [[ -n "${MM_ATTENTION_BACKEND}" ]]; then
  mm_attention_args+=(--mm-attention-backend "${MM_ATTENTION_BACKEND}")
fi

custom_all_reduce_args=()
custom_all_reduce_status=enabled
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
  custom_all_reduce_args+=(--disable-custom-all-reduce)
  custom_all_reduce_status=disabled
fi

tbo_args=()
if [[ "${ENABLE_TWO_BATCH_OVERLAP}" == "1" ]]; then
  tbo_args+=(--enable-two-batch-overlap)
fi

radix_cache_args=()
radix_cache_status=enabled
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_cache_args+=(--disable-radix-cache)
  radix_cache_status=disabled
fi

echo "Attention hybrid: prefill-flydsl=${SGLANG_FLYDSL_MIMO_PREFILL}, full-target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, target-swa=${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL}, draft-swa=${SGLANG_AITER_DFLASH_SWA_IMPL}"
echo "Configuration: model=${MODEL}, target-format=${MIMO_TARGET_WEIGHT_FORMAT}, max-running=${MAX_RUNNING_REQUESTS}, page=${PAGE_SIZE}, chunked-prefill=${CHUNKED_PREFILL_SIZE}, moe-runner=${MOE_RUNNER_BACKEND}, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=disabled, mixed-router=${SGLANG_MIMO_MIXED_ROUTER}, tf32-cublas-override=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE}, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, tbo=${ENABLE_TWO_BATCH_OVERLAP}, tbo-min-isl=${SGLANG_TBO_MIM_SEQ_LEN}, overlap-plan-stream=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM}"
echo "AITER root/configs: ${AITER_ROOT}; fmoe=$(basename -- "${AITER_CONFIG_FMOE}"); gemm=$(basename -- "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}"); bpreshuffle=$(basename -- "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}")"
echo "DFlash: draft-tokens=${SPECULATIVE_NUM_DRAFT_TOKENS}, draft-model=${SPECULATIVE_DRAFT_MODEL}, draft-attention=${SPECULATIVE_DRAFT_ATTENTION_BACKEND}, draft-kv=${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}, draft-window=${SPECULATIVE_DRAFT_WINDOW_SIZE}"
echo "DFlash acceptance: $([[ ${SGLANG_DFLASH_PERFORMANCE_MODE} == 1 ]] && echo simulated:${SGLANG_SIMULATE_ACC_LEN}/${SGLANG_SIMULATE_ACC_METHOD} || echo real)"
echo "Multimodal attention: ${MM_ATTENTION_BACKEND:-auto}"
echo "Custom all-reduce: ${custom_all_reduce_status}; radix cache: ${radix_cache_status}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"

python3 -u -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp-size 8 \
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}" \
  "${seed_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length "${CONTEXT_LENGTH}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
  --attention-backend aiter \
  "${decode_attention_args[@]}" \
  "${mm_attention_args[@]}" \
  --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
  --aiter-mxfp4-stage2-output-dtype "${AITER_MXFP4_STAGE2_OUTPUT_DTYPE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --page-size "${PAGE_SIZE}" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL}" \
  --speculative-draft-attention-backend "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}" \
  --speculative-draft-kv-cache-dtype "${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}" \
  --speculative-draft-window-size "${SPECULATIVE_DRAFT_WINDOW_SIZE}" \
  --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}" \
  "${tbo_args[@]}" \
  "${radix_cache_args[@]}" \
  "${cuda_graph_args[@]}" \
  "${custom_all_reduce_args[@]}" \
  2>&1 | tee "${LOG_DIR}/${LOG_FILE}"

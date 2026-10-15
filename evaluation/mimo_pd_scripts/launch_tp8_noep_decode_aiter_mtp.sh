#!/usr/bin/env bash
set -euo pipefail
# MiMo-v2.5-Pro SGLang 1P1D disaggregated decode node.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# KV/attention settings must match prefill
# (page 64, kv auto, vectorized_5d). CUDA graph stays on for decode.

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1

# Native InfiniBand use GID 0 is valid. GID 1 used for RoCE.
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-10.2.80.29}"
export HOST_IP="${HOST_IP:-${SGLANG_HOST_IP}}"
export MC_GID_INDEX="${MC_GID_INDEX:-1}"
#export DISAGGREGATION_IB_DEVICE="${DISAGGREGATION_IB_DEVICE:-mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7}"
export DISAGGREGATION_IB_DEVICE="${DISAGGREGATION_IB_DEVICE:-rocep121s0,rocep9s0,rocep105s0,rocep25s0,rocep249s0,rocep137s0,rocep233s0,rocep153s0}"
export MC_TE_METRIC=1

export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=5000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=5000
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1
export SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY="${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY:-1}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1

if [[ "${SGLANG_ABLATION_ENABLE_QUICK_REDUCE:-0}" == "1" ]]; then
  export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT8}"
  echo "ROCM_QUICK_REDUCE_QUANTIZATION enabled for ablation: ${ROCM_QUICK_REDUCE_QUANTIZATION}"
elif [[ -v ROCM_QUICK_REDUCE_QUANTIZATION ]]; then
  echo "ROCM_QUICK_REDUCE_QUANTIZATION was set to '${ROCM_QUICK_REDUCE_QUANTIZATION}', unsetting for accuracy"
  unset ROCM_QUICK_REDUCE_QUANTIZATION
else
  echo "ROCM_QUICK_REDUCE_QUANTIZATION is unset"
fi

export SGLANG_MIMO_MIXED_ROUTER="${SGLANG_MIMO_MIXED_ROUTER:-0}"
export SGLANG_FLYPA_MIMO_PREFILL="${SGLANG_FLYPA_MIMO_PREFILL:-0}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_FLYDSL_MIMO_PREFILL="${SGLANG_FLYDSL_MIMO_PREFILL:-0}"
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.2}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"

export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"

unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD

export MODEL="${MODEL:-/models/MiMo-V2.5-Pro}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/server_log-${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-pd_decode_server.log}"

echo "Attention hybrid: prefill-flydsl=${SGLANG_FLYDSL_MIMO_PREFILL}, target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, KV-layout=${SGLANG_AITER_KV_CACHE_LAYOUT}"
echo "Configuration: max-running=${MAX_RUNNING_REQUESTS}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}"
echo "PD decode: mooncake, ib-device=${DISAGGREGATION_IB_DEVICE}, gid-index=${MC_GID_INDEX}, host-ip=${SGLANG_HOST_IP}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"
echo "MTP: EAGLE, steps=3, top-k=1, draft-tokens=4, multi-layer=enabled"

mkdir -p "${LOG_DIR}"

cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a cuda_graph_bs <<< "${CUDA_GRAPH_BS_DECODE}"
  cuda_graph_args+=(--cuda-graph-bs-decode "${cuda_graph_bs[@]}")
fi

kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  kv_cache_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

radix_cache_args=()
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_cache_args+=(--disable-radix-cache)
  echo "Radix cache: disabled"
else
  echo "Radix cache: enabled"
fi

python3 -u -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp-size 8 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host 0.0.0.0 \
  --port 30001 \
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
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "${DISAGGREGATION_IB_DEVICE}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  --log-level-http warning \
  --watchdog-time 1000000  \
  --tokenizer-worker-num 64 \
  --enable-cache-report \
  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
  --collect-tokens-histogram \
  --enable-metrics \
  --bucket-time-to-first-token 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 800 1200 1800 2400 3600 5400 7200 \
  --bucket-e2e-request-latency 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 600 1200 1800 2400 3600 5400 7200 \
  --load-balance-method round_robin \
  --allow-auto-truncate \
  --enable-metrics-for-all-schedulers \
  --disable-tokenizer-batch-decode \
  "${radix_cache_args[@]}" \
  "${cuda_graph_args[@]}" \
  2>&1 | tee "${LOG_DIR}/${LOG_FILE}"

#!/usr/bin/env bash
set -uo pipefail

# Sweep the exact MiMo-V2.5 TP4/EP4 MegaMoE shape on GPU 0-3.
# Modes: stage1, stage1_refine, stage1_dcu_refine, stage2,
# stage2_refine, combine, confirm, final_compare.

MODE="${1:-stage1}"
SGLANG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FLYDSL_ROOT="${FLYDSL_ROOT:-/root/workspace/xiaomi/FlyDSL}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SGLANG_ROOT}/evaluation/logs/megamoe_ep4_tuning_${RUN_ID}}"
ITERS="${ITERS:-30}"
N_SEEDS="${N_SEEDS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
STAGE1_BASE_OVERRIDE="${STAGE1_OVERRIDE-}"
if [[ -z "${STAGE1_BASE_OVERRIDE}" ]]; then
  STAGE1_BASE_OVERRIDE='{}'
fi
STAGE2_BASE_OVERRIDE="${STAGE2_OVERRIDE-}"
if [[ -z "${STAGE2_BASE_OVERRIDE}" ]]; then
  STAGE2_BASE_OVERRIDE='{}'
fi

mkdir -p "${LOG_DIR}"

run_case() {
  local label="$1"
  local stage1_override="$2"
  local stage2_override="$3"
  local combine_override="${4-}"
  if [[ -z "${combine_override}" ]]; then
    combine_override='{}'
  fi
  local log_file="${LOG_DIR}/${MODE}_${label}.log"
  local json_file="${LOG_DIR}/${MODE}_${label}.jsonl"

  echo "Running ${MODE}/${label}: stage1=${stage1_override} stage2=${stage2_override} combine=${combine_override}"
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH="${FLYDSL_ROOT}" \
  ARCH=gfx950 \
  FLYDSL_GPU_ARCH=gfx950 \
  MORI_SHMEM_HEAP_SIZE=8G \
  MORI_SOCKET_IFNAME=lo \
  GLOO_SOCKET_IFNAME=lo \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  FLYDSL_MEGA_MOE_STAGE1_OVERRIDE="${stage1_override}" \
  FLYDSL_MEGA_MOE_STAGE2_OVERRIDE="${stage2_override}" \
  FLYDSL_MEGA_MOE_COMBINE_OVERRIDE="${combine_override}" \
  timeout "${TIMEOUT_SECONDS}" \
    python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
      "${SGLANG_ROOT}/evaluation/run_flydsl_megamoe_config_override.py" \
      --mega-only \
      --network v4_flash \
      --quant a8w4 \
      --tokens 1024 \
      --mtpr 1024 \
      --topk 8 \
      --experts-per-rank 64 \
      --stage2-p2p-quant auto \
      --iters "${ITERS}" \
      --n-seeds "${N_SEEDS}" \
      --measure-perf \
      --skip-acc \
      --json-out "${json_file}" \
      2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  if (( status != 0 )); then
    echo "FAILED ${MODE}/${label}: exit=${status}" >&2
  fi
}

case "${MODE}" in
  stage1)
    candidates=(
      'base|{}'
      'sbm32|{"sort_block_m":32}'
      'sbm128|{"sort_block_m":128}'
      'tn256_w4|{"tile_n":256,"num_waves":4}'
      'tn256_w8|{"tile_n":256,"num_waves":8}'
      'tn512_w4|{"tile_n":512,"num_waves":4}'
      'grid1|{"grid_mult":1}'
      'grid3|{"grid_mult":3}'
      'grid4|{"grid_mult":4}'
      'dcu64|{"num_dispatch_cu":64}'
      'dcu96|{"num_dispatch_cu":96}'
      'dcu160|{"num_dispatch_cu":160}'
      'dcu192|{"num_dispatch_cu":192}'
      'tile_resource|{"use_tile_resource":true}'
      'bnt3|{"b_nt":3}'
      'wpe1|{"waves_per_eu_hint":1}'
      'work_shards4|{"work_shards":4}'
      'no_pipe_weights|{"pipe_weights":false}'
      'no_swizzle|{"swizzle_a":false}'
      'sync_a|{"async_a_copy":false,"mfma_amajor":false}'
    )
    for entry in "${candidates[@]}"; do
      label="${entry%%|*}"
      override="${entry#*|}"
      run_case "${label}" "${override}" '{}'
    done
    ;;
  stage1_refine)
    candidates=(
      'dcu32|{"num_dispatch_cu":32}'
      'dcu40|{"num_dispatch_cu":40}'
      'dcu48|{"num_dispatch_cu":48}'
      'dcu56|{"num_dispatch_cu":56}'
      'dcu72|{"num_dispatch_cu":72}'
      'dcu80|{"num_dispatch_cu":80}'
      'dcu88|{"num_dispatch_cu":88}'
      'dcu64_grid1|{"num_dispatch_cu":64,"grid_mult":1}'
      'dcu64_grid3|{"num_dispatch_cu":64,"grid_mult":3}'
      'dcu64_grid4|{"num_dispatch_cu":64,"grid_mult":4}'
      'dcu64_tile_resource|{"num_dispatch_cu":64,"use_tile_resource":true}'
      'dcu64_wpe1|{"num_dispatch_cu":64,"waves_per_eu_hint":1}'
      'dcu64_work_shards2|{"num_dispatch_cu":64,"work_shards":2}'
      'dcu64_work_shards4|{"num_dispatch_cu":64,"work_shards":4}'
      'dcu64_chunk256|{"num_dispatch_cu":64,"payload_chunk_rows":256,"payload_tile_ready":true}'
      'dcu64_chunk512|{"num_dispatch_cu":64,"payload_chunk_rows":512,"payload_tile_ready":true}'
      'dcu64_chunk1024|{"num_dispatch_cu":64,"payload_chunk_rows":1024,"payload_tile_ready":true}'
    )
    for entry in "${candidates[@]}"; do
      label="${entry%%|*}"
      override="${entry#*|}"
      run_case "${label}" "${override}" '{}'
    done
    ;;
  stage1_dcu_refine)
    candidates=(
      'dcu8|{"num_dispatch_cu":8}'
      'dcu16|{"num_dispatch_cu":16}'
      'dcu20|{"num_dispatch_cu":20}'
      'dcu24|{"num_dispatch_cu":24}'
      'dcu28|{"num_dispatch_cu":28}'
      'dcu36|{"num_dispatch_cu":36}'
      'dcu32_grid1|{"num_dispatch_cu":32,"grid_mult":1}'
      'dcu32_grid3|{"num_dispatch_cu":32,"grid_mult":3}'
      'dcu32_grid4|{"num_dispatch_cu":32,"grid_mult":4}'
      'dcu32_tile_resource|{"num_dispatch_cu":32,"use_tile_resource":true}'
      'dcu32_wpe1|{"num_dispatch_cu":32,"waves_per_eu_hint":1}'
      'dcu32_work_shards2|{"num_dispatch_cu":32,"work_shards":2}'
      'dcu32_work_shards4|{"num_dispatch_cu":32,"work_shards":4}'
      'dcu32_bnt3|{"num_dispatch_cu":32,"b_nt":3}'
    )
    for entry in "${candidates[@]}"; do
      label="${entry%%|*}"
      override="${entry#*|}"
      run_case "${label}" "${override}" '{}'
    done
    ;;
  stage2)
    candidates=(
      'base|{}'
      'bm16|{"block_m":16}'
      'bm64|{"block_m":64}'
      'bn128|{"block_n":128}'
      'bn512|{"block_n":512}'
      'pcu128|{"persist_cu":128}'
      'pcu160|{"persist_cu":160}'
      'pcu192|{"persist_cu":192}'
      'pcu224|{"persist_cu":224}'
      'pcu256|{"persist_cu":256}'
      'contiguous|{"persist_strided":false}'
      'use_nt|{"use_nt":true}'
      'shallow_a|{"deep_a_pipeline":false}'
      'no_bhoist|{"b_hoist":false}'
      'no_ascale_prefetch|{"ascale_prefetch":false}'
      'bm64_single_b_stage|{"block_m":64,"block_n":256,"b2stage":false,"deep_a_pipeline":false}'
      'nonpersistent_sp402|{"persist":false}'
      'nonpersistent_linear|{"persist":false,"spatial_partition":0}'
      'bf16_lds|{"bf16_lds":true}'
    )
    for entry in "${candidates[@]}"; do
      label="${entry%%|*}"
      override="${entry#*|}"
      run_case "${label}" "${STAGE1_BASE_OVERRIDE}" "${override}"
    done
    ;;
  stage2_refine)
    candidates=(
      'bn128_pcu128|{"block_n":128,"persist_cu":128}'
      'bn128_pcu160|{"block_n":128,"persist_cu":160}'
      'bn128_pcu192|{"block_n":128,"persist_cu":192}'
      'bn128_pcu208|{"block_n":128,"persist_cu":208}'
      'bn128_pcu224|{"block_n":128,"persist_cu":224}'
      'bn128_pcu240|{"block_n":128,"persist_cu":240}'
      'bn128_pcu256|{"block_n":128,"persist_cu":256}'
      'bn128_contiguous|{"block_n":128,"persist_strided":false}'
      'bn128_use_nt|{"block_n":128,"use_nt":true}'
      'bn128_shallow_a|{"block_n":128,"deep_a_pipeline":false}'
      'bn128_no_bhoist|{"block_n":128,"b_hoist":false}'
      'bn128_no_ascale_prefetch|{"block_n":128,"ascale_prefetch":false}'
      'bn128_bf16_lds|{"block_n":128,"bf16_lds":true}'
      'bm16_bn128|{"block_m":16,"block_n":128}'
      'bm64_bn128|{"block_m":64,"block_n":128}'
    )
    for entry in "${candidates[@]}"; do
      label="${entry%%|*}"
      override="${entry#*|}"
      run_case "${label}" "${STAGE1_BASE_OVERRIDE}" "${override}"
    done
    ;;
  confirm)
    candidates=(
      'base|{}|{}'
      'dcu32|{"num_dispatch_cu":32}|{}'
      'dcu32_tile|{"num_dispatch_cu":32,"use_tile_resource":true}|{}'
      'dcu32_bn128|{"num_dispatch_cu":32}|{"block_n":128}'
      'dcu32_tile_bn128|{"num_dispatch_cu":32,"use_tile_resource":true}|{"block_n":128}'
      'dcu36_bn128|{"num_dispatch_cu":36}|{"block_n":128}'
      'dcu32_pcu256|{"num_dispatch_cu":32}|{"persist_cu":256}'
    )
    for entry in "${candidates[@]}"; do
      label="${entry%%|*}"
      rest="${entry#*|}"
      stage1_override="${rest%%|*}"
      stage2_override="${rest#*|}"
      run_case "${label}" "${stage1_override}" "${stage2_override}"
    done
    ;;
  combine)
    candidates=(
      'b32_w4|{"block_num":32,"warp_num_per_block":4}'
      'b32_w8|{"block_num":32,"warp_num_per_block":8}'
      'b32_w16|{"block_num":32,"warp_num_per_block":16}'
      'b64_w4|{"block_num":64,"warp_num_per_block":4}'
      'b64_w8|{"block_num":64,"warp_num_per_block":8}'
      'b64_w16|{"block_num":64,"warp_num_per_block":16}'
      'b96_w4|{"block_num":96,"warp_num_per_block":4}'
      'b96_w8|{"block_num":96,"warp_num_per_block":8}'
      'b128_w4|{"block_num":128,"warp_num_per_block":4}'
      'b128_w8|{"block_num":128,"warp_num_per_block":8}'
      'b128_w16|{"block_num":128,"warp_num_per_block":16}'
      'b192_w4|{"block_num":192,"warp_num_per_block":4}'
      'b192_w8|{"block_num":192,"warp_num_per_block":8}'
      'b256_w4|{"block_num":256,"warp_num_per_block":4}'
      'b256_w8|{"block_num":256,"warp_num_per_block":8}'
      'b256_w16|{"block_num":256,"warp_num_per_block":16}'
    )
    for entry in "${candidates[@]}"; do
      label="${entry%%|*}"
      combine_override="${entry#*|}"
      run_case "${label}" "${STAGE1_BASE_OVERRIDE}" "${STAGE2_BASE_OVERRIDE}" "${combine_override}"
    done
    ;;
  final_compare)
    candidates=(
      'base|{}|{}'
      'dcu32_bn128|{"num_dispatch_cu":32}|{"block_n":128}'
      'dcu36_bn128|{"num_dispatch_cu":36}|{"block_n":128}'
    )
    for entry in "${candidates[@]}"; do
      label="${entry%%|*}"
      rest="${entry#*|}"
      stage1_override="${rest%%|*}"
      stage2_override="${rest#*|}"
      run_case "${label}" "${stage1_override}" "${stage2_override}"
    done
    ;;
  *)
    echo "Usage: $0 stage1|stage1_refine|stage1_dcu_refine|stage2|stage2_refine|confirm|combine|final_compare" >&2
    exit 2
    ;;
esac

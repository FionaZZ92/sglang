#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export MODEL="${MODEL:-/models/MiMo-V2.5-Pro-FP4-DFlash}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-30001}"
export DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/scripts/ShareGPT_V3_unfiltered_cleaned_split.json}"

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/peak_output_matrix_${RUN_ID}}"
export OUTPUT_TOKENS="${OUTPUT_TOKENS:-1024}"
export WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}"
# Four full request waves give multiple steady full-residency plateaus. A fixed
# NUM_PROMPTS override remains available for a deliberately longer run.
export PROMPT_WAVES="${PROMPT_WAVES:-4}"
export NUM_PROMPTS="${NUM_PROMPTS:-}"
export BENCHMARK_PRESET="${BENCHMARK_PRESET:-baseline_highest}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1, observed '${DRY_RUN}'" >&2
  exit 2
fi
if ! [[ "${PROMPT_WAVES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROMPT_WAVES must be a positive integer, observed '${PROMPT_WAVES}'" >&2
  exit 2
fi
if [[ -n "${NUM_PROMPTS}" ]] && ! [[ "${NUM_PROMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROMPTS must be a positive integer when set, observed '${NUM_PROMPTS}'" >&2
  exit 2
fi
if ! [[ "${WARMUP_REQUESTS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "WARMUP_REQUESTS must be a non-negative integer, observed '${WARMUP_REQUESTS}'" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "0" ]]; then
  mkdir -p "${LOG_DIR}"
  curl -fsS --max-time 5 "http://${HOST}:${PORT}/health" >/dev/null
fi

# supplied IDs | actual input after four MiMo special tokens | label | concurrency
# CASE_SPECS may override this with newline-separated entries in the same
# format, allowing one targeted case without editing this file.
if [[ -n "${CASE_SPECS:-}" ]]; then
  mapfile -t cases <<<"${CASE_SPECS}"
else
  case "${BENCHMARK_PRESET}" in
    customer)
      cases=(
          "65532|65536|64k|204"
          "262140|262144|256k|63"
          "65532|65536|64k|196"
      )
      ;;
    baseline_highest)
      cases=(
          "65532|65536|64k|204"
          "262140|262144|256k|70"
          "524284|524288|512k|34"
          "786428|786428|768K|23"
          "1047548|1047552|1m|17"
      )
      ;;
    optimized_highest)
      cases=(
          "65532|65536|64k|258"
          "262140|262144|256k|141"
          "524284|524288|512k|70"
          "786428|786428|768K|47"
          "1047548|1047552|1m|35"
      )
      ;;
    sweep)
      cases=(
          "65532|65536|64k|16 32 48 64 80 88 96 112 114 116 118 120 124 126 128"
          "262140|262144|256k|16 32 36 38 39 40"
          "524284|524288|512k|16 18 19 20"
          "786428|786428|768K|12 14 13 16 17"
          "1047548|1047552|1m|8 9 10"
      )
      ;;
    *)
      echo "BENCHMARK_PRESET must be customer, baseline_highest, optimized_highest, or sweep, observed '${BENCHMARK_PRESET}'" >&2
      exit 2
      ;;
  esac
fi

test_count=0
for case_spec in "${cases[@]}"; do
  IFS='|' read -r supplied_ids actual_input label concurrency_spec <<<"${case_spec}"
  if ! [[ "${supplied_ids}" =~ ^[1-9][0-9]*$ && "${actual_input}" =~ ^[1-9][0-9]*$ && -n "${label}" ]]; then
    echo "Invalid CASE_SPECS entry: '${case_spec}'" >&2
    exit 2
  fi
  read -r -a concurrencies <<<"${concurrency_spec}"
  if (( ${#concurrencies[@]} == 0 )); then
    echo "CASE_SPECS entry has no concurrency values: '${case_spec}'" >&2
    exit 2
  fi

  for concurrency in "${concurrencies[@]}"; do
    if ! [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Concurrency must be a positive integer, observed '${concurrency}'" >&2
      exit 2
    fi
    if [[ -n "${NUM_PROMPTS}" ]]; then
      num_prompts="${NUM_PROMPTS}"
    else
      num_prompts="$((PROMPT_WAVES * concurrency))"
    fi
    log_file="${LOG_DIR}/benchmark_${label}_con${concurrency}.log"
    echo "================ Running ${label}: input=${actual_input}, output=${OUTPUT_TOKENS}, concurrency=${concurrency}, prompts=${num_prompts} ================"
    benchmark_cmd=(python3 -m sglang.bench_serving \
      --backend sglang \
      --model "${MODEL}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --dataset-name random \
      --random-input-len "${supplied_ids}" \
      --random-output-len "${OUTPUT_TOKENS}" \
      --random-range-ratio 1.0 \
      --dataset-path "${DATASET_PATH}" \
      --flush-cache \
      --seed 12345 \
      --num-prompts "${num_prompts}" \
      --warmup-requests "${WARMUP_REQUESTS}" \
      --max-concurrency "${concurrency}" \
      --tokenize-prompt \
      --fake-prefill)
    printf 'Command:'
    printf ' %q' "${benchmark_cmd[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == "0" ]]; then
      "${benchmark_cmd[@]}" 2>&1 | tee "${log_file}"
    fi
    test_count=$((test_count + 1))
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete; no benchmark requests sent."
else
  echo "Completed ${test_count} decoding-server benchmark tests with fake prefill"
fi

#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_SHAREGPT_DATASET="${SCRIPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_DATASET_URL="${SHAREGPT_DATASET_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-${DEFAULT_SHAREGPT_DATASET}}"

if [[ "${SHAREGPT_DATASET}" == "${DEFAULT_SHAREGPT_DATASET}" && ! -f "${SHAREGPT_DATASET}" ]]; then
  tmp_dataset="$(mktemp "${SCRIPT_DIR}/.ShareGPT_V3_unfiltered_cleaned_split.json.tmp.XXXXXX")"
  echo "ShareGPT dataset not found in ${SCRIPT_DIR}; downloading to ${SHAREGPT_DATASET}"
  curl -L --fail --retry 3 --connect-timeout 10 \
    --output "${tmp_dataset}" \
    "${SHAREGPT_DATASET_URL}"
  mv "${tmp_dataset}" "${SHAREGPT_DATASET}"
fi

if [[ ! -f "${SHAREGPT_DATASET}" ]]; then
  echo "ShareGPT dataset not found: ${SHAREGPT_DATASET}" >&2
  echo "Set SHAREGPT_DATASET to an existing file, or allow download to ${DEFAULT_SHAREGPT_DATASET}." >&2
  exit 2
fi

TOKEN_LIST=(8192 65536 262144)
output_tokens=1024
concurrency_list=(16 32 64 128)
LOG_DIR="${LOG_DIR:-logs/pd-disagg-benchmark-decode}"
mkdir -p "$LOG_DIR"

# Loop execution: run each length multiple times (1 time), each run with a different concurrency
for input_tokens in "${TOKEN_LIST[@]}"; do
  for concurrency in "${concurrency_list[@]}"; do # iterate over the concurrency list
    # Run 1 round of test for each length and concurrency combination
    run=1 # only run one round
    echo -e "\n============================================================"
    echo "Testing: Input Token = ${input_tokens}, Concurrency = ${concurrency} | Run ${run}"
    echo "Log file: benchmark_decode_${input_tokens}_con${concurrency}_${run}.log"
    echo "============================================================"

    # Run benchmark: display in terminal + write to a separate log
    python3 -m sglang.bench_serving \
        --backend sglang \
        --model /models/MiMo-V2.5-Pro \
        --host 0.0.0.0 \
        --port 40000 \
        --dataset-name random \
        --random-input-len ${input_tokens} \
        --random-output-len ${output_tokens} \
        --random-range-ratio 1.0 \
        --dataset-path "${SHAREGPT_DATASET}" \
        --seed 12345 \
        --num-prompts 256 \
        --warmup-requests 32 \
        --max-concurrency ${concurrency} \
        --pd-separated \
        2>&1 | tee "$LOG_DIR/benchmark_${input_tokens}_con${concurrency}.log"

    echo -e "============================================================\n"
  done
done

echo "All lengths and concurrency tests completed!"

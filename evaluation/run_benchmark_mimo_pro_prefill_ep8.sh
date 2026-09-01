#!/usr/bin/env bash
set -euo pipefail

# Client benchmark for an already-running FlyDSL MegaMoE server (TP8/EP8 by default).
# Start it first with ./launch_tp8_ep8_flydsl_megamoe_prefill.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.5/}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-30001}"
EXPECTED_TP_SIZE="${EXPECTED_TP_SIZE:-${TP_SIZE:-8}}"
EXPECTED_EP_SIZE="${EXPECTED_EP_SIZE:-${EP_SIZE:-${EXPECTED_TP_SIZE}}}"
if ! [[ "${EXPECTED_TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_TP_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ -n "${EXPECTED_DP_SIZE:-}" ]]; then
  EXPECTED_DP_SIZE="${EXPECTED_DP_SIZE}"
elif [[ -n "${DP_SIZE:-}" ]]; then
  EXPECTED_DP_SIZE="${DP_SIZE}"
else
  EXPECTED_DP_SIZE=$((EXPECTED_TP_SIZE / 4))
fi
for parallel_size in "${EXPECTED_DP_SIZE}" "${EXPECTED_EP_SIZE}"; do
  if ! [[ "${parallel_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_DP_SIZE and EXPECTED_EP_SIZE must be positive integers" >&2
    exit 2
  fi
done

read -r -a TOKEN_LIST <<< "${TOKEN_LIST_OVERRIDE:-4096 8192 16384 32768 65536 131068 262144 524284 786428 1047548}"
output_tokens=1
small_input_concurrency_list="${SMALL_INPUT_CONCURRENCY_LIST_OVERRIDE:-${SHORT_CONCURRENCY_LIST_OVERRIDE:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32}}"
short_concurrency_list="${SHORT_CONCURRENCY_LIST_OVERRIDE:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}"
long_concurrency_list="${LONG_CONCURRENCY_LIST_OVERRIDE:-1}"
warmup_requests="${WARMUP_REQUESTS_OVERRIDE:-4}"
small_input_num_prompts="${SMALL_INPUT_NUM_PROMPTS_OVERRIDE:-64}"
prompt_waves="${PROMPT_WAVES:-4}"
min_num_prompts="${MIN_NUM_PROMPTS:-32}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/benchmark_tp${EXPECTED_TP_SIZE}_ep${EXPECTED_EP_SIZE}_flydsl_megamoe_prefill_${RUN_ID}}"
mkdir -p "${LOG_DIR}"

if ! [[ "${small_input_num_prompts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMALL_INPUT_NUM_PROMPTS_OVERRIDE must be a positive integer, observed '${small_input_num_prompts}'" >&2
  exit 2
fi

server_info_file="${LOG_DIR}/server_info.json"
curl --max-time 10 --fail --silent --show-error \
  "http://${SERVER_HOST}:${SERVER_PORT}/server_info" > "${server_info_file}"
python3 - "${server_info_file}" "${MODEL_PATH}" "${EXPECTED_TP_SIZE}" "${EXPECTED_DP_SIZE}" "${EXPECTED_EP_SIZE}" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    info = json.load(file)

ep_size = info.get("ep_size")
tp_size = info.get("tp_size")
dp_size = info.get("dp_size")
a2a_backend = info.get("moe_a2a_backend")
served_model = os.path.normpath(info.get("model_path", ""))
expected_model = os.path.normpath(sys.argv[2])
expected_tp_size = int(sys.argv[3])
expected_dp_size = int(sys.argv[4])
expected_ep_size = int(sys.argv[5])
if (
    tp_size != expected_tp_size
    or dp_size != expected_dp_size
    or ep_size != expected_ep_size
    or a2a_backend != "megamoe"
    or served_model != expected_model
):
    raise SystemExit(
        "Unexpected FlyDSL MegaMoE server configuration: "
        f"model={served_model!r}, tp={tp_size!r}, dp={dp_size!r}, "
        f"ep={ep_size!r}, moe_a2a_backend={a2a_backend!r}"
    )
print(
    "Connected server: "
    f"tp={tp_size}, dp={dp_size}, ep={ep_size}, "
    f"a2a={a2a_backend}, runner={info.get('moe_runner_backend')}"
)
PY

concurrency_spec_for_input() {
  local input_tokens="$1"
  if [[ -n "${CONCURRENCY_LIST_OVERRIDE:-}" ]]; then
    echo "${CONCURRENCY_LIST_OVERRIDE}"
  elif (( input_tokens <= 8192 )); then
    echo "${small_input_concurrency_list}"
  elif (( input_tokens <= 65536 )); then
    echo "${short_concurrency_list}"
  else
    echo "${long_concurrency_list}"
  fi
}

for input_tokens in "${TOKEN_LIST[@]}"; do
  read -r -a concurrency_list <<< "$(concurrency_spec_for_input "${input_tokens}")"
  for concurrency in "${concurrency_list[@]}"; do
    run=1
    if [[ -n "${NUM_PROMPTS_OVERRIDE:-}" ]]; then
      num_prompts="${NUM_PROMPTS_OVERRIDE}"
    elif (( input_tokens <= 8192 )); then
      num_prompts="${small_input_num_prompts}"
    else
      num_prompts=$((prompt_waves * concurrency))
      if (( num_prompts < min_num_prompts )); then
        num_prompts="${min_num_prompts}"
      fi
    fi

    log_file="${LOG_DIR}/benchmark_${input_tokens}_con${concurrency}.log"
    json_file="${LOG_DIR}/benchmark_${input_tokens}_con${concurrency}.jsonl"
    echo -e "\n============================================================"
    echo "Testing EP${EXPECTED_EP_SIZE}: Input Token = ${input_tokens}, Concurrency = ${concurrency} | Run ${run}"
    echo "Measured prompts = ${num_prompts}, warmups = ${warmup_requests}"
    echo "Log file: ${log_file}"
    echo "============================================================"

    python3 -m sglang.bench_serving \
      --backend sglang \
      --model "${MODEL_PATH}" \
      --host "${SERVER_HOST}" \
      --port "${SERVER_PORT}" \
      --dataset-name random \
      --random-input-len "${input_tokens}" \
      --random-output-len "${output_tokens}" \
      --random-range-ratio 1.0 \
      --flush-cache \
      --seed 12345 \
      --num-prompts "${num_prompts}" \
      --warmup-requests "${warmup_requests}" \
      --max-concurrency "${concurrency}" \
      --tokenize-prompt \
      --output-file "${json_file}" \
      2>&1 | tee "${log_file}"
    echo -e "============================================================\n"
  done
done

echo "All EP${EXPECTED_EP_SIZE} lengths and concurrency tests completed. Results: ${LOG_DIR}"

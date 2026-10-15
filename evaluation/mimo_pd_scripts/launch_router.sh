#!/bin/bash
set -euo pipefail
# launch_router.sh — SGLang PD router
#
# Run this on the PREFILL node after both P and D servers are ready.

# Defaults are the verified host IPs.
# Override with environment variables when running on a different network.
PREFILL_HOST_IP="${PREFILL_HOST_IP:-10.2.80.21}"
DECODE_HOST_IP="${DECODE_HOST_IP:-10.2.80.29}"
LOG_DIR="${LOG_DIR:-logs/pd-disagg-router}"

mkdir -p "$LOG_DIR"

echo "=== Launching SGLang PD Router ==="
echo "Prefill: http://$PREFILL_HOST_IP:30000"
echo "Decode:  http://$DECODE_HOST_IP:30001"

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://$PREFILL_HOST_IP:30000" \
  --decode "http://$DECODE_HOST_IP:30001" \
  --host 0.0.0.0 --port 40000 \
  2>&1 | tee "$LOG_DIR/router.log"

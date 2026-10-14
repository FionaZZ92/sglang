## 1. Prepare docker image

### gfx942

```bash
docker run -it --name mimo-sgl-opt-gfx942 --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v $HOME:/root/workspace  -v /data/models:/models  rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510
```

### gfx950

```bash
docker run -it --name mimo-sgl-opt-gfx950 --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v $HOME:/root/workspace  -v /data/models:/models  rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260510
```

## 2. Environment setup

### Clean up pre-installed environment

```bash
pip uninstall sglang sglang-kernel sgl-kernel amd-aiter flydsl mimo-flydsl-kernels -y
rm -rf /sgl-workspace/sglang /sgl-workspace/aiter
```

### Install sglang

```bash
cd /sgl-workspace
git clone https://github.com/sammysun0711/sglang -b mimo-opt-mxfp4-dflash
cd sglang && pip install --upgrade pip && cd sgl-kernel && python3 setup_rocm.py install
cd ..  && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml && pip install -e "python[all_hip]"
cd ..
```

### Install AITER & pyhip dependency

```bash
git clone https://github.com/sammysun0711/aiter -b integration/mimo-fp4-dflash
cd aiter
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
pip install -e .
cd ..
```

### Install FlyDSL

Install FlyDSL runtime version that compatible with aiter

```bash
pip install flydsl==0.3.2
```

## 3. Run baseline single node prefill benchmark

Baseline prefill keeps quick-reduce disabled, mixed router disabled, FlyDSL prefill disabled, Gluon decode, BF16 KV cache

### Launch TBO-off server

```bash
cd /sgl-workspace/sglang/evaluation/mimo_dflash_scripts
MEM_FRACTION_STATIC=0.80 \
CHUNKED_PREFILL_SIZE=65536 \
DISABLE_RADIX_CACHE=1 \
ENABLE_TWO_BATCH_OVERLAP=0 \
MAX_RUNNING_REQUESTS=96 \
SGLANG_DFLASH_PERFORMANCE_MODE=1 \
./launch_tp8_noep_aiter_dflash_accuracy_baseline.sh
```

### Run prefill benchmark

```bash
cd /sgl-workspace/sglang/evaluation/
./run_benchmark_mimo_pro_prefill.sh
```

Stop the TBO-off server, verify that the GPUs are clear, then launch a fresh
TBO-on server and run the same client matrix:

```bash
cd /sgl-workspace/sglang/evaluation/mimo_dflash_scripts
MEM_FRACTION_STATIC=0.75 \
CHUNKED_PREFILL_SIZE=65536 \
DISABLE_RADIX_CACHE=1 \
ENABLE_TWO_BATCH_OVERLAP=1 \
SGLANG_TBO_MIM_SEQ_LEN=2000 \
MAX_RUNNING_REQUESTS=32 \
SGLANG_DFLASH_PERFORMANCE_MODE=1 \
./launch_tp8_noep_aiter_dflash_accuracy_baseline.sh
```

The qualified TBO-on memory fraction is `0.75`. The `0.80` configuration can
exhaust transient HSA memory even though the static KV pools themselves are
not full.

## 4. Run baseline single node decode benchmark with fake prefill

Baseline fake-prefill decode keeps quick-reduce disabled, mixed router disabled,
FlyDSL prefill disabled, FlyDSL decode, and BF16 target/draft KV cache.

The fake-prefill decode capacity must fit both the full-token and SWA-token
pools:

```text
(C + 1) * (ISL + OSL) <= max_total_num_tokens
(C + 1) * OSL         <= swa_layer_tokens
```

Read `max_total_num_tokens` from `/server_info` and `swa_layer_tokens` from
the server initialization log. If SWA is the smaller bound for a short-ISL
case, increase `SWA_FULL_TOKENS_RATIO` only until the SWA slot count is at
least the full-token slot count (or the desired concurrency cap). Allocating
more than that reduces the full-token pool without increasing concurrency.

### Launch server

```bash
cd /sgl-workspace/sglang/evaluation/mimo_dflash_scripts
MEM_FRACTION_STATIC=0.85 \
DISABLE_RADIX_CACHE=1 \
SWA_FULL_TOKENS_RATIO=0.016 \
MAX_RUNNING_REQUESTS=286 \
  ./launch_tp8_noep_aiter_dflash_decode_fake_prefill_baseline.sh
```

### Run the matching decode point

```bash
cd /sgl-workspace/sglang/evaluation
CASE_SPECS='65532|65536|64k|285' \
PROMPT_WAVES=4 \
WARMUP_REQUESTS=32 \
./run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh
```

### Run decode throughput analysis

```bash
python3 analyze_server_output_throughput.py \
  <path-to-server-log> \
  --target-bs <C> \
  --exact-bs \
  --min-tps 40 \
  --max-tpot 25
```

Use the analyzer's filtered server median as the reported output throughput.
The analyzer reports the following median-consistent derived values:

```text
TPS/request = filtered server median / C
TPOT (ms)   = 1000 * C / filtered server median
```

The analyzer also retains `mean_tpot_ms` and `mean_tps_per_request` as explicit
mean-based reference values.

## 5. Profiling & Analysis

```bash
cd /sgl-workspace/sglang/evaluation
./run_sglang_profile.sh
```

## 8. Run real-MTP ShareGPT dataset accuracy gate

```bash
cd /sgl-workspace/sglang/evaluation
./run_sharegpt_mtp_accuracy_test.sh
```

## 9. Run swe-bench & accuracy benchmark test

Follow up customer's swe-bench accuracy verification guide.

## 10. H200 performance evaluation

Follow up customer's shared performance data

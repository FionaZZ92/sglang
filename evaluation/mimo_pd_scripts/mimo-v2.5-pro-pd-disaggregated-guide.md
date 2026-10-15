## 1. Prepare docker image for gfx950 with AINIC
### Prefill container on node 0
```bash
docker run -it --name mimo-sgl-opt-gfx950-prefill --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem --device=/dev/infiniband -v $HOME:/root/workspace  -v /data/models:/models  -v /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0:/lib/x86_64-linux-gnu/libibverbs.so.1 -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so -v /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so -v /etc/libibverbs.d:/etc/libibverbs.d:ro rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260510
```

### Decode container on node 1
```bash
docker run -it --name mimo-sgl-opt-gfx950-decode --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem --device=/dev/infiniband -v $HOME:/root/workspace  -v /data/models:/models  -v /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0:/lib/x86_64-linux-gnu/libibverbs.so.1 -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so -v /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so -v /etc/libibverbs.d:/etc/libibverbs.d:ro rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260510
```
## 2. Environment setup for both prefill container and decode container

### Clean up pre-installed environment
```bash
pip uninstall sglang sglang-kernel sgl-kernel amd-aiter flydsl mimo-flydsl-kernels -y
rm -rf /sgl-workspace/sglang /sgl-workspace/aiter
```

### Install sglang
```bash
cd /sgl-workspace/
git clone https://github.com/sammysun0711/sglang -b mimo-opt
cd sglang && pip install --upgrade pip && cd sgl-kernel && python3 setup_rocm.py install
cd ..  && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml && pip install -e "python[all_hip]"
cd ..
```
### Install AITER & pyhip dependency
```bash
git clone https://github.com/sammysun0711/aiter -b mimo-opt
cd aiter
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
pip install -e .
cd ..
```

### Install FlyDSL
FlyDSL installation should be after AITER installation, since it will update FlyDSL 0.2.4 runtime and FlyDSL FA & PA decode kernels.
```bash
git clone https://github.com/sammysun0711/FlyDSL -b mimo-opt
cd FlyDSL/wheels && python3 -m pip install --no-deps --force-reinstall flydsl-0.2.4-*.whl mimo_flydsl_kernels-0.1.3-*.whl
cd ../..
```

### Check RDMA status
```bash
rdma link show
ibv_devinfo -v | head -30
```

### Setup passwordless ssh connection
```bash
git clone https://github.com/sammysun0711/llm-distributed-inference.git
cd llm-distributed-inference/sglang/
# On node 0, create remote ssh access in prefill container to node 1 decode container and vice versa
./scripts/setup_docker_passwdless_ssh.sh mi355-gpu-14
```
checkout that node 0 and node 1 can access via ssh without password.
```shell
ssh mi355-gpu-14
```

## 3. Launch PD disagg prefill server on node 0
Please note: update `SGLANG_HOST_IP`, `MC_GID_INDEX` and `DISAGGREGATION_IB_DEVICE` based on network settings.

TBO on prefill node is disable by default, enable via `ENABLE_TWO_BATCH_OVERLAP=1`
```bash
cd /sgl-workspace/sglang/evaluation/mimo_pd_scripts
./launch_tp8_noep_prefill_aiter_mtp.sh
```

## 4. Launch PD disagg decode server on node 1
Please note: update `SGLANG_HOST_IP`, `MC_GID_INDEX` and `DISAGGREGATION_IB_DEVICE` based on network settings
```bash
cd /sgl-workspace/sglang/evaluation/mimo_pd_scripts
./launch_tp8_noep_decode_aiter_mtp.sh
```

## 5. Launch PD disagg router on node 0
Please note: update `PREFILL_HOST_IP` and `DECODE_HOST_IP` based on network settings
```bash
cd /sgl-workspace/sglang/evaluation/mimo_pd_scripts
./launch_router.sh
```

## 6. Run PD disagg benchmark
### Run prefill benchmark on node 0
```bash
cd /sgl-workspace/sglang/evaluation/mimo_pd_scripts
./run_benchmark_mimo_pro_prefill.sh
```

### Run decode benchmark on node 0
```bash
cd /sgl-workspace/sglang/evaluation/mimo_pd_scripts
./run_benchmark_mimo_pro_decode.sh
```

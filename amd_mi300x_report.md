# Event-SAE OpenPI on AMD MI300X

This report summarizes the AMD Developer Cloud replication work for
Event-SAE with the OpenPI backbone. It is intended to help someone
understand the purpose of the project, reproduce the setup, and verify
the same pipeline milestones on an AMD AI platform.

For the chronological lab notebook with every intermediate command,
error, and fix, see [amd_mi300x_openpi.md](amd_mi300x_openpi.md).

## Purpose

The goal was to show that the Event-SAE OpenPI workflow can run on AMD
AI hardware, specifically an AMD MI300X GPU droplet.

The validated pipeline is:

```text
OpenPI policy server on AMD MI300X
  -> LIBERO simulation client
  -> dense activation collection
  -> BatchTopK SAE training on ROCm PyTorch
  -> top-k SAE serving
```

This is a simulation and infrastructure pipeline. A physical robot was
not connected during this setup. LIBERO simulation was used as the
client environment for policy rollout and activation collection.

## Result Summary

The AMD pipeline is functional for setup-scale runs.

Validated:

```text
ROCm sees the MI300X GPU
JAX sees RocmDevice(id=0)
OpenPI pi05_libero checkpoint loads on AMD
LIBERO client connects to OpenPI server over websocket
OpenPI actions are finite after the AMD/JAX precision fix
Dense activation collection writes finite layer shards
Tiny BatchTopK SAE training runs on ROCm PyTorch
Top-k SAE serving loads the trained SAE and writes top feature rows
```

Important caveat:

```text
The SAE trained here is intentionally tiny and is only a pipeline smoke
test. It is not a scientifically meaningful SAE checkpoint.
```

## AMD Cloud Configuration

Droplet:

```text
Name: 0.4.35-gpu-mi300x1-192gb-devcloud-atl1
Region: ATL1
Image: JAX 0.4.35 on Ubuntu 24.04
GPU: 1x MI300X
VRAM: 192 GB
vCPU: 20
RAM: 240 GB
Boot disk: 720 GB NVMe
Scratch disk: 5 TB NVMe
Public IPv4: 134.199.206.96
Cost: $1.99/hr
```

Host ROCm validation:

```bash
rocm-smi
```

Observed MI300X device:

```text
AMD Instinct MI300X VF
```

## Repository Layout

Important local paths:

```text
README.md                         project overview
docs/openpi.md                    upstream OpenPI runbook
docs/amd_mi300x_openpi.md         chronological AMD setup log
docs/amd_mi300x_report.md         this replication report

scripts/openpi/serve_policy.py    OpenPI policy server with Event-SAE modes
scripts/openpi/eval_libero.py     LIBERO simulation client for OpenPI
scripts/train_sae.py              offline SAE training CLI

event_sae/openpi/activations.py   top-k OpenPI activation collector
event_sae/openpi/eval/runner.py   OpenPI LIBERO client loop
event_sae/train.py                offline BatchTopK SAE trainer

external/openpi-event-sae         OpenPI fork used by Event-SAE
external/dictionary_learning      SAE training dependency
external/awe                      keyframe extraction dependency
```

Main OpenPI serving modes:

```text
dense      collect dense hidden activations from selected layers
topk       run trained SAE online and store top feature IDs/values
intervene  apply SAE reconstruction or feature perturbation hooks
```

Validated in this AMD setup:

```text
dense: done
topk: done
intervene: not yet tested
```

## Code Changes Made

### AMD JAX Precision Fix

File:

```text
scripts/openpi/serve_policy.py
```

The server now sets this before importing OpenPI/JAX:

```python
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
```

Why:

The default ROCm/JAX matmul behavior on the MI300X produced XLA GEMM
autotuner mismatch warnings and NaN OpenPI actions. Setting highest
matmul precision made actions finite and removed the warnings in the
validated smoke tests.

Rejected workaround:

```bash
XLA_FLAGS=--xla_gpu_autotune_level=0
```

This caused:

```text
miopenStatusUnknownError
```

### LIBERO Init-State Compatibility

File:

```text
event_sae/openpi/eval/runner.py
```

PyTorch 2.6 changed `torch.load` to default to `weights_only=True`.
LIBERO init-state files are older trusted benchmark assets and need the
legacy behavior. The code now scopes `weights_only=False` only around
the LIBERO init-state load.

### Tiny SAE Training Compatibility

File:

```text
event_sae/train.py
```

The full trainer default uses `warmup_steps=1000`, which is correct for
large runs but invalid for tiny smoke tests such as `steps=20`. The
trainer now computes proportional warmup/decay for short runs while
preserving the full-run behavior.

## Container Strategy

Two runtime styles were used.

### JAX/OpenPI Serving Container

Persistent container:

```text
event-sae-jax-server
```

Committed image:

```text
event-sae-pipeline-env:latest
```

Purpose:

```text
OpenPI server
LIBERO client smoke tests
dense activation collection
top-k SAE serving
```

Key Python runtime:

```text
Python 3.11.13
JAX 0.5.0
jaxlib 0.5.0
jax-rocm60-plugin 0.5.0
NumPy 1.26.4
```

ROCm compatibility note:

The container needed this local compatibility symlink:

```bash
ln -sf /opt/rocm-6.4.2/lib/libamd_comgr.so.3 \
  /opt/rocm-6.4.2/lib/libamd_comgr.so.2
```

### PyTorch ROCm Training Container

Image:

```text
rocm/pytorch:latest
```

Validated:

```text
torch 2.10.0+rocm7.2.3.git1a270074
torch.cuda.is_available() True
device AMD Instinct MI300X VF
```

Purpose:

```text
offline BatchTopK SAE training
```

## Reproduction Steps

The commands below assume the droplet already has:

```text
/workspace/Event-SAE
/workspace/Event-SAE/external/openpi-event-sae
/workspace/Event-SAE/external/dictionary_learning
/workspace/Event-SAE/external/awe
```

and the JAX/OpenPI serving environment has been created as described in
the chronological log.

### 1. Start Dense OpenPI Server

```bash
cd /workspace/Event-SAE

PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py \
  --mode dense \
  --env libero \
  --port 8000 \
  --output-root /workspace/Event-SAE/logs/openpi/sae_collection \
  --run-name amd_dense_task0_3trial_25step
```

Expected server signal:

```text
server listening on 0.0.0.0:8000
```

### 2. Run LIBERO Client For Dense Collection

In another shell inside the same serving container:

```bash
cd /workspace/Event-SAE

PYENV_VERSION=3.11.13 \
PYTHONPATH=/workspace/Event-SAE/external/openpi-event-sae/third_party/libero:$PYTHONPATH \
python scripts/openpi/eval_libero.py \
  --config configs/examples/openpi/eval_libero_spatial.yaml \
  --override server.host=127.0.0.1 \
  --override server.port=8000 \
  --override env.task_ids=[0] \
  --override env.num_trials_per_task=3 \
  --override env.max_steps=25 \
  --override env.num_steps_wait=1 \
  --override logging.save_video=false \
  --override logging.run_tag=amd_dense_task0_3trial_25step_client \
  --override sae_collect.enabled=true \
  --override sae_collect.capture_target=action_expert \
  --override sae_collect.layer_idxs=0,5,11,17 \
  --libero-root external/openpi-event-sae/third_party/libero
```

Validated output:

```text
total_episodes=3
activation_index_records 600
predicted_action_has_nan False
layer_00_shard_000000.pt (1500, 1024) finite True
layer_05_shard_000000.pt (1500, 1024) finite True
layer_11_shard_000000.pt (1500, 1024) finite True
layer_17_shard_000000.pt (1500, 1024) finite True
SERVER_WARN_GEMM_MISMATCH=0
SERVER_ERR=0
```

Dense collection output:

```text
/workspace/Event-SAE/logs/openpi/sae_collection/amd_dense_task0_3trial_25step
```

### 3. Train Tiny SAE On Layer 17

Run in a ROCm PyTorch container:

```bash
docker run --rm -i \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size=32G \
  --security-opt seccomp=unconfined \
  -v /workspace/Event-SAE:/workspace/Event-SAE \
  -w /workspace/Event-SAE \
  rocm/pytorch:latest \
  bash
```

Inside the container:

```bash
python -m pip install -e external/dictionary_learning

PYTHONPATH=/workspace/Event-SAE python scripts/train_sae.py \
  --config configs/examples/openpi/train_sae_libero_spatial.yaml \
  --save-dir logs/openpi/sae/amd_tiny_layer17 \
  --override data_dir=logs/openpi/sae_collection/amd_dense_task0_3trial_25step/sae_activations/post_mlp_residual \
  --override layer_idx=17 \
  --override activation_dim=1024 \
  --override dict_size=1024 \
  --override k=16 \
  --override steps=20 \
  --override batch_size=64 \
  --override num_workers=0 \
  --override pin_memory=false \
  --override device=cuda:0 \
  --override run_tag=amd_tiny_layer17
```

Validated checkpoint:

```text
/workspace/Event-SAE/logs/openpi/sae/amd_tiny_layer17/trainer_0/ae.pt
```

Checkpoint contents:

```text
b_dec (1024,) torch.float32 finite True
k () torch.int32 finite True
threshold () torch.float32 finite True
decoder.weight (1024, 1024) torch.float32 finite True
encoder.weight (1024, 1024) torch.float32 finite True
encoder.bias (1024,) torch.float32 finite True
```

### 4. Run TopK SAE Serving Smoke

Server:

```bash
cd /workspace/Event-SAE

PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py \
  --mode topk \
  --env libero \
  --port 8000 \
  --output-root /workspace/Event-SAE/logs/openpi/sae_collection \
  --run-name amd_topk_tiny_layer17_smoke \
  --capture-target action_expert \
  --layer-indices 17 \
  --sae-checkpoint /workspace/Event-SAE/logs/openpi/sae/amd_tiny_layer17/trainer_0/ae.pt \
  --topk 16 \
  --rows-per-shard 1000
```

Client:

```bash
cd /workspace/Event-SAE

PYENV_VERSION=3.11.13 \
PYTHONPATH=/workspace/Event-SAE/external/openpi-event-sae/third_party/libero:$PYTHONPATH \
python scripts/openpi/eval_libero.py \
  --config configs/examples/openpi/eval_libero_spatial.yaml \
  --override server.host=127.0.0.1 \
  --override server.port=8000 \
  --override env.task_ids=[0] \
  --override env.num_trials_per_task=1 \
  --override env.max_steps=5 \
  --override env.num_steps_wait=1 \
  --override logging.save_video=false \
  --override logging.run_tag=amd_topk_tiny_layer17_client \
  --override sae_collect.enabled=true \
  --override sae_collect.mode=topk \
  --override sae_collect.capture_target=action_expert \
  --override "sae_collect.layer_idxs='17'" \
  --libero-root external/openpi-event-sae/third_party/libero
```

Validated output:

```text
activation_records=10
num_rows=100
CLIENT_STATUS=0
SERVER_WARN_GEMM_MISMATCH=0
SERVER_ERR=0
```

TopK output:

```text
/workspace/Event-SAE/logs/openpi/sae_collection/amd_topk_tiny_layer17_smoke/sae_activations/post_mlp_residual/shard_000000.pt
```

The top-k shard contains:

```text
top_feature_ids (100, 16)
top_feature_vals (100, 16)
```

## Known Warnings

### Robosuite EGL Cleanup

LIBERO smoke tests emitted EGL cleanup warnings at process shutdown:

```text
OpenGL.raw.EGL._errors.EGLError: EGL_NOT_INITIALIZED
```

The runs completed with `CLIENT_STATUS=0`, and artifacts were written.
This warning appears during GL context cleanup, not during policy
inference or activation collection.

### Gym Deprecation Warning

The LIBERO dependency stack uses old `gym`, which emits a warning about
Gym being unmaintained. This is expected for the current LIBERO stack.

### Tiny SAE Is Not Scientific

The tiny SAE run uses:

```text
steps=20
batch_size=64
k=16
```

It exists only to prove the training and top-k serving path. Full
experiments should use larger collections and paper-scale training
settings.

## What Remains

Recommended next work:

```text
1. Test --mode intervene using the tiny SAE.
2. Convert successful commands into scripts.
3. Create a Dockerfile or setup script for event-sae-pipeline-env.
4. Scale dense collection to real task/trial counts.
5. Train real SAEs with paper-scale settings.
6. Run event extraction, clustering, scoring, and intervention ranking.
7. Only later, consider physical robot integration.
```

The robot is not required for the current AMD compatibility milestone.
The validated target here is simulated OpenPI/Event-SAE reproducibility
on AMD MI300X.

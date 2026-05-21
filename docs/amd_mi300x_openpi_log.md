# Event-SAE OpenPI on AMD MI300X

This is a living setup log for reproducing the Event-SAE OpenPI workflow
on AMD Developer Cloud. Only completed or verified steps are recorded
here. Add the next section only after that step is actually done.

## Local Repo

Event-SAE was cloned locally into a separate workspace:

```text
/home/trossen-ai/event-sae-workspace/Event-SAE
```

Clone source:

```text
https://github.com/xc-j/Event-SAE.git
```

Current local repo state when cloned:

```text
Branch: main
Latest commit: f7a0000 Add paper link;
```

## AMD Droplet Created

AMD Developer Cloud droplet:

```text
Name: 0.4.35-gpu-mi300x1-192gb-devcloud-atl1
Status: Active
Region: ATL1
Project: default-atl1
Image: JAX 0.4.35 on Ubuntu 24.04
GPU: 1x MI300X
VRAM: 192 GB
vCPU: 20
RAM: 240 GB
Boot disk: 720 GB NVMe
Scratch disk: 5 TB NVMe
Public IPv4: 134.199.206.96
Private IP: 10.128.0.2
Cost: $1.99/hr
```

The cloud console showed a security notice recommending system package
updates and a reboot before production use.

## SSH Connectivity Verified

SSH connectivity from this workstation to the droplet was verified with:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  root@134.199.206.96 'hostname && uname -a'
```

Observed output:

```text
0
Linux 0 6.8.0-60-generic #63-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 15 19:04:15 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
```

## System Update Started

The cloud image security notice recommended:

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo reboot
```

During `openssh-server` package configuration, the package manager asked
what to do with a locally modified `/etc/ssh/sshd_config`.

Choice made:

```text
keep the local version currently installed
```

Reasoning:

- This is a remote cloud droplet.
- The provider image may have SSH settings needed for root/key access.
- Replacing `sshd_config` with the package maintainer version could
  accidentally change remote login behavior.
- Keeping the local version is the conservative option when connected
  over SSH.

The package upgrade proceeded after that choice. A reboot was requested.

## Host ROCm Verified

After reconnecting, ROCm SMI was run on the host:

```bash
rocm-smi
```

Observed concise output:

```text
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
0       1     0x74b5,   21947  38.0°C      155.0W    NPS1, SPX, 0        131Mhz  900Mhz  0%   auto  750.0W  0%     0%
```

Reasoning:

- The host can see the MI300X through ROCm.
- VRAM usage and GPU utilization were both `0%`, meaning the GPU was
  visible and idle.
- This verifies the low-level ROCm driver path before checking any
  Python framework.

## Host Python Does Not Contain JAX

The first JAX check was attempted on the host:

```bash
python - <<'PY'
import jax
print("jax", jax.__version__)
print("hi")
print(jax.devices())
PY
```

Observed result:

```text
Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3
```

The command was retried with `python3`:

```bash
python3 - <<'PY'
import jax
print("jax", jax.__version__)
print("hi")
print(jax.devices())
PY
```

Observed result:

```text
ModuleNotFoundError: No module named 'jax'
```

Reasoning:

- The Ubuntu host has system Python as `python3`, not `python`.
- JAX is not installed into the host system Python.
- This does not mean the JAX image is broken. It means the marketplace
  image likely keeps the JAX environment somewhere else, most likely in
  its prebuilt container or Jupyter setup.

## JAX Environment Identified As Docker-Based

From this workstation, the droplet was inspected over SSH. The host
contains Docker and ROCm directories:

```text
/usr/bin/python3
/usr/bin/docker
/opt/rocm -> /etc/alternatives/rocm
/opt/rocm-6.4.1
```

Docker containers/images were listed:

```bash
docker ps -a
docker images
```

Observed:

```text
CONTAINER ID   IMAGE     COMMAND                  STATUS                       NAMES
020c9fe6a9f4   rocm      "/bin/sh -c 'jupyter…"   Exited (137) 6 minutes ago   rocm

IMAGE             ID             DISK USAGE
rocm/jax:latest   3eaecd08f997    20.4GB
rocm:latest       db36a5b95f27     21GB
```

The `rocm` container logs show JupyterLab starting from a Python
environment under:

```text
/pyenv/versions/3.10.18
```

Reasoning:

- The AMD JAX quick-start image is Docker/Jupyter based.
- The system Python on the host is not the intended JAX runtime.
- JAX should be tested inside the `rocm` container, not directly on the
  host.
- The `rocm` container exited after reboot, so it must be restarted
  before checking JAX inside it.

## JAX Verified Inside ROCm Container

The existing `rocm` container was initially stopped:

```bash
docker ps -a
```

Observed:

```text
CONTAINER ID   IMAGE     COMMAND                  STATUS                       NAMES
020c9fe6a9f4   rocm      "/bin/sh -c 'jupyter..." Exited (137) 7 minutes ago   rocm
```

The container was restarted and entered:

```bash
docker start rocm
docker exec -it rocm bash
```

Inside the container, JAX was tested:

```bash
python - <<'PY'
import jax
print("jax", jax.__version__)
print(jax.devices())
PY
```

Observed output:

```text
jax 0.4.35
[RocmDevice(id=0)]
```

Reasoning:

- This confirms the AMD Developer Cloud JAX quick-start image works as a
  containerized environment.
- JAX is not installed on the host Python, but it is installed in the
  `rocm` container.
- `RocmDevice(id=0)` confirms JAX sees the MI300X through ROCm.
- Future OpenPI/JAX work should start inside this container unless we
  intentionally create a separate ROCm-compatible environment.

## PyTorch Not Present In JAX Container

Inside the same `rocm` container, PyTorch was tested:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda api available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Observed output:

```text
ModuleNotFoundError: No module named 'torch'
```

Reasoning:

- The AMD JAX quick-start container has working JAX/ROCm support, but it
  does not include PyTorch.
- This matters because Event-SAE uses JAX for the OpenPI policy path, but
  SAE training and several Event-SAE utilities use PyTorch.
- We need a ROCm-compatible PyTorch environment before training SAEs.
- The next implementation choice is whether to install ROCm PyTorch into
  this JAX container or use a second PyTorch ROCm container/environment
  for the PyTorch-only Event-SAE steps.

## Docker Image Inventory Checked

After exiting the `rocm` container, Docker images on the host were
listed:

```bash
docker images
```

Observed:

```text
IMAGE             ID             DISK USAGE   CONTENT SIZE   EXTRA
rocm/jax:latest   3eaecd08f997       20.4GB             0B
rocm:latest       db36a5b95f27         21GB             0B    U
```

Reasoning:

- No PyTorch ROCm image is currently present on the droplet.
- The existing working image should be kept for JAX/OpenPI validation.
- A separate ROCm PyTorch image is the cleaner path for SAE training,
  because it avoids modifying the known-good JAX container.

## PyTorch ROCm Image Pulled

The official ROCm PyTorch image was pulled:

```bash
docker pull rocm/pytorch:latest
```

Observed result:

```text
Digest: sha256:a68ea05633759a7e33b7542321fc9df542bfe8f93e33bddc499099e20bd2b1c3
Status: Downloaded newer image for rocm/pytorch:latest
docker.io/rocm/pytorch:latest
```

The first PyTorch test command used `-it` with a heredoc:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --group-add=render \
  --ipc=host \
  --shm-size=16G \
  --security-opt seccomp=unconfined \
  rocm/pytorch:latest \
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda api available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Observed result:

```text
cannot attach stdin to a TTY-enabled container because stdin is not a terminal
```

Reasoning:

- The image pull succeeded.
- The test did not reach Python or PyTorch.
- The failure came from combining Docker's interactive TTY flags (`-it`)
  with shell heredoc input.
- The next test should either remove `-it` or run an interactive shell
  first and execute Python inside it.

## PyTorch Test Hit Missing `render` Group

The PyTorch container test was retried without `-it`:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --group-add=render \
  --ipc=host \
  --shm-size=16G \
  --security-opt seccomp=unconfined \
  rocm/pytorch:latest \
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda api available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Observed result:

```text
docker: Error response from daemon: Unable to find group render: no matching entries in group file
```

Reasoning:

- The container still did not reach Python or PyTorch.
- Docker failed while resolving the named supplemental group `render`.
- On this host, ROCm device nodes are visible under `/dev/kfd` and
  `/dev/dri`; because the container runs as root, the next retry can
  omit `--group-add=render`.
- If group access becomes necessary later, use the numeric host group ID
  from `getent group render` instead of the group name.

## PyTorch Verified In Separate ROCm Container

A retry without `--group-add=render` but also without `-i` produced no
output. The reason is that Docker did not keep stdin open for the
heredoc.

The successful test used `-i` without `-t`:

```bash
docker run --rm -i \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size=16G \
  --security-opt seccomp=unconfined \
  rocm/pytorch:latest \
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda api available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Observed output:

```text
torch 2.10.0+rocm7.2.3.git1a270074
cuda api available: True
AMD Instinct MI300X VF
```

Reasoning:

- PyTorch works on the MI300X through ROCm.
- PyTorch still reports device availability through the `torch.cuda`
  API, which is expected on ROCm builds.
- The JAX quick-start container remains the verified OpenPI/JAX runtime.
- The separate `rocm/pytorch:latest` image is the verified PyTorch/SAE
  training runtime.
- Keeping JAX and PyTorch in separate containers avoids disturbing the
  known-good JAX container.

## Event-SAE Cloned On Droplet

A persistent workspace was created on the droplet host:

```bash
mkdir -p /workspace
cd /workspace
```

Event-SAE was cloned:

```bash
git clone https://github.com/xc-j/Event-SAE.git
cd Event-SAE
git status --short
git log -1 --oneline
```

Observed output:

```text
Cloning into 'Event-SAE'...
remote: Enumerating objects: 119, done.
remote: Counting objects: 100% (119/119), done.
remote: Compressing objects: 100% (85/85), done.
remote: Total 119 (delta 31), reused 119 (delta 31), pack-reused 0 (from 0)
Receiving objects: 100% (119/119), 110.72 KiB | 3.16 MiB/s, done.
Resolving deltas: 100% (31/31), done.
f7a0000 (HEAD -> main, origin/main, origin/HEAD) Add paper link;
```

Reasoning:

- `/workspace/Event-SAE` is now the droplet-side Event-SAE checkout.
- The droplet checkout matches the local clone's commit:
  `f7a0000 Add paper link;`.
- `git status --short` produced no output, so the droplet checkout was
  clean immediately after cloning.

## External Repositories Cloned On Droplet

From `/workspace/Event-SAE`, the external source repositories were
cloned:

```bash
mkdir -p external
cd external

git clone https://github.com/xc-j/openpi-event-sae.git
git clone https://github.com/saprmarks/dictionary_learning.git
git clone https://github.com/xc-j/awe.git

cd openpi-event-sae
git submodule update --init --recursive

cd /workspace/Event-SAE
find external -maxdepth 2 -type d -name .git -print
```

Observed repositories:

```text
external/openpi-event-sae/.git
external/dictionary_learning/.git
external/awe/.git
```

OpenPI fork submodules were initialized:

```text
Submodule 'third_party/aloha' (https://github.com/Physical-Intelligence/aloha.git)
Submodule 'third_party/libero' (https://github.com/Lifelong-Robot-Learning/LIBERO.git)
Submodule path 'third_party/aloha': checked out 'd1dc83afd89ded4379851257fe5d85632d31d5ec'
Submodule path 'third_party/libero': checked out 'f78abd68ee283de9f9be3c8f7e2a9ad60246e95c'
```

Reasoning:

- Event-SAE's OpenPI path needs the `xc-j/openpi-event-sae` fork, not a
  vanilla OpenPI checkout, because the fork contains the SAE collection
  and intervention hooks used by `scripts/openpi/serve_policy.py`.
- `dictionary_learning` is needed for BatchTopK SAE training.
- `awe` is needed for kinematic keyframe extraction later in the
  Event-SAE pipeline.
- OpenPI's LIBERO submodule is needed for the LIBERO simulation client
  workflow.

## JAX Container Mounts Inspected

The existing `rocm` JAX container mounts were inspected:

```bash
docker inspect rocm --format '{{json .Mounts}}'
```

Observed output:

```json
[{"Type":"bind","Source":"/shared-docker","Destination":"/shared-docker","Mode":"","RW":true,"Propagation":"rprivate"}]
```

Reasoning:

- The current JAX container only bind-mounts `/shared-docker`.
- The droplet-side Event-SAE checkout is at `/workspace/Event-SAE`.
- Therefore the existing JAX container cannot directly see the cloned
  source tree at `/workspace/Event-SAE`.
- We need either to copy/sync the source tree into `/shared-docker` or
  create a new JAX container with `/workspace/Event-SAE` mounted.
- A new explicitly mounted container is cleaner and keeps paths obvious.

## Mounted JAX Container Created

A new JAX container was created so the Event-SAE source tree is mounted
directly:

```bash
docker run -dit \
  --name event-sae-jax \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size=32G \
  --security-opt seccomp=unconfined \
  -v /workspace/Event-SAE:/workspace/Event-SAE \
  -w /workspace/Event-SAE/external/openpi-event-sae \
  rocm/jax:latest \
  bash
```

The container was created:

```text
2d00537e614c4717e472a15fd3c5934f8ed0d40a07b0ef5e2a99cb1e7d6c4ab3
```

Repository visibility inside the container was verified:

```text
/workspace/Event-SAE/external/openpi-event-sae
LICENSE
README.md
configs
docs
environment-openpi.lock.yml
environment-openvla.lock.yml
environment-openvla.yml
event_sae
external
scripts
```

The first JAX check inside this new container failed:

```text
ModuleNotFoundError: No module named 'jax'
```

Reasoning:

- The mounted container can see the Event-SAE source tree.
- The failure is not a GPU visibility result; Python failed before
  importing JAX.
- The original `rocm` Jupyter container had JAX through pyenv at
  `/pyenv/versions/3.10.18/bin/python`.
- The new `event-sae-jax` container appears to need explicit pyenv
  version/environment selection before using JAX.
- Next step is to test `PYENV_VERSION=3.10.18 python` inside
  `event-sae-jax`, but remote SSH became unavailable before that check
  completed.

## SSH Temporarily Unavailable

After the `event-sae-jax` container was created, subsequent SSH attempts
from this workstation returned:

```text
ssh: connect to host 134.199.206.96 port 22: Connection refused
```

Reasoning:

- This is a host connectivity/service issue, not an Event-SAE install
  result.
- Further remote setup cannot be performed from this workstation until
  SSH is accepting connections again.
- If an interactive console is still open on the droplet, check:

```bash
systemctl status ssh --no-pager
docker ps
```

Later, SSH was reachable again. The host showed `ssh.service` active and
both containers running:

```text
NAMES           IMAGE             STATUS
event-sae-jax   rocm/jax:latest   Up 3 minutes
rocm            rocm              Up 33 minutes
```

The temporary refusal was treated as a transient connectivity/service
availability issue. The user's interactive SSH session was kept open as
a safety fallback.

## Mounted JAX Container Python Diagnosed

Inside `event-sae-jax`, pyenv was inspected:

```bash
docker exec event-sae-jax bash -lc "pyenv versions; pyenv version; which python; python -V"
```

Observed:

```text
pyenv: version `3.11' is not installed (set by /workspace/Event-SAE/external/openpi-event-sae/.python-version)
  system
  3.10.18
python=/pyenv/shims/python
Python 3.10.12
```

Then JAX was tested by explicitly selecting the image's available pyenv
runtime:

```bash
docker exec event-sae-jax bash -lc "PYENV_VERSION=3.10.18 python - <<'PY'
import sys
print(sys.executable)
import jax
print('jax', jax.__version__)
print(jax.devices())
PY"
```

Observed:

```text
/pyenv/versions/3.10.18/bin/python
jax 0.4.35
[RocmDevice(id=0)]
```

Reasoning:

- `event-sae-jax` can see the mounted source tree and can use JAX/ROCm.
- The OpenPI fork directory contains a `.python-version` that requests
  Python `3.11`.
- The AMD JAX image currently has pyenv `3.10.18`, not `3.11`.
- Plain `python` in the OpenPI fork directory is confused by the missing
  `3.11` pyenv version.
- For now, JAX commands inside this container must use
  `PYENV_VERSION=3.10.18 python ...` unless we install/build Python 3.11
  with compatible ROCm JAX packages.

## OpenPI Python Requirement Checked

The OpenPI fork's `pyproject.toml` was inspected inside
`event-sae-jax`:

```toml
[project]
name = "openpi"
requires-python = ">=3.11"
dependencies = [
    "jax[cuda12]==0.5.3",
    "torch==2.7.1",
    ...
]
```

The fork also has:

```text
/workspace/Event-SAE/external/openpi-event-sae/.python-version
```

with content:

```text
3.11
```

Reasoning:

- The Event-SAE OpenPI fork expects Python 3.11 or newer.
- The AMD JAX quick-start container's working ROCm JAX runtime is Python
  3.10.18.
- The OpenPI dependency list is CUDA-oriented (`jax[cuda12]`) and cannot
  be used directly on AMD.
- This is the first real compatibility mismatch to solve before running
  the OpenPI server.

## ROCm JAX Package Versions Recorded

The working JAX runtime inside `event-sae-jax` was inspected with:

```bash
PYENV_VERSION=3.10.18 python -m pip show \
  jax jaxlib jax-rocm60-plugin jax-rocm60-pjrt
```

Observed:

```text
jax                  0.4.35
jaxlib               0.4.35
jax-rocm60-plugin    0.4.35
jax-rocm60-pjrt      0.4.35
```

Location:

```text
/pyenv/versions/3.10.18/lib/python3.10/site-packages
```

Reasoning:

- AMD's JAX image provides ROCm JAX through the `jax-rocm60-*` plugin
  packages.
- The available JAX version is `0.4.35`, while the OpenPI fork pins
  CUDA JAX `0.5.3`.
- Any OpenPI setup on this droplet must avoid replacing the working ROCm
  JAX stack with CUDA wheels.
- The next setup choice is either:
  1. create a Python 3.11 environment and install matching ROCm JAX
     packages there, or
  2. test whether the OpenPI fork can run under Python 3.10 with local
     requirement overrides.

## Python 3.11 ROCm JAX Runtime Created

Because the OpenPI fork requires Python `>=3.11`, Python 3.11.13 was
installed inside the mounted JAX container:

```bash
docker exec event-sae-jax bash -lc "pyenv install -s 3.11.13"
```

Observed:

```text
Installed Python-3.11.13 to /pyenv/versions/3.11.13
```

Python and pip were verified:

```text
Python 3.11.13
pip 24.0 from /pyenv/versions/3.11.13/lib/python3.11/site-packages/pip
```

Then the ROCm JAX stack matching the original image was installed into
Python 3.11.13:

```bash
PYENV_VERSION=3.11.13 python -m pip install --upgrade pip
PYENV_VERSION=3.11.13 python -m pip install \
  jax==0.4.35 \
  jaxlib==0.4.35 \
  jax-rocm60-plugin==0.4.35 \
  jax-rocm60-pjrt==0.4.35
```

Observed installed packages:

```text
Successfully installed jax-0.4.35 jax-rocm60-pjrt-0.4.35
jax-rocm60-plugin-0.4.35 jaxlib-0.4.35 ...
```

The Python 3.11 JAX runtime was tested from the OpenPI fork directory:

```bash
cd /workspace/Event-SAE/external/openpi-event-sae
PYENV_VERSION=3.11.13 python - <<'PY'
import sys
import jax
print(sys.executable)
print('jax', jax.__version__)
print(jax.devices())
PY
```

Observed:

```text
/pyenv/versions/3.11.13/bin/python
jax 0.4.35
[RocmDevice(id=0)]
```

Reasoning:

- This resolves the OpenPI fork's Python `>=3.11` requirement while
  preserving ROCm JAX support.
- The OpenPI fork still pins CUDA JAX in `pyproject.toml`, so future
  package installation must avoid replacing this working ROCm JAX stack.
- The current working JAX/OpenPI base command prefix is:

```bash
PYENV_VERSION=3.11.13 python ...
```

## OpenPI Dependencies Installed With AMD Overrides

The OpenPI fork's dependencies could not be installed with `uv sync`
because the fork pins CUDA packages:

```toml
jax[cuda12]==0.5.3
torch==2.7.1
```

Instead, the main dependencies from `pyproject.toml` were installed with
these exclusions:

```text
jax
torch
openpi-client
lerobot
```

Reasoning:

- JAX must remain ROCm-backed, not CUDA-backed.
- PyTorch in this JAX container only needs to satisfy imports for dense
  OpenPI serving; full PyTorch/SAE training is handled in the separate
  `rocm/pytorch:latest` container.
- `openpi-client` is a local workspace package.
- `lerobot` is supplied through a git source in OpenPI's uv config and
  needs separate handling.

CPU PyTorch was installed into the JAX container to satisfy imports:

```bash
PYENV_VERSION=3.11.13 python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.7.1+cpu
```

Then the filtered OpenPI dependency set was installed. During dependency
resolution, pip upgraded JAX to CPU-oriented `0.6.2`, so the ROCm JAX
stack had to be restored manually.

## ROCm JAX 0.5.0 Compatibility Fix

ROCm JAX `0.5.0` was available for Python 3.11:

```text
jax-rocm60-plugin available versions: 0.5.0, 0.4.35
jax-rocm60-pjrt available versions:   0.5.0, 0.4.35
```

The `0.5.0` stack was installed:

```bash
PYENV_VERSION=3.11.13 python -m pip install --force-reinstall \
  jax==0.5.0 \
  jaxlib==0.5.0 \
  jax-rocm60-plugin==0.5.0 \
  jax-rocm60-pjrt==0.5.0 \
  numpy==1.26.4
```

Initial import failed to use ROCm because the plugin looked for:

```text
libamd_comgr.so.2
```

but the image provides:

```text
/opt/rocm-6.4.2/lib/libamd_comgr.so.3
/opt/rocm-6.4.2/lib/libamd_comgr.so.3.0.60402
/opt/rocm-6.4.2/lib/libamd_comgr.so
```

A container-local compatibility symlink was added:

```bash
ln -sf /opt/rocm-6.4.2/lib/libamd_comgr.so.3 \
  /opt/rocm-6.4.2/lib/libamd_comgr.so.2
```

After that, Python 3.11 JAX was verified:

```bash
PYENV_VERSION=3.11.13 python - <<'PY'
import jax
print('jax', jax.__version__)
print(jax.devices())
PY
```

Observed:

```text
jax 0.5.0
[RocmDevice(id=0)]
```

Reasoning:

- OpenPI wants JAX `0.5.x`; the AMD quick-start image originally shipped
  JAX `0.4.35`.
- ROCm JAX `0.5.0` is the closest available ROCm package set on PyPI for
  this image.
- The symlink is local to the container and resolves the SONAME mismatch
  between the plugin package and the ROCm 6.4.2 library layout.

## Local OpenPI Packages Installed

The local OpenPI workspace packages were installed editable without
dependency resolution:

```bash
cd /workspace/Event-SAE/external/openpi-event-sae
PYENV_VERSION=3.11.13 python -m pip install -e packages/openpi-client --no-deps
PYENV_VERSION=3.11.13 python -m pip install -e . --no-deps
```

Observed:

```text
Successfully installed openpi-client-0.1.0
Successfully installed openpi-0.1.0
```

Reasoning:

- `--no-deps` was used intentionally.
- The OpenPI fork's dependency metadata still references CUDA JAX.
- Dependency installation was already handled manually with AMD/ROCm
  overrides.

## Core OpenPI/Event-SAE Imports Verified

Inside `event-sae-jax`, the core import path was tested from
`/workspace/Event-SAE`:

```bash
PYENV_VERSION=3.11.13 python - <<'PY'
import jax, torch
print('jax', jax.__version__, jax.devices())
print('torch', torch.__version__)
import openpi
print('openpi import ok')
from openpi.training import config as _config
print('config import ok', _config.get_config('pi05_libero').name)
from event_sae.openpi.activations import TopKActivationCollector
print('event_sae openpi import ok')
PY
```

Observed:

```text
jax 0.5.0 [RocmDevice(id=0)]
torch 2.7.1+cpu
openpi import ok
config import ok pi05_libero
event_sae openpi import ok
```

Reasoning:

- The Python 3.11 runtime can import JAX and sees the MI300X.
- CPU PyTorch is sufficient for import-time requirements in the JAX
  server container.
- The OpenPI training config registry loads and can resolve
  `pi05_libero`.
- Event-SAE's OpenPI activation module imports successfully.
- This is the first successful integrated import check for the AMD JAX
  container.

## Policy Server Runtime Dependencies Added

The Event-SAE policy server entrypoint was tested:

```bash
cd /workspace/Event-SAE
PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py --help
```

The first attempts exposed missing import-time dependencies from
OpenPI, LeRobot, and the websocket serving path. These packages were
added incrementally:

```bash
PYENV_VERSION=3.11.13 python -m pip install \
  git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
  --no-deps

PYENV_VERSION=3.11.13 python -m pip install datasets==4.1.1
PYENV_VERSION=3.11.13 python -m pip install jsonlines==4.0.0
PYENV_VERSION=3.11.13 python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torchvision==0.22.1+cpu
PYENV_VERSION=3.11.13 python -m pip install --force-reinstall draccus==0.10.0
PYENV_VERSION=3.11.13 python -m pip install av==14.2.0
PYENV_VERSION=3.11.13 python -m pip install websockets
```

Observed installed versions:

```text
lerobot      0.1.0
datasets     4.1.1
jsonlines    4.0.0
torch        2.7.1+cpu
torchvision  0.22.1+cpu
draccus      0.10.0
av           14.2.0
websockets   16.0
```

Reasoning:

- LeRobot was installed from the pinned OpenPI git revision, matching
  the OpenPI fork's dependency source.
- LeRobot was installed with `--no-deps` so pip would not disturb the
  AMD-specific JAX and CPU Torch overrides.
- `datasets`, `jsonlines`, `torchvision`, `draccus`, `av`, and
  `websockets` were added only after the server entrypoint required
  them at import time.
- `torchvision==0.22.1+cpu` was chosen to match
  `torch==2.7.1+cpu`.
- `draccus` was corrected from the latest release to `0.10.0` because
  LeRobot pins that version.

The environment still has intentional compatibility warnings:

```text
openpi 0.1.0 requires jax[cuda12]==0.5.3, but this AMD container uses jax==0.5.0 with ROCm plugins.
gcsfs 2026.4.0 requires fsspec>=2026.4.0, while datasets installed fsspec==2025.9.0.
```

These were not fixed at this step because:

- Installing `jax[cuda12]` would replace the working ROCm JAX stack with
  CUDA packages.
- The `fsspec`/`gcsfs` mismatch should be adjusted only if checkpoint or
  dataset download actually fails, because changing it early could
  disturb the currently working server import path.

## Policy Server Help Verified

After the runtime dependency additions, the Event-SAE OpenPI policy
server help command succeeded inside `event-sae-jax`:

```bash
cd /workspace/Event-SAE
PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py --help
```

Observed output:

```text
usage: serve_policy.py [-h] [--env {libero}] [--config CONFIG]
                       [--checkpoint-dir CHECKPOINT_DIR]
                       [--default-prompt DEFAULT_PROMPT] [--port PORT]
                       [--mode {dense,topk,intervene}]
                       [--output-root OUTPUT_ROOT] [--run-name RUN_NAME]
                       [--capture-target {action_expert,paligemma}]
                       [--layer-indices LAYER_INDICES]
                       [--flush-every-rows FLUSH_EVERY_ROWS]
                       [--sae-checkpoint SAE_CHECKPOINT] [--topk TOPK]
                       [--rows-per-shard ROWS_PER_SHARD]
                       [--layer-idx LAYER_IDX]
                       [--feature-indices FEATURE_INDICES]
                       [--feature-alpha FEATURE_ALPHA]
                       [--recon-alpha RECON_ALPHA]
                       [--active-feature-drop-count ACTIVE_FEATURE_DROP_COUNT]
                       [--active-feature-drop-seed ACTIVE_FEATURE_DROP_SEED]

Serve openpi policy with event-grounded SAE collection.
```

Reasoning:

- The server entrypoint imports successfully in the AMD JAX container.
- The Event-SAE-specific serving modes are visible:
  `dense`, `topk`, and `intervene`.
- This verifies the command-line surface before attempting checkpoint
  loading or live policy serving.

## Configured JAX Container Preserved

After the Python 3.11, ROCm JAX, OpenPI, LeRobot, and serving
dependencies were installed in `event-sae-jax`, the container was
committed to a local Docker image:

```bash
docker commit event-sae-jax event-sae-jax-env:latest
```

Observed image:

```text
IMAGE                      ID             DISK USAGE
event-sae-jax-env:latest   40c297a9ca73   25.5GB
```

Reasoning:

- The package setup lives in the container filesystem, not in
  `/workspace/Event-SAE`.
- Committing the container preserves the working AMD/ROCm Python
  environment so a new runtime container can be started without
  repeating the manual dependency work.
- The original `event-sae-jax` setup container was not created with a
  published policy-server port, so a new serving container is needed for
  host-accessible serving.

## Port-Published Serving Container Created

A runtime container was started from the committed image:

```bash
docker run -dit \
  --name event-sae-jax-server \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size=32G \
  --security-opt seccomp=unconfined \
  -p 8001:8000 \
  -v /workspace/Event-SAE:/workspace/Event-SAE \
  -w /workspace/Event-SAE \
  event-sae-jax-env:latest \
  bash
```

Observed container:

```text
NAMES                  IMAGE                      STATUS  PORTS
event-sae-jax-server   event-sae-jax-env:latest   Up      0.0.0.0:8001->8000/tcp
```

The runtime container was checked with:

```bash
cd /workspace/Event-SAE
PYENV_VERSION=3.11.13 python - <<'PY'
import jax
print('jax', jax.__version__, jax.devices())
from openpi.training import config as _config
print('config', _config.get_config('pi05_libero').name)
PY
PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py --help
```

Observed:

```text
jax 0.5.0 [RocmDevice(id=0)]
config pi05_libero
usage: serve_policy.py [-h] [--env {libero}] ...
```

Reasoning:

- The runtime container has the same verified ROCm JAX/OpenPI
  environment as the setup container.
- Host port `8001` maps to container port `8000`, avoiding the existing
  `rocm` Jupyter container's host port mappings.
- Future policy serving should run in `event-sae-jax-server`.

## Dense Policy Server Startup Smoke Test

A bounded startup test was run in `event-sae-jax-server`:

```bash
cd /workspace/Event-SAE
timeout 120s env PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py \
  --mode dense \
  --env libero \
  --port 8000 \
  --output-root /workspace/Event-SAE/logs/openpi/sae_collection \
  --run-name amd_smoke_dense
```

Observed startup milestones:

```text
Downloading gs://openpi-assets/checkpoints/pi05_libero to /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero
Progress on: 11.6GiB/11.6GiB
Loading model...
Restoring checkpoint from /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/params.
Finished restoring checkpoint in 3.76 seconds
Downloading gs://big_vision/paligemma_tokenizer.model to /root/.cache/openpi/big_vision/paligemma_tokenizer.model
Loaded norm stats from /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero
Enabled SAE collection: {'enabled': True, 'schema_version': 'openpi_sae_collection_v1', ...}
server listening on 0.0.0.0:8000
server closing
server closed
```

The timeout exit status was:

```text
POLICY_STARTUP_STATUS=124
```

Reasoning:

- Exit status `124` is expected because the `timeout 120s` wrapper
  intentionally stopped the server after it reached a listening state.
- The checkpoint and tokenizer downloaded successfully without GCP
  credentials, using anonymous `gcsfs` access.
- The model restored on the AMD JAX/ROCm runtime.
- The server reached `server listening on 0.0.0.0:8000`, which means it
  would be reachable on the droplet host at port `8001` through the
  Docker port mapping.
- No client rollout was run in this step, so the dense activation index
  is expected to be empty.

Generated smoke files:

```text
/workspace/Event-SAE/logs/openpi/sae_collection/amd_smoke_dense/activation_index.jsonl 0 bytes
/workspace/Event-SAE/logs/openpi/sae_collection/amd_smoke_dense/collection.log 252 bytes
/workspace/Event-SAE/logs/openpi/sae_collection/amd_smoke_dense/run_metadata.json 2131 bytes
/workspace/Event-SAE/logs/openpi/sae_collection/amd_smoke_dense/notes.md 1896 bytes
```

## LIBERO Client Compatibility Patch Added

The first one-episode LIBERO client smoke connected to the server and
shared the server SAE run directory, but failed while loading LIBERO
initial states:

```text
_pickle.UnpicklingError: Weights only load failed.
In PyTorch 2.6, we changed the default value of the weights_only argument in torch.load from False to True.
```

The Event-SAE client runner was patched in:

```text
event_sae/openpi/eval/runner.py
```

The patch wraps only the LIBERO init-state call:

```python
with _libero_legacy_torch_load():
    initial_states = task_suite.get_task_init_states(task_id)
```

Reasoning:

- The init-state files are trusted local LIBERO benchmark assets from
  the checked-out OpenPI/LIBERO submodule.
- Newer PyTorch defaults to `weights_only=True`, but these older LIBERO
  assets need the legacy loading behavior.
- The compatibility override is scoped to the LIBERO init-state load,
  rather than changing `torch.load` globally for the whole process.

## LIBERO Client Dependencies Added

The next client smoke reached environment creation and exposed missing
simulation dependencies. The following packages and libraries were added
inside `event-sae-jax-server`:

```bash
PYENV_VERSION=3.11.13 python -m pip install \
  robosuite==1.4.1 \
  bddl==1.0.1 \
  gym==0.25.2 \
  easydict==1.9 \
  cloudpickle==2.1.0 \
  future==0.18.2 \
  termcolor==2.4.0 \
  opencv-python-headless \
  matplotlib==3.8.4

PYENV_VERSION=3.11.13 python -m pip install --force-reinstall numpy==1.26.4

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libegl1 \
  libopengl0 \
  libgl1 \
  libglx0 \
  libosmesa6 \
  mesa-utils \
  libglu1-mesa
```

After the sim dependency install, pip upgraded NumPy to `2.4.6`.
NumPy was restored to `1.26.4` and JAX was rechecked:

```text
numpy 1.26.4
jax 0.5.0 [RocmDevice(id=0)]
```

Reasoning:

- The canonical LIBERO sim env in `docs/openpi.md` is Python 3.8 and
  CUDA-era, which does not match this AMD JAX container.
- For the AMD pipeline smoke, only the missing runtime packages were
  added, while preserving the working OpenPI/JAX stack.
- NumPy `1.26.4` is kept because OpenPI and `openpi-client` require
  NumPy `<2.0`.
- The GL packages are required for MuJoCo/robosuite offscreen rendering.

## Dense Pipeline Smoke Test Completed

The dense policy server was started in `event-sae-jax-server`:

```bash
cd /workspace/Event-SAE
PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py \
  --mode dense \
  --env libero \
  --port 8000 \
  --output-root /workspace/Event-SAE/logs/openpi/sae_collection \
  --run-name amd_pipeline_smoke_dense
```

The one-episode LIBERO client smoke was then run against the local
server:

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
  --override logging.run_tag=amd_pipeline_smoke_client \
  --override sae_collect.enabled=true \
  --override sae_collect.capture_target=action_expert \
  --override sae_collect.layer_idxs=0,5,11,17 \
  --libero-root external/openpi-event-sae/third_party/libero
```

Observed client result:

```text
Finalized SAE episode=0 keep=True result={'status': 'ok', 'episode_num': 0, 'kept': True, 'activation_records': 40, 'num_rows': 400}
run_dir=/workspace/Event-SAE/logs/openpi/sae_collection/amd_pipeline_smoke_dense
total_episodes=1
total_successes=0
success_rate=0.0000
CLIENT_STATUS=0
```

Generated pipeline files:

```text
logs/openpi/sae_collection/amd_pipeline_smoke_dense/actions.jsonl 4510 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/activation_index.jsonl 60784 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/chunk_records.jsonl 769 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/collection.log 1150 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/config.yaml 1560 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/prompt_records.jsonl 275 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/results.json 227 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/run_metadata.json 2393 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/sae_activations/post_mlp_residual/layer_00_shard_000000.pt 411275 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/sae_activations/post_mlp_residual/layer_05_shard_000000.pt 411275 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/sae_activations/post_mlp_residual/layer_11_shard_000000.pt 411275 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/sae_activations/post_mlp_residual/layer_17_shard_000000.pt 411275 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/success.csv 167 bytes
logs/openpi/sae_collection/amd_pipeline_smoke_dense/trajectory_records.jsonl 4510 bytes
```

Reasoning:

- This verifies the AMD pipeline wiring end to end:
  OpenPI policy server -> LIBERO client -> websocket inference ->
  server-side SAE collection -> activation shard files.
- The short rollout intentionally used `max_steps=5`, so a task success
  was not expected.
- The important pipeline signal is that the client completed with status
  `0`, the server finalized the episode, and four layer shard files were
  created.

## AMD/JAX Numerical Issue Observed

During the successful dense pipeline smoke, the server log emitted XLA
GEMM autotuner mismatch warnings:

```text
Results do not match the reference. This is likely a bug/unexpected loss of precision.
```

The activation index recorded `NaN` values for the first predicted
action:

```text
"predicted_action_first": [NaN, NaN, NaN, NaN, NaN, NaN, NaN]
```

An attempted rerun with:

```bash
XLA_FLAGS=--xla_gpu_autotune_level=0
```

was not usable. It failed during inference with:

```text
jaxlib.xla_extension.XlaRuntimeError: INTERNAL: Failed to enqueue convolution: miopenStatusUnknownError
```

Reasoning:

- The pipeline setup is functional, but the current ROCm JAX runtime is
  not yet numerically clean for OpenPI inference.
- Disabling the GPU autotuner is not a valid workaround on this image,
  because it breaks convolution execution.
- Before full experiments, the next AMD-specific task is to find a ROCm
  JAX/XLA setting or image version that produces finite OpenPI actions.

## AMD/JAX Numerical Fix Validated

The NaN source was isolated by running the upstream vanilla OpenPI server
without SAE collection:

```bash
PYENV_VERSION=3.11.13 python external/openpi-event-sae/scripts/serve_policy.py \
  --env LIBERO \
  --port 8000
```

Then a one-episode client smoke was run with `sae_collect.enabled=false`.
It still produced NaN actions:

```text
has_nan True
first_action [nan, nan, nan, nan, nan, nan, nan]
```

Reasoning:

- The NaNs are not caused by Event-SAE activation collection hooks.
- The issue is in core OpenPI inference on the current ROCm/JAX stack.

Several runtime settings were tested against vanilla inference:

```text
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
  client_status=0
  has_nan=False

XLA_FLAGS=--xla_gpu_autotune_level=1
  client_status=0
  has_nan=True

XLA_FLAGS=--xla_gpu_autotune_level=2
  client_status=0
  has_nan=True

JAX_DEFAULT_MATMUL_PRECISION=highest
  client_status=0
  has_nan=False

XLA_FLAGS=--xla_gpu_enable_triton_gemm=false JAX_DEFAULT_MATMUL_PRECISION=highest
  client_status=0
  has_nan=False
```

The gentlest working fix was chosen:

```bash
JAX_DEFAULT_MATMUL_PRECISION=highest
```

This was baked into the Event-SAE server wrapper:

```text
scripts/openpi/serve_policy.py
```

by setting it before OpenPI/JAX imports:

```python
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
```

Reasoning:

- The setting fixes the NaN action output without disabling Triton GEMM
  globally.
- It avoids the MIOpen convolution failure seen with
  `--xla_gpu_autotune_level=0`.
- It is applied only if the user has not explicitly provided another
  `JAX_DEFAULT_MATMUL_PRECISION` value.

## Dense Pipeline Smoke Re-Run With Fix

After patching `scripts/openpi/serve_policy.py`, the dense pipeline smoke
was rerun without manually passing `JAX_DEFAULT_MATMUL_PRECISION`:

```bash
cd /workspace/Event-SAE
PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py \
  --mode dense \
  --env libero \
  --port 8000 \
  --output-root /workspace/Event-SAE/logs/openpi/sae_collection \
  --run-name amd_pipeline_smoke_dense_codefix
```

Client smoke:

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
  --override logging.run_tag=amd_pipeline_smoke_client_codefix \
  --override sae_collect.enabled=true \
  --override sae_collect.capture_target=action_expert \
  --override sae_collect.layer_idxs=0,5,11,17 \
  --libero-root external/openpi-event-sae/third_party/libero
```

Observed result:

```text
CLIENT_STATUS=0
activation_index_records 40
predicted_action_has_nan False
first_predicted_action [0.1022166449859141, 0.2551838759567737, 0.10230367527079576, -0.0033411225995843424, -0.014904850221343802, -0.033109428373303385, -0.9973183260858058]
shard_count 4
SERVER_WARN_GEMM_MISMATCH=0
SERVER_ERR=0
```

Run directory:

```text
/workspace/Event-SAE/logs/openpi/sae_collection/amd_pipeline_smoke_dense_codefix
```

Reasoning:

- The patched server wrapper now produces finite actions on AMD MI300X.
- Dense SAE collection still finalizes and writes four layer shards.
- The previous XLA GEMM autotuner mismatch warning did not appear.
- No MIOpen runtime error appeared.
- The AMD pipeline is now both wired and numerically clean for this
  one-episode smoke test.

## Pipeline Environment Preserved

After the LIBERO client dependencies and GL libraries were added, the
runtime container was committed:

```bash
docker commit event-sae-jax-server event-sae-pipeline-env:latest
```

Observed image:

```text
IMAGE                           ID             DISK USAGE
event-sae-pipeline-env:latest   e47b554e1462   39.5GB
```

Reasoning:

- `event-sae-pipeline-env:latest` preserves the current AMD pipeline
  environment, including OpenPI, the LIBERO client dependencies, and
  system GL libraries.
- The source tree and generated logs remain mounted under
  `/workspace/Event-SAE`; they are not baked into the image.

## Longer Dense Collection Completed

After the NaN fix was validated, a slightly larger dense collection was
run to produce enough rows for a tiny SAE training smoke:

```bash
cd /workspace/Event-SAE
PYENV_VERSION=3.11.13 python scripts/openpi/serve_policy.py \
  --mode dense \
  --env libero \
  --port 8000 \
  --output-root /workspace/Event-SAE/logs/openpi/sae_collection \
  --run-name amd_dense_task0_3trial_25step
```

Client:

```bash
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

Observed:

```text
episode 0: activation_records=200, num_rows=2000
episode 1: activation_records=200, num_rows=2000
episode 2: activation_records=200, num_rows=2000
total_episodes=3
CLIENT_STATUS=0
activation_index_records 600
predicted_action_has_nan False
layer_00_shard_000000.pt (1500, 1024) finite True
layer_05_shard_000000.pt (1500, 1024) finite True
layer_11_shard_000000.pt (1500, 1024) finite True
layer_17_shard_000000.pt (1500, 1024) finite True
SERVER_WARN_GEMM_MISMATCH=0
SERVER_ERR=0
```

Run directory:

```text
/workspace/Event-SAE/logs/openpi/sae_collection/amd_dense_task0_3trial_25step
```

Reasoning:

- This is still a small setup run, not a paper-scale experiment.
- It proves finite OpenPI actions and finite dense activation shards
  over multiple episodes.
- The layer-17 shard has enough rows for a tiny SAE training smoke.

## Tiny SAE Training Completed

The SAE trainer originally used a hardcoded `warmup_steps=1000`, which
is valid for full runs but invalid for a tiny `steps=20` smoke. The
trainer config builder was patched in:

```text
event_sae/train.py
```

to use proportional warmup/decay for small runs:

```python
warmup_steps = min(1000, max(1, cfg.steps // 10))
decay_start = max(warmup_steps + 1, int(cfg.steps * 0.8))
```

Tiny SAE training was then run in the ROCm PyTorch container:

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

Inside that container:

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

Observed:

```text
torch 2.10.0+rocm7.2.3.git1a270074
cuda available True
device AMD Instinct MI300X VF
Average mean squared norm: 1233130.125
Norm factor: 1110.4638671875
Step 0: L0 = 16.0, frac_variance_explained = -0.048242926597595215
```

Output checkpoint:

```text
/workspace/Event-SAE/logs/openpi/sae/amd_tiny_layer17/trainer_0/ae.pt
```

Checkpoint verification:

```text
trainer_0/ae.pt 8399193 bytes
trainer_0/config.json 618 bytes
keys ['b_dec', 'decoder.weight', 'encoder.bias', 'encoder.weight', 'k', 'threshold']
b_dec (1024,) torch.float32 finite True
k () torch.int32 finite True
threshold () torch.float32 finite True
decoder.weight (1024, 1024) torch.float32 finite True
encoder.weight (1024, 1024) torch.float32 finite True
encoder.bias (1024,) torch.float32 finite True
```

Reasoning:

- This proves SAE training can run on the AMD PyTorch ROCm image.
- The run is intentionally tiny and not scientifically meaningful.
- The output `ae.pt` is sufficient to test the OpenPI `topk` serving
  path.

## TopK SAE Serving Smoke Completed

The JAX serving container needed `dictionary_learning` available to load
the trained SAE checkpoint:

```bash
PYENV_VERSION=3.11.13 python -m pip install -e external/dictionary_learning --no-deps
PYENV_VERSION=3.11.13 python -m pip install nnsight==0.3.7 einops tqdm
```

TopK server:

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

Observed:

```text
Finalized SAE episode=0 keep=True result={'status': 'ok', 'episode_num': 0, 'kept': True, 'activation_records': 10, 'num_rows': 100}
CLIENT_STATUS=0
SERVER_WARN_GEMM_MISMATCH=0
SERVER_ERR=0
```

Run directory:

```text
/workspace/Event-SAE/logs/openpi/sae_collection/amd_topk_tiny_layer17_smoke
```

TopK artifact:

```text
sae_activations/post_mlp_residual/shard_000000.pt 25049 bytes
```

The shard contains:

```text
episode_num (100,)
step_in_episode (100,)
global_forward_idx (100,)
forward_type (100,)
forward_step_idx (100,)
expert_idx (100,)
batch_idx (100,)
token_idx (100,)
chunk_start_step (100,)
executed_chunk_len (100,)
top_feature_ids (100, 16)
top_feature_vals (100, 16)
```

Reasoning:

- This proves the trained SAE checkpoint can be loaded by the OpenPI
  server on AMD.
- The server can run in `topk` mode and write top feature IDs/values.
- `activation_index.jsonl` is dense-mode-oriented and remained empty in
  this top-k smoke; the top-k result is stored in the shard dictionary.

## Pipeline Environment Updated

After adding the serving-side `dictionary_learning` loader dependencies,
the runtime container was committed again:

```bash
docker commit event-sae-jax-server event-sae-pipeline-env:latest
```

Observed image:

```text
IMAGE                           ID             DISK USAGE
event-sae-pipeline-env:latest   e49e50c79643   39.6GB
```

Reasoning:

- `event-sae-pipeline-env:latest` now includes the dependencies needed
  for dense serving, LIBERO client smoke tests, and top-k SAE serving.

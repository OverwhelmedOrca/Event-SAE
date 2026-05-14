# openVLA pipeline

Closed-loop interpretability pipeline for the openVLA backbone on the
LIBERO simulation suites.

## Reproducibility check

We re-ran the openVLA + LIBERO-Spatial zero-out sweep (top-5 features
per ranking, α = 0, 5 trials per task, seed 0) in three configurations
and compared each to the reference libero_spatial numbers from the
original research code. All four rankings match the reference mean
ΔSR within ~3 percentage points.

| Configuration                  | SAE       | Feature lists  | Baseline SR | Δ event-aligned | Δ window-mean | Δ task-mean | Δ random-alive |
|---|---|---|---:|---:|---:|---:|---:|
| Reference                      | original  | original       | 80.0%       | −28.4           | −7.2          | −8.0        | −6.8           |
| This codebase, both reused     | original  | original       | 82.0%       | −26.8           | −9.2          | −9.6        | −9.2           |
| This codebase, SAE only reused | original  | this codebase  | 80.0%       | −25.2           | −6.4          | −8.0        | −8.8           |

ΔSR is in percentage points relative to each row's own baseline run.

The third row is the strongest end-to-end check: only the SAE is held
fixed, and every other step runs through this repository. Differences
stay within the ~2 pp cuDNN / float32 noise floor.

## Installation

Two steps; run them in order.

### Step 1: Conda environment

```bash
conda env create -f environment.yml
conda activate event-sae-openvla
```

Optional flash-attn for faster inference (openVLA falls back to `sdpa`
if skipped):

```bash
pip install flash-attn==2.5.5 --no-build-isolation
```

### Step 2: External libraries

Three libraries installed editable into the conda env. Clone under
`external/` at the repo root (already gitignored):

```bash
cd external
```

**(a) LIBERO** — sim benchmark:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
touch libero/__init__.py libero/lifelong/models/modules/__init__.py   # missing in upstream
pip install -e .
pip install robosuite==1.4.0 bddl==1.0.1 robomimic==0.2.0 mujoco \
            gym==0.25.2 easydict==1.9 cloudpickle==2.1.0 future
cd ..
```

**(b) `dictionary_learning`** — SAE training library:

```bash
git clone https://github.com/saprmarks/dictionary_learning.git
pip install -e ./dictionary_learning
```

Tested against commit `60ec6bf`. If upstream breaks, pin it:
`(cd dictionary_learning && git checkout 60ec6bf)`.

**(c) AWE** — kinematic keyframe extraction (fork with packaging
fixes; see the fork's NOTICE):

```bash
git clone https://github.com/xc-j/awe.git
pip install -e ./awe
```

## Usage

The pipeline has four stages: **(1)** SAE training, **(2)** kinematic
keyframe extraction, **(3)** event clustering with VLM annotation,
**(4)** closed-loop intervention. Between (3) and (4) a feature
ranking step picks candidate features to intervene on.

Every command below chains off a single rollout. After step (a) runs,
export the run directory name so later commands can derive their
paths:

```bash
export EVAL_RUN=EVAL-libero_spatial-openvla-<DATE_TIME>   # name of the dir under logs/openvla/
```

For `$SAE_CKPT`, either train one in step (b) or download the
paper's pretrained checkpoints from the Hugging Face Hub:

```python
# pip install huggingface_hub
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("mr-cabbage/event-sae-openvla-libero",
                       "libero_spatial/ae.pt")
print(ckpt)   # → export SAE_CKPT=<that path>
```

The Hub repo holds one SAE per LIBERO suite (`libero_spatial`,
`libero_object`, `libero_goal`, `libero_10`), each at openVLA
layer 31.

## Phase 1 — SAE training

Roll out openVLA on LIBERO, save the residual-stream activations, then
train a BatchTopK SAE on them.

### (a) Collect openVLA activations during a LIBERO rollout

Edit `configs/examples/openvla/collect_libero_spatial.yaml` (or copy
and adapt). Requires GPU and LIBERO assets (`LIBERO_CONFIG_PATH`).

```bash
python scripts/openvla/collect_activations.py \
    --config configs/examples/openvla/collect_libero_spatial.yaml
```

Outputs under `logs/openvla/$EVAL_RUN/sae_activations/`:
- dense `.pt` shards — input to step (b)
- `activation_index.jsonl` — input to step (h)

To skip step (h), use the **online top-k mode**: set
`sae_collect.mode: "topk"` and `sae_collect.sae_checkpoint: <path>`
in the YAML and the rollout writes sparse shards directly.

### (b) Train an SAE on collected shards

Edit `configs/examples/openvla/train_sae_layer31.yaml` and point
`data_dir` at the shard directory from step (a). Set `wandb_project`
to log to wandb, or leave empty to disable.

```bash
python scripts/train_sae.py \
    --config configs/examples/openvla/train_sae_layer31.yaml \
    --save-dir logs/openvla/sae/libero_spatial_layer31
```

Output: `ae.pt` + `config.json` under
`logs/openvla/sae/libero_spatial_layer31/trainer_0/`.

Skip this step if you use the pretrained checkpoints from the Hugging
Face Hub.

## Phase 2 — Kinematic keyframe extraction

Pick a small number of waypoints per episode from the end-effector
trajectory. These waypoints anchor the events used in Phase 3 and are
independent of the SAE.

### (c) Extract AWE kinematic keyframes from rollout trajectories

AWE picks waypoints along the end-effector trajectory. CPU-only.
Defaults (`pos_only`, error budget η = 0.05) are baked into the CLI.

```bash
python scripts/extract_keyframes.py \
    --trajectory-records-path logs/openvla/$EVAL_RUN/trajectory_records.jsonl
```

Output: `waypoint_summary.json` under
`logs/openvla/keyframes/$EVAL_RUN/dp_pos_only_err0p05/`.

## Phase 3 — Event clustering with VLM annotation

Group waypoint windows into per-task event clusters, then ask Gemini
to label each cluster with a short phrase and one of six phase tags
(`pre_grasp`, `immobilization`, `contact`, `detach`, `post_grasp`,
`transition`).

### (d) Render 5-frame bundles around each keyframe

Save 5 PNG frames per waypoint at offsets `-4, -2, 0, 2, 4` plus a
short MP4 over the same window. Requires step (a) to have saved
rollout videos (`logging.save_video: true`).

```bash
python scripts/extract_keyframe_media.py \
    --waypoint-summary-path logs/openvla/keyframes/$EVAL_RUN/dp_pos_only_err0p05/waypoint_summary.json
```

Outputs under `logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/`:
- per-sample PNG frames — input to step (e)
- per-sample MP4 clips — for human inspection
- `samples.jsonl` — sample manifest

### (e) Build vision embeddings + state vectors per sample

Encode each 5-frame bundle through a frozen vision encoder (default
SigLIP), L2-normalize, then concatenate the end-effector pose at the
waypoint. Requires GPU.

```bash
python scripts/build_event_features.py \
    --samples-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/samples.jsonl
```

Output: `event_features.jsonl` next to `samples.jsonl`.

### (f) Task-local agglomerative clustering of event features

Cluster samples per task by cosine-distance agglomerative clustering
on the weighted [vision, state, progress] descriptor. CPU-only, runs
in seconds. Defaults: cosine threshold 0.18, weights 1.0 / 0.5 / 0.4,
5 exemplars per cluster.

```bash
python scripts/cluster_events.py \
    --event-features-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/event_features.jsonl
```

Outputs under `clusters/` next to `event_features.jsonl`:
- `cluster_assignments.jsonl` — sample → cluster_id map
- `clusters.jsonl` — per-cluster members + exemplars
- `summary.json` — overall stats

### (g) Annotate clusters with Gemini

Send each cluster's representative 5-frame sequences to Gemini and
parse a `{phrase, phase}` JSON response. `phase` is one of the six
tags from the Phase 3 intro. Default model: `gemini-2.5-flash`
(override with `--model`).

```bash
export GEMINI_API_KEY=<your-key>
python scripts/annotate_clusters.py \
    --clusters-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/clusters.jsonl
```

Output: `gemini-2_5-flash_cluster_annotations.jsonl` next to
`clusters.jsonl`.

## Feature ranking (bridge between Phase 3 and Phase 4)

Encode the saved activations through the trained SAE, then score each
event cluster against the SAE features. The result is a ranked list of
candidate features for Phase 4.

### (h) Top-k SAE encoding (offline)

Apply the SAE to the dense shards from step (a) and write sparse
top-k shards. Encoding is decoupled from collection, so re-encoding
with a different SAE or layer needs no fresh rollout.

```bash
python scripts/extract_topk.py \
    --dense-dir logs/openvla/$EVAL_RUN/sae_activations/post_mlp_residual \
    --sae-checkpoint $SAE_CKPT \
    --layer-idx 31
```

Output: top-k shards + `manifest.json` under
`logs/openvla/$EVAL_RUN/topk_activations/`.

### (i) Event-feature score matrix

For each VLM-labeled cluster, score every SAE feature on how
strongly its activation lines up with that cluster's events. The
score is the max projection onto three temporal templates — pulse,
step-up, step-down — inside a ±5-step window around each event,
averaged across episodes. CPU-only.

```bash
python scripts/score_cluster_features.py \
    --topk-run-dir logs/openvla/$EVAL_RUN/topk_activations \
    --event-features-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/event_features.jsonl \
    --cluster-assignments-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/cluster_assignments.jsonl \
    --cluster-annotations-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/gemini-2_5-flash_cluster_annotations.jsonl \
    --output-path logs/openvla/scores/$EVAL_RUN/event_feature_scores.pt
```

Output: one `.pt` payload — `(num_clusters, dict_size)` `matrix` plus
`row_keys`, `row_results`, `templates`, `selection_counts`,
`selected_events`, `source`.

## Phase 4 — Closed-loop intervention

At inference time, edit one SAE feature at a time and check how the
policy's success rate changes. For chosen feature index `i` and
scalar `α`:

    z' = α · z    for index i
    x' = x + Dec(z') − Dec(z)

`α = 0` zeros the feature out, `α = 1` leaves the hidden state
unchanged, intermediate values give partial suppression, `α > 1`
amplifies. The SAE reconstruction error on the un-edited code is
preserved.

### (j) Build candidate feature lists from the four rankings

Surface the top-K features under four ranking strategies. CPU-only.

```bash
python scripts/build_feature_rankings.py \
    --scores-pt logs/openvla/scores/$EVAL_RUN/event_feature_scores.pt \
    --topk-run-dir logs/openvla/$EVAL_RUN/topk_activations \
    --prompt-records-path logs/openvla/$EVAL_RUN/prompt_records.jsonl \
    --output-dir logs/openvla/rankings/$EVAL_RUN \
    --top-k 5
```

The four rankings:

- **event-aligned** — mean of the score matrix across cluster rows.
- **window-mean** — per-row window-mean vectors weighted by event count.
- **task-mean** — per-task feature means weighted by per-task step count.
- **random-alive** — uniform sample over alive features, excluding any
  feature already chosen by the three informed rankings.

Outputs under `--output-dir`:

- per-ranking JSONL: `event_aligned.jsonl`, `window_mean.jsonl`,
  `task_mean.jsonl`, `random_alive.jsonl`
- `candidates.jsonl` — flat list of `4 × K` `(ranking, rank,
  feature_id, score)` rows that feeds step (k)

### (k) Run single-feature intervention on LIBERO

Run a closed-loop LIBERO eval with the residual-preserving hook
applied at the SAE's layer. Repeat the command once per `feature_id`
in `candidates.jsonl`. Requires GPU.

```bash
python scripts/openvla/intervene.py \
    --config configs/examples/openvla/collect_libero_spatial.yaml \
    --sae-checkpoint $SAE_CKPT \
    --layer-idx 31 \
    --feature-id <FEATURE_ID> \
    --alpha 0.0
```

Add one no-hook baseline:

```bash
python scripts/openvla/collect_activations.py \
    --config configs/examples/openvla/collect_libero_spatial.yaml \
    --override sae_collect.enabled=false
```

For each ranking, take the mean of `SR_hook − SR_baseline` across
its K features — this is how much zeroing that ranking's features
hurts the policy.

Each intervention run also writes
`intervene_feat<N>_alpha<A>_records.jsonl` (per-step feature
activation before / after the edit) for verifying the hook fired.

## Frozen environment snapshot

If `environment.yml` ever resolves into a broken env (for example
after an upstream package change), use the pinned snapshot in
`environment.lock.yml` instead:

```bash
conda env create -f environment.lock.yml -n event-sae-openvla
```

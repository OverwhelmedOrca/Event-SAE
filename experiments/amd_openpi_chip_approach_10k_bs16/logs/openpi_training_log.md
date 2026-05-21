# OpenPI AMD MI300X Training Log

## 2026-05-21 - Setup Started

Objective: fine-tune OpenPI on the 100-episode Trossen solo-arm chip-approach dataset using the AMD Developer Cloud MI300X droplet.

Chosen dataset:

- `approach_red_yellow_chip_single_arm_v1_drop_0_26`

Reasoning:

- It matches the existing OpenPI config `pi05_trossen_solo_chip_lora`.
- It contains 100 cleaned episodes for red/yellow chip approach behavior.
- It has the expected Trossen solo-arm observation and action schema: two RGB camera streams, 7D robot state, and 7D actions.

Chosen training config:

- `pi05_trossen_solo_chip_lora`

Reasoning:

- It already targets this dataset through `repo_id="approach_red_yellow_chip_single_arm_v1_drop_0_26"`.
- It uses the Trossen solo policy transforms.
- It uses LoRA fine-tuning from OpenPI pi05 base weights, which is the practical route for this dataset size.
- It already uses the requested global batch size of `16`.

Requested run settings:

- Batch size: `16`
- Steps: `10_000`
- W&B: disabled

Status:

- Remote OpenPI repo mirror completed at `/workspace/openpi`.
- Remote LeRobot dataset cache completed under `/root/.cache/huggingface/lerobot/test`.
- Dedicated training container created: `openpi-chip-train`.
- ROCm/JAX confirmed `RocmDevice(id=0)`.

## Environment Fixes

The AMD container initially had a compatible ROCm/JAX environment, but the LeRobot data stack did not match this OpenPI checkout.

Observed issue:

- `datasets==4.1.1` caused LeRobot parquet loading to fail with a `torch.stack(... Column ...)` type mismatch.

Fix:

- Installed `datasets==3.6.0`, matching the OpenPI `uv.lock`.
- Installed LeRobot from `huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`, matching the OpenPI `pyproject.toml`.
- Cleared the generated HuggingFace parquet cache so it could be rebuilt under the compatible `datasets` version.

## Config Fix

The chip-approach config had the right dataset id and policy transform, but it did not repack raw LeRobot keys into the OpenPI policy schema.

Observed issue:

- `TrossenSoloInputs` expected `data["images"]`, but the raw LeRobot sample contained `observation.images.cam_high` and `observation.images.cam_right_wrist`.

Fix:

- Added a repack transform to `pi05_trossen_solo_chip_lora`.
- Added the same repack transform to `pi05_trossen_solo_chip_uniform_prompt_lora` for consistency.
- Included `prompt: prompt` in the repack map so tokenization receives the task prompt produced by `prompt_from_task=True`.

## Normalization Stats

Command:

```bash
python scripts/compute_norm_stats.py --config-name pi05_trossen_solo_chip_lora
```

Result:

- Wrote stats to `/workspace/openpi/assets/pi05_trossen_solo_chip_lora/approach_red_yellow_chip_single_arm_v1_drop_0_26`.
- Runtime: `302` seconds.

## Sanity Check

The one-batch sanity check passed on AMD.

Key output:

```text
config=pi05_trossen_solo_chip_lora
exp_name=amd_chip_approach_10k_bs16
batch_size=16
num_train_steps=10000
dataset_episodes=100
dataset_frames=19946
jax_devices=[RocmDevice(id=0)]
batch_actions_shape=(16, 50, 32)
state_shape=(16, 32)
image_keys=['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
```

## Training Run

The full requested training run was launched with:

- Config: `pi05_trossen_solo_chip_lora`
- Experiment name: `amd_chip_approach_10k_bs16`
- Batch size: `16`
- Requested steps: `10_000`
- W&B: disabled

The run successfully:

- Loaded the dataset and normalization stats.
- Downloaded and restored `gs://openpi-assets/checkpoints/pi05_base/params`.
- Initialized the pi05 LoRA train state on MI300X.
- Entered the training loop and trained past 2.4k steps.

Observed speed:

- About `1.1 steps/sec`.
- About `17.6 samples/sec` at batch size `16`.

Observed loss:

```text
Step 0: loss=0.0364
Step 1000: loss=0.0131
Step 2000: loss=0.0082
Step 2400: loss=0.0073
```

Stopped result:

- Stopped after validation to preserve AMD credits.
- Total tracked wall time: `2495.13` seconds, or `41.59` minutes.
- Last logged step: `2400`.
- Last observed progress before interrupt: about `2.47k / 10k`.

Checkpoint result:

- No checkpoint was written.
- Reason: `save_interval=10000`; OpenPI would save at the final 10k step, not at the stopped 2.47k point.
- The checkpoint directory existed but contained no checkpoint files:
  `/workspace/openpi/checkpoints/pi05_trossen_solo_chip_lora/amd_chip_approach_10k_bs16`

## 2026-05-21 - Progress Recheck

The AMD droplet/container was checked again after the training process had been interrupted.

Commands inspected:

- `pgrep -af "run_training.py|scripts/train.py"`
- `/workspace/openpi_experiments/amd_openpi_chip_approach_10k_bs16/logs/metrics.json`
- tail of `/workspace/openpi_experiments/amd_openpi_chip_approach_10k_bs16/logs/train.log`
- checkpoint directory under `/workspace/openpi/checkpoints/pi05_trossen_solo_chip_lora/amd_chip_approach_10k_bs16`

Current state:

- No active OpenPI training process was running.
- `metrics.json` reports `status="stopped_by_user_after_validation"`.
- Last logged step remains `2400`.
- Last logged loss remains `0.0073`.
- Last observed progress line was about `2.47k / 10k`.
- No checkpoint files were present.

Interpretation:

The run did not continue in the background after interruption. The validated result is the same: the AMD MI300X OpenPI training pipeline works and trained with descending loss, but the run was stopped before the 10k-step save point.

# AMD OpenPI Chip Approach Training Run

This directory documents the second AMD MI300X validation test: fine-tuning OpenPI on the 100-episode Trossen solo-arm chip-approach dataset.

## Goal

Show that the OpenPI training pipeline used locally can be reproduced on the AMD Developer Cloud MI300X platform and produce a usable checkpoint from the chip-approach dataset.

## Target Run

- Platform: AMD Developer Cloud GPU Droplet
- GPU: 1x AMD Instinct MI300X, 192 GB VRAM
- Droplet: `0.4.35-gpu-mi300x1-192gb-devcloud-atl1`
- Training repo on droplet: `/workspace/openpi`
- Dataset on droplet: `/root/.cache/huggingface/lerobot/test/approach_red_yellow_chip_single_arm_v1_drop_0_26`
- OpenPI config: `pi05_trossen_solo_chip_lora`
- Dataset repo id used by config: `approach_red_yellow_chip_single_arm_v1_drop_0_26`
- Batch size: `16`
- Training steps: `10_000`
- Logging: Weights & Biases disabled for reproducibility/offline operation.

## Dataset

The selected dataset is the cleaned 100-episode LeRobot dataset:

`approach_red_yellow_chip_single_arm_v1_drop_0_26`

Local source path:

`/home/trossen-ai/.cache/huggingface/lerobot/test/approach_red_yellow_chip_single_arm_v1_drop_0_26`

Important metadata from `meta/info.json`:

- Episodes: `100`
- Frames: `19,946`
- FPS: `30`
- Robot type: `trossen_ai_solo`
- State dimension: `7`
- Action dimension: `7`
- Cameras:
  - `observation.images.cam_high`
  - `observation.images.cam_right_wrist`

## Outputs

Runtime logs and metrics are recorded in:

- `logs/openpi_training_log.md`
- `logs/metrics.json`
- `logs/compute_norm_stats.log`
- `logs/sanity_check.log`
- `logs/train.log`

## Result

Status: validated and stopped before final checkpoint to preserve AMD credits.

The training pipeline was proven end to end on MI300X:

- OpenPI source mirrored to the AMD droplet.
- 100-episode LeRobot dataset mirrored to the AMD droplet.
- ROCm/JAX detected `RocmDevice(id=0)`.
- Dataset normalization stats were computed on the droplet.
- One-batch sanity check passed.
- Full OpenPI pi05 LoRA training started successfully.
- Training ran for about `41.6` minutes total and reached the last logged step `2400`.
- Observed iteration speed was about `1.1 steps/sec`.
- Effective sample throughput was about `17.6 samples/sec` with batch size `16`.
- Training loss decreased from `0.0364` at step `0` to `0.0073` at step `2400`.

No checkpoint was written for this stopped run. The active config had `save_interval=10000`, so OpenPI would only save at the 10k final step. Stopping at about 2.47k steps validated compatibility but did not preserve the in-memory train state.

## Latest Progress Check

Checked on `2026-05-21` after the run was interrupted:

- No `run_training.py` or OpenPI `scripts/train.py` process is running in the AMD container.
- Remote metrics still report `status="stopped_by_user_after_validation"`.
- Last logged training step: `2400`.
- Last observed progress line: about `2.47k / 10k`.
- Last logged loss: `0.0073`.
- Observed iteration speed: about `1.1 steps/sec`.
- Checkpoint written: `false`.
- Checkpoint directory exists but contains no checkpoint files:
  `/workspace/openpi/checkpoints/pi05_trossen_solo_chip_lora/amd_chip_approach_10k_bs16`

Conclusion: the OpenPI training run successfully demonstrated AMD MI300X compatibility and loss descent, but it was stopped before a checkpoint could be saved.

## Code Changes Needed

The OpenPI chip-approach config needed a repack transform so the raw LeRobot keys match the Trossen solo policy transform:

- `observation.images.cam_high` -> `images.cam_high`
- `observation.images.cam_right_wrist` -> `images.cam_right_wrist`
- `observation.state` -> `state`
- `action` -> `actions`
- `prompt` -> `prompt`

This was applied locally in:

`/home/trossen-ai/openpi/src/openpi/training/config.py`

and synced to:

`/workspace/openpi/src/openpi/training/config.py`

The AMD container also needed the data stack aligned to the OpenPI lockfile:

- `datasets==3.6.0`
- `lerobot` from `huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`

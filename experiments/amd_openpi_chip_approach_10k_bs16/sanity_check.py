import dataclasses
import json
import pathlib

import jax

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding


def main() -> None:
    cfg = _config.get_config("pi05_trossen_solo_chip_lora")
    cfg = dataclasses.replace(
        cfg,
        exp_name="amd_chip_approach_10k_bs16",
        num_train_steps=10_000,
        batch_size=16,
        num_workers=0,
        overwrite=True,
        wandb_enabled=False,
    )

    meta_path = pathlib.Path(
        "/root/.cache/huggingface/lerobot/test/"
        "approach_red_yellow_chip_single_arm_v1_drop_0_26/meta/info.json"
    )
    meta = json.loads(meta_path.read_text())

    print(f"config={cfg.name}")
    print(f"exp_name={cfg.exp_name}")
    print(f"batch_size={cfg.batch_size}")
    print(f"num_train_steps={cfg.num_train_steps}")
    print(f"dataset_episodes={meta['total_episodes']}")
    print(f"dataset_frames={meta['total_frames']}")
    print(f"jax_devices={jax.devices()}")
    print(f"assets_dirs={cfg.assets_dirs}")
    print(f"checkpoint_dir={cfg.checkpoint_dir}")

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    loader = _data_loader.create_data_loader(
        cfg,
        sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)),
        shuffle=True,
    )
    obs, actions = next(iter(loader))
    print(f"batch_actions_shape={actions.shape}")
    print(f"state_shape={obs.state.shape}")
    print(f"image_keys={list(obs.images.keys())}")


if __name__ == "__main__":
    main()

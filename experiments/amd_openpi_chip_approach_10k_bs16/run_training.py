import dataclasses
import json
import os
import pathlib
import sys
import time
import traceback


EXPERIMENT_DIR = pathlib.Path("/workspace/openpi_experiments/amd_openpi_chip_approach_10k_bs16")
METRICS_PATH = EXPERIMENT_DIR / "logs" / "metrics.json"


def main() -> None:
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    os.environ.setdefault("WANDB_MODE", "disabled")

    sys.path.insert(0, "/workspace/openpi/scripts")

    import train
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_trossen_solo_chip_lora")
    cfg = dataclasses.replace(
        cfg,
        exp_name="amd_chip_approach_10k_bs16",
        num_train_steps=10_000,
        batch_size=16,
        overwrite=True,
        resume=False,
        wandb_enabled=False,
    )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    metrics = {
        "status": "running",
        "config": cfg.name,
        "exp_name": cfg.exp_name,
        "batch_size": cfg.batch_size,
        "num_train_steps": cfg.num_train_steps,
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "assets_dirs": str(cfg.assets_dirs),
        "start_time_unix": start,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    try:
        train.main(cfg)
    except BaseException as exc:
        end = time.time()
        metrics.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "end_time_unix": end,
                "duration_seconds": end - start,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
        raise

    end = time.time()
    metrics.update(
        {
            "status": "completed",
            "end_time_unix": end,
            "duration_seconds": end - start,
            "duration_minutes": (end - start) / 60.0,
        }
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()

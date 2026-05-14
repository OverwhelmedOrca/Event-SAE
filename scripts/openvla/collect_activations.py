"""CLI: run a LIBERO closed-loop openVLA eval with SAE activation collection.

Activations are written under
`{cfg.logging.root_dir}/{run_id}/sae_activations/post_mlp_residual/layer_NN_shard_MMMMMM.pt`,
the filename format consumed by `event_sae.train.ActivationShardDataLoader`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.openvla.eval import eval_libero, load_config, parse_overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (dot notation), e.g. env.num_trials_per_task=5",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=parse_overrides(args.override))
    eval_libero(cfg)


if __name__ == "__main__":
    main()

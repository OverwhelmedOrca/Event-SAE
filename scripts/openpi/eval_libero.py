"""CLI: run a LIBERO closed-loop eval for openpi (client side).

Expects an openpi policy server to be running at the configured
host:port. If the server advertises SAE collection in its metadata,
each chunk inference request will include the rollout context so
the server-side collector can tag activations correctly.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.openpi.eval import eval_libero, load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="openpi LIBERO eval (client side).")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted key=value overrides for the YAML (repeatable)",
    )
    parser.add_argument(
        "--libero-root",
        default="external/openpi-event-sae/third_party/libero",
        help="Path to the LIBERO submodule clone (third_party/libero in the openpi fork).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    libero_root = Path(args.libero_root).resolve()

    cfg = load_config(config_path, overrides=parse_overrides(args.override))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    result = eval_libero(cfg, config_path=config_path, libero_root=libero_root)
    print(
        f"run_dir={result.run_dir}\n"
        f"total_episodes={result.total_episodes}\n"
        f"total_successes={result.total_successes}\n"
        f"success_rate={result.success_rate:.4f}"
    )


if __name__ == "__main__":
    main()

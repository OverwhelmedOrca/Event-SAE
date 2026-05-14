"""CLI: run an openVLA LIBERO eval with a single-feature SAE intervention hook.

For each ``(feature_id, alpha)`` pair, runs ``eval_libero`` with the
residual-preserving latent-edit hook applied at the configured SAE layer
and reports closed-loop success rate. Compare against a baseline run
(same config, no ``--intervene-feature-id``) to get the SR delta.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from event_sae.openvla.eval.config import load_config, parse_overrides
from event_sae.openvla.eval.runner import eval_libero
from event_sae.openvla.intervene import apply_resid_post_feature_perturb_hook


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LIBERO eval with a single-feature SAE intervention.")
    ap.add_argument("--config", required=True, help="YAML eval config (same schema as collect_activations).")
    ap.add_argument(
        "--override", action="append", default=[], help="Dotted key=value overrides for the YAML (repeatable)."
    )
    ap.add_argument("--sae-checkpoint", required=True, help="Path to a post_mlp_residual BatchTopKSAE ae.pt.")
    ap.add_argument("--layer-idx", type=int, required=True, help="Decoder layer index to hook.")
    ap.add_argument("--feature-id", type=int, required=True, help="SAE feature column index to perturb.")
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Scaling applied to z[feature_id]: 0 zeros out, (0,1) suppresses, 1 no-op, >1 amplifies.",
    )
    ap.add_argument(
        "--hook-start-step",
        type=int,
        default=0,
        help="Step in episode at which the hook becomes active (set >0 to skip a warm-up).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.override))
    cfg.sae_collect.enabled = False

    def applier(*, model, cfg, run_dir, log_file):
        handle = apply_resid_post_feature_perturb_hook(
            model=model,
            layer_idx=args.layer_idx,
            sae_checkpoint_path=args.sae_checkpoint,
            feature_idx=args.feature_id,
            alpha=args.alpha,
            hook_start_step=args.hook_start_step,
            run_dir=run_dir,
            log_file=log_file,
        )
        return [handle]

    result = eval_libero(cfg, extra_hook_applier=applier)
    print(
        json.dumps(
            {
                "run_dir": result.run_dir,
                "success_rate": result.success_rate,
                "feature_id": args.feature_id,
                "alpha": args.alpha,
                "layer_idx": args.layer_idx,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

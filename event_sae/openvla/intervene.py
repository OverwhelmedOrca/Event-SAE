"""Residual-preserving single-feature SAE intervention hook (paper Section 4.5).

For a target feature index ``i`` and scalar ``alpha``:

    z' = α · z  for i ∈ S, z'_j = z_j otherwise
    x' = x + Dec(z') − Dec(z)

The residual SAE-reconstruction error ``err(x) = x − Dec(Enc(x))`` is
preserved by construction: ``x' = Dec(z') + err(x)``. With α = 0 the
target feature is zeroed; α ∈ (0, 1) softly suppresses; α = 1 recovers
``x`` exactly; α > 1 amplifies.

The hook reads per-step metadata from ``model._sae_hook_context`` (set
by the runner) and only activates once ``step_in_episode ≥
hook_start_step`` so a warm-up phase can run unhooked.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from event_sae.openvla.activations import load_batch_topk_sae


class _InterveneHookHandle:
    def __init__(self, hook, records_file):
        self._hook = hook
        self._records_file = records_file

    def remove(self) -> None:
        self._hook.remove()
        self._records_file.close()


def apply_resid_post_feature_perturb_hook(
    *,
    model,
    layer_idx: int,
    sae_checkpoint_path: str,
    feature_idx: int,
    alpha: float,
    hook_start_step: int,
    run_dir: str,
    log_file,
) -> _InterveneHookHandle:
    device = str(model.language_model.lm_head.weight.device)
    sae, sae_config = load_batch_topk_sae(Path(sae_checkpoint_path), device=device)
    trainer_cfg = sae_config["trainer"]
    if trainer_cfg.get("submodule_name") != "post_mlp_residual":
        raise ValueError(
            f"Expected a post_mlp_residual SAE checkpoint, got submodule_name="
            f"{trainer_cfg.get('submodule_name')!r}"
        )
    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    if not (0 <= feature_idx < dict_size):
        raise IndexError(f"feature_idx {feature_idx} out of range [0, {dict_size})")

    decoder_layers = model.language_model.model.layers
    if not (0 <= layer_idx < len(decoder_layers)):
        raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(decoder_layers)})")

    records_path = Path(run_dir) / f"intervene_feat{feature_idx}_alpha{alpha}_records.jsonl"
    records_file = records_path.open("w", encoding="utf-8")
    global_forward_idx = {"value": 0}

    def hook_fn(module, inputs, output):
        del module, inputs
        context = getattr(model, "_sae_hook_context", {})
        episode_num = context.get("episode_num")
        step_in_episode = context.get("step_in_episode")
        if episode_num is None or step_in_episode is None:
            return output
        if step_in_episode < hook_start_step:
            return output

        if not isinstance(output, tuple):
            raise ValueError(f"Expected decoder layer output tuple, got {type(output)!r}")
        hidden = output[0]
        if hidden.ndim != 3 or hidden.shape[-1] != activation_dim:
            raise ValueError(
                f"Hidden shape {tuple(hidden.shape)} incompatible with SAE activation_dim={activation_dim}"
            )

        flat = hidden.reshape(-1, hidden.shape[-1]).to(dtype=torch.float32)
        encoded = sae.encode(flat)
        perturbed = encoded.clone()
        feature_before = encoded[:, feature_idx].clone()
        perturbed[:, feature_idx] = perturbed[:, feature_idx] * float(alpha)
        feature_after = perturbed[:, feature_idx]

        recon_orig = sae.decode(encoded)
        recon_pert = sae.decode(perturbed)
        updated = flat + (recon_pert - recon_orig)
        updated = updated.to(dtype=hidden.dtype).reshape_as(hidden)

        global_forward_idx["value"] += 1
        record = {
            "episode_num": int(episode_num),
            "step_in_episode": int(step_in_episode),
            "task_description": context.get("task_description", ""),
            "global_forward_idx": int(global_forward_idx["value"]),
            "feature_idx": int(feature_idx),
            "alpha": float(alpha),
            "feature_activation_mean_before": float(feature_before.mean().item()),
            "feature_activation_mean_after": float(feature_after.mean().item()),
            "feature_activation_max_before": float(feature_before.max().item()),
            "feature_activation_max_after": float(feature_after.max().item()),
            "num_tokens": int(encoded.shape[0]),
        }
        records_file.write(json.dumps(record) + "\n")
        records_file.flush()
        return (updated, *output[1:])

    hook = decoder_layers[layer_idx].register_forward_hook(hook_fn)
    log_file.write(
        f"Resid-post intervention hook: layer={layer_idx} feature_idx={feature_idx} "
        f"alpha={alpha} hook_start_step={hook_start_step} sae_checkpoint={sae_checkpoint_path}\n"
    )
    log_file.flush()
    return _InterveneHookHandle(hook=hook, records_file=records_file)

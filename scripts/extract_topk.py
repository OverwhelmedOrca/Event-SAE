"""CLI: offline extract top-k SAE activations from dense shards.

Reads:
  - Dense residual shards `{run_dir}/sae_activations/post_mlp_residual/layer_NN_shard_MMMMMM.pt`
  - Metadata `{run_dir}/sae_activations/post_mlp_residual/activation_index.jsonl`
  - Trained `BatchTopKSAE` checkpoint (`ae.pt` with sibling `config.json`)

Writes (under `--output-dir`, default `{run_dir}/topk_activations/`):
  - `shard_NNNNNN.pt` with sparse top-k rows + metadata
  - `manifest.json` in `token_topk_sparse_v1` format (same as online mode)

The output is byte-format-compatible with `event_sae.openvla.activations.apply_sae_topk_collect_hooks`,
so downstream scoring can consume either online or offline shards uniformly.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.openvla.activations import load_batch_topk_sae


def _load_index(index_path: Path, layer_idx: int) -> dict[str, list[dict]]:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if int(record["layer_idx"]) != layer_idx:
                continue
            by_shard[str(record["shard_path"])].append(record)
    for records in by_shard.values():
        records.sort(key=lambda r: int(r["row_start"]))
    return by_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SAE encode of dense activation shards.")
    parser.add_argument(
        "--dense-dir",
        required=True,
        help="Directory containing dense layer_NN_shard_*.pt + activation_index.jsonl.",
    )
    parser.add_argument("--sae-checkpoint", required=True, help="Path to trained ae.pt")
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: {dense_dir}/../../topk_activations/).",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu).")
    args = parser.parse_args()

    dense_dir = Path(args.dense_dir).resolve()
    index_path = dense_dir / "activation_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (dense_dir.parent.parent / "topk_activations").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sae, config = load_batch_topk_sae(Path(args.sae_checkpoint), device=device)
    trainer_cfg = config["trainer"]
    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    if not (1 <= args.topk <= dict_size):
        raise ValueError(f"topk must be in [1, {dict_size}], got {args.topk}")

    index = _load_index(index_path, layer_idx=args.layer_idx)
    if not index:
        raise RuntimeError(f"No index records found for layer {args.layer_idx} in {index_path}")

    manifest = {
        "format": "token_topk_sparse_v1",
        "layer": args.layer_idx,
        "sae_path": str(args.sae_checkpoint),
        "dict_size": dict_size,
        "activation_dim": activation_dim,
        "topk": args.topk,
        "num_shards": 0,
        "total_rows": 0,
        "shards": [],
    }

    shard_names = sorted(index.keys())
    total_rows = 0
    for src_shard_name in shard_names:
        src_shard_path = dense_dir / src_shard_name
        if not src_shard_path.is_file():
            raise FileNotFoundError(f"Missing dense shard: {src_shard_path}")
        dense = torch.load(src_shard_path, map_location="cpu").to(torch.float32)
        if dense.ndim != 2 or int(dense.shape[1]) != activation_dim:
            raise ValueError(
                f"Unexpected dense shard shape {tuple(dense.shape)} in {src_shard_path}; "
                f"expected (N, {activation_dim})"
            )
        records = index[src_shard_name]
        n_rows = int(dense.shape[0])

        episode_num = torch.zeros((n_rows,), dtype=torch.int64)
        step_in_episode = torch.zeros((n_rows,), dtype=torch.int64)
        global_forward_idx = torch.zeros((n_rows,), dtype=torch.int64)
        token_idx = torch.zeros((n_rows,), dtype=torch.int64)
        batch_idx = torch.zeros((n_rows,), dtype=torch.int64)
        for record in records:
            r0, r1 = int(record["row_start"]), int(record["row_end"])
            episode_num[r0:r1] = int(record.get("episode_num") or 0)
            step_in_episode[r0:r1] = int(record.get("step_in_episode") or 0)
            global_forward_idx[r0:r1] = int(record.get("global_forward_idx") or 0)
            token_idx[r0:r1] = torch.arange(r1 - r0, dtype=torch.int64)

        with torch.no_grad():
            encoded = sae.encode(dense.to(device))
            values, indices = torch.topk(encoded, k=args.topk, dim=-1)
            values = values.float().cpu()
            indices = indices.to(torch.int32).cpu()

        out_shard_name = f"shard_{manifest['num_shards']:06d}.pt"
        torch.save(
            {
                "episode_num": episode_num,
                "step_in_episode": step_in_episode,
                "global_forward_idx": global_forward_idx,
                "batch_idx": batch_idx,
                "token_idx": token_idx,
                "top_feature_ids": indices,
                "top_feature_vals": values,
            },
            output_dir / out_shard_name,
        )
        manifest["shards"].append(
            {
                "shard_idx": manifest["num_shards"],
                "path": out_shard_name,
                "num_rows": n_rows,
                "row_start": total_rows,
                "row_end": total_rows + n_rows,
            }
        )
        manifest["num_shards"] += 1
        total_rows += n_rows
        print(f"Encoded {src_shard_name} → {out_shard_name}  rows={n_rows}")

    manifest["total_rows"] = total_rows
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()

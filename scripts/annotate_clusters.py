"""CLI: Gemini VLM annotation of event clusters.

Reads `clusters.jsonl` from `scripts/cluster_events.py`, writes one
JSONL row per cluster with the assigned `phrase` + `phase`.

API key: `GEMINI_API_KEY` env var (preferred), or `--api-key-path` pointing
at a text file with the key.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.events.annotate import annotate_clusters, load_api_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate event clusters with Gemini.")
    parser.add_argument("--clusters-path", required=True, help="Path to clusters.jsonl")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output JSONL path (default: {model}_cluster_annotations.jsonl next to clusters.jsonl)",
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument(
        "--api-key-path",
        default=None,
        help="Optional path to text file with API key (fallback if GEMINI_API_KEY env var unset)",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-clusters", type=int, default=None)
    args = parser.parse_args()

    clusters_path = Path(args.clusters_path).resolve()
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path is not None
        else clusters_path.with_name(
            f"{args.model.replace('.', '_')}_cluster_annotations.jsonl"
        ).resolve()
    )

    api_key = load_api_key(Path(args.api_key_path) if args.api_key_path else None)

    annotate_clusters(
        clusters_path=clusters_path,
        output_path=output_path,
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_clusters=args.max_clusters,
    )
    print(f"Clusters: {clusters_path}")
    print(f"Annotations: {output_path}")


if __name__ == "__main__":
    main()

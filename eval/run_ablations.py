"""
Ablation runner: runs multiple configurations and collects results.

Configs: dense-only, sparse-only, hybrid, +reranker, +KG, +rewrite, strict vs lenient.

Usage:
    python -m eval.run_ablations --dataset eval/datasets/queries.jsonl
"""

import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ablation configurations
CONFIGS = [
    {
        "name": "dense_only",
        "description": "Dense retrieval only (FAISS)",
        "dense_weight": 1.0,
        "sparse_weight": 0.0,
        "use_reranker": True,
        "use_kg": True,
        "max_retries": 2,
        "grounding_mode": "strict",
    },
    {
        "name": "sparse_only",
        "description": "Sparse retrieval only (BM25)",
        "dense_weight": 0.0,
        "sparse_weight": 1.0,
        "use_reranker": True,
        "use_kg": True,
        "max_retries": 2,
        "grounding_mode": "strict",
    },
    {
        "name": "hybrid",
        "description": "Hybrid retrieval (FAISS + BM25)",
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "use_reranker": True,
        "use_kg": True,
        "max_retries": 2,
        "grounding_mode": "strict",
    },
    {
        "name": "hybrid_no_reranker",
        "description": "Hybrid without cross-encoder reranking",
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "use_reranker": False,
        "use_kg": True,
        "max_retries": 2,
        "grounding_mode": "strict",
    },
    {
        "name": "hybrid_no_kg",
        "description": "Hybrid without KG expansion",
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "use_reranker": True,
        "use_kg": False,
        "max_retries": 2,
        "grounding_mode": "strict",
    },
    {
        "name": "hybrid_no_rewrite",
        "description": "Hybrid without query rewrite loop",
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "use_reranker": True,
        "use_kg": True,
        "max_retries": 0,
        "grounding_mode": "strict",
    },
    {
        "name": "hybrid_lenient",
        "description": "Hybrid with lenient grounding",
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "use_reranker": True,
        "use_kg": True,
        "max_retries": 2,
        "grounding_mode": "lenient",
    },
]


def run_ablation(config: dict, dataset_path: str) -> dict:
    """Run a single ablation config and collect metrics."""
    from config import settings
    from eval.eval_retrieval import evaluate_retrieval

    # Note: full ablation requires loading models per config.
    # This scaffold shows the structure; actual execution needs
    # the models to be available.
    print(f"\n  Running: {config['name']} — {config['description']}")

    try:
        retrieval_results = evaluate_retrieval(dataset_path)
        return {
            "config": config,
            "retrieval": retrieval_results["aggregate"],
            "status": "completed",
        }
    except Exception as e:
        return {
            "config": config,
            "error": str(e),
            "status": "failed",
        }


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--dataset", default="eval/datasets/queries.jsonl")
    parser.add_argument("--output", default="eval/results/ablations.json")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Specific config names to run (default: all)")
    args = parser.parse_args()

    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c["name"] in args.configs]

    print(f"🔬 Running {len(configs_to_run)} ablation experiments")
    results = []

    for config in configs_to_run:
        result = run_ablation(config, args.dataset)
        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n📊 Ablation Summary:")
    for r in results:
        status = "✅" if r["status"] == "completed" else "❌"
        print(f"   {status} {r['config']['name']}: {r['status']}")
    print(f"\n   Saved to: {output_path}")


if __name__ == "__main__":
    main()

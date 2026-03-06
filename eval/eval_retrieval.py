"""
Retrieval evaluation: Recall@k, MRR, nDCG.

Ground-truth format (queries.jsonl):
  {"query": "...", "relevant_doc_ids": ["doc_id_1", "doc_id_2", ...]}

Usage:
    python -m eval.eval_retrieval --dataset eval/datasets/queries.jsonl --output eval/results/retrieval.json
"""

import json
import math
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute nDCG@k (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], 1)
        if doc_id in relevant_ids
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_ids), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_retrieval(
    dataset_path: str,
    retriever=None,
    top_k_values: list[int] = [3, 5, 10],
) -> dict:
    """
    Run retrieval evaluation over a ground-truth dataset.

    Args:
        dataset_path: Path to queries.jsonl
        retriever: HybridRetriever instance (if None, loads from config)
        top_k_values: k values for metrics

    Returns:
        Dict with per-query and aggregate metrics
    """
    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    queries = [json.loads(line) for line in dataset.read_text(encoding="utf-8").strip().split("\n")]

    if retriever is None:
        from config import settings
        from agribot.ingestion.index_builder import IndexBundle
        from agribot.retrieval.hybrid import HybridRetriever
        from sentence_transformers import SentenceTransformer

        bundle = IndexBundle.load(settings.INDEX_DIR)
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        retriever = HybridRetriever(
            index_bundle=bundle,
            embedding_model=embedding_model,
            dense_top_k=settings.DENSE_TOP_K,
            sparse_top_k=settings.SPARSE_TOP_K,
        )

    max_k = max(top_k_values)
    per_query = []

    for item in queries:
        query = item["query"]
        relevant = set(item.get("relevant_doc_ids", []))

        results = retriever.retrieve(query, top_n=max_k)
        retrieved_ids = [r.citation for r in results]

        metrics = {}
        for k in top_k_values:
            metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)
        metrics["mrr"] = mrr(retrieved_ids, relevant)

        per_query.append({"query": query, **metrics})

    # Aggregate
    aggregate = {}
    for key in per_query[0]:
        if key == "query":
            continue
        aggregate[f"avg_{key}"] = sum(q[key] for q in per_query) / len(per_query)

    return {"aggregate": aggregate, "per_query": per_query, "n_queries": len(queries)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--dataset", default="eval/datasets/queries.jsonl")
    parser.add_argument("--output", default="eval/results/retrieval.json")
    args = parser.parse_args()

    results = evaluate_retrieval(args.dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n📊 Retrieval Results ({results['n_queries']} queries):")
    for k, v in results["aggregate"].items():
        print(f"   {k}: {v:.4f}")
    print(f"\n   Saved to: {output_path}")


if __name__ == "__main__":
    main()

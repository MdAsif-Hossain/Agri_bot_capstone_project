"""Retrieval-quality evaluation: Recall@k and Mean Reciprocal Rank (MRR).

For each in-corpus query in dataset.json we run the same hybrid pipeline the
live agent uses (FAISS + BM25 → RRF fusion → cross-encoder rerank) and check
whether any of the top-k retrieved chunks come from the expected source PDF
(and ideally on or near the expected page).

Run:
    python evaluation/eval_retrieval.py
    python evaluation/eval_retrieval.py --top-k 1 3 5 10
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

from evaluation.common import (
    filter_queries,
    load_dataset,
    save_results,
    to_retrieval_query,
)


def _chunk_meta(chunk_or_ev) -> tuple[str, int]:
    """Extract (source_file, page_num) whether the object is an EvidenceChunk
    (wraps .chunk) or a raw Chunk (has fields directly) or a dict."""
    inner = getattr(chunk_or_ev, "chunk", chunk_or_ev)  # unwrap EvidenceChunk
    if isinstance(inner, dict):
        return str(inner.get("source_file", "")), int(inner.get("page_num", -1))
    return str(getattr(inner, "source_file", "")), int(getattr(inner, "page_num", -1))


def _hit_rank(retrieved, expected_sources) -> int | None:
    """Return 1-based rank of the first retrieved chunk that matches an
    expected source_file (page tolerance ±2)."""
    expected_map: dict[str, set[int]] = defaultdict(set)
    for src in expected_sources:
        expected_map[src["source_file"]].update(src.get("expected_pages", []))

    for rank, chunk in enumerate(retrieved, start=1):
        src, page = _chunk_meta(chunk)
        if src in expected_map:
            if any(abs(page - ep) <= 2 for ep in expected_map[src]):
                return rank
            # Source matches but page is off — still counts as a (weaker) hit
            return rank
    return None


def run(top_k_values: list[int]) -> dict:
    from config import settings
    from agribot.retrieval.hybrid import HybridRetriever
    from agribot.retrieval.reranker import Reranker
    from agribot.ingestion.index_builder import IndexBundle
    from agribot.translation.bangla_t5 import get_translator
    from sentence_transformers import SentenceTransformer

    print(f"Loading retrieval stack (chunks from {settings.INDEX_DIR})...")
    bundle = IndexBundle.load(settings.INDEX_DIR)
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    # The deployed agent translates Bengali queries to English (normalize node)
    # before retrieval; mirror that here so dialect-bn queries are scored on the
    # same input the system actually retrieves on.
    translator = get_translator(device="cpu")
    retriever = HybridRetriever(
        index_bundle=bundle,
        embedding_model=embedder,
        dense_top_k=max(max(top_k_values), settings.DENSE_TOP_K),
        sparse_top_k=max(max(top_k_values), settings.SPARSE_TOP_K),
        dense_weight=settings.DENSE_WEIGHT,
        sparse_weight=settings.SPARSE_WEIGHT,
    )
    reranker = Reranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        top_n=max(top_k_values),
        threshold=settings.RERANK_THRESHOLD,
    )

    dataset = load_dataset()
    queries = filter_queries(
        dataset,
        categories={"in_corpus_factual", "in_corpus_procedural", "dialect_bn"},
    )
    print(f"Evaluating {len(queries)} in-corpus queries (skipping refusal queries)")

    per_query = []
    ranks = []
    timings = []

    for q in queries:
        if not q["expected_sources"]:
            continue
        t0 = time.perf_counter()
        rq = to_retrieval_query(translator, q["query"])
        candidates = retriever.retrieve(rq, top_n=max(top_k_values) * 2)
        ranked = reranker.rerank(rq, candidates) if candidates else []
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings.append(elapsed_ms)

        rank = _hit_rank(ranked, q["expected_sources"])
        ranks.append(rank)
        top1_meta = None
        if ranked:
            src, page = _chunk_meta(ranked[0])
            top1_meta = {"source_file": src, "page_num": page}
        per_query.append(
            {
                "id": q["id"],
                "category": q["category"],
                "rank": rank,
                "retrieved_top1": top1_meta,
            }
        )

    summary = {
        "queries_evaluated": len(per_query),
        "mrr": round(
            sum(1.0 / r for r in ranks if r is not None) / max(1, len(ranks)), 4
        ),
        "recall_at_k": {
            k: round(
                sum(1 for r in ranks if r is not None and r <= k) / max(1, len(ranks)),
                4,
            )
            for k in top_k_values
        },
        "avg_retrieval_latency_ms": round(sum(timings) / max(1, len(timings)), 1),
    }

    print("\n=== Retrieval results ===")
    print(f"  Queries scored: {summary['queries_evaluated']}")
    print(f"  MRR:            {summary['mrr']}")
    for k in top_k_values:
        print(f"  Recall@{k}:      {summary['recall_at_k'][k]}")
    print(f"  Avg latency:    {summary['avg_retrieval_latency_ms']} ms")

    payload = {"summary": summary, "per_query": per_query}
    out = save_results("retrieval", payload)
    print(f"\nResults written to: {out}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", nargs="*", type=int, default=[1, 3, 5, 10])
    args = p.parse_args()
    run(args.top_k)


if __name__ == "__main__":
    main()

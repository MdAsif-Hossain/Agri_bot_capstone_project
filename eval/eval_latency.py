"""
Latency evaluation: p50/p95 from agent timings_ms.

Reads pre-computed results with timings or runs queries live.

Usage:
    python -m eval.eval_latency --dataset eval/datasets/queries.jsonl --output eval/results/latency.json
"""

import json
import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[f]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


def evaluate_latency(dataset_path: str, agent=None) -> dict:
    """Run latency evaluation."""
    dataset = Path(dataset_path)
    queries = [json.loads(line) for line in dataset.read_text(encoding="utf-8").strip().split("\n")]

    if agent is None:
        from config import settings
        from agribot.llm.engine import get_llm
        from agribot.ingestion.index_builder import IndexBundle
        from agribot.retrieval.hybrid import HybridRetriever
        from agribot.retrieval.reranker import Reranker
        from agribot.knowledge_graph.schema import KnowledgeGraph
        from agribot.knowledge_graph.entity_linker import EntityLinker
        from agribot.knowledge_graph.seed_data import seed_knowledge_graph
        from agribot.agent.graph import build_agent_graph
        from agribot.translation.bangla_t5 import get_translator
        from sentence_transformers import SentenceTransformer

        llm = get_llm(model_path=str(settings.MODEL_PATH), n_ctx=settings.LLM_N_CTX, n_gpu_layers=settings.LLM_N_GPU_LAYERS)
        bundle = IndexBundle.load(settings.INDEX_DIR)
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        retriever = HybridRetriever(index_bundle=bundle, embedding_model=embedding_model)
        reranker = Reranker(threshold=settings.RERANK_THRESHOLD)
        kg = KnowledgeGraph(settings.KG_DB_PATH)
        seed_knowledge_graph(kg)
        linker = EntityLinker(kg)
        translator = get_translator(device="cpu")
        agent = build_agent_graph(llm=llm, retriever=retriever, reranker=reranker, entity_linker=linker, translator=translator)

    import time
    per_query = []
    for item in queries:
        query = item["query"]
        state = {
            "query_original": query, "query_language": "", "query_normalized": "",
            "query_expanded": "", "kg_entities": [], "evidences": [],
            "evidence_texts": "", "evidence_grade": "", "answer": "", "answer_bn": "",
            "citations": [], "is_verified": False, "verification_reason": "",
            "retry_count": 0, "should_refuse": False, "input_mode": "text",
            "input_audio_path": "", "error": "", "trace_id": "", "timings_ms": {},
            "grounding_action": "pass", "follow_up_suggestions": [],
        }

        start = time.perf_counter()
        result = agent.invoke(state)
        total_ms = (time.perf_counter() - start) * 1000

        timings = result.get("timings_ms", {})
        timings["total"] = round(total_ms, 1)

        per_query.append({"query": query, "timings_ms": timings, "total_ms": round(total_ms, 1)})

    totals = [q["total_ms"] for q in per_query]
    node_names = set()
    for q in per_query:
        node_names.update(q["timings_ms"].keys())

    aggregate = {
        "total_p50": percentile(totals, 50),
        "total_p95": percentile(totals, 95),
        "total_mean": statistics.mean(totals) if totals else 0,
    }

    for node in sorted(node_names - {"total"}):
        vals = [q["timings_ms"].get(node, 0) for q in per_query]
        aggregate[f"{node}_p50"] = percentile(vals, 50)
        aggregate[f"{node}_p95"] = percentile(vals, 95)

    return {"aggregate": aggregate, "per_query": per_query, "n_queries": len(per_query)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline latency")
    parser.add_argument("--dataset", default="eval/datasets/queries.jsonl")
    parser.add_argument("--output", default="eval/results/latency.json")
    args = parser.parse_args()

    results = evaluate_latency(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n⏱️  Latency Results ({results['n_queries']} queries):")
    for k, v in results["aggregate"].items():
        print(f"   {k}: {v:.1f}ms")
    print(f"\n   Saved to: {output_path}")


if __name__ == "__main__":
    main()

"""
Refusal evaluation: correctness on out-of-scope (OOS) queries.

Measures whether the system correctly refuses to answer OOS queries.

Dataset format (oos_queries.jsonl):
  {"query": "...", "expected": "refuse"}

Usage:
    python -m eval.eval_refusal --dataset eval/datasets/oos_queries.jsonl --output eval/results/refusal.json
"""

import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REFUSAL_INDICATORS = [
    "i don't know",
    "cannot provide",
    "not available",
    "no evidence",
    "outside my scope",
    "please consult",
    "cannot answer",
    "not enough information",
]


def is_refusal(answer: str) -> bool:
    """Detect if an answer is a refusal response."""
    lower = answer.lower()
    return any(indicator in lower for indicator in REFUSAL_INDICATORS)


def evaluate_refusal(dataset_path: str, agent=None) -> dict:
    """Run refusal evaluation over OOS dataset."""
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

    per_query = []
    for item in queries:
        query = item["query"]
        expected = item.get("expected", "refuse")

        state = {
            "query_original": query, "query_language": "", "query_normalized": "",
            "query_expanded": "", "kg_entities": [], "evidences": [],
            "evidence_texts": "", "evidence_grade": "", "answer": "", "answer_bn": "",
            "citations": [], "is_verified": False, "verification_reason": "",
            "retry_count": 0, "should_refuse": False, "input_mode": "text",
            "input_audio_path": "", "error": "", "trace_id": "", "timings_ms": {},
            "grounding_action": "pass", "follow_up_suggestions": [],
        }
        result = agent.invoke(state)

        answer = result.get("answer", "")
        action = result.get("grounding_action", "pass")
        refused = is_refusal(answer) or action == "refuse"

        correct = (expected == "refuse" and refused) or (expected != "refuse" and not refused)

        per_query.append({
            "query": query,
            "expected": expected,
            "refused": refused,
            "correct": correct,
            "grounding_action": action,
        })

    correct_count = sum(1 for q in per_query if q["correct"])
    aggregate = {
        "accuracy": correct_count / len(per_query),
        "refusal_rate": sum(1 for q in per_query if q["refused"]) / len(per_query),
        "correct": correct_count,
        "total": len(per_query),
    }

    return {"aggregate": aggregate, "per_query": per_query}


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOS refusal correctness")
    parser.add_argument("--dataset", default="eval/datasets/oos_queries.jsonl")
    parser.add_argument("--output", default="eval/results/refusal.json")
    args = parser.parse_args()

    results = evaluate_refusal(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    agg = results["aggregate"]
    print(f"\n📊 Refusal Results ({agg['total']} queries):")
    print(f"   Accuracy:     {agg['accuracy']:.2%}")
    print(f"   Refusal rate: {agg['refusal_rate']:.2%}")
    print(f"\n   Saved to: {output_path}")


if __name__ == "__main__":
    main()

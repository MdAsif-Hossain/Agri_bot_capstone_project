"""
Grounding evaluation: citation coverage + unsupported claim detection.

Uses agent pipeline responses to measure how well answers are grounded.

Usage:
    python -m eval.eval_grounding --dataset eval/datasets/queries.jsonl --output eval/results/grounding.json
"""

import json
import re
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def citation_coverage(answer: str, citations: list[str]) -> float:
    """Fraction of answer sentences that reference a citation."""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    if not sentences:
        return 0.0

    cited_count = 0
    for sent in sentences:
        for cit in citations:
            # Check if any part of the citation text matches
            if any(word.lower() in sent.lower() for word in cit.split()[:3] if len(word) > 3):
                cited_count += 1
                break

    return cited_count / len(sentences)


def unsupported_claim_ratio(answer: str, evidence_texts: str) -> float:
    """Proxy for unsupported claims: sentences with low evidence overlap."""
    if not answer or not evidence_texts:
        return 1.0

    evidence_lower = evidence_texts.lower()
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    unsupported = 0

    for sent in sentences:
        words = set(re.findall(r'\b\w{4,}\b', sent.lower()))
        if not words:
            continue
        overlap = sum(1 for w in words if w in evidence_lower)
        coverage = overlap / len(words) if words else 0
        if coverage < 0.2:
            unsupported += 1

    return unsupported / len(sentences) if sentences else 1.0


def evaluate_grounding(dataset_path: str, agent=None) -> dict:
    """Run grounding evaluation over a dataset."""
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

        cov = citation_coverage(result.get("answer", ""), result.get("citations", []))
        unsup = unsupported_claim_ratio(result.get("answer", ""), result.get("evidence_texts", ""))

        per_query.append({
            "query": query,
            "citation_coverage": round(cov, 4),
            "unsupported_ratio": round(unsup, 4),
            "is_verified": result.get("is_verified", False),
            "grounding_action": result.get("grounding_action", ""),
        })

    aggregate = {
        "avg_citation_coverage": sum(q["citation_coverage"] for q in per_query) / len(per_query),
        "avg_unsupported_ratio": sum(q["unsupported_ratio"] for q in per_query) / len(per_query),
        "verified_rate": sum(1 for q in per_query if q["is_verified"]) / len(per_query),
    }

    return {"aggregate": aggregate, "per_query": per_query, "n_queries": len(per_query)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate answer grounding")
    parser.add_argument("--dataset", default="eval/datasets/queries.jsonl")
    parser.add_argument("--output", default="eval/results/grounding.json")
    args = parser.parse_args()

    results = evaluate_grounding(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n📊 Grounding Results ({results['n_queries']} queries):")
    for k, v in results["aggregate"].items():
        print(f"   {k}: {v:.4f}")
    print(f"\n   Saved to: {output_path}")


if __name__ == "__main__":
    main()

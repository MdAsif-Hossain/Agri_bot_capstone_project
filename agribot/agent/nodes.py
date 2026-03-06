"""
Agent graph nodes: each function implements one step of the agentic RAG pipeline.

Flow: normalize → kg_link → retrieve → rerank → grade → (rewrite | generate) → translate → verify → enforce_policy → END

Each node records timing in state["timings_ms"] for observability.
"""

import re
import time
import logging

from llama_cpp import Llama

from agribot.agent.state import AgentState
from agribot.retrieval.hybrid import HybridRetriever, EvidenceChunk
from agribot.retrieval.reranker import Reranker
from agribot.knowledge_graph.entity_linker import EntityLinker
from agribot.translation.bangla_t5 import BanglaTranslator
from agribot.llm.engine import (
    grade_evidence,
    generate_answer,
    rewrite_query,
    verify_answer,
)

logger = logging.getLogger(__name__)


# --- Timing helper ---

def _timed(node_name: str, fn, state: AgentState) -> dict:
    """Wrap a node function with timing instrumentation."""
    start = time.perf_counter()
    result = fn(state)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Merge timing into result
    timings = dict(state.get("timings_ms", {}))
    timings[node_name] = round(elapsed_ms, 1)
    result["timings_ms"] = timings

    logger.debug(
        "Node %s completed in %.1fms",
        node_name, elapsed_ms,
        extra={"trace_id": state.get("trace_id", "")},
    )
    return result


# --- Node factory functions ---
# Each returns a closure bound to the required services.


def make_normalize_node(translator: BanglaTranslator | None = None):
    """Create the query normalization node. Optionally translates Bengali queries."""

    def _normalize(state: AgentState) -> dict:
        query = state["query_original"].strip()
        query = re.sub(r"\s+", " ", query)

        has_bengali = bool(re.search(r"[\u0980-\u09FF]", query))
        lang = "bn" if has_bengali else "en"

        normalized_query = query
        if lang == "bn" and translator is not None:
            try:
                translated = translator.translate_bn_to_en(query)
                if translated and translated != query:
                    logger.info("BN→EN query translation: '%s' → '%s'", query[:60], translated[:60])
                    normalized_query = translated
                else:
                    logger.warning("BN→EN translation returned empty/same; using original")
            except Exception as e:
                logger.error("Translation fallback: %s", e)
                # Fallback: use original query

        logger.info("Normalized query (lang=%s): %s", lang, normalized_query[:100])
        return {
            "query_normalized": normalized_query,
            "query_language": lang,
        }

    def normalize(state: AgentState) -> dict:
        return _timed("normalize", _normalize, state)

    return normalize


def make_kg_link_node(entity_linker: EntityLinker):
    """Create the KG entity linking + query expansion node."""

    def _kg_link(state: AgentState) -> dict:
        query = state["query_normalized"]

        try:
            entities = entity_linker.link_entities(query)
            entity_info = [
                {"bn": e.canonical_bn, "en": e.canonical_en, "type": e.entity_type}
                for e in entities
            ]
            expanded = entity_linker.expand_query(query)

            logger.info("KG linked %d entities, expanded: %s", len(entities), expanded[:100])
            return {
                "query_expanded": expanded,
                "kg_entities": entity_info,
            }
        except Exception as e:
            # Fallback: skip KG expansion on failure
            logger.error("KG linking fallback: %s", e)
            return {
                "query_expanded": query,
                "kg_entities": [],
            }

    def kg_link(state: AgentState) -> dict:
        return _timed("kg_link", _kg_link, state)

    return kg_link


def make_retrieve_node(retriever: HybridRetriever):
    """Create the hybrid retrieval node."""

    def _retrieve(state: AgentState) -> dict:
        query = state.get("query_expanded") or state["query_normalized"]
        try:
            evidences = retriever.retrieve(query, top_n=15)
            logger.info("Retrieved %d evidence chunks", len(evidences))
            return {"evidences": evidences, "error": ""}
        except Exception as e:
            logger.error("Retrieval error: %s", e)
            return {"evidences": [], "error": f"Retrieval error: {e}"}

    def retrieve(state: AgentState) -> dict:
        return _timed("retrieve", _retrieve, state)

    return retrieve


def make_rerank_node(reranker: Reranker):
    """Create the reranking node."""

    def _rerank(state: AgentState) -> dict:
        query = state.get("query_expanded") or state["query_normalized"]
        evidences = state.get("evidences", [])

        if not evidences:
            return {
                "evidences": [],
                "evidence_texts": "",
                "should_refuse": True,
            }

        try:
            reranked = reranker.rerank(query, evidences)

            evidence_parts = []
            citations = []
            for ev in reranked:
                citation = ev.citation
                evidence_parts.append(f"[{citation}]: {ev.text}")
                if citation not in citations:
                    citations.append(citation)

            evidence_text = "\n\n".join(evidence_parts)
            logger.info("Reranked to %d evidences", len(reranked))
            return {
                "evidences": reranked,
                "evidence_texts": evidence_text,
                "citations": citations,
                "should_refuse": len(reranked) == 0,
            }
        except Exception as e:
            # Fallback: use top-k retrieval without reranking
            logger.error("Reranking fallback to top-k: %s", e)
            fallback = evidences[:5]
            return {
                "evidences": fallback,
                "evidence_texts": "\n\n".join(ev.text for ev in fallback),
                "citations": list({ev.citation for ev in fallback}),
                "should_refuse": False,
                "error": f"Reranking error (using top-k fallback): {e}",
            }

    def rerank(state: AgentState) -> dict:
        return _timed("rerank", _rerank, state)

    return rerank


def make_grade_node(llm: Llama):
    """Create the evidence grading node."""

    def _grade(state: AgentState) -> dict:
        if state.get("should_refuse"):
            return {"evidence_grade": "INSUFFICIENT"}

        query = state["query_normalized"]
        context = state.get("evidence_texts", "")

        if not context.strip():
            return {"evidence_grade": "INSUFFICIENT"}

        grade_result, confidence = grade_evidence(llm, query, context)
        logger.info("Evidence grade: %s (confidence=%.2f)", grade_result, confidence)
        return {"evidence_grade": grade_result}

    def grade(state: AgentState) -> dict:
        return _timed("grade", _grade, state)

    return grade


def make_rewrite_node(llm: Llama):
    """Create the query rewrite node for retry."""

    def _rewrite(state: AgentState) -> dict:
        retry_count = state.get("retry_count", 0) + 1
        original_query = state["query_normalized"]
        failed_context = state.get("evidence_texts", "")

        rewritten = rewrite_query(llm, original_query, failed_context)
        logger.info("Rewritten query (retry %d): %s", retry_count, rewritten[:100])

        return {
            "query_expanded": rewritten,
            "retry_count": retry_count,
        }

    def rewrite(state: AgentState) -> dict:
        return _timed("rewrite", _rewrite, state)

    return rewrite


def make_generate_node(llm: Llama, max_tokens: int = 512):
    """Create the answer generation node."""

    def _gen(state: AgentState) -> dict:
        if state.get("should_refuse"):
            return {
                "answer": "I don't know based on the provided documents.",
                "answer_bn": "আমি দেওয়া নথি থেকে এই তথ্য পাইনি।",
                "is_verified": True,
                "verification_reason": "Refusal — no evidence",
            }

        query = state["query_normalized"]
        context = state.get("evidence_texts", "")

        try:
            answer = generate_answer(llm, query, context, max_tokens=max_tokens)
            if not answer.strip():
                answer = "I don't know based on the provided documents."

            logger.info("Generated answer (%d chars)", len(answer))
            return {"answer": answer}
        except Exception as e:
            logger.error("Generation error: %s", e)
            return {
                "answer": "An error occurred during answer generation.",
                "error": f"Generation error: {e}",
            }

    def gen(state: AgentState) -> dict:
        return _timed("generate", _gen, state)

    return gen


def make_verify_node(llm: Llama):
    """Create the answer verification node."""

    def _verify(state: AgentState) -> dict:
        answer = state.get("answer", "")
        context = state.get("evidence_texts", "")

        if state.get("should_refuse") or not answer.strip():
            return {
                "is_verified": True,
                "verification_reason": "Skipped (refusal)",
            }

        try:
            is_verified, reason = verify_answer(llm, answer, context)
            logger.info("Verification: %s — %s", is_verified, reason)
            return {
                "is_verified": is_verified,
                "verification_reason": reason,
            }
        except Exception as e:
            # Fallback: mark as unverified, policy will handle
            logger.error("Verification fallback: %s", e)
            return {
                "is_verified": False,
                "verification_reason": f"Verification error (fallback): {e}",
            }

    def verify(state: AgentState) -> dict:
        return _timed("verify", _verify, state)

    return verify


def make_translate_node(translator: BanglaTranslator):
    """Create the BanglaT5 translation node."""

    def _translate(state: AgentState) -> dict:
        answer = state.get("answer", "")

        if state.get("should_refuse") or state.get("answer_bn"):
            return {}

        if not answer.strip():
            return {"answer_bn": ""}

        try:
            answer_bn = translator.translate_en_to_bn(answer)
            logger.info("Translated to Bengali (%d chars)", len(answer_bn))
            return {"answer_bn": answer_bn}
        except Exception as e:
            # Fallback: return empty with logged reason
            logger.error("Translation fallback: %s", e)
            return {"answer_bn": ""}

    def translate(state: AgentState) -> dict:
        return _timed("translate", _translate, state)

    return translate

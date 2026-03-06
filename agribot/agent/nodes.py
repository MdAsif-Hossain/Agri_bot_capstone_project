"""
Agent graph nodes: each function implements one step of the agentic RAG pipeline.

Flow: normalize → kg_link → retrieve → rerank → grade → (rewrite | generate) → translate → verify → respond
"""

import re
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


# --- Node factory functions ---
# Each returns a closure bound to the required services.


def make_normalize_node(translator: BanglaTranslator | None = None):
    """Create the query normalization node. Optionally translates Bengali queries."""

    def normalize(state: AgentState) -> dict:
        query = state["query_original"].strip()

        # Basic normalization
        # Remove excessive whitespace
        query = re.sub(r"\s+", " ", query)

        # Detect language (simple heuristic: Bengali Unicode range)
        has_bengali = bool(re.search(r"[\u0980-\u09FF]", query))
        lang = "bn" if has_bengali else "en"

        # If Bengali input, translate to English for retrieval/LLM
        normalized_query = query
        if lang == "bn" and translator is not None:
            translated = translator.translate_bn_to_en(query)
            if translated and translated != query:
                logger.info("BN→EN query translation: '%s' → '%s'", query[:60], translated[:60])
                normalized_query = translated
            else:
                logger.warning("BN→EN translation returned empty/same; using original")

        logger.info("Normalized query (lang=%s): %s", lang, normalized_query[:100])
        return {
            "query_normalized": normalized_query,
            "query_language": lang,
        }

    return normalize


def make_kg_link_node(entity_linker: EntityLinker):
    """Create the KG entity linking + query expansion node."""

    def kg_link(state: AgentState) -> dict:
        query = state["query_normalized"]

        # Link entities
        entities = entity_linker.link_entities(query)
        entity_info = [
            {"bn": e.canonical_bn, "en": e.canonical_en, "type": e.entity_type}
            for e in entities
        ]

        # Expand query
        expanded = entity_linker.expand_query(query)

        logger.info(
            "KG linked %d entities, expanded: %s",
            len(entities), expanded[:100],
        )
        return {
            "query_expanded": expanded,
            "kg_entities": entity_info,
        }

    return kg_link


def make_retrieve_node(retriever: HybridRetriever):
    """Create the hybrid retrieval node."""

    def retrieve(state: AgentState) -> dict:
        query = state.get("query_expanded") or state["query_normalized"]
        try:
            evidences = retriever.retrieve(query, top_n=15)
            logger.info("Retrieved %d evidence chunks", len(evidences))
            return {"evidences": evidences, "error": ""}
        except Exception as e:
            logger.error("Retrieval error: %s", e)
            return {"evidences": [], "error": f"Retrieval error: {e}"}

    return retrieve


def make_rerank_node(reranker: Reranker):
    """Create the reranking node."""

    def rerank(state: AgentState) -> dict:
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

            # Build concatenated evidence text with citations
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
            logger.error("Reranking error: %s", e)
            return {
                "evidences": evidences[:5],
                "evidence_texts": "\n\n".join(ev.text for ev in evidences[:5]),
                "should_refuse": False,
                "error": f"Reranking error: {e}",
            }

    return rerank


def make_grade_node(llm: Llama):
    """Create the evidence grading node."""

    def grade(state: AgentState) -> dict:
        if state.get("should_refuse"):
            return {"evidence_grade": "INSUFFICIENT"}

        query = state["query_normalized"]
        context = state.get("evidence_texts", "")

        if not context.strip():
            return {"evidence_grade": "INSUFFICIENT"}

        grade_result, confidence = grade_evidence(llm, query, context)
        logger.info("Evidence grade: %s (confidence=%.2f)", grade_result, confidence)
        return {"evidence_grade": grade_result}

    return grade


def make_rewrite_node(llm: Llama):
    """Create the query rewrite node for retry."""

    def rewrite(state: AgentState) -> dict:
        retry_count = state.get("retry_count", 0) + 1
        original_query = state["query_normalized"]
        failed_context = state.get("evidence_texts", "")

        rewritten = rewrite_query(llm, original_query, failed_context)
        logger.info("Rewritten query (retry %d): %s", retry_count, rewritten[:100])

        return {
            "query_expanded": rewritten,
            "retry_count": retry_count,
        }

    return rewrite


def make_generate_node(llm: Llama, max_tokens: int = 512):
    """Create the answer generation node."""

    def gen(state: AgentState) -> dict:
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

    return gen


def make_verify_node(llm: Llama):
    """Create the answer verification node."""

    def verify(state: AgentState) -> dict:
        answer = state.get("answer", "")
        context = state.get("evidence_texts", "")

        # Skip verification for refusals
        if state.get("should_refuse") or not answer.strip():
            return {
                "is_verified": True,
                "verification_reason": "Skipped (refusal)",
            }

        try:
            is_verified, reason = verify_answer(llm, answer, context)
            logger.info("Verification: %s — %s", is_verified, reason)

            if not is_verified:
                # Add safety disclaimer
                disclaimer = (
                    "\n\n⚠️ Note: Some claims in this answer could not be fully "
                    "verified against the source documents. Please cross-check "
                    "with your local agricultural extension officer."
                )
                answer = state["answer"] + disclaimer

            return {
                "is_verified": is_verified,
                "verification_reason": reason,
                "answer": answer,
            }
        except Exception as e:
            logger.error("Verification error: %s", e)
            return {
                "is_verified": False,
                "verification_reason": f"Verification error: {e}",
            }

    return verify


def make_translate_node(translator: BanglaTranslator):
    """Create the BanglaT5 translation node."""

    def translate(state: AgentState) -> dict:
        answer = state.get("answer", "")

        # Skip translation for refusals (already has hardcoded BN)
        if state.get("should_refuse") or state.get("answer_bn"):
            return {}

        if not answer.strip():
            return {"answer_bn": ""}

        try:
            answer_bn = translator.translate_en_to_bn(answer)
            logger.info("Translated to Bengali (%d chars)", len(answer_bn))
            return {"answer_bn": answer_bn}
        except Exception as e:
            logger.error("Translation error: %s", e)
            return {"answer_bn": "(অনুবাদ ত্রুটি / Translation error)"}

    return translate

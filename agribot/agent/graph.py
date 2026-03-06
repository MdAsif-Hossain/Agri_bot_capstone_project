"""
LangGraph workflow definition.

Wires all agent nodes into a bounded self-correction loop:
normalize → kg_link → retrieve → rerank → grade →
  ├─ INSUFFICIENT (retry < max) → rewrite → retrieve → rerank → grade → ...
  └─ SUFFICIENT (or max retries) → generate → translate (BanglaT5) → verify → END
"""

import logging

from llama_cpp import Llama
from langgraph.graph import StateGraph, END

from agribot.agent.state import AgentState
from agribot.agent.nodes import (
    make_normalize_node,
    make_kg_link_node,
    make_retrieve_node,
    make_rerank_node,
    make_grade_node,
    make_rewrite_node,
    make_generate_node,
    make_translate_node,
    make_verify_node,
)
from agribot.retrieval.hybrid import HybridRetriever
from agribot.retrieval.reranker import Reranker
from agribot.knowledge_graph.entity_linker import EntityLinker
from agribot.translation.bangla_t5 import BanglaTranslator

logger = logging.getLogger(__name__)


def _grade_router(state: AgentState) -> str:
    """Route based on evidence grade and retry count."""
    grade = state.get("evidence_grade", "INSUFFICIENT")
    retry = state.get("retry_count", 0)
    max_retries = 2

    if grade == "SUFFICIENT":
        return "generate"
    elif retry >= max_retries:
        logger.info("Max retries reached (%d), generating with available evidence", retry)
        return "generate"
    else:
        logger.info("Evidence insufficient, rewriting (retry %d)", retry + 1)
        return "rewrite"


def build_agent_graph(
    llm: Llama,
    retriever: HybridRetriever,
    reranker: Reranker,
    entity_linker: EntityLinker,
    translator: BanglaTranslator,
    max_tokens: int = 512,
) -> StateGraph:
    """
    Build and compile the LangGraph agent workflow.

    Args:
        llm: Loaded Llama model
        retriever: Hybrid retriever
        reranker: Cross-encoder reranker
        entity_linker: KG entity linker
        translator: BanglaT5 EN→BN translator
        max_tokens: Max tokens for answer generation

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create nodes bound to their services
    normalize = make_normalize_node(translator=translator)
    kg_link = make_kg_link_node(entity_linker)
    retrieve = make_retrieve_node(retriever)
    rerank = make_rerank_node(reranker)
    grade = make_grade_node(llm)
    rewrite = make_rewrite_node(llm)
    generate = make_generate_node(llm, max_tokens=max_tokens)
    translate = make_translate_node(translator)
    verify = make_verify_node(llm)

    # Build graph
    workflow = StateGraph(AgentState)

    workflow.add_node("normalize", normalize)
    workflow.add_node("kg_link", kg_link)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("grade", grade)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("translate", translate)
    workflow.add_node("verify", verify)

    # Linear pipeline until grade
    workflow.set_entry_point("normalize")
    workflow.add_edge("normalize", "kg_link")
    workflow.add_edge("kg_link", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade")

    # Conditional: grade → generate or rewrite
    workflow.add_conditional_edges(
        "grade",
        _grade_router,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    # Rewrite loops back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Generation → Translation → Verification → END
    workflow.add_edge("generate", "translate")
    workflow.add_edge("translate", "verify")
    workflow.add_edge("verify", END)

    compiled = workflow.compile()
    logger.info("Agent graph compiled successfully (with BanglaT5 translation)")
    return compiled

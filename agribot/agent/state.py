"""
Agent state definition for the LangGraph workflow.
"""

from typing import TypedDict


class AgentState(TypedDict):
    """Full state for the agentic RAG pipeline."""

    # --- Input ---
    query_original: str  # Raw user query
    query_language: str  # Detected language (bn/en)

    # --- Processing ---
    query_normalized: str  # Cleaned query
    query_expanded: str  # After KG entity linking + expansion
    kg_entities: list[dict]  # Linked KG entity info [{bn, en, type}]

    # --- Retrieval ---
    evidences: list  # Retrieved EvidenceChunk objects
    evidence_texts: str  # Concatenated evidence text for LLM
    evidence_grade: str  # SUFFICIENT / INSUFFICIENT

    # --- Generation ---
    answer: str  # Generated answer (English)
    answer_bn: str  # Bengali translation (BanglaT5)
    citations: list[str]  # List of citation strings

    # --- Verification ---
    is_verified: bool  # Whether answer passed verification
    verification_reason: str  # Reason if not verified

    # --- Grounding Policy ---
    grounding_action: str  # "pass" | "disclaimer" | "cited_facts_only" | "refuse"
    follow_up_suggestions: list[str]  # Suggested follow-up queries

    # --- Observability ---
    trace_id: str  # UUID per request for log correlation
    timings_ms: dict[str, float]  # Per-node timing {node_name: duration_ms}

    # --- Multimodal Input ---
    input_mode: str  # "text", "voice", or "image"
    input_audio_path: str  # Path to recorded audio (if voice input)

    # --- Vision context (set by API layer when input_mode == "image") ---
    image_top_label: str  # Top condition label from classifier (empty if none)
    image_top_confidence: float  # Top condition confidence in [0, 1]; 0.0 if none
    image_symptom_hints: list[str]  # Heuristic symptom hints from PIL/NumPy analyzer

    # --- Control ---
    retry_count: int  # Current retry iteration
    should_refuse: bool  # Whether to refuse answering
    error: str  # Error message if any

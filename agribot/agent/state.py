"""
Agent state definition for the LangGraph workflow.
"""

from typing import TypedDict


class AgentState(TypedDict):
    """Full state for the agentic RAG pipeline."""

    # --- Input ---
    query_original: str          # Raw user query
    query_language: str          # Detected language (bn/en)

    # --- Processing ---
    query_normalized: str        # Cleaned query
    query_expanded: str          # After KG entity linking + expansion
    kg_entities: list            # Linked KG entity info

    # --- Retrieval ---
    evidences: list              # Retrieved EvidenceChunk objects
    evidence_texts: str          # Concatenated evidence text for LLM
    evidence_grade: str          # SUFFICIENT / INSUFFICIENT

    # --- Generation ---
    answer: str                  # Generated answer (English)
    answer_bn: str               # Bengali translation (BanglaT5)
    citations: list              # List of citation strings

    # --- Verification ---
    is_verified: bool            # Whether answer passed verification
    verification_reason: str     # Reason if not verified

    # --- Voice ---
    input_mode: str              # "text", "voice", or "image"
    input_audio_path: str        # Path to recorded audio (if voice input)

    # --- Control ---
    retry_count: int             # Current retry iteration
    should_refuse: bool          # Whether to refuse answering
    error: str                   # Error message if any

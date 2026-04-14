"""
Tests for the FastAPI backend endpoints.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestAPIModels:
    """Tests for API request/response models (no server needed)."""

    def test_chat_request_validation(self):
        from api import ChatRequest

        req = ChatRequest(query="What is rice blast?")
        assert req.query == "What is rice blast?"

    def test_chat_request_rejects_empty(self):
        from api import ChatRequest

        with pytest.raises(Exception):
            ChatRequest(query="")

    def test_chat_response_defaults(self):
        from api import ChatResponse

        resp = ChatResponse(answer="Rice blast is a fungal disease.")
        assert resp.answer == "Rice blast is a fungal disease."
        assert resp.answer_bn == ""
        assert resp.citations == []
        assert resp.is_verified is False

    def test_health_response(self):
        from api import HealthResponse

        resp = HealthResponse(
            status="ok",
            chunk_count=100,
            kg_entities=25,
            kg_aliases=80,
            kg_relations=30,
        )
        assert resp.status == "ok"
        assert resp.chunk_count == 100

    def test_initial_state_builder(self):
        from api import _build_initial_state

        state = _build_initial_state("test query", "voice")
        assert state["query_original"] == "test query"
        assert state["input_mode"] == "voice"
        assert state["retry_count"] == 0
        assert state["should_refuse"] is False

"""
Tests for the agent state and graph structure.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agribot.agent.state import AgentState


class TestAgentState:
    """Tests for agent state structure."""

    def test_state_has_required_fields(self):
        """AgentState should have all required fields."""
        required_fields = [
            "query_original",
            "query_language",
            "query_normalized",
            "query_expanded",
            "kg_entities",
            "evidences",
            "evidence_texts",
            "evidence_grade",
            "answer",
            "answer_bn",
            "citations",
            "is_verified",
            "verification_reason",
            "retry_count",
            "should_refuse",
            "error",
            "input_mode",
            "input_audio_path",
        ]
        annotations = AgentState.__annotations__
        for field in required_fields:
            assert field in annotations, f"Missing field: {field}"

    def test_initial_state_template(self):
        """A valid initial state should be constructible."""
        state: AgentState = {
            "query_original": "What is rice blast?",
            "query_language": "",
            "query_normalized": "",
            "query_expanded": "",
            "kg_entities": [],
            "evidences": [],
            "evidence_texts": "",
            "evidence_grade": "",
            "answer": "",
            "answer_bn": "",
            "citations": [],
            "is_verified": False,
            "verification_reason": "",
            "retry_count": 0,
            "should_refuse": False,
            "input_mode": "text",
            "input_audio_path": "",
            "error": "",
        }
        assert state["query_original"] == "What is rice blast?"
        assert state["retry_count"] == 0


class TestGradeRouter:
    """Tests for the grade routing logic."""

    def test_sufficient_routes_to_generate(self):
        from agribot.agent.graph import _make_grade_router

        router = _make_grade_router()
        state = {"evidence_grade": "SUFFICIENT", "retry_count": 0}
        assert router(state) == "generate"

    def test_insufficient_routes_to_rewrite(self):
        from agribot.agent.graph import _make_grade_router

        router = _make_grade_router()
        state = {"evidence_grade": "INSUFFICIENT", "retry_count": 0}
        assert router(state) == "rewrite"

    def test_max_retries_routes_to_generate(self):
        from agribot.agent.graph import _make_grade_router

        router = _make_grade_router(max_retries=2)
        state = {"evidence_grade": "INSUFFICIENT", "retry_count": 2}
        assert router(state) == "generate"

    def test_max_retries_boundary(self):
        from agribot.agent.graph import _make_grade_router

        router = _make_grade_router(max_retries=2)
        state = {"evidence_grade": "INSUFFICIENT", "retry_count": 1}
        assert router(state) == "rewrite"  # Still below max

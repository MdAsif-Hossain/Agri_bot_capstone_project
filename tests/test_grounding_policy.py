"""
Tests for the grounding policy module.

Covers: strict refuse, cited_facts_only, disclaimer, risk detection, lenient mode.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agribot.agent.grounding_policy import (
    is_risky_query,
    _extract_cited_facts,
    make_enforce_policy_node,
    REFUSE_EN,
    REFUSE_BN,
    DISCLAIMER_EN,
)


# =============================================================================
# Risk Detection
# =============================================================================


class TestRiskDetection:
    """Tests for the risk pattern detector."""

    def test_detects_dosage_units(self):
        assert is_risky_query("Apply 50 ml per liter of water") is True
        assert is_risky_query("Use 2 gm per kg soil") is True
        assert is_risky_query("Mix 100 ppm solution") is True

    def test_detects_dosage_keywords(self):
        assert is_risky_query("What is the dosage for neem oil?") is True
        assert is_risky_query("Spray rate for carbendazim") is True

    def test_detects_application_quantities(self):
        assert is_risky_query("Apply 500 grams per acre") is True
        assert is_risky_query("Dilute 20 ml in water") is True

    def test_safe_queries_pass(self):
        assert is_risky_query("What causes rice blast disease?") is False
        assert is_risky_query("How to identify leaf blight?") is False
        assert is_risky_query("Best practices for rice farming") is False
        assert is_risky_query("What is compost?") is False

    def test_pesticide_rate_queries(self):
        assert is_risky_query("pesticide amount for tomato") is True
        assert is_risky_query("fungicide rate for wheat") is True


# =============================================================================
# Cited Facts Extraction
# =============================================================================


class TestCitedFacts:
    """Tests for the cited facts extraction."""

    def test_extracts_supported_sentences(self):
        evidence = "Rice blast disease is caused by Magnaporthe oryzae fungus"
        answer = (
            "Rice blast disease is caused by Magnaporthe oryzae. "
            "Aliens from Mars sometimes visit rice paddies."
        )
        result = _extract_cited_facts(answer, evidence)
        assert "Magnaporthe" in result
        assert "Aliens" not in result

    def test_empty_evidence_returns_fallback(self):
        result = _extract_cited_facts("Some answer", "")
        assert result == ""

    def test_empty_answer_returns_empty(self):
        result = _extract_cited_facts("", "some evidence")
        assert result == ""


# =============================================================================
# Policy Enforcement
# =============================================================================


class TestEnforcePolicyStrict:
    """Tests for strict grounding mode."""

    def _make_state(self, **overrides):
        base = {
            "query_original": "test query",
            "query_normalized": "test query",
            "answer": "Test answer about crops.",
            "answer_bn": "ফসল সম্পর্কে পরীক্ষা উত্তর।",
            "is_verified": True,
            "should_refuse": False,
            "evidence_texts": "Test evidence about crops and farming.",
            "trace_id": "test-trace",
            "grounding_action": "pass",
            "follow_up_suggestions": [],
        }
        base.update(overrides)
        return base

    def test_verified_answer_passes(self):
        """Verified answers pass through unchanged."""
        node = make_enforce_policy_node("strict", "refuse")
        state = self._make_state(is_verified=True)
        result = node(state)
        assert result["grounding_action"] == "pass"

    def test_unverified_strict_refuse(self):
        """Strict + refuse: replaces answer with refusal."""
        node = make_enforce_policy_node("strict", "refuse")
        state = self._make_state(is_verified=False)
        result = node(state)
        assert result["grounding_action"] == "refuse"
        assert result["answer"] == REFUSE_EN
        assert result["answer_bn"] == REFUSE_BN

    def test_unverified_strict_disclaimer(self):
        """Strict + disclaimer: appends disclaimer to answer."""
        node = make_enforce_policy_node("strict", "disclaimer")
        state = self._make_state(is_verified=False)
        result = node(state)
        assert result["grounding_action"] == "disclaimer"
        assert "answer" in result
        assert DISCLAIMER_EN in result["answer"]

    def test_unverified_strict_cited_facts_only(self):
        """Strict + cited_facts_only: strips unsupported claims."""
        node = make_enforce_policy_node("strict", "cited_facts_only")
        state = self._make_state(
            is_verified=False,
            answer="Crops need water. Aliens farm on Mars.",
            evidence_texts="Crops need water for growth and photosynthesis.",
        )
        result = node(state)
        assert result["grounding_action"] == "cited_facts_only"
        assert "answer" in result
        assert "verified sources" in result["answer"].lower()

    def test_risky_query_unverified_always_refuses(self):
        """Risky dosage queries without verification always refuse."""
        node = make_enforce_policy_node("lenient", "disclaimer")  # Even lenient!
        state = self._make_state(
            query_original="Apply 50 ml of pesticide per liter",
            is_verified=False,
        )
        result = node(state)
        assert result["grounding_action"] == "refuse"
        assert result["answer"] == REFUSE_EN

    def test_should_refuse_returns_refuse(self):
        """Pre-existing refusal flag is honored."""
        node = make_enforce_policy_node("strict", "refuse")
        state = self._make_state(should_refuse=True)
        result = node(state)
        assert result["grounding_action"] == "refuse"

    def test_follow_up_suggestions_on_refuse(self):
        """Refusals include follow-up suggestions."""
        node = make_enforce_policy_node("strict", "refuse")
        state = self._make_state(is_verified=False)
        result = node(state)
        assert len(result["follow_up_suggestions"]) > 0


class TestEnforcePolicyLenient:
    """Tests for lenient grounding mode."""

    def _make_state(self, **overrides):
        base = {
            "query_original": "test query",
            "query_normalized": "test query",
            "answer": "Test answer.",
            "answer_bn": "পরীক্ষা উত্তর।",
            "is_verified": False,
            "should_refuse": False,
            "evidence_texts": "Some evidence.",
            "trace_id": "test-trace",
            "grounding_action": "pass",
            "follow_up_suggestions": [],
        }
        base.update(overrides)
        return base

    def test_lenient_adds_disclaimer(self):
        """Lenient mode always adds disclaimer, never refuses."""
        node = make_enforce_policy_node("lenient", "refuse")  # refuse ignored in lenient
        state = self._make_state()
        result = node(state)
        assert result["grounding_action"] == "disclaimer"
        assert "answer" in result
        assert DISCLAIMER_EN in result["answer"]

"""
Grounding policy enforcement for AgriBot.

Controls how the system handles unverified or risky outputs:
- STRICT mode: never outputs "full advice" when verification fails.
- LENIENT mode: adds disclaimers but still returns the answer.

Risk detector catches high-risk dosage/chemical queries and refuses
to guess answers without explicit evidence.
"""

import re
import logging
from typing import Literal

from agribot.agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
GroundingMode = Literal["strict", "lenient"]
OnVerifyFailAction = Literal["disclaimer", "cited_facts_only", "refuse"]

# ---------------------------------------------------------------------------
# Risk patterns — queries involving dosage, chemical application, etc.
# These require explicit evidence; never guess.
# ---------------------------------------------------------------------------
RISK_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # English patterns
        r"\b\d+\s*(?:ml|mg|gm?|kg|l|cc|ppm)\b",  # numeric dose units
        r"\bdos(?:age|e)\b",
        r"\bspray\s+(?:rate|schedule|amount)\b",
        r"\bapply\s+\d+",
        r"\b(?:mix|dilute)\s+\d+",
        r"\bconcentration\b",
        r"\bpesticide\s+(?:amount|quantity|rate)\b",
        r"\bfungicide\s+(?:amount|quantity|rate)\b",
        r"\bherbicide\s+(?:amount|quantity|rate)\b",
        r"\binsecticide\s+(?:amount|quantity|rate)\b",
        r"\bper\s+(?:acre|hectare|bigha|liter|litre)\b",
        # Bengali patterns (মাত্রা = dosage, কীটনাশক = pesticide, etc.)
        r"মাত্রা",  # dosage
        r"কীটনাশক",  # pesticide
        r"ছত্রাকনাশক",  # fungicide
        r"আগাছানাশক",  # herbicide
        r"কতটুকু.*দিতে হবে",  # "how much to apply"
        r"\d+\s*(?:মিলি|গ্রাম|কেজি|লিটার)",  # Bengali numeric units
        r"প্রতি\s*(?:একর|হেক্টর|বিঘা|লিটার)",  # per acre/hectare/bigha
        r"স্প্রে\s*(?:হার|পরিমাণ)",  # spray rate/amount
    ]
]

DISCLAIMER_EN = (
    "\n\n⚠️ Note: Some claims in this answer could not be fully "
    "verified against the source documents. Please cross-check "
    "with your local agricultural extension officer."
)

DISCLAIMER_BN = (
    "\n\n⚠️ দ্রষ্টব্য: এই উত্তরের কিছু তথ্য উৎস নথি থেকে সম্পূর্ণভাবে "
    "যাচাই করা সম্ভব হয়নি। অনুগ্রহ করে আপনার স্থানীয় কৃষি সম্প্রসারণ "
    "কর্মকর্তার সাথে যাচাই করুন।"
)

REFUSE_EN = (
    "I cannot provide a confident answer to this question based on "
    "the available documents. The information could not be verified, "
    "and providing unverified agricultural advice may be harmful. "
    "Please consult your local agricultural extension officer."
)

REFUSE_BN = (
    "উপলব্ধ নথির ভিত্তিতে এই প্রশ্নের একটি আত্মবিশ্বাসী উত্তর দেওয়া "
    "সম্ভব হচ্ছে না। তথ্য যাচাই করা যায়নি এবং অযাচাইকৃত কৃষি পরামর্শ "
    "ক্ষতিকর হতে পারে। অনুগ্রহ করে আপনার স্থানীয় কৃষি সম্প্রসারণ "
    "কর্মকর্তার সাথে পরামর্শ করুন।"
)

ESCALATION_EN = (
    "⚠️ This question involves chemical dosage or application rates. "
    "For safety, please consult:\n"
    "• Your local agricultural extension officer\n"
    "• The product label instructions\n"
    "• Bangladesh Agricultural Research Institute (BARI): +880-2-9270534"
)

ESCALATION_BN = (
    "⚠️ এই প্রশ্নটি রাসায়নিক মাত্রা বা প্রয়োগের হার সংক্রান্ত। "
    "নিরাপত্তার জন্য অনুগ্রহ করে পরামর্শ করুন:\n"
    "• আপনার স্থানীয় কৃষি সম্প্রসারণ কর্মকর্তা\n"
    "• পণ্যের লেবেল নির্দেশাবলী\n"
    "• বাংলাদেশ কৃষি গবেষণা ইনস্টিটিউট (BARI): +৮৮০-২-৯২৭০৫৩৪"
)


def is_risky_query(query: str) -> bool:
    """Check if a query involves high-risk dosage/chemical advice."""
    return any(pat.search(query) for pat in RISK_PATTERNS)


def _extract_cited_facts(answer: str, evidence_texts: str) -> str:
    """
    Extract only sentences from the answer that have direct support
    in the evidence texts (simple overlap heuristic).
    """
    if not answer or not evidence_texts:
        return ""

    evidence_lower = evidence_texts.lower()
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    supported = []

    for sentence in sentences:
        # A sentence is "supported" if it shares significant keyword overlap
        words = set(re.findall(r"\b\w{4,}\b", sentence.lower()))
        if not words:
            continue
        overlap = sum(1 for w in words if w in evidence_lower)
        if overlap >= max(2, len(words) * 0.3):
            supported.append(sentence)

    if supported:
        return " ".join(supported)
    return "No claims could be directly supported by the source documents."


def make_enforce_policy_node(
    grounding_mode: GroundingMode = "strict",
    on_verify_fail: OnVerifyFailAction = "disclaimer",
):
    """
    Create the grounding policy enforcement node.

    This runs after verify and before END, applying safety controls.
    """

    def enforce_policy(state: AgentState) -> dict:
        answer = state.get("answer", "")
        answer_bn = state.get("answer_bn", "")
        is_verified = state.get("is_verified", False)
        should_refuse = state.get("should_refuse", False)
        evidence_texts = state.get("evidence_texts", "")
        query = state.get("query_original", "") or state.get("query_normalized", "")

        # Default: pass through
        action = "pass"
        follow_ups: list[str] = []

        # Already a refusal — no further processing needed
        if should_refuse:
            return {
                "grounding_action": "refuse",
                "follow_up_suggestions": [
                    "Can you rephrase your question?",
                    "What specific crop are you asking about?",
                ],
            }

        # Check risk for dosage/chemical queries
        risky = is_risky_query(query)

        if risky and not is_verified:
            # High-risk query without verification → always refuse
            logger.warning(
                "Risky query detected without verification — refusing",
                extra={"trace_id": state.get("trace_id", "")},
            )
            return {
                "answer": REFUSE_EN,
                "answer_bn": REFUSE_BN,
                "grounding_action": "refuse",
                "follow_up_suggestions": [
                    "What is the recommended dosage for [specific product]?",
                    "What pesticides are approved for [specific crop]?",
                ],
            }

        # Handle verification failure
        if not is_verified:
            if grounding_mode == "strict":
                if on_verify_fail == "refuse":
                    action = "refuse"
                    answer = REFUSE_EN
                    answer_bn = REFUSE_BN
                    follow_ups = [
                        "Can you ask about a specific crop or disease?",
                        "What symptoms are you observing?",
                    ]

                elif on_verify_fail == "cited_facts_only":
                    action = "cited_facts_only"
                    cited = _extract_cited_facts(answer, evidence_texts)
                    answer = (
                        f"Based on verified sources only:\n\n{cited}{DISCLAIMER_EN}"
                    )
                    follow_ups = [
                        "Would you like more details on a specific point?",
                    ]

                else:  # disclaimer
                    action = "disclaimer"
                    if DISCLAIMER_EN not in answer:
                        answer = answer + DISCLAIMER_EN
                    if answer_bn and DISCLAIMER_BN not in answer_bn:
                        answer_bn = answer_bn + DISCLAIMER_BN
                    follow_ups = []

            else:  # lenient mode
                action = "disclaimer"
                if DISCLAIMER_EN not in answer:
                    answer = answer + DISCLAIMER_EN
                if answer_bn and DISCLAIMER_BN not in answer_bn:
                    answer_bn = answer_bn + DISCLAIMER_BN

        result: dict = {
            "grounding_action": action,
            "follow_up_suggestions": follow_ups,
        }

        if answer != state.get("answer", ""):
            result["answer"] = answer
        if answer_bn != state.get("answer_bn", ""):
            result["answer_bn"] = answer_bn

        logger.info(
            "Grounding policy applied: action=%s, risky=%s",
            action,
            risky,
            extra={"trace_id": state.get("trace_id", "")},
        )
        return result

    return enforce_policy

"""
Structured output schema for AgriBot image analysis pipeline.

Defines the canonical result types used by both baseline (OCR + heuristics)
and optional classifier-assisted analysis paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PossibleCondition:
    """A predicted disease/condition with confidence score."""

    label: str
    confidence: float  # 0.0 - 1.0

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3)}


@dataclass
class ImageAnalysisResult:
    """
    Structured result from image analysis pipeline.

    Attributes:
        pipeline_used: Which analysis path produced this result.
        ocr_text: Extracted text from the image (Tesseract OCR).
        symptom_hints: List of detected visual symptom descriptions.
        quality_flags: Image quality indicators (e.g., 'blurry', 'dark').
        limitations: List of analysis limitations to disclose.
        keywords: Extracted keywords useful for RAG query expansion.
        possible_conditions: Predicted conditions (from classifier, if enabled).
    """

    pipeline_used: Literal["ocr_baseline", "classifier_assisted", "ocr_fallback"] = (
        "ocr_baseline"
    )
    ocr_text: str = ""
    symptom_hints: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    limitations: list[str] = field(
        default_factory=lambda: [
            "Heuristic color analysis only; not a clinical diagnosis",
            "Accuracy depends on image quality and lighting",
        ]
    )
    keywords: list[str] = field(default_factory=list)
    possible_conditions: list[PossibleCondition] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pipeline_used": self.pipeline_used,
            "ocr_text": self.ocr_text,
            "symptom_hints": self.symptom_hints,
            "quality_flags": self.quality_flags,
            "limitations": self.limitations,
            "keywords": self.keywords,
            "possible_conditions": [c.to_dict() for c in self.possible_conditions],
        }

    def build_query_text(self, user_query: str = "") -> str:
        """
        Build a combined query string for RAG retrieval.

        Retrieval is embedding-based, so the query must be DOMINATED by the
        strongest signal — the classifier's predicted disease — phrased as a
        natural disease query. Generic color-heuristic keywords (e.g.
        "dark/necrotic, areas, detected") are deliberately excluded from the
        retrieval text because they dilute the embedding away from
        disease-specific documents. Those heuristic hints are still surfaced
        to the user via the image-analysis block; they just don't drive search.
        """
        parts: list[str] = []

        # 1. Lead with the predicted disease as a rich, natural phrase. The
        #    top label carries the most weight; secondary labels follow.
        if self.possible_conditions:
            top = self.possible_conditions[0].label
            parts.append(
                f"{top} disease in rice: symptoms, cause, and management of {top}."
            )
            others = [c.label for c in self.possible_conditions[1:3]]
            if others:
                parts.append(f"It may also be {', '.join(others)}.")

        # 2. The user's actual question (if any) — kept, but after the disease
        #    anchor so it refines rather than dominates.
        if user_query.strip():
            parts.append(user_query.strip())

        # 3. OCR text can carry real disease/product names from labels — keep a
        #    short slice. (Generic symptom_hints/keywords are intentionally
        #    omitted from retrieval; see docstring.)
        if self.ocr_text.strip():
            parts.append(f"Text from image: {self.ocr_text[:200]}")

        if not parts:
            return "Crop disease identification from an uploaded leaf image."

        return " ".join(parts)

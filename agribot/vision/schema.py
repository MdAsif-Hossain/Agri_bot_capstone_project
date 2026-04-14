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

        Merges user text + OCR keywords + symptom hints + classifier labels.
        """
        parts = []

        if user_query.strip():
            parts.append(user_query.strip())

        if self.symptom_hints:
            parts.append(f"Observed symptoms: {'; '.join(self.symptom_hints)}")

        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")

        if self.possible_conditions:
            top_labels = [c.label for c in self.possible_conditions[:3]]
            parts.append(f"Possible conditions: {', '.join(top_labels)}")

        if self.ocr_text:
            parts.append(f"Text from image: {self.ocr_text[:200]}")

        if not parts:
            return "Crop image uploaded; unable to extract specific features."

        return " | ".join(parts)

"""
PDF Loader with noise filtering.

Extracts text page-by-page, detects/removes repeated headers/footers,
and classifies page types (content, TOC, reference, etc.).
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """A single extracted PDF page with metadata."""
    source_file: str
    page_num: int
    text: str
    page_type: str = "content"  # content | toc | reference | low_signal
    keep_weight: float = 1.0


def _extract_pages(pdf_path: Path) -> list[PageData]:
    """Extract raw text from each page of a PDF."""
    pages: list[PageData] = []
    source_name = pdf_path.name

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", pdf_path, e)
        return pages

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Failed to extract page %d from %s: %s", i + 1, source_name, e)
            text = ""

        pages.append(PageData(
            source_file=source_name,
            page_num=i + 1,
            text=text,
        ))

    logger.info("Extracted %d pages from %s", len(pages), source_name)
    return pages


def _detect_repeated_lines(
    pages: list[PageData],
    freq_threshold: float = 0.5,
) -> set[str]:
    """
    Find lines that appear in more than `freq_threshold` fraction of pages.
    These are likely headers/footers.
    """
    if len(pages) < 3:
        return set()

    line_page_count: Counter = Counter()
    total_pages = len(pages)

    for page in pages:
        # Deduplicate within a single page
        unique_lines = set()
        for line in page.text.splitlines():
            stripped = line.strip()
            if stripped:
                # Normalize: remove page numbers (standalone digits)
                normalized = re.sub(r"^\d+$", "", stripped).strip()
                if normalized and len(normalized) > 3:
                    unique_lines.add(normalized)
        for line in unique_lines:
            line_page_count[line] += 1

    repeated = {
        line for line, count in line_page_count.items()
        if count / total_pages >= freq_threshold
    }

    if repeated:
        logger.info("Detected %d repeated header/footer lines", len(repeated))

    return repeated


def _remove_repeated_lines(text: str, repeated_lines: set[str]) -> str:
    """Remove lines identified as headers/footers."""
    if not repeated_lines:
        return text

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        normalized = re.sub(r"^\d+$", "", stripped).strip()
        if normalized not in repeated_lines:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _classify_page_type(text: str, toc_keywords: list[str]) -> tuple[str, float]:
    """
    Classify a page as content, toc, reference, or low_signal.
    Returns (page_type, keep_weight).
    """
    text_lower = text.lower().strip()
    word_count = len(text_lower.split())

    # Almost empty page
    if word_count < 15:
        return "low_signal", 0.1

    # Check for TOC / Index / References patterns
    for keyword in toc_keywords:
        if keyword in text_lower[:200]:  # Check first 200 chars
            # Count how many numbered entries or dot leaders there are
            dot_leader_count = len(re.findall(r"\.{3,}", text))
            numbered_entries = len(re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE))
            if dot_leader_count > 3 or numbered_entries > 5:
                return "toc", 0.2

    # Check for references/bibliography pages
    ref_patterns = [
        r"^\s*\[\d+\]",  # [1] style references
        r"^\s*\d+\.\s+[A-Z]",  # 1. Author style
    ]
    ref_matches = sum(
        len(re.findall(p, text, re.MULTILINE)) for p in ref_patterns
    )
    if ref_matches > 5:
        return "reference", 0.3

    return "content", 1.0


def load_pdfs(
    pdf_dir: Path,
    freq_threshold: float = 0.5,
    toc_keywords: list[str] | None = None,
) -> list[PageData]:
    """
    Load all PDFs from a directory with noise filtering.

    Steps:
    1. Extract raw text from all pages
    2. Detect repeated headers/footers across pages (per PDF)
    3. Remove repeated lines
    4. Classify page types and assign keep_weight

    Args:
        pdf_dir: Directory containing PDF files
        freq_threshold: Fraction of pages a line must appear in to be
                        considered a header/footer
        toc_keywords: Keywords indicating TOC/index pages

    Returns:
        List of PageData objects, cleaned and classified
    """
    if toc_keywords is None:
        toc_keywords = [
            "table of contents", "contents", "index",
            "bibliography", "references", "appendix",
        ]

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return []

    logger.info("Found %d PDF files in %s", len(pdf_files), pdf_dir)

    all_pages: list[PageData] = []

    for pdf_path in pdf_files:
        # 1. Extract raw pages
        pages = _extract_pages(pdf_path)
        if not pages:
            continue

        # 2. Detect repeated headers/footers for this PDF
        repeated_lines = _detect_repeated_lines(pages, freq_threshold)

        # 3. Clean and classify each page
        for page in pages:
            page.text = _remove_repeated_lines(page.text, repeated_lines)
            page.page_type, page.keep_weight = _classify_page_type(
                page.text, toc_keywords
            )

        all_pages.extend(pages)

    content_count = sum(1 for p in all_pages if p.page_type == "content")
    logger.info(
        "Loaded %d total pages (%d content, %d filtered/downweighted)",
        len(all_pages),
        content_count,
        len(all_pages) - content_count,
    )

    return all_pages

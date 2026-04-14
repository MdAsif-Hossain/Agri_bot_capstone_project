"""
PDF Loader with layout-aware extraction (Marker/Surya) and OCR fallback.

Uses Marker for structured content recovery (reading order, headings, tables,
figures) and Tesseract OCR for scanned pages. Includes noise filtering to
detect/remove repeated headers/footers and classify page types.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """A single extracted PDF page with metadata."""

    source_file: str
    page_num: int
    text: str
    page_type: str = "content"  # content | toc | reference | low_signal
    keep_weight: float = 1.0
    extraction_method: str = "marker"  # marker | ocr | fallback


def _extract_with_marker(pdf_path: Path) -> list[PageData]:
    """
    Extract pages using Marker for layout-aware PDF understanding.

    Marker recovers reading order, headings, tables (as Markdown),
    and handles complex multi-column layouts.
    """
    source_name = pdf_path.name
    pages: list[PageData] = []

    try:
        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser

        config_parser = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(config=config_parser.generate_config_dict())
        rendered = converter(str(pdf_path))

        # Marker returns a single document — split by page markers if available
        full_text = rendered.markdown

        # Marker includes page breaks as horizontal rules or headers
        # Split into rough page segments using double-newline blocks
        # For per-page provenance, we estimate page boundaries
        raw_pages = _split_marker_output_by_pages(full_text, pdf_path)

        for i, page_text in enumerate(raw_pages):
            pages.append(
                PageData(
                    source_file=source_name,
                    page_num=i + 1,
                    text=page_text,
                    extraction_method="marker",
                )
            )

        logger.info(
            "Marker extracted %d pages from %s (%d chars total)",
            len(pages),
            source_name,
            len(full_text),
        )

    except ImportError:
        logger.warning(
            "marker-pdf not installed. Falling back to basic extraction for %s. "
            "Install with: pip install marker-pdf",
            source_name,
        )
        return _extract_with_fallback(pdf_path)
    except Exception as e:
        logger.error(
            "Marker extraction failed for %s: %s. Using fallback.", source_name, e
        )
        return _extract_with_fallback(pdf_path)

    return pages


def _split_marker_output_by_pages(full_text: str, pdf_path: Path) -> list[str]:
    """
    Split Marker's single-document output into per-page segments.

    Strategy: use the actual PDF page count and distribute content evenly,
    or split on Marker's page-break indicators (horizontal rules).
    """
    # Try to get page count from pypdf for accurate page numbering
    page_count = _get_pdf_page_count(pdf_path)

    # Marker often inserts horizontal rules (---) or page markers between pages
    # Split on those first
    segments = re.split(r"\n-{3,}\n", full_text)

    # If segments roughly match page count, use them
    if segments and abs(len(segments) - page_count) <= 2:
        return segments

    # Otherwise, if we have fewer segments than pages, return segments as-is
    # (Marker merged some pages — acceptable for chunking)
    if segments and len(segments) > 1:
        return segments

    # Fallback: return the whole text as one "page"
    if full_text.strip():
        return [full_text]
    return []


def _get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF (lightweight check)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return 1


def _extract_with_ocr(pdf_path: Path) -> list[PageData]:
    """
    Extract text from scanned PDF pages using Tesseract OCR.

    Converts each PDF page to an image, then runs OCR.
    Used as fallback for pages where Marker produces no text.
    """
    source_name = pdf_path.name
    pages: list[PageData] = []

    try:
        import pytesseract
        from PIL import Image
        import fitz  # PyMuPDF — often bundled or available

        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc):
            # Render page to image
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # OCR with Bengali + English
            text = pytesseract.image_to_string(img, lang="eng+ben")

            pages.append(
                PageData(
                    source_file=source_name,
                    page_num=i + 1,
                    text=text.strip(),
                    extraction_method="ocr",
                )
            )

        doc.close()
        logger.info("OCR extracted %d pages from %s", len(pages), source_name)

    except ImportError as e:
        logger.warning(
            "OCR dependencies not available (%s). Install pytesseract and PyMuPDF.",
            e,
        )
    except Exception as e:
        logger.error("OCR extraction failed for %s: %s", source_name, e)

    return pages


def _extract_with_fallback(pdf_path: Path) -> list[PageData]:
    """
    Basic fallback extraction using pypdf when Marker is unavailable.
    """
    source_name = pdf_path.name
    pages: list[PageData] = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(
                    "Failed to extract page %d from %s: %s", i + 1, source_name, e
                )
                text = ""

            pages.append(
                PageData(
                    source_file=source_name,
                    page_num=i + 1,
                    text=text,
                    extraction_method="fallback",
                )
            )

        logger.info("Fallback extracted %d pages from %s", len(pages), source_name)

    except Exception as e:
        logger.error("Fallback extraction failed for %s: %s", source_name, e)

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
        line
        for line, count in line_page_count.items()
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
    ref_matches = sum(len(re.findall(p, text, re.MULTILINE)) for p in ref_patterns)
    if ref_matches > 5:
        return "reference", 0.3

    return "content", 1.0


def _ocr_empty_pages(pages: list[PageData], pdf_path: Path) -> list[PageData]:
    """
    For pages that have very little text after Marker extraction,
    attempt OCR to recover scanned content.
    """
    empty_indices = [
        i
        for i, p in enumerate(pages)
        if len(p.text.strip()) < 30 and p.extraction_method == "marker"
    ]

    if not empty_indices:
        return pages

    logger.info(
        "Attempting OCR on %d empty/low-text pages from %s",
        len(empty_indices),
        pdf_path.name,
    )

    try:
        import pytesseract
        from PIL import Image
        import fitz

        doc = fitz.open(str(pdf_path))

        for idx in empty_indices:
            page_num = pages[idx].page_num - 1  # 0-indexed for fitz
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang="eng+ben")

            if len(ocr_text.strip()) > len(pages[idx].text.strip()):
                pages[idx].text = ocr_text.strip()
                pages[idx].extraction_method = "ocr"
                logger.info(
                    "OCR recovered %d chars for page %d",
                    len(ocr_text),
                    pages[idx].page_num,
                )

        doc.close()

    except ImportError:
        logger.info("OCR dependencies not available; skipping OCR for empty pages")
    except Exception as e:
        logger.warning("OCR fallback failed: %s", e)

    return pages


def load_pdfs(
    pdf_dir: Path,
    freq_threshold: float = 0.5,
    toc_keywords: list[str] | None = None,
) -> list[PageData]:
    """
    Load all PDFs from a directory with layout-aware extraction and noise filtering.

    Pipeline:
    1. Extract structured content using Marker (layout-aware, tables, figures)
    2. OCR fallback for scanned/empty pages via Tesseract
    3. Detect and remove repeated headers/footers (per PDF)
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
            "table of contents",
            "contents",
            "index",
            "bibliography",
            "references",
            "appendix",
        ]

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return []

    logger.info("Found %d PDF files in %s", len(pdf_files), pdf_dir)

    all_pages: list[PageData] = []

    for pdf_path in pdf_files:
        # 1. Extract with Marker (layout-aware)
        pages = _extract_with_marker(pdf_path)
        if not pages:
            continue

        # 2. OCR fallback for empty/scanned pages
        pages = _ocr_empty_pages(pages, pdf_path)

        # 3. Detect repeated headers/footers for this PDF
        repeated_lines = _detect_repeated_lines(pages, freq_threshold)

        # 4. Clean and classify each page
        for page in pages:
            page.text = _remove_repeated_lines(page.text, repeated_lines)
            page.page_type, page.keep_weight = _classify_page_type(
                page.text, toc_keywords
            )

        all_pages.extend(pages)

    content_count = sum(1 for p in all_pages if p.page_type == "content")
    marker_count = sum(1 for p in all_pages if p.extraction_method == "marker")
    ocr_count = sum(1 for p in all_pages if p.extraction_method == "ocr")
    logger.info(
        "Loaded %d total pages (%d content, %d marker, %d ocr, %d filtered/downweighted)",
        len(all_pages),
        content_count,
        marker_count,
        ocr_count,
        len(all_pages) - content_count,
    )

    return all_pages

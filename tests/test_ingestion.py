"""
Tests for the ingestion pipeline: PDF loading, chunking, and index building.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agribot.ingestion.pdf_loader import (
    PageData,
    _detect_repeated_lines,
    _remove_repeated_lines,
    _classify_page_type,
)
from agribot.ingestion.chunker import Chunk, chunk_pages


class TestPageClassification:
    """Tests for page type classification."""

    def test_content_page(self):
        text = (
            "Rice blast is a serious disease caused by the fungus Magnaporthe oryzae. "
            * 5
        )
        page_type, weight = _classify_page_type(
            text, ["table of contents", "references"]
        )
        assert page_type == "content"
        assert weight == 1.0

    def test_low_signal_page(self):
        text = "Page 42"
        page_type, weight = _classify_page_type(text, ["table of contents"])
        assert page_type == "low_signal"
        assert weight < 0.5

    def test_toc_page(self):
        lines = ["table of contents\n"]
        lines += [f"Chapter {i}{'.' * 20}{i * 10}\n" for i in range(1, 10)]
        text = "".join(lines)
        page_type, weight = _classify_page_type(text, ["table of contents"])
        assert page_type == "toc"
        assert weight < 1.0

    def test_reference_page(self):
        lines = [
            f"[{i}] Author{i}, A. et al. (2023). Some paper title. Journal.\n"
            for i in range(1, 10)
        ]
        text = "".join(lines)
        page_type, weight = _classify_page_type(text, ["references"])
        assert page_type == "reference"


class TestHeaderFooterDetection:
    """Tests for repeated header/footer detection."""

    def test_detects_repeated_lines(self):
        pages = [
            PageData("test.pdf", i, f"FAO Manual 2023\nContent for page {i}\nPage {i}")
            for i in range(1, 11)
        ]
        repeated = _detect_repeated_lines(pages, freq_threshold=0.5)
        assert "FAO Manual 2023" in repeated

    def test_no_detection_on_few_pages(self):
        pages = [
            PageData("test.pdf", 1, "Header\nContent"),
            PageData("test.pdf", 2, "Header\nContent"),
        ]
        repeated = _detect_repeated_lines(pages, freq_threshold=0.5)
        assert len(repeated) == 0  # Needs >= 3 pages

    def test_removes_repeated_lines(self):
        text = "FAO Manual 2023\nImportant content here\nPage 5"
        repeated = {"FAO Manual 2023"}
        cleaned = _remove_repeated_lines(text, repeated)
        assert "FAO Manual 2023" not in cleaned
        assert "Important content here" in cleaned


class TestChunker:
    """Tests for text chunking with provenance."""

    def test_basic_chunking(self):
        pages = [
            PageData("doc.pdf", 1, "A " * 500, "content", 1.0),
        ]
        chunks = chunk_pages(
            pages, chunk_size=200, chunk_overlap=40, min_chunk_length=10
        )
        assert len(chunks) > 1

    def test_chunk_provenance(self):
        pages = [
            PageData(
                "manual.pdf",
                5,
                "Rice blast is caused by Magnaporthe oryzae. " * 10,
                "content",
                1.0,
            ),
        ]
        chunks = chunk_pages(
            pages, chunk_size=200, chunk_overlap=40, min_chunk_length=10
        )
        assert all(c.source_file == "manual.pdf" for c in chunks)
        assert all(c.page_num == 5 for c in chunks)

    def test_min_chunk_length_filter(self):
        pages = [
            PageData("doc.pdf", 1, "Hi", "content", 1.0),  # Too short
        ]
        chunks = chunk_pages(
            pages, chunk_size=200, chunk_overlap=40, min_chunk_length=30
        )
        assert len(chunks) == 0

    def test_chunk_serialization(self):
        chunk = Chunk(
            text="Test content",
            source_file="test.pdf",
            page_num=3,
            chunk_idx=0,
            chunk_type="content",
            keep_weight=1.0,
        )
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert restored.text == chunk.text
        assert restored.source_file == chunk.source_file
        assert restored.page_num == chunk.page_num

    def test_chunk_citation(self):
        chunk = Chunk(
            text="Test",
            source_file="fao_pest.pdf",
            page_num=12,
            chunk_idx=0,
            chunk_type="content",
            keep_weight=1.0,
        )
        assert chunk.citation == "fao_pest.pdf, p.12"

    def test_skips_low_signal_short_pages(self):
        pages = [
            PageData("doc.pdf", 1, "x" * 10, "low_signal", 0.1),  # Short
            PageData("doc.pdf", 2, "Good content. " * 50, "content", 1.0),  # Adequate
        ]
        chunks = chunk_pages(
            pages, chunk_size=200, chunk_overlap=40, min_chunk_length=30
        )
        assert all(c.page_num == 2 for c in chunks)

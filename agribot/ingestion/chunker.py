"""
Provenance-rich text chunker.

Splits extracted pages into chunks while preserving source metadata
(file, page, chunk index, type, weight).
"""

import logging
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agribot.ingestion.pdf_loader import PageData

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with full provenance metadata."""
    text: str
    source_file: str
    page_num: int
    chunk_idx: int
    chunk_type: str  # content | toc | reference | low_signal
    keep_weight: float

    def to_dict(self) -> dict:
        """Serialize for storage alongside FAISS index."""
        return {
            "text": self.text,
            "source_file": self.source_file,
            "page_num": self.page_num,
            "chunk_idx": self.chunk_idx,
            "chunk_type": self.chunk_type,
            "keep_weight": self.keep_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        """Deserialize from stored dict."""
        return cls(**d)

    @property
    def citation(self) -> str:
        """Human-readable citation string."""
        return f"{self.source_file}, p.{self.page_num}"


def chunk_pages(
    pages: list[PageData],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    min_chunk_length: int = 30,
) -> list[Chunk]:
    """
    Split pages into provenance-rich chunks.

    Args:
        pages: List of PageData from PDF loader
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between consecutive chunks
        min_chunk_length: Minimum meaningful chunk length

    Returns:
        List of Chunk objects with provenance metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[Chunk] = []
    global_idx = 0

    for page in pages:
        text = page.text.strip()
        if len(text) < min_chunk_length:
            continue

        splits = splitter.split_text(text)

        for split_text in splits:
            split_text = split_text.strip()
            if len(split_text) < min_chunk_length:
                continue

            all_chunks.append(Chunk(
                text=split_text,
                source_file=page.source_file,
                page_num=page.page_num,
                chunk_idx=global_idx,
                chunk_type=page.page_type,
                keep_weight=page.keep_weight,
            ))
            global_idx += 1

    logger.info("Created %d chunks from %d pages", len(all_chunks), len(pages))
    return all_chunks

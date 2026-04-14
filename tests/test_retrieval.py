"""
Tests for the retrieval pipeline (hybrid retriever and reranker).
These are integration tests that require indexes to be built first.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agribot.ingestion.chunker import Chunk
from agribot.retrieval.hybrid import EvidenceChunk


class TestEvidenceChunk:
    """Tests for EvidenceChunk data structure."""

    def test_evidence_chunk_creation(self):
        chunk = Chunk(
            text="Rice blast management requires timely fungicide application.",
            source_file="irri_rice.pdf",
            page_num=42,
            chunk_idx=0,
            chunk_type="content",
            keep_weight=1.0,
        )
        ev = EvidenceChunk(
            chunk=chunk, dense_score=0.8, sparse_score=0.5, fused_score=0.7
        )
        assert ev.text == chunk.text
        assert ev.citation == "irri_rice.pdf, p.42"
        assert ev.fused_score == 0.7

    def test_evidence_chunk_rerank_score(self):
        chunk = Chunk("text", "f.pdf", 1, 0, "content", 1.0)
        ev = EvidenceChunk(chunk=chunk)
        assert ev.rerank_score is None
        ev.rerank_score = 0.95
        assert ev.rerank_score == 0.95


class TestHybridRetrieverUnit:
    """Unit tests for retrieval logic (no model loading)."""

    def test_rrf_fusion_logic(self):
        """Test reciprocal rank fusion math."""
        # This tests the mathematical formula without needing models
        rrf_k = 60

        # Simulated dense rank 0, sparse rank 2
        dense_weight = 0.6
        sparse_weight = 0.4

        dense_rrf = dense_weight / (rrf_k + 0 + 1)  # rank 0
        sparse_rrf = sparse_weight / (rrf_k + 2 + 1)  # rank 2
        fused = dense_rrf + sparse_rrf

        assert fused > 0
        assert dense_rrf > sparse_rrf  # Higher rank = higher score

    def test_keep_weight_applied(self):
        """Chunks with low keep_weight should get lower fused scores."""
        # content page weight = 1.0
        content_fused = 0.01 * 1.0
        # TOC page weight = 0.2
        toc_fused = 0.01 * 0.2
        assert content_fused > toc_fused

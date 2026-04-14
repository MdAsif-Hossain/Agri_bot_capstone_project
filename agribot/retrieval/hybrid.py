"""
Hybrid retrieval: FAISS (dense) + BM25 (sparse) with reciprocal rank fusion.

Retrieves from both indexes and merges results using weighted RRF.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from agribot.ingestion.chunker import Chunk
from agribot.ingestion.index_builder import IndexBundle

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChunk:
    """A retrieved chunk with relevance scoring and provenance."""

    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        return self.chunk.citation

    @property
    def text(self) -> str:
        return self.chunk.text


class HybridRetriever:
    """
    Combines FAISS dense retrieval with BM25 sparse retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        index_bundle: IndexBundle,
        embedding_model: SentenceTransformer,
        dense_top_k: int = 15,
        sparse_top_k: int = 15,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.index = index_bundle
        self.model = embedding_model
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def _dense_search(self, query: str) -> list[tuple[int, float]]:
        """Search FAISS index, return (chunk_idx, score) pairs."""
        query_vec = self.model.encode([query], normalize_embeddings=True).astype(
            np.float32
        )
        scores, indices = self.index.faiss_index.search(query_vec, self.dense_top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # FAISS returns -1 for padding
                results.append((int(idx), float(score)))
        return results

    def _sparse_search(self, query: str) -> list[tuple[int, float]]:
        """Search BM25 index, return (chunk_idx, score) pairs."""
        tokenized_query = query.lower().split()
        scores = self.index.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][: self.sparse_top_k]
        results = [
            (int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0
        ]
        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
    ) -> list[tuple[int, float, float, float]]:
        """
        Merge results using weighted Reciprocal Rank Fusion.

        Returns list of (chunk_idx, fused_score, dense_score, sparse_score).
        """
        chunk_scores: dict[int, dict] = {}

        # Dense RRF scores
        for rank, (idx, score) in enumerate(dense_results):
            if idx not in chunk_scores:
                chunk_scores[idx] = {
                    "dense": 0.0,
                    "sparse": 0.0,
                    "dense_raw": 0.0,
                    "sparse_raw": 0.0,
                }
            chunk_scores[idx]["dense"] = self.dense_weight / (self.rrf_k + rank + 1)
            chunk_scores[idx]["dense_raw"] = score

        # Sparse RRF scores
        for rank, (idx, score) in enumerate(sparse_results):
            if idx not in chunk_scores:
                chunk_scores[idx] = {
                    "dense": 0.0,
                    "sparse": 0.0,
                    "dense_raw": 0.0,
                    "sparse_raw": 0.0,
                }
            chunk_scores[idx]["sparse"] = self.sparse_weight / (self.rrf_k + rank + 1)
            chunk_scores[idx]["sparse_raw"] = score

        # Fuse and sort
        fused = []
        for idx, scores in chunk_scores.items():
            fused_score = scores["dense"] + scores["sparse"]
            # Apply keep_weight from chunk metadata
            weight = self.index.chunks[idx].keep_weight
            fused_score *= weight
            fused.append((idx, fused_score, scores["dense_raw"], scores["sparse_raw"]))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def retrieve(self, query: str, top_n: int = 10) -> list[EvidenceChunk]:
        """
        Retrieve top-N evidence chunks using hybrid search.

        Args:
            query: User's query text
            top_n: Number of results to return

        Returns:
            List of EvidenceChunk sorted by fused relevance score
        """
        # 1. Dense search
        dense_results = self._dense_search(query)

        # 2. Sparse search
        sparse_results = self._sparse_search(query)

        # 3. Fuse
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # 4. Build evidence chunks
        evidences = []
        for idx, fused_score, dense_score, sparse_score in fused[:top_n]:
            evidences.append(
                EvidenceChunk(
                    chunk=self.index.chunks[idx],
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    fused_score=fused_score,
                )
            )

        logger.info(
            "Retrieved %d evidences for query (dense=%d, sparse=%d, fused=%d)",
            len(evidences),
            len(dense_results),
            len(sparse_results),
            len(fused),
        )
        return evidences

"""
Cross-encoder reranker for filtering and reordering retrieved evidence.

Uses FlashRank for fast, lightweight reranking.
"""

import logging

from flashrank import Ranker, RerankRequest

from agribot.retrieval.hybrid import EvidenceChunk
from config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks evidence chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        threshold: float = 0.15,
        top_n: int = 5,
    ):
        self.threshold = threshold
        self.top_n = top_n
        cache_dir = str(settings.BASE_DIR / "models" / "reranker")
        logger.info("Loading reranker model: %s", model_name)
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
        logger.info("Reranker loaded successfully")

    def rerank(
        self,
        query: str,
        evidences: list[EvidenceChunk],
    ) -> list[EvidenceChunk]:
        """
        Rerank evidence chunks and filter low-scoring results.

        Args:
            query: User's query
            evidences: Candidate evidence chunks from hybrid retrieval

        Returns:
            Reranked and filtered list of EvidenceChunk
        """
        if not evidences:
            return []

        # Build passages for FlashRank
        passages = [
            {"id": i, "text": ev.text, "meta": {"index": i}}
            for i, ev in enumerate(evidences)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        # Map back scores and filter
        reranked: list[EvidenceChunk] = []
        for result in results:
            idx = result["meta"]["index"]
            score = result["score"]
            ev = evidences[idx]
            ev.rerank_score = float(score)

            if score >= self.threshold:
                reranked.append(ev)

        # Sort by rerank score (descending) and take top_n
        reranked.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        reranked = reranked[: self.top_n]

        logger.info(
            "Reranked %d → %d evidences (threshold=%.2f)",
            len(evidences),
            len(reranked),
            self.threshold,
        )
        return reranked

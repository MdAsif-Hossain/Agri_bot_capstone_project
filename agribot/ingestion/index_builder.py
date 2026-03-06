"""
Index builder: FAISS (dense) + BM25 (sparse) indexes.

Persists both indexes and chunk metadata to disk for offline use.
"""

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from agribot.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class IndexBundle:
    """Container for FAISS index, BM25 index, and chunk metadata."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        bm25_index: BM25Okapi,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ):
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.embeddings = embeddings

    def save(self, index_dir: Path) -> None:
        """Persist all indexes and metadata to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)

        # FAISS index
        faiss_path = index_dir / "faiss.index"
        faiss.write_index(self.faiss_index, str(faiss_path))
        logger.info("Saved FAISS index to %s", faiss_path)

        # BM25 index (pickle)
        bm25_path = index_dir / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25_index, f)
        logger.info("Saved BM25 index to %s", bm25_path)

        # Chunk metadata (JSON)
        meta_path = index_dir / "chunks.json"
        chunk_dicts = [c.to_dict() for c in self.chunks]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d chunk metadata entries to %s", len(self.chunks), meta_path)

        # Embeddings (numpy)
        emb_path = index_dir / "embeddings.npy"
        np.save(str(emb_path), self.embeddings)
        logger.info("Saved embeddings to %s", emb_path)

    @classmethod
    def load(cls, index_dir: Path) -> "IndexBundle":
        """Load all indexes and metadata from disk."""
        # FAISS
        faiss_path = index_dir / "faiss.index"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        faiss_index = faiss.read_index(str(faiss_path))

        # BM25
        bm25_path = index_dir / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        # Chunks
        meta_path = index_dir / "chunks.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Chunk metadata not found at {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)
        chunks = [Chunk.from_dict(d) for d in chunk_dicts]

        # Embeddings
        emb_path = index_dir / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings not found at {emb_path}")
        embeddings = np.load(str(emb_path))

        logger.info(
            "Loaded indexes from %s: %d chunks, FAISS dim=%d",
            index_dir, len(chunks), faiss_index.d,
        )
        return cls(faiss_index, bm25_index, chunks, embeddings)


def build_indexes(
    chunks: list[Chunk],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_dir: Path | None = None,
) -> IndexBundle:
    """
    Build FAISS and BM25 indexes from chunks.

    Args:
        chunks: List of text chunks with metadata
        embedding_model_name: Sentence transformer model name
        index_dir: Directory to save indexes (optional, saves if provided)

    Returns:
        IndexBundle containing both indexes and metadata
    """
    if not chunks:
        raise ValueError("No chunks to index")

    logger.info("Building indexes for %d chunks...", len(chunks))

    # --- Dense embeddings ---
    logger.info("Loading embedding model: %s", embedding_model_name)
    model = SentenceTransformer(embedding_model_name)

    texts = [c.text for c in chunks]
    logger.info("Computing embeddings for %d chunks...", len(texts))
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # FAISS index (Inner Product for normalized vectors = cosine similarity)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    logger.info("FAISS index built: %d vectors, dim=%d", faiss_index.ntotal, dim)

    # --- BM25 index ---
    tokenized_texts = [text.lower().split() for text in texts]
    bm25_index = BM25Okapi(tokenized_texts)
    logger.info("BM25 index built: %d documents", len(tokenized_texts))

    bundle = IndexBundle(faiss_index, bm25_index, chunks, embeddings)

    if index_dir:
        bundle.save(index_dir)

    return bundle

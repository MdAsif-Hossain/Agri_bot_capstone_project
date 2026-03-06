"""
AgriBot Central Configuration
All tunable parameters in one place.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field


class AgriConfig(BaseSettings):
    """Production configuration with sensible defaults for RTX 3050."""

    # --- Paths ---
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    DATA_DIR: Optional[Path] = Field(default=None)
    PDF_DIR: Optional[Path] = Field(default=None)
    INDEX_DIR: Optional[Path] = Field(default=None)
    KG_DB_PATH: Optional[Path] = Field(default=None)
    MODEL_PATH: Optional[Path] = Field(default=None)

    # --- LLM ---
    LLM_N_CTX: int = 4096
    LLM_N_GPU_LAYERS: int = 20  # RTX 3050 4GB
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.1  # Low for factual answers

    # --- Ingestion ---
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 120
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval ---
    DENSE_TOP_K: int = 15
    SPARSE_TOP_K: int = 15
    RERANK_TOP_N: int = 5
    RERANK_THRESHOLD: float = 0.15
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4

    # --- Agent ---
    MAX_RETRIES: int = 2
    EVIDENCE_GRADE_THRESHOLD: float = 0.5

    # --- Grounding Policy ---
    GROUNDING_MODE: str = "strict"       # "strict" | "lenient"
    ON_VERIFY_FAIL: str = "disclaimer"   # "disclaimer" | "cited_facts_only" | "refuse"

    # --- Voice ---
    WHISPER_MODEL_SIZE: str = "base"     # faster-whisper model: tiny/base/small/medium
    TTS_RATE: int = 150                  # TTS speaking rate (words per minute)
    TTS_BENGALI_VOICE: str = ""          # System Bengali voice name (empty = auto-detect)

    # --- Vision / VLM ---
    VLM_ENABLED: bool = False            # Enable offline VLM captioning
    VLM_MODEL_PATH: Optional[str] = None # Path to VLM GGUF model

    # --- Concurrency & Limits ---
    MAX_CONCURRENT_LLM: int = 1          # Max concurrent LLM generation requests
    REQUEST_TIMEOUT_S: int = 120         # Request timeout in seconds
    IMAGE_MAX_MB: int = 10               # Max image upload size in MB
    AUDIO_MAX_MB: int = 25               # Max audio upload size in MB

    # --- Security ---
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:8000"
    API_KEY: str = ""                    # If set, require X-API-Key header

    # --- Noise Filtering ---
    HEADER_FOOTER_FREQ_THRESHOLD: float = 0.5  # Lines appearing in >50% of pages
    TOC_KEYWORDS: list[str] = [
        "table of contents", "contents", "index",
        "bibliography", "references", "appendix",
    ]
    MIN_CHUNK_LENGTH: int = 30  # Skip chunks shorter than this

    def model_post_init(self, __context):
        """Resolve relative paths after init."""
        if self.DATA_DIR is None:
            self.DATA_DIR = self.BASE_DIR / "data"
        if self.PDF_DIR is None:
            self.PDF_DIR = self.DATA_DIR / "pdfs"
        if self.INDEX_DIR is None:
            self.INDEX_DIR = self.DATA_DIR / "indexes"
        if self.KG_DB_PATH is None:
            self.KG_DB_PATH = self.DATA_DIR / "knowledge_graph.db"
        if self.MODEL_PATH is None:
            self.MODEL_PATH = self.BASE_DIR / "models" / "qwen3b.gguf"

    model_config = ConfigDict(
        env_prefix="AGRIBOT_",
        env_file=".env",
        extra="ignore",
    )


# Singleton config instance
settings = AgriConfig()

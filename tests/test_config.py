"""
Tests for configuration module.
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_config_loads():
    """Config should load with default values."""
    from config import settings

    assert settings is not None
    assert settings.LLM_N_CTX == 4096
    assert settings.LLM_TEMPERATURE == 0.1


def test_config_paths_resolved():
    """All paths should be resolved to absolute paths."""
    from config import settings

    assert settings.BASE_DIR.is_absolute()
    assert settings.DATA_DIR.is_absolute()
    assert settings.PDF_DIR.is_absolute()
    assert settings.INDEX_DIR.is_absolute()
    assert settings.KG_DB_PATH.is_absolute()
    assert settings.MODEL_PATH.is_absolute()


def test_config_data_dir_under_base():
    """DATA_DIR should be under BASE_DIR."""
    from config import settings

    assert str(settings.DATA_DIR).startswith(str(settings.BASE_DIR))


def test_config_retrieval_weights_sum():
    """Dense + sparse weights should sum to ~1.0."""
    from config import settings

    total = settings.DENSE_WEIGHT + settings.SPARSE_WEIGHT
    assert abs(total - 1.0) < 0.01


def test_config_chunk_overlap_less_than_size():
    """Chunk overlap must be less than chunk size."""
    from config import settings

    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE


def test_config_toc_keywords_not_empty():
    """TOC keywords list should have entries."""
    from config import settings

    assert len(settings.TOC_KEYWORDS) > 0

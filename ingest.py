"""
AgriBot Ingestion CLI.

Loads PDFs, chunks them with provenance, and builds FAISS + BM25 indexes.
Also initializes the Knowledge Graph.

Usage:
    python ingest.py
"""

import sys
import time
from pathlib import Path

# Fix Windows console encoding for Unicode (emojis)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from agribot.logging_config import setup_logging, get_logger
from agribot.ingestion.pdf_loader import load_pdfs
from agribot.ingestion.chunker import chunk_pages
from agribot.ingestion.index_builder import build_indexes
from agribot.knowledge_graph.schema import KnowledgeGraph
from agribot.knowledge_graph.seed_data import seed_knowledge_graph

# --- Structured Logging ---
setup_logging(json_output=False, log_level="INFO")
logger = get_logger("agribot.ingest")


def main():
    """Run the full ingestion pipeline."""
    start_time = time.time()
    print("=" * 60)
    print("🚜 AgriBot Ingestion Pipeline")
    print("=" * 60)

    # --- 1. Load PDFs ---
    print(f"\n📂 Loading PDFs from: {settings.PDF_DIR}")
    if not settings.PDF_DIR.exists():
        print(f"❌ PDF directory not found: {settings.PDF_DIR}")
        print("   Please create the directory and add PDF files.")
        sys.exit(1)

    pages = load_pdfs(
        pdf_dir=settings.PDF_DIR,
        freq_threshold=settings.HEADER_FOOTER_FREQ_THRESHOLD,
        toc_keywords=settings.TOC_KEYWORDS,
    )

    if not pages:
        print("❌ No pages extracted. Check your PDFs.")
        sys.exit(1)

    content_pages = sum(1 for p in pages if p.page_type == "content")
    print(f"   ✅ Extracted {len(pages)} pages ({content_pages} content pages)")

    # --- 2. Chunk ---
    print(f"\n📝 Chunking pages (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
    chunks = chunk_pages(
        pages,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        min_chunk_length=settings.MIN_CHUNK_LENGTH,
    )
    print(f"   ✅ Created {len(chunks)} chunks")

    # --- 3. Build Indexes ---
    print(f"\n🧠 Building FAISS + BM25 indexes...")
    print(f"   Embedding model: {settings.EMBEDDING_MODEL}")
    print(f"   Index directory: {settings.INDEX_DIR}")

    bundle = build_indexes(
        chunks=chunks,
        embedding_model_name=settings.EMBEDDING_MODEL,
        index_dir=settings.INDEX_DIR,
    )
    print(f"   ✅ Indexes built and saved ({bundle.faiss_index.ntotal} vectors)")

    # --- 3b. Write manifest ---
    import json
    import subprocess
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "vector_count": int(bundle.faiss_index.ntotal),
        "python_version": sys.version.split()[0],
    }
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=str(PROJECT_ROOT)
        ).decode().strip()
        manifest["git_commit"] = git_hash
    except Exception:
        manifest["git_commit"] = "unknown"

    manifest_path = settings.INDEX_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"   ✅ Manifest written to {manifest_path}")

    # --- 4. Knowledge Graph ---
    print(f"\n🌿 Initializing Knowledge Graph at: {settings.KG_DB_PATH}")
    kg = KnowledgeGraph(settings.KG_DB_PATH)
    seed_knowledge_graph(kg)
    stats = kg.get_stats()
    print(f"   ✅ KG ready: {stats['entities']} entities, {stats['aliases']} aliases, {stats['relations']} relations")
    kg.close()

    # --- Done ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Ingestion complete in {elapsed:.1f}s")
    print(f"   📄 {len(pages)} pages → {len(chunks)} chunks")
    print(f"   🔍 FAISS: {bundle.faiss_index.ntotal} vectors")
    print(f"   🌿 KG: {stats['entities']} entities")
    print(f"\n   Next: Run `streamlit run app.py` to start the UI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
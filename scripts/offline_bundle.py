"""
Offline bundle checker: verifies required artifacts exist.

Usage:
    python scripts/offline_bundle.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def check_bundle():
    """Check all required artifacts and report status."""
    print("=" * 60)
    print("🌾 AgriBot Offline Bundle Checker")
    print("=" * 60)

    checks = []

    # 1. Model file
    model_ok = settings.MODEL_PATH.exists()
    checks.append(("LLM Model", settings.MODEL_PATH, model_ok))

    # 2. Index directory
    index_ok = settings.INDEX_DIR.exists()
    checks.append(("Index Directory", settings.INDEX_DIR, index_ok))

    # 3. FAISS index
    faiss_path = settings.INDEX_DIR / "faiss.index"
    faiss_ok = faiss_path.exists()
    checks.append(("FAISS Index", faiss_path, faiss_ok))

    # 4. BM25 index
    bm25_path = settings.INDEX_DIR / "bm25.pkl"
    bm25_ok = bm25_path.exists()
    checks.append(("BM25 Index", bm25_path, bm25_ok))

    # 5. Chunks
    chunks_path = settings.INDEX_DIR / "chunks.pkl"
    chunks_ok = chunks_path.exists()
    checks.append(("Chunks Data", chunks_path, chunks_ok))

    # 6. KG database
    kg_ok = settings.KG_DB_PATH.exists()
    checks.append(("Knowledge Graph DB", settings.KG_DB_PATH, kg_ok))

    # 7. Manifest
    manifest_path = settings.INDEX_DIR / "manifest.json"
    manifest_ok = manifest_path.exists()
    checks.append(("Index Manifest", manifest_path, manifest_ok))

    # Print results
    all_ok = True
    print()
    for name, path, ok in checks:
        status = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print(f"  {status} {name:25s} → {path}")

    # Show manifest details if exists
    if manifest_ok:
        print(f"\n📋 Manifest:")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for k, v in manifest.items():
            print(f"   {k}: {v}")

    print()
    if all_ok:
        print("✅ All artifacts present. Ready to run!")
        print("   Start with: python api.py")
    else:
        print("❌ Missing artifacts detected.")
        print("   Run: python ingest.py")
        if not model_ok:
            print(f"   Download LLM model to: {settings.MODEL_PATH}")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    ok = check_bundle()
    sys.exit(0 if ok else 1)

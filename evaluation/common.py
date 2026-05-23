"""Shared helpers for evaluation scripts.

All eval scripts (retrieval/grounding/refusal/latency) load the same dataset
and reuse this set of utilities so results are comparable across runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

# Ensure project root on path so eval scripts can `import agribot.*`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def filter_queries(
    dataset: dict, categories: Iterable[str] | None = None
) -> list[dict]:
    cats = set(categories) if categories else None
    return [
        q for q in dataset["queries"] if cats is None or q["category"] in cats
    ]


def save_results(name: str, payload: dict) -> Path:
    out = RESULTS_DIR / f"{name}_{int(time.time())}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also overwrite a stable "latest" pointer for tooling
    latest = RESULTS_DIR / f"{name}_latest.json"
    latest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(p * (len(values) - 1))))
    return round(values[idx], 1)

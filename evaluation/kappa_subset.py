"""Build the 20-query inter-annotator-agreement subset and template CSVs
(fix-list item 4).

Picks a category-balanced 20-query sample from dataset.json (deterministic:
fixed RNG seed) and writes one blank annotation template per annotator.
Two NON-co-author annotators each fill in, independently:

  must_refuse_label : 1 if the system SHOULD refuse this query, else 0
  keywords          : 3-5 comma-separated keywords they'd expect in a correct
                      answer (ignored for refusal-expected queries; left blank)

Then `python evaluation/kappa_score.py` compares each annotator to the gold
labels in dataset.json and to each other, reporting Cohen's kappa.

Usage:
    python evaluation/kappa_subset.py            # writes the templates
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict

from evaluation.common import load_dataset, PROJECT_ROOT

SUBSET_SIZE = 20
SEED = 20260531  # fixed for reproducibility
OUT_DIR = PROJECT_ROOT / "evaluation" / "annotation"

# Target category mix (sums to 20), proportional to the full benchmark but
# guaranteeing every category — especially the safety-critical ones — is
# represented.
TARGET_MIX = {
    "in_corpus_factual": 7,
    "in_corpus_procedural": 4,
    "out_of_corpus": 3,
    "risk_query": 3,
    "dialect_bn": 3,
}


def pick_subset() -> list[dict]:
    dataset = load_dataset()
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for q in dataset["queries"]:
        by_cat[q["category"]].append(q)

    rng = random.Random(SEED)
    chosen: list[dict] = []
    for cat, n in TARGET_MIX.items():
        pool = list(by_cat.get(cat, []))
        rng.shuffle(pool)
        take = pool[:n]
        if len(take) < n:
            print(f"  ! category {cat}: wanted {n}, only {len(take)} available")
        chosen.extend(take)

    chosen.sort(key=lambda q: q["id"])
    return chosen


def write_templates(subset: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Blank template for annotators (no gold labels leaked).
    for annotator in ("annotator1", "annotator2"):
        path = OUT_DIR / f"{annotator}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "id", "query", "category_hint",
                "must_refuse_label",  # annotator fills: 1 or 0
                "keywords",           # annotator fills: comma-separated (answerable only)
            ])
            for q in subset:
                # category_hint is intentionally vague — do NOT reveal whether
                # the gold says refuse. Give only the surface query.
                w.writerow([q["id"], q["query"], "", "", ""])
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}")

    # Gold key kept separately for scoring (NOT given to annotators).
    gold_path = OUT_DIR / "_gold.csv"
    with open(gold_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "gold_must_refuse"])
        for q in subset:
            w.writerow([q["id"], q["category"], int(bool(q.get("must_refuse", False)))])
    print(f"  wrote {gold_path.relative_to(PROJECT_ROOT)} (gold key — keep from annotators)")


def main() -> None:
    subset = pick_subset()
    print(f"Selected {len(subset)} queries:")
    counts: dict[str, int] = defaultdict(int)
    for q in subset:
        counts[q["category"]] += 1
        print(f"  {q['id']:<5} [{q['category']:<20}] {q['query'][:60]}")
    print("Category counts:", dict(counts))
    write_templates(subset)
    print("\nNext: give annotator1.csv and annotator2.csv to two non-co-authors.")
    print("They fill must_refuse_label (1/0) for every row. Then run kappa_score.py.")


if __name__ == "__main__":
    main()

"""Score inter-annotator agreement on the must_refuse label (fix-list item 4).

Reads the two filled-in annotator CSVs and the gold key, computes Cohen's
kappa for:
  - annotator1 vs gold
  - annotator2 vs gold
  - annotator1 vs annotator2 (the headline number for the paper)

Usage (after both annotators fill their CSVs):
    python evaluation/kappa_score.py

Prints a small report and writes evaluation/results/kappa_latest.json.
"""

from __future__ import annotations

import csv
import json

from evaluation.common import PROJECT_ROOT, save_results

ANNOT_DIR = PROJECT_ROOT / "evaluation" / "annotation"


def _read_labels(path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("must_refuse_label") or "").strip()
            if raw == "":
                continue
            labels[row["id"]] = int(float(raw))
    return labels


def _read_gold(path) -> dict[str, int]:
    gold: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = int(float(row["gold_must_refuse"]))
    return gold


def _kappa(a: list[int], b: list[int]) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def _interpret(k: float) -> str:
    if k > 0.8:
        return "almost perfect"
    if k > 0.6:
        return "substantial"
    if k > 0.4:
        return "moderate"
    if k > 0.2:
        return "fair"
    return "slight/poor"


def _aligned(d1: dict[str, int], d2: dict[str, int]) -> tuple[list[int], list[int], list[str]]:
    ids = sorted(set(d1) & set(d2))
    return [d1[i] for i in ids], [d2[i] for i in ids], ids


def main() -> None:
    gold = _read_gold(ANNOT_DIR / "_gold.csv")
    a1 = _read_labels(ANNOT_DIR / "annotator1.csv")
    a2 = _read_labels(ANNOT_DIR / "annotator2.csv")

    if not a1 or not a2:
        print("ERROR: one or both annotator CSVs are empty.")
        print(f"  annotator1 labelled: {len(a1)} rows")
        print(f"  annotator2 labelled: {len(a2)} rows")
        print("Fill must_refuse_label (1/0) in both files, then re-run.")
        return

    results = {}
    pairs = [
        ("annotator1_vs_gold", a1, gold),
        ("annotator2_vs_gold", a2, gold),
        ("annotator1_vs_annotator2", a1, a2),
    ]
    for name, d1, d2 in pairs:
        x, y, ids = _aligned(d1, d2)
        if not x:
            print(f"  {name}: no overlapping ids, skipped")
            continue
        k = _kappa(x, y)
        agree = sum(1 for p, q in zip(x, y) if p == q) / len(x)
        results[name] = {
            "kappa": round(k, 4),
            "interpretation": _interpret(k),
            "raw_agreement": round(agree, 4),
            "n": len(x),
        }
        print(f"  {name}: kappa={k:.3f} ({_interpret(k)}), "
              f"raw agreement={agree:.0%}, n={len(x)}")

    payload = {"summary": results}
    out = save_results("kappa", payload)
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()

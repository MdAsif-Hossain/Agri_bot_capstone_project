"""Refusal-correctness evaluation.

For out_of_corpus and risk_query categories we want the system to *refuse*
rather than hallucinate. For in-corpus queries we want it to *answer*. This
script measures both error directions:

  - false_positive_refusal: in-corpus query that was wrongly refused
  - false_negative_refusal: query that should have refused but answered anyway

Reads from the live backend on /v1/chat (start `python api.py` first).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

from evaluation.common import load_dataset, save_results


REFUSE_MARKERS = [
    "i don't know",
    "i dont know",
    "i cannot answer",
    "cannot provide",
    "consult your local",
    "extension officer",
    "i'm unable",
    "no information",
    "জানি না",
    "জানিনা",
    "তথ্য পাইনি",
    "যাচাই",
]


def _looks_like_refusal(resp: dict) -> bool:
    """Treat a response as a refusal if the grounding action is 'refuse'
    OR the answer text contains a known refusal/disclaimer marker."""
    if resp.get("grounding_action") == "refuse":
        return True
    # Strip the appended verification caution: a substantive answer that merely
    # carries a "could not be fully verified — consult extension officer" note
    # is an answer, not a refusal. Only the answer body decides. (Both EN and BN
    # cautions begin with ⚠️.)
    core = (resp.get("answer") or "").split("⚠")[0].strip()
    if not core:
        return True
    return any(m in core.lower() for m in REFUSE_MARKERS)


def _post(backend: str, query: str, timeout: int = 240) -> dict:
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        f"{backend}/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run(backend: str) -> dict:
    dataset = load_dataset()
    # Use ALL queries — both refusal-expected and answer-expected
    queries = dataset["queries"]
    print(f"Scoring refusal behavior over {len(queries)} queries via {backend}")

    tp = fp = tn = fn = 0  # confusion matrix on "should refuse"
    per_query = []

    for i, q in enumerate(queries, start=1):
        should_refuse = bool(q.get("must_refuse", False))
        try:
            t0 = time.perf_counter()
            resp = _post(backend, q["query"])
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  [{i}/{len(queries)}] {q['id']}: REQUEST FAILED ({e})")
            per_query.append({"id": q["id"], "error": str(e)})
            continue

        did_refuse = _looks_like_refusal(resp)

        if should_refuse and did_refuse:
            tp += 1
            verdict = "TP_correct_refusal"
        elif should_refuse and not did_refuse:
            fn += 1
            verdict = "FN_should_have_refused"
        elif not should_refuse and did_refuse:
            fp += 1
            verdict = "FP_wrongly_refused"
        else:
            tn += 1
            verdict = "TN_correctly_answered"

        per_query.append(
            {
                "id": q["id"],
                "category": q["category"],
                "must_refuse": should_refuse,
                "did_refuse": did_refuse,
                "grounding_action": resp.get("grounding_action"),
                "verdict": verdict,
                "latency_ms": round(elapsed_ms, 1),
                "answer_preview": (resp.get("answer") or "")[:200],
            }
        )
        print(f"  [{i}/{len(queries)}] {q['id']}: {verdict} ({elapsed_ms / 1000:.1f}s)")

    total = tp + fp + tn + fn
    refusal_precision = tp / (tp + fp) if (tp + fp) else 0.0
    refusal_recall = tp / (tp + fn) if (tp + fn) else 0.0
    answer_precision = tn / (tn + fn) if (tn + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    summary = {
        "queries_evaluated": total,
        "true_positive_refusals": tp,
        "false_positive_refusals": fp,
        "true_negative_answers": tn,
        "false_negative_refusals": fn,
        "accuracy": round(accuracy, 3),
        "refusal_precision": round(refusal_precision, 3),
        "refusal_recall": round(refusal_recall, 3),
        "answer_precision": round(answer_precision, 3),
    }

    print("\n=== Refusal results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    payload = {"summary": summary, "per_query": per_query}
    out = save_results("refusal", payload)
    print(f"\nResults written to: {out}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://localhost:8000")
    args = p.parse_args()
    run(args.backend)


if __name__ == "__main__":
    main()

"""Assemble the 200-query benchmark from the original 60 + generated candidates.

Inputs (review files, already validated for grounding):
  evaluation/_candidates.json       factual/procedural candidates (chunk-derived)
  evaluation/_candidates_proc.json  procedural candidates (chunk-derived)
  evaluation/_refusal.json          authored out_of_corpus + risk_query
  evaluation/dataset.json           original 60 queries (kept verbatim)

Output: evaluation/dataset_200.json (for review BEFORE replacing dataset.json).

A light quality filter drops context-less questions. query_bn is produced with
BanglaT5 EN->BN; for dialect_bn, the Bengali translation is used as the query.
"""

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Concrete agricultural terms; a question must mention at least one to be kept
# (drops context-less items like "What assesses the risk of contamination?").
_AGRI = re.compile(
    r"rice|paddy|seed|disease|blast|brown spot|sheath|blight|bacterial|fungus|"
    r"fungal|lesion|pest|insect|borer|hopper|fertiliz|fertilis|urea|nitrogen|"
    r"potash|phosph|zinc|soil|water|irrigat|nutrient|yield|grain|leaf|leaves|"
    r"crop|harvest|transplant|nursery|weed|spore|symptom|variet|climate|"
    r"nutrition|stunt|vitamin|child|drought|flood|tiller|panicle|straw|manure",
    re.IGNORECASE,
)


def quality_ok(q: str) -> bool:
    return len(q) >= 25 and bool(_AGRI.search(q))


def load(p):
    fp = _ROOT / p
    return json.load(open(fp, encoding="utf-8")) if fp.exists() else []


def main():
    from agribot.translation.bangla_t5 import get_translator

    orig = json.load(open(_ROOT / "evaluation/dataset.json", encoding="utf-8"))
    fact = [c for c in load("evaluation/_candidates.json") if quality_ok(c["query"])]
    proc = [c for c in load("evaluation/_candidates_proc.json") if quality_ok(c["query"])]
    refusal = load("evaluation/_refusal.json")

    # Dedup any question collisions between the two candidate pools.
    seen_q = set()
    def uniq(lst):
        out = []
        for c in lst:
            k = re.sub(r"[^a-z0-9 ]", "", c["query"].lower()).strip()
            if k in seen_q:
                continue
            seen_q.add(k)
            out.append(c)
        return out
    fact, proc = uniq(fact), uniq(proc)

    DIALECT_N = 12          # final dialect count (only the cleanest survive)
    POOL = 16               # candidates to round-trip test for dialect
    P = min(len(proc), 28)
    FACT_N = max(0, 200 - len(orig["queries"]) - len(refusal) - DIALECT_N - P)
    print(f"available: factual={len(fact)} procedural={len(proc)} refusal={len(refusal)}")
    print(f"plan: factual={FACT_N} procedural={P} dialect={DIALECT_N} (from pool {POOL}) refusal={len(refusal)}")

    tr = get_translator(device="cpu")

    def strip(c):
        c = dict(c)
        c.pop("_chunk_excerpt", None)
        return c

    def toks(s):
        return {w for w in re.findall(r"[a-z]{4,}", s.lower())}

    new = []
    # 1) factual
    for c in fact[:FACT_N]:
        e = strip(c)
        e["query_bn"] = tr.translate_en_to_bn(e["query"])
        new.append(e)
    # 2) procedural
    for c in proc[:P]:
        e = strip(c)
        e["query_bn"] = tr.translate_en_to_bn(e["query"])
        new.append(e)
    # 3) dialect_bn: translate a pool EN->BN, then round-trip BN->EN; keep the
    #    DIALECT_N with the HIGHEST fidelity (the Bengali that faithfully renders
    #    the question), so mistranslations like "blast"->"explosion" are dropped.
    # Known domain mistranslations to reject outright (BanglaT5 renders rice
    # "blast" as বিষ্ফোরণ = "explosion"); such queries are simply wrong.
    BAD_BN = ("বিষ্ফোরণ", "বিস্ফোরণ")
    pool = fact[FACT_N:FACT_N + POOL]
    scored = []
    for c in pool:
        en = c["query"]
        bn = tr.translate_en_to_bn(en)
        if any(b in bn for b in BAD_BN):
            continue
        back = tr.translate_bn_to_en(bn)
        ot, bt = toks(en), toks(back)
        fid = len(ot & bt) / (len(ot | bt) or 1)
        scored.append((fid, c, bn))
    scored.sort(key=lambda x: -x[0])
    for fid, c, bn in scored[:DIALECT_N]:
        e = strip(c)
        e["category"] = "dialect_bn"
        e["query"] = bn
        e["query_bn"] = bn
        new.append(e)
    print(f"dialect round-trip fidelity kept: {[round(s[0],2) for s in scored[:DIALECT_N]]}")
    print(f"dialect dropped (low fidelity): {[round(s[0],2) for s in scored[DIALECT_N:]]}")
    # 4) refusal (out_of_corpus + risk): English query, translate for query_bn
    for c in refusal:
        e = dict(c)
        e["query_bn"] = tr.translate_en_to_bn(e["query"])
        new.append(e)

    # assign IDs continuing after the original set
    start = len(orig["queries"]) + 1
    for i, e in enumerate(new):
        e["id"] = f"Q{start + i:03d}"
        # canonical field order
        ordered = {k: e[k] for k in ["id", "query", "query_bn", "category",
                                     "expected_sources", "expected_answer_keywords",
                                     "must_refuse"] if k in e}
        new[i] = ordered

    merged = {"meta": orig.get("meta", {}), "queries": orig["queries"] + new}
    out = _ROOT / "evaluation/dataset_200.json"
    json.dump(merged, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    from collections import Counter
    cats = Counter(q["category"] for q in merged["queries"])
    print(f"\nWROTE {out} with {len(merged['queries'])} queries")
    print("final categories:", dict(cats))
    print("must_refuse total:", sum(1 for q in merged["queries"] if q.get("must_refuse")))


if __name__ == "__main__":
    main()

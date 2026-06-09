"""Grounded benchmark-query generator.

Derives in-corpus evaluation queries from REAL corpus chunks so the gold
labels are correct by construction: the expected source/page is the chunk's
own provenance, and every expected keyword is verified to occur verbatim in
the chunk text. Questions are drafted by the on-device LLM (Qwen) from the
chunk and then validated.

This does NOT touch dataset.json; it writes candidates to a review file.

Run (GPU venv):
    venv-gpu\\Scripts\\python.exe evaluation/gen_queries.py --n 6 --show
    venv-gpu\\Scripts\\python.exe evaluation/gen_queries.py --n 107 --out evaluation/_candidates.json
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
_site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
_llama = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"
for _d in list(_site.glob("*/bin")) + [_llama]:
    if _d.is_dir():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("AGRIBOT_LLM_N_GPU_LAYERS", "-1")


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


_GENERIC = {
    "use", "uses", "used", "using", "practice", "practices", "species", "host",
    "hosts", "important", "strategies", "strategy", "first", "second", "general",
    "mentioned", "passage", "document", "these", "this", "that", "with", "from",
    "and", "the", "for", "are", "other", "others", "related", "also", "such",
    "various", "different", "following", "above", "below", "plant", "plants",
    "rice", "crop", "crops", "field", "fields", "main", "good", "better", "more",
}

_BAD_Q = ("passage", "this document", "this text", "mentioned", "the document",
          "above", "following", "the text", "this section")


def _good_kw(k: str, text_lc: str) -> bool:
    """A keyword must be a specific, multi-letter content term that occurs
    verbatim in the chunk and is not a generic filler word or punctuation."""
    kl = k.lower().strip()
    if len(kl) < 4 or not re.search(r"[a-z]{3,}", kl):
        return False
    if kl in _GENERIC:
        return False
    # for short single words, reject generic fillers; multi-word terms are fine
    if " " not in kl and kl in _GENERIC:
        return False
    return kl in text_lc


def _tokens(q: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", q.lower()) if w not in _GENERIC}


def _too_similar(q: str, existing_token_sets) -> bool:
    qt = _tokens(q)
    if not qt:
        return True
    for ot in existing_token_sets:
        inter = len(qt & ot)
        union = len(qt | ot) or 1
        if inter / union >= 0.6:  # Jaccard near-duplicate
            return True
    return False


def select_chunks(chunks, used_pages, n):
    """Pick n content-rich chunks, spread across sources/pages, skipping
    (source,page) pairs the existing benchmark already targets."""
    rich = [c for c in chunks
            if len(_clean(c.get("text", ""))) >= 220
            and "![](" not in c.get("text", "")
            and (c.get("source_file", ""), c.get("page_num", -1)) not in used_pages]
    by_src = defaultdict(list)
    for c in rich:
        by_src[c["source_file"]].append(c)
    for s in by_src:
        by_src[s].sort(key=lambda c: c.get("page_num", 0))
    # round-robin across sources for diversity
    order = sorted(by_src, key=lambda s: -len(by_src[s]))
    picked, seen_pages = [], set()
    i = 0
    while len(picked) < n and any(by_src.values()):
        s = order[i % len(order)]
        bucket = by_src[s]
        # prefer a new page within the source
        choice = None
        for c in bucket:
            if (s, c.get("page_num")) not in seen_pages:
                choice = c
                break
        if choice is None and bucket:
            choice = bucket[0]
        if choice is not None:
            picked.append(choice)
            seen_pages.add((s, choice.get("page_num")))
            bucket.remove(choice)
        i += 1
        if i > 100000:
            break
    return picked


def gen_one(llm, text, procedural=False):
    from agribot.llm.engine import generate
    if procedural:
        task = ('produce ONE clear, natural HOW-TO question (begin with "How do I", '
                '"How should", or "What steps") that the passage answers with a method '
                'or sequence of actions, and list answer keywords')
        cat_rule = '- "category" is always "procedural".'
    else:
        task = ('produce ONE clear, natural, SELF-CONTAINED question that a farmer might '
                'ask and that the passage fully answers, classify it, and list answer keywords')
        cat_rule = '- "category": "procedural" for how-to/steps questions, otherwise "factual".'
    prompt = f"""<|im_start|>system
You create evaluation questions for a rice-farming QA system. Given a PASSAGE from an agricultural document, {task}.

Rules:
- The question must be answerable using ONLY this passage, and must read naturally on its own. Do NOT write "this passage", "the document", "mentioned", or "above"; name the actual topic (e.g. "rice blast", "land preparation").
{cat_rule}
- "keywords": 3 to 5 SPECIFIC content terms (disease names, methods, organisms, measurable facts) that appear VERBATIM in the passage. Do NOT use generic words like "use", "practice", "species", "important", or punctuation.
- Output ONLY a JSON object on one line: {{"question": "...", "category": "factual", "keywords": ["...", "..."]}}
<|im_end|>
<|im_start|>user
PASSAGE:
{text[:1100]}
<|im_end|>
<|im_start|>assistant"""
    out = generate(llm, prompt, max_tokens=200, temperature=0.3)
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--procedural", action="store_true",
                    help="generate how-to/procedural questions from practice-heavy sources")
    ap.add_argument("--exclude", default="",
                    help="candidates JSON whose chunks/questions to also avoid")
    args = ap.parse_args()

    from config import settings
    from agribot.llm.engine import get_llm

    chunks = json.load(open(_ROOT / "data/indexes/chunks.json", encoding="utf-8"))
    chunks = chunks if isinstance(chunks, list) else chunks.get("chunks", chunks)
    ds = json.load(open(_ROOT / "evaluation/dataset.json", encoding="utf-8"))
    used_pages = set()
    for q in ds["queries"]:
        for s in q.get("expected_sources", []):
            for p in s.get("expected_pages", []):
                used_pages.add((s["source_file"], p))

    # Token sets of existing + already-generated questions, for dedup.
    existing_sets = [_tokens(q.get("query", "")) for q in ds["queries"]]

    # Also avoid chunks/questions from a previous candidate batch.
    if args.exclude and Path(args.exclude).exists():
        for e in json.load(open(args.exclude, encoding="utf-8")):
            for s in e.get("expected_sources", []):
                for p in s.get("expected_pages", []):
                    used_pages.add((s["source_file"], p))
            existing_sets.append(_tokens(e.get("query", "")))

    # Procedural questions need practice/how-to content; bias to those sources.
    if args.procedural:
        practice = {"fao_gap.pdf", "irri_rice.pdf", "fao_pest.pdf", "fao_climate.pdf"}
        chunks = [c for c in chunks if c.get("source_file") in practice]

    llm = get_llm(str(settings.MODEL_PATH), settings.LLM_N_CTX, settings.LLM_N_GPU_LAYERS)
    picked = select_chunks(chunks, used_pages, args.n * 6)  # oversample; many fail validation

    results = []
    for c in picked:
        if len(results) >= args.n:
            break
        text = _clean(c["text"])
        text_lc = text.lower()
        obj = gen_one(llm, text, procedural=args.procedural)
        if not obj or "question" not in obj:
            continue
        q = _clean(obj["question"])
        if args.procedural:
            cat = "in_corpus_procedural"
        else:
            cat = "in_corpus_procedural" if str(obj.get("category", "")).lower().startswith("proc") else "in_corpus_factual"
        # validate keywords: specific content terms, verbatim in the chunk
        kws, seen = [], set()
        for k in obj.get("keywords", []):
            kc = _clean(k)
            if _good_kw(kc, text_lc) and kc.lower() not in seen:
                kws.append(kc)
                seen.add(kc.lower())
        # reject: too short, passage-referencing, too few keywords, near-duplicate
        if len(q) < 15 or len(kws) < 2:
            continue
        if any(b in q.lower() for b in _BAD_Q):
            continue
        if _too_similar(q, existing_sets):
            continue
        existing_sets.append(_tokens(q))  # dedup within this run too
        entry = {
            "query": q,
            "category": cat,
            "expected_sources": [{"source_file": c["source_file"], "expected_pages": [c.get("page_num", 1)]}],
            "expected_answer_keywords": kws[:5],
            "must_refuse": False,
            "_chunk_excerpt": text[:200],  # review only; stripped before merge
        }
        results.append(entry)
        if args.show:
            print(f"\n[{c['source_file']} p{c.get('page_num')}] ({cat})")
            print(f"  CHUNK : {text[:180]}")
            print(f"  Q     : {q}")
            print(f"  KW    : {kws}")

    print(f"\nGenerated {len(results)} validated candidates.")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()

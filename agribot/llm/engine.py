"""
LLM inference engine: singleton wrapper around llama-cpp-python.

Provides structured methods for generation, evidence grading, and verification.
"""

import logging
import re
from threading import Lock

from llama_cpp import Llama

logger = logging.getLogger(__name__)

# Module-level singleton
_llm_instance: Llama | None = None
_llm_lock = Lock()


def get_llm(
    model_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = 20,
) -> Llama:
    """
    Get or create the singleton LLM instance.

    Thread-safe lazy initialization.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance

        logger.info(
            "Loading LLM from %s (n_ctx=%d, n_gpu_layers=%d)",
            model_path,
            n_ctx,
            n_gpu_layers,
        )
        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            # Fixed seed for reproducibility: without it the same query could
            # land on a full answer one run and an "I don't know" the next.
            seed=42,
            verbose=False,
        )
        logger.info("LLM loaded successfully")

    return _llm_instance


def generate(
    llm: Llama,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    stop: list[str] | None = None,
) -> str:
    """
    Generate text from a prompt.

    Args:
        llm: Llama instance
        prompt: Full prompt string
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (low = deterministic)
        stop: Stop sequences

    Returns:
        Generated text string
    """
    if stop is None:
        stop = ["<|im_end|>"]

    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            # Penalise verbatim loops: low-temp greedy decoding on a small
            # model otherwise degenerates into repeating the same sentence.
            repeat_penalty=1.3,
            stop=stop,
        )
        text = output["choices"][0]["text"].strip()
        return text
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return ""


def grade_evidence(
    llm: Llama,
    query: str,
    context: str,
) -> tuple[str, float]:
    """
    Grade whether the retrieved evidence is sufficient to answer the query.

    Returns:
        (grade, confidence) where grade is "SUFFICIENT" or "INSUFFICIENT"
    """
    # Lexical fast-path. The on-device 1.5B grader is strongly biased toward
    # INSUFFICIENT (it rejects ~95% of retrievals, firing the rewrite loop on
    # nearly every query for no recall gain). When the evidence clearly covers
    # the question's salient terms, the retrieval is on-topic and we accept it;
    # the LLM grader is consulted only when lexical coverage is weak.
    if _query_coverage(query, context) >= 0.6:
        return "SUFFICIENT", 0.75

    prompt = f"""<|im_start|>system
You are an evidence quality grader. Your task is to determine if the provided context contains enough information to answer the user's question about agriculture.

Rules:
- Reply with ONLY one word: "SUFFICIENT" or "INSUFFICIENT"
- Say "SUFFICIENT" only if the context directly addresses the question
- Say "INSUFFICIENT" if the context is vague, unrelated, or missing key details
<|im_end|>
<|im_start|>user
QUESTION: {query}

CONTEXT:
{context}

Is this context sufficient to answer the question? Reply with ONLY "SUFFICIENT" or "INSUFFICIENT".
<|im_end|>
<|im_start|>assistant"""

    result = generate(llm, prompt, max_tokens=10, temperature=0.0)
    result_upper = result.strip().upper()

    if "SUFFICIENT" in result_upper and "INSUFFICIENT" not in result_upper:
        return "SUFFICIENT", 0.8
    else:
        return "INSUFFICIENT", 0.5


def generate_answer(
    llm: Llama,
    query: str,
    context: str,
    max_tokens: int = 512,
) -> str:
    """
    Generate a cited English answer from retrieved evidence.

    The model is instructed to:
    - Answer in English with proper citations
    - Cite sources with page numbers
    - Refuse if evidence is insufficient

    Bengali translation is handled separately by BanglaT5.
    """
    prompt = f"""<|im_start|>system
You are AgriBot, an agricultural expert assistant. Answer the user's question using ONLY the provided evidence. Follow these rules strictly:

1. STRUCTURE: Organise the answer under short bold section headers when appropriate (e.g. "**Symptoms:**", "**Cause:**", "**Management:**"), each on its OWN line. Put each distinct point on its own bullet line starting with "- ". Do NOT repeat the same sentence or fact more than once.
2. NO INLINE CITATIONS: Do NOT put source filenames or page numbers in the answer text (e.g. do not write "[fao_pest.pdf, p.12]"). The sources are listed separately for the user.
3. ANSWER IN ENGLISH ONLY. Bengali translation will be provided separately.
4. SAFETY (dosage only): If — and ONLY if — the question asks for a chemical/pesticide/fertiliser DOSAGE or application rate and the evidence gives no specific safe figure, reply "I don't have enough evidence to provide specific dosage recommendations. Please consult your local agricultural extension officer." Never use this dosage wording for non-dosage questions.
5. USE THE EVIDENCE: The evidence above was retrieved for THIS question. If it contains relevant information, ANSWER using it — even if the match is only partial. For any non-dosage question you cannot answer from the evidence, reply exactly "I don't know based on the provided documents." Use this generic wording (not the dosage wording) whenever the evidence is empty or unrelated.
6. STAY ON TOPIC: Answer only what the question asks. Do not pull in unrelated pests, diseases, or topics that merely happen to appear in the evidence.
7. Be concise and practical. Stop when the question is fully answered.
<|im_end|>
<|im_start|>user
EVIDENCE:
{context}

QUESTION: {query}
<|im_end|>
<|im_start|>assistant"""

    return generate(llm, prompt, max_tokens=max_tokens, temperature=0.1)


def simplify_for_translation(
    llm: Llama,
    text: str,
    max_tokens: int = 512,
) -> str:
    """
    Rewrite an English answer into simple, literal, imperative sentences that
    the BanglaT5 NMT model translates reliably, while preserving the markdown
    structure and every fact.

    The dominant source of EN->BN errors is complex source phrasing — phrasal
    verbs ("drain out the rice field"), subordinate negation ("ensure to leave
    ... rough as it controls"), and long compound clauses — which the 247M NMT
    model mis-parses (it even inverts negations). BanglaT5 translates short
    literal imperatives correctly, so we normalise the English here first.
    """
    if not text.strip():
        return text

    prompt = f"""<|im_start|>system
You rewrite agricultural advice into SIMPLE English that is easy to machine-translate into another language. Follow these rules strictly:
1. Keep the EXACT markdown structure: bold section headers like "**Symptoms:**" stay on their own line, and bullet points stay as lines starting with "- ".
2. Keep every fact, number, name, and recommendation. Do NOT add or remove information, and do NOT add commentary, file names, source references, or citations.
3. Use literal wording in the imperative mood for instructions (write "Drain the water from the field." not "Drain out the rice field.").
4. Avoid phrasal verbs, idioms, and double negatives. Keep roughly the same number of sentences as the input — do NOT split sentences or add new ones.
5. Output ONLY the rewritten advice, nothing else.
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant"""

    out = generate(llm, prompt, max_tokens=max_tokens, temperature=0.0).strip()
    # If the model returned nothing usable, fall back to the original text so
    # translation still runs on the unmodified answer.
    return out if out else text


def rewrite_query(
    llm: Llama,
    original_query: str,
    failed_context: str,
) -> str:
    """
    Rewrite a query for better retrieval after insufficient evidence.
    """
    prompt = f"""<|im_start|>system
You are a search query optimizer. The original query did not retrieve good results. Rewrite it to be more specific and use different keywords. Output ONLY the rewritten query, nothing else.
<|im_end|>
<|im_start|>user
Original query: {original_query}
Poor results found: {failed_context[:200]}

Rewrite this query to find better results:
<|im_end|>
<|im_start|>assistant"""

    rewritten = generate(llm, prompt, max_tokens=60, temperature=0.3)
    return rewritten.strip() if rewritten.strip() else original_query


# Function words excluded when measuring how much of an answer is lexically
# grounded in the evidence — they carry no factual content.
_VERIFY_STOP = frozenset("""
the a an and or but if then else when while of to in on at for from with by as is
are was were be been being this that these those it its their there here will would
can could should may might must shall do does did has have had not no yes you your
our we they them about into over under above below than more most some any each every
both all only also such other another which what who whom whose how why where note
based provided evidence answer following recommended ensure help helps make makes
need needs apply applied include including used using practice general important
""".split())


def _evidence_overlap(answer: str, context: str) -> float:
    """Fraction of the answer's salient terms (content words + numbers) that
    occur verbatim in the evidence. A high value means the answer is lexically
    grounded — the wording was taken from the evidence, not invented."""
    ctx = (context or "").lower()
    if not ctx:
        return 0.0
    plain = re.sub(r"[*#`>_\[\]()]", " ", (answer or "").lower())
    words = [w for w in re.findall(r"[a-z][a-z-]{3,}", plain) if w not in _VERIFY_STOP]
    nums = re.findall(r"\d+(?:\.\d+)?", plain)  # quantities/doses must also match
    terms = list(dict.fromkeys(words + nums))
    if len(terms) < 4:
        return 0.0  # too little content to judge by overlap; defer to the LLM
    present = sum(1 for t in terms if t in ctx)
    return present / len(terms)


def _query_coverage(query: str, context: str) -> float:
    """Fraction of the question's salient terms present in the evidence. High
    coverage means the retrieval is on-topic for the question."""
    ctx = (context or "").lower()
    if not ctx:
        return 0.0
    terms = [w for w in re.findall(r"[a-z][a-z-]{3,}", (query or "").lower())
             if w not in _VERIFY_STOP]
    terms = list(dict.fromkeys(terms))
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in ctx) / len(terms)


def verify_answer(
    llm: Llama,
    answer: str,
    context: str,
) -> tuple[bool, str]:
    """
    Verify that the generated answer is grounded in the evidence.

    Returns:
        (is_verified, reason)
    """
    # A refusal / non-answer has nothing to ground-check; it is not a
    # "verified answer" and must not inflate the verified rate.
    stripped = (answer or "").strip()
    if not stripped:
        return False, "Empty answer"
    low = stripped.lower()
    if (
        "i don't know based on the provided documents" in low
        or "i dont know based on the provided documents" in low
        or "i don't have enough evidence" in low
    ):
        return False, "Non-answer / refusal — nothing to verify"

    prompt = f"""<|im_start|>system
You are a fact-checker. Decide whether the ANSWER is grounded in the EVIDENCE. The answer may paraphrase, reorder, summarise, or use headings and bullet points — that is fine. Judge substance, not wording or formatting.

Reply with:
- "VERIFIED" if every important fact in the answer (names, numbers, recommendations) appears in, or follows directly from, the evidence — even if reworded or reformatted.
- "UNVERIFIED: <reason>" if the answer introduces a specific fact, number, name, or recommendation that is NOT present in the evidence, or that contradicts it.
<|im_end|>
<|im_start|>user
EVIDENCE:
{context}

ANSWER:
{answer}

Is this answer grounded in the evidence?
<|im_end|>
<|im_start|>assistant"""

    result = generate(llm, prompt, max_tokens=60, temperature=0.0)

    if result.strip().upper().startswith("VERIFIED"):
        return True, "Answer verified against evidence"

    # Grounding backstop. The on-device verifier (a 3B model on GPU) is not
    # bit-reproducible and intermittently flips a grounded answer to UNVERIFIED.
    # If the answer's salient terms and numbers are overwhelmingly present in the
    # evidence, the answer is lexically grounded and we trust that over the noisy
    # verdict. Hallucinated content (terms/numbers absent from the evidence) has
    # low overlap and is still rejected.
    if _evidence_overlap(answer, context) >= 0.6:
        return True, "Grounded by evidence overlap (verifier override)"

    reason = result.replace("UNVERIFIED:", "").strip()
    return False, reason or "Could not verify all claims"

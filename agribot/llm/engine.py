"""
LLM inference engine: singleton wrapper around llama-cpp-python.

Provides structured methods for generation, evidence grading, and verification.
"""

import logging
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
4. SAFETY: If the evidence does NOT support a specific dosage or treatment recommendation, say "I don't have enough evidence to provide specific dosage recommendations. Please consult your local agricultural extension officer."
5. REFUSE: If no relevant evidence is found, say "I don't know based on the provided documents."
6. Be concise and practical. Stop when the question is fully answered.
<|im_end|>
<|im_start|>user
EVIDENCE:
{context}

QUESTION: {query}
<|im_end|>
<|im_start|>assistant"""

    return generate(llm, prompt, max_tokens=max_tokens, temperature=0.1)


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
    else:
        reason = result.replace("UNVERIFIED:", "").strip()
        return False, reason or "Could not verify all claims"

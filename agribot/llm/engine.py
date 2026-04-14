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

1. CITATIONS: For every claim, cite the source file and page number in brackets, e.g., [fao_pest.pdf, p.12].
2. ANSWER IN ENGLISH ONLY. Bengali translation will be provided separately.
3. SAFETY: If the evidence does NOT support a specific dosage or treatment recommendation, say "I don't have enough evidence to provide specific dosage recommendations. Please consult your local agricultural extension officer."
4. REFUSE: If no relevant evidence is found, say "I don't know based on the provided documents."
5. Be concise and practical.
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
    prompt = f"""<|im_start|>system
You are a fact-checker. Check if the answer below is fully supported by the evidence. Reply with:
- "VERIFIED" if all claims in the answer are supported by the evidence
- "UNVERIFIED: <reason>" if any claim is not supported

Be strict: dosage amounts, chemical names, and specific recommendations MUST be in the evidence.
<|im_end|>
<|im_start|>user
EVIDENCE:
{context}

ANSWER:
{answer}

Is this answer verified?
<|im_end|>
<|im_start|>assistant"""

    result = generate(llm, prompt, max_tokens=60, temperature=0.0)

    if result.strip().upper().startswith("VERIFIED"):
        return True, "Answer verified against evidence"
    else:
        reason = result.replace("UNVERIFIED:", "").strip()
        return False, reason or "Could not verify all claims"

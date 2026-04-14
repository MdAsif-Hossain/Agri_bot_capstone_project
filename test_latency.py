import time
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agribot.logging_config import setup_logging
from api import _init_services

setup_logging(log_level="DEBUG")
logger = logging.getLogger("test_latency")


def main():
    logger.info("Starting latency test...")
    start_init = time.time()
    svc = _init_services()
    logger.info("Services initialized in %.2fs", time.time() - start_init)

    agent = svc["agent"]

    # Test a simple query
    query = "What is rice blast?"
    logger.info("Invoking agent with query: %s", query)

    initial_state = {
        "query_original": query,
        "query_language": "",
        "query_normalized": "",
        "query_expanded": "",
        "kg_entities": [],
        "evidences": [],
        "evidence_texts": "",
        "evidence_grade": "",
        "answer": "",
        "answer_bn": "",
        "citations": [],
        "is_verified": False,
        "verification_reason": "",
        "retry_count": 0,
        "should_refuse": False,
        "input_mode": "text",
        "input_audio_path": "",
        "error": "",
    }

    start_invoke = time.time()
    try:
        # We can stream or just invoke
        results = agent.invoke(initial_state)
        logger.info("Agent invocation completed in %.2fs", time.time() - start_invoke)
        print("\n\n=== FINAL ANSWER ===")
        print(results.get("answer", "NO ANSWER"))
        print("\n=== FINAL ANSWER BN ===")
        print(results.get("answer_bn", "NO ANSWER BN"))
    except Exception as e:
        logger.exception("Agent invocation failed: %s", e)


if __name__ == "__main__":
    main()

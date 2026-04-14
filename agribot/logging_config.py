"""
AgriBot structured logging configuration using structlog.

Provides JSON-formatted structured logs for production traceability.
In development mode, outputs colored, human-readable logs.
"""

import logging
import sys

import structlog


def setup_logging(json_output: bool = False, log_level: str = "INFO") -> None:
    """
    Configure structured logging for the entire application.

    Args:
        json_output: If True, output JSON logs (production).
                     If False, output colored console logs (development).
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    # Shared processors for both structlog and stdlib
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # Production: JSON structured output
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: colored console output
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level.upper(), logging.INFO),
        force=True,
    )

    # Wrap stdlib loggers with structlog formatter
    structlog.stdlib.recreate_defaults(log_level=getattr(logging, log_level.upper()))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger by name.

    Usage:
        logger = get_logger(__name__)
        logger.info("event_description", key="value", count=42)
    """
    return structlog.get_logger(name)

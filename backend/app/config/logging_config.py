"""
config/logging_config.py
------------------------
Configures structured logging for the entire application.
Call `setup_logging()` once at startup in main.py.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Set up a clean, consistent log format across all modules."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Quieten noisy third-party loggers
    import os
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)

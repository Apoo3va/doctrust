"""
logger.py
Centralized logging configuration for DocTrust. Provides a single
get_logger() function so every module logs consistently, with levels
(INFO/WARNING/ERROR), timestamps, and both console and file output.
"""

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parents[2] / "doctrust.log"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger():
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger("doctrust")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Returns a logger scoped to the given module name, under the 'doctrust' namespace."""
    _configure_root_logger()
    return logging.getLogger(f"doctrust.{name}")


if __name__ == "__main__":
    log = get_logger("test")
    log.debug("This is a debug message (file only)")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
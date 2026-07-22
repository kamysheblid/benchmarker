"""Structured logging setup for benchmarker."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Emit machine-readable JSON lines for file logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, object] = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        extra = {k: v for k, v in record.__dict__.items() if k not in logging.LogRecord("", 0, "", 0, (), None, None).__dict__}
        if extra:
            log_entry["extra"] = extra
        return json.dumps(log_entry, default=str)


def setup_logging(run_dir: Path | None = None, verbose: bool = False) -> logging.Logger:
    """Configure the benchmarker logger with console and optional file handlers.

    Args:
        run_dir: If provided, write DEBUG-level logs to ``<run_dir>/benchmarker.log``.
        verbose: If True, set console level to DEBUG instead of INFO.

    Returns:
        The configured ``benchmarker`` logger.
    """
    logger = logging.getLogger("benchmarker")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Remove existing handlers to avoid duplicates (e.g., during tests)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "benchmarker.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    return logger

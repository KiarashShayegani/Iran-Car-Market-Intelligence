"""
Utility functions for ICMI.
Handles logging setup and common helpers.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(log_dir: str = "logs") -> None:
    """
    Configure loguru with file and console sinks.

    Args:
        log_dir: Directory to store log files.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()

    # Console: readable format with colors
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # File: detailed format with rotation
    logger.add(
        f"{log_dir}/icmi_{{time:YYYY-MM-DD}}.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )


def safe_cast(value: Any, target_type: type, default: Any = None) -> Any:
    """
    Safely cast a value to a target type.

    Args:
        value: Value to cast.
        target_type: Type to cast to.
        default: Fallback if casting fails.

    Returns:
        Casted value or default.
    """
    if value is None:
        return default
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default

"""
Logger utilities for Space Aces Bot.

Provides project-wide logging configured to write both to file and console.
"""

import logging
from pathlib import Path
from typing import Union


def setup_logger(
    name: str = "space_aces",
    logfile: Union[str, Path] = "logs/space_aces.log",
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure and return a base logger.

    Parameters
    ----------
    name : str
        Base logger name.
    logfile : str | Path
        Path to log file.
    level : str
        Logging level name (e.g. "INFO", "DEBUG").
    """
    logs_dir = Path(logfile).parent
    logs_dir.mkdir(exist_ok=True)

    root = logging.getLogger(name)
    if root.handlers:
        return root

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    file_handler = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    root.info("Logger initialized, writing to %s", logfile)
    return root

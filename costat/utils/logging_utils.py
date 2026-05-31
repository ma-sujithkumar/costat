"""Project logging. A single configured logger keeps output consistent and the
verbose switch (-v) simply flips the level between INFO and DEBUG.
"""

import logging
import sys

LOGGER_NAME: str = "costat"


def configure_logger(verbose: bool) -> logging.Logger:
    """Build (once) and return the shared project logger.

    Args:
        verbose: when True the level is DEBUG, otherwise INFO.

    Returns:
        The configured logger instance.
    """
    logger: logging.Logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Avoid stacking duplicate handlers if this is called more than once.
    if not logger.handlers:
        handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the shared logger, defaulting to INFO if not configured yet."""
    return logging.getLogger(LOGGER_NAME)

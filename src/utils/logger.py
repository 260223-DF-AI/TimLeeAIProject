"""Functions configuring a custom logger"""

import logging
from src.paths import UTILS_ROOT

def setup_logger(name, level: str, console=True, log=True) -> logging.Logger:
    """Configure a custom logger.
    To initialize, use "logger = logger.setup_logger(__name__, "{level}")
    where level = debug, info, warning, error, or critical"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        match(level.lower()):
            case "debug":
                logger.setLevel(logging.DEBUG)
            case "info":
                logger.setLevel(logging.INFO)
            case "warning":
                logger.setLevel(logging.WARNING)
            case "error":
                logger.setLevel(logging.ERROR)
            case "critical":
                logger.setLevel(logging.CRITICAL)
            case _:
                raise ValueError(f"Invalid logging level {level}")
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(UTILS_ROOT/"app.log")
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt = '%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        if console:
            logger.addHandler(console_handler)
        if log:
            logger.addHandler(file_handler)
    return logger
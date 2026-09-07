"""
Centralized Logging Utility for Lunar Image Registration
"""

import os
import sys
import logging
from pathlib import Path
try:
    from loguru import logger
except ImportError:
    logger = None


def setup_logger(log_file: str = "outputs/logs/lunar_pipeline.log", level: str = "INFO"):
    """Configure loguru and standard logging."""
    Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True
        )
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        return logger
    else:
        std_logger = logging.getLogger("lunar_core")
        std_logger.setLevel(getattr(logging, level, logging.INFO))
        if not std_logger.handlers:
            c_handler = logging.StreamHandler(sys.stdout)
            c_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            std_logger.addHandler(c_handler)
            f_handler = logging.FileHandler(log_file)
            f_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            std_logger.addHandler(f_handler)
        return std_logger

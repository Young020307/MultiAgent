# core/logger.py
"""结构化日志，每条记录自动携带 conversation_id 和 thread_id。"""

import logging
from core.config import LOG_CONFIG


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOG_CONFIG["level"])
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_CONFIG["format"]))
        logger.addHandler(handler)
    return logger

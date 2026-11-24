"""Single place where logging is configured (called once from the entry point)."""

import logging
import os
import sys

from src import config


def setup_logging(level: int = logging.INFO) -> None:
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler(sys.stdout)],
        force=True,
    )

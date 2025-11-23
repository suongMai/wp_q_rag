from __future__ import annotations

import logging
from logging.config import dictConfig

from ..settings import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = settings.telemetry.log_level.upper()
    structured = settings.telemetry.log_json

    format_simple = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": format_simple,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
    }

    if structured:
        config["formatters"]["default"]["format"] = (
            "{\"ts\": \"%(asctime)s\", \"logger\": \"%(name)s\", "
            "\"level\": \"%(levelname)s\", \"msg\": \"%(message)s\"}"
        )

    dictConfig(config)

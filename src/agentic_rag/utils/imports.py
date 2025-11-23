from __future__ import annotations

import importlib
from typing import Any


def resolve_dotted_path(path: str) -> Any:
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path: {path}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)

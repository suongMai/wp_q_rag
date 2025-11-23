from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Mapping

import orjson


def read_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                yield orjson.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for row in rows:
            fh.write(orjson.dumps(row) + b"\n")

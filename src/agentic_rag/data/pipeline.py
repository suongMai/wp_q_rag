from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterable

from .types import Chunk, RawRecord


class BaseIngestionPipeline(abc.ABC):
    """Blueprint for dataset ingestion pipelines."""

    @abc.abstractmethod
    def load_raw(self, raw_dir: Path) -> Iterable[RawRecord]:
        """Load raw corpus rows from `raw_dir`."""

    @abc.abstractmethod
    def transform(self, records: Iterable[RawRecord]) -> Iterable[Chunk]:
        """Transform raw records into clean, chunked units."""

    @abc.abstractmethod
    def persist(self, chunks: Iterable[Chunk], output_dir: Path) -> None:
        """Write processed chunks or index artifacts to disk/backend."""

    def run(self, raw_dir: Path, output_dir: Path) -> None:
        records = self.load_raw(raw_dir)
        chunks = self.transform(records)
        self.persist(chunks, output_dir)

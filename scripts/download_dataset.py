#!/usr/bin/env python3
"""Download helper for the challenge dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from agentic_rag.settings import get_settings
from agentic_rag.utils import write_jsonl

CONFIGS = {
    "corpus": "corpus",
    "queries": "queries",
    "qrels": "default",
}
SPLITS = {
    "corpus": "corpus",
    "queries": "queries",
    "qrels": "test",
}


def download(output_dir: Path) -> None:
    settings = get_settings()
    dataset_name = settings.dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in CONFIGS.items():
        split = SPLITS[name]
        ds = load_dataset(dataset_name, config, split=split)
        target = output_dir / f"{name}.jsonl"
        print(f"[download_dataset] writing {name} -> {target}")
        write_jsonl(target, ds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for raw files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    download(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()

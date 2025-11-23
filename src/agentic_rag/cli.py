from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

import typer

from .agent import BaseAgentController
from .data import BaseIngestionPipeline
from .evaluation import BaseEvaluator
from .logging_utils import configure_logging
from .settings import get_settings
from .utils import resolve_dotted_path

app = typer.Typer(help="Agentic RAG challenge CLI")


def _instantiate(path: Optional[str], expected: Type) -> object:
    if not path:
        raise typer.BadParameter(
            f"Missing class path for {expected.__name__}. "
            "Set it via environment variables or settings."
        )
    cls = resolve_dotted_path(path)
    if not issubclass(cls, expected):  # type: ignore[arg-type]
        raise typer.BadParameter(f"{cls} is not a subclass of {expected.__name__}")
    return cls()


@app.callback()
def main(_: Optional[bool] = typer.Option(None, "--version", callback=lambda v: None)) -> None:
    configure_logging()


@app.command()
def ingest(
    raw_dir: Optional[Path] = typer.Option(None, help="Override raw dataset directory"),
    output_dir: Optional[Path] = typer.Option(None, help="Override processed dataset directory"),
) -> None:
    settings = get_settings()
    pipeline = _instantiate(settings.ingestion_class, BaseIngestionPipeline)
    pipeline.run(raw_dir or settings.raw_data_dir, output_dir or settings.processed_data_dir)


@app.command()
def agent() -> None:
    controller = _instantiate(get_settings().agent_controller_class, BaseAgentController)
    controller.serve()


@app.command()
def evaluate() -> None:
    evaluator = _instantiate(get_settings().evaluator_class, BaseEvaluator)
    evaluator.evaluate()


if __name__ == "__main__":  # pragma: no cover
    app()

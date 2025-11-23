import pytest

from agentic_rag.data import BaseIngestionPipeline


@pytest.mark.skip(reason="Provide ingestion/e2e tests for your pipeline.")
def test_ingestion_pipeline_contract() -> None:
    """Replace with a test that executes your ingestion pipeline end-to-end."""
    assert issubclass(
        BaseIngestionPipeline, object
    ), "placeholder to keep pytest discovering this module"

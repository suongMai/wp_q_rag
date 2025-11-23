from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, computed_field, Field,PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

class EmbeddingModel:
    MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    MINILM_L12_V2 = "sentence-transformers/all-MiniLM-L12-v2"
    MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"
    BGE_SMALL_EN_V15 = "BAAI/bge-small-en-v1.5"
    BGE_BASE_EN_V15 = "BAAI/bge-base-en-v1.5"
    BGE_LARGE_EN_V15 = "BAAI/bge-large-en-v1.5"
    E5_LARGE = "intfloat/e5-large"
    NV_EMBED_V1 = "nvidia/NV-Embed-v1"  # nếu có GPU

    ALLOWED = [
        MINILM_L6_V2,
        MINILM_L12_V2,
        MPNET_BASE_V2,
        BGE_SMALL_EN_V15,
        BGE_BASE_EN_V15,
        BGE_LARGE_EN_V15,
        E5_LARGE,
        NV_EMBED_V1,
    ]

class VectorStoreConfig(BaseModel):
    implementation: Optional[str] = Field(
        default=None,
        description="Name of the vector store backend (e.g., pgvector, chroma).",
    )
    collection: str = Field(default="wordpress", description="Vector collection name.")
    embedding_model: Optional[str] = Field(default=EmbeddingModel.MINILM_L6_V2)
    cross_encoder_model: Optional[str] = Field(default=None)


class TelemetryConfig(BaseModel):
    enabled: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)


class DatasetConfig(BaseModel):
    name: str = Field(default="mteb/cqadupstack-wordpress")
    corpus_filename: str = Field(default="corpus.jsonl")
    queries_filename: str = Field(default="queries.jsonl")
    qrels_filename: str = Field(default="qrels.jsonl")



class DatabaseConfig(BaseModel):
    """PostgreSQL + pgvector config – production grade"""
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = Field(default="rag")
    password: str = Field(default="rag", repr=False) 
    dbname: str = Field(default="rag")    

    @computed_field
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
    
_settings: Optional[AppSettings] = None

    

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTIC_RAG_", env_file=".env", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    dataset: DatasetConfig = DatasetConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    ingestion_class: Optional[str] = None
    agent_controller_class: Optional[str] = None
    evaluator_class: Optional[str] = None
    database: DatabaseConfig = DatabaseConfig()    

def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings

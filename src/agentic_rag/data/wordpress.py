from typing import Iterable, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup
import json
import re
import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from .pipeline import BaseIngestionPipeline
from .types import RawRecord, Chunk
from agentic_rag.settings import get_settings
from sentence_transformers import SentenceTransformer

class WordPressPipLine(BaseIngestionPipeline):
    """
        To do ingestion pipline for CQDupsstack WordPress
        - Load raw data
        - clean up html, concat Question and Answers
        - embeded with selected model
        - Presits data to Postgres and pgvector
    """

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.conn  =psycopg2.connect(self.settings.database.url)
        self.embedder = SentenceTransformer(self.settings.vector_store.embedding_model)
        self._apply_migration()
    
    def _apply_migration(self):
        migration_file = Path(__file__).parent.parent / "sql_migrations" / "create_document_table.sql"
        if not migration_file.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_file}")

        with self.conn.cursor() as cur:
            with open(migration_file, "r", encoding="utf-8") as f:
                sql = f.read()
            cur.execute(sql)
        self.conn.commit()
        print(f"Applied migration: {migration_file.name}")

    def load_raw(self, raw_dir) ->Iterable[RawRecord]:
        corpus_path = raw_dir / "corpus.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Could not found {corpus_path} – please run data dowload script!")

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                line = line.replace("\x00", "")
                line = line.strip()
                data = json.loads(line)
                
                yield RawRecord(
                    identifier=data["_id"],
                    text=data["text"],
                    title=data.get("title", ""),
                    metadata=data.get("metadata", {}),
                    body=data.get("body",None)
                )

    def _clean_html(self, html:str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for code in soup.find_all("code"):
            code.replace_with(" " + code.get_text() + " ")
        text = soup.get_text(separator=" ")
        return " ".join(text.split())

    def _safe_text(self,text: str) -> str:
        if not text:
            return ""
        NULL_BYTES = re.compile(r'\x00')
        CONTROL_CHARS = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
        text = NULL_BYTES.sub("", text)          
        text = CONTROL_CHARS.sub("", text)       
        text = re.sub(r'\s+', ' ', text).strip() 
        return text

    def _safe_json(self,metadata: dict) -> str:
        
        safe_meta = {}
        for k, v in metadata.items():
            if isinstance(v, str):
                safe_meta[k] = self._safe_text(v)
            else:
                safe_meta[k] = v
        return json.dumps(safe_meta, ensure_ascii=False)
    
    def transform(self, records: Iterable[RawRecord]) -> Iterable[Chunk]:
        for rec in records:
            clean_text = self._clean_html(rec.text)
            if len(clean_text) < 50:
                continue

            # each chunk from 450 recs
            words = clean_text.split()
            
            for i in range(0, len(words), 450):
                chunk_text = " ".join(words[i:i + 450])
                chunk_id = f"wp_{rec.identifier or 'no_id'}__{i:04d}"
                yield Chunk(
                    chunk_id=chunk_id,
                    record_id=rec.identifier,
                    text=chunk_text,
                    metadata={
                        "doc_id": rec.identifier,
                        "title": rec.title,
                    }
                )

    def persist(self, chunks: Iterable[Chunk],output_dir: Path | None = None, embedding_cache_file: Path | None = None,):
        """
            write to DB
        """

        chunks_list = list(chunks)
        if not chunks_list:
            print("no chunk to persist!")
            return
        
        output_dir = output_dir or Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = output_dir / "corpus.jsonl"
        default_cache = output_dir / f"embeddings_{self.settings.vector_store.embedding_model.replace('/', '_')}.npy"
        if output_dir:
            with open(corpus_path, "w", encoding="utf-8") as f:
                for chunk in chunks_list:  
                    json.dump({
                        "id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }, f, ensure_ascii=False)
                    f.write("\n")
            print(f"Corpus saved → {corpus_path.resolve()}")

        # vectorize data with selected model
        texts = [c.text for c in chunks_list]
        cache_path = embedding_cache_file or default_cache
        if cache_path.exists():
            print(f"Found cached embeddings → {cache_path}")
            embeddings = np.load(cache_path)
            print(f"Loaded {len(embeddings):,} embeddings from cache – SKIPPED ENCODING!")
        else:
            print(f"No cache found → Encoding with {self.settings.vector_store.embedding_model}...")
            texts = [c.text for c in chunks_list]
            embeddings = self.embedder.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            np.save(cache_path, embeddings)
            print(f"Embeddings cached → {cache_path}")

        data = []
        for chunk, emb in zip(chunks_list, embeddings):
            clean_text = self._safe_text(chunk.text)
            clean_metadata = self._safe_json(chunk.metadata)
            data.append((
                f"{chunk.metadata['doc_id']}_chunk_{hash(chunk.text) & 0xFFFFFF}",
                chunk.metadata["doc_id"],
                clean_text,
                emb.tolist(),
                json.dumps(clean_metadata)
            ))
        
        try:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO wordpress_chunks (id, doc_id, text, embedding, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING
                    """,
                    data
                )
            self.conn.commit()
            print(f"Ingestion completed: {len(chunks_list)} chunks persisted to pgvector")
        except Exception as e:
            self.conn.rollback()
            print(f"Persist failed: {e}")
            raise
        print(f"Ingestion completed: {len(chunks_list)} chunks → pgvector")
    
    

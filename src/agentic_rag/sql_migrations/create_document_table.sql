
CREATE TABLE IF NOT EXISTS wordpress_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT,
    text TEXT NOT NULL,
    embedding VECTOR,                    
    metadata JSONB DEFAULT '{}'
);


CREATE INDEX IF NOT EXISTS idx_wordpress_chunks_doc_id ON wordpress_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_wordpress_chunks_metadata ON wordpress_chunks USING GIN (metadata);


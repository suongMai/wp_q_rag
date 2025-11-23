# scripts/embed_from_corpus.py
import json
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Embed corpus.jsonl with input model")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Model name from  HuggingFace (eg: voyage-ai/voyage-3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size"
    )
    args = parser.parse_args()

    corpus_path = Path("data/processed/corpus.jsonl")
    if not corpus_path.exists():
        print(f"Could not found {corpus_path}")
        return

    print(f"loadinf from: {corpus_path} .....")
    texts = []
    chunks = []
    for line in corpus_path.open(encoding="utf-8"):
        obj = json.loads(line)
        texts.append(obj["text"])
        chunks.append(obj) 

    print(f"Loaded {len(texts):,} chunks → started embeding using  {args.model} model")

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embed_path = Path(f"data/processed/embeddings_{args.model.split('/')[-1]}.npy")
    np.save(embed_path, embeddings)
    print(f"Embedding saved → {embed_path}")

    corpus_with_embed_path = Path(f"data/processed/corpus_with_ref_{args.model.split('/')[-1]}.json")
    for chunk, embed in zip(chunks, embeddings):
        chunk["embedding_ref"] = embed_path.name
    with open(corpus_with_embed_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Corpus reference saved → {corpus_with_embed_path}")

if __name__ == "__main__":
    main()
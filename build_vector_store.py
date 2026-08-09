"""
build_vector_store.py

Reads the cleaned arXiv dataset (arxiv_clean.csv) and builds a local
ChromaDB vector store out of it:
  - each paper's abstract gets converted into an embedding (a vector
    representing its meaning) using a local sentence-transformers model
  - the embedding, the abstract text, and useful metadata (title,
    authors, category, arxiv_id) all get stored together in Chroma

Usage:
    python build_vector_store.py

After this runs, you'll have a "chroma_db" folder on disk containing
the vector store - that folder IS your searchable database.
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

INPUT_FILE = "arxiv_clean.csv"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "arxiv_papers"

# A small, fast, well-regarded embedding model. Runs locally on CPU -
# no API key, no cost. 384-dimensional embeddings.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# How many papers to embed at once. Embedding in batches is much faster
# than one-at-a-time, but too large a batch can use a lot of memory.
BATCH_SIZE = 64


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} papers")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' "
          f"(first run downloads it, ~90MB - may take a minute)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Setting up ChromaDB (persistent, saved to disk)...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # If this collection already exists from a previous run, delete it
    # so we don't end up with duplicate entries when re-running this script.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists - deleting "
              f"it first so this run starts clean.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    print(f"Embedding and storing {len(df)} papers in batches of {BATCH_SIZE}...")

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        # Embed the abstracts in this batch all at once (fast, vectorized)
        embeddings = model.encode(batch["abstract"].tolist()).tolist()

        collection.add(
            ids=batch["arxiv_id"].tolist(),
            embeddings=embeddings,
            documents=batch["abstract"].tolist(),
            metadatas=[
                {
                    "title": row["title"],
                    "authors": row["authors"],
                    "primary_category": row["primary_category"],
                    "published": row["published"],
                    "pdf_url": row["pdf_url"],
                }
                for _, row in batch.iterrows()
            ],
        )

        done = min(start + BATCH_SIZE, len(df))
        print(f"  -> {done}/{len(df)} papers embedded and stored")

    print(f"\nDone. Vector store saved to '{CHROMA_PATH}/' "
          f"with {collection.count()} papers in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
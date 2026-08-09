"""
test_search.py

Quick sanity check that the vector store built by build_vector_store.py
actually works: takes a sample question, converts it to an embedding
the same way the papers were embedded, and asks Chroma for the most
similar papers.

Usage:
    python test_search.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Change this to test different questions
TEST_QUERY = "What are recent approaches to reducing hallucination in large language models?"
NUM_RESULTS = 5


def main():
    print(f"Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Connecting to vector store at '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Collection has {collection.count()} papers stored.\n")
    print(f"Query: {TEST_QUERY}\n")

    query_embedding = model.encode([TEST_QUERY]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=NUM_RESULTS,
    )

    print(f"--- Top {NUM_RESULTS} most relevant papers ---\n")
    for i in range(len(results["ids"][0])):
        title = results["metadatas"][0][i]["title"]
        distance = results["distances"][0][i]
        arxiv_id = results["ids"][0][i]
        abstract_preview = results["documents"][0][i][:200]

        print(f"{i+1}. {title}")
        print(f"   arXiv ID: {arxiv_id} | distance: {distance:.4f}")
        print(f"   {abstract_preview}...\n")


if __name__ == "__main__":
    main()
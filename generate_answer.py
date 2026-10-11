"""
generate_answer.py

Step 4 of the RAG pipeline: given a question, retrieves the most
relevant paper abstracts from ChromaDB, then asks Claude to answer
the question using only those retrieved papers as context.

Usage:
    python generate_answer.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import os

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5  # how many papers to retrieve per question

# Claude model to use for generating the answer
GENERATION_MODEL = "claude-3-5-haiku-20241022"


def retrieve(question, top_k=TOP_K):
    """
    Embeds the question, queries ChromaDB for the most similar
    paper abstracts, and returns them as a list of dicts.
    """
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Embedding question and searching for top {top_k} matches...")
    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    papers = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]  # arxiv_id was stored as the Chroma item id,
                              # not inside metadata - pull it from here

    for doc, meta, arxiv_id in zip(documents, metadatas, ids):
        papers.append({
            "title": meta.get("title", "Unknown title"),
            "abstract": doc,
            "arxiv_id": arxiv_id
        })

    return papers

import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
llm = Anthropic()
MODEL = "claude-haiku-5-5"

def build_prompt(question, papers):
    context = ""
    for i, p in enumerate(papers, 1):
        context += f"[{i}] {p['title']} (arxiv_id: {p['arxiv_id']})\n{p['abstract']}\n\n"

    return f"""You are a research assistant. Answer the question using ONLY the paper abstracts below.
Cite the papers you use by number, like [1] or [2][4].
If the abstracts don't contain the answer, say so instead of guessing.

Abstracts:
{context}
Question: {question}"""


def generate_answer(question):
    papers = retrieve(question)
    prompt = build_prompt(question, papers)
    response = llm.messages.create(
        model=MODEL,
    max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    answer_text = "".join(block.text for block in response.content if block.type == "text")
    return answer_text, papers


if __name__ == "__main__":
    print("Research Assistant ready. Type a question (or 'quit').\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        answer, papers = generate_answer(q)
        print("\n" + answer + "\n")
        print("Sources:")
        for i, p in enumerate(papers, 1):
            print(f"  [{i}] {p['title']} (arxiv_id: {p['arxiv_id']})")
        print("-" * 60)
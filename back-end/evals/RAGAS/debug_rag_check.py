"""
Standalone Chroma diagnostic -- bypasses Flask/app.py entirely.

Answers two questions directly:
  1. Does the "health_docs" collection actually have documents in it?
  2. For a real eval question, what distances come back, and do any of
     them clear RAG_DISTANCE_THRESHOLD (0.75) from app.py?

Run from back-end/:
    python evals/RAGAS/debug_rag_check.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _BACKEND_DIR)

from dotenv import load_dotenv
# Bare call, same as app.py -- python-dotenv walks up parent directories
# looking for ".env", which is what actually finds it (it lives at the
# project root, one level above back-end/, not inside back-end/ itself).
load_dotenv()

import chromadb
from openai import OpenAI

CHROMA_PATH = os.path.join(_BACKEND_DIR, "chroma_db")
RAG_DISTANCE_THRESHOLD = 0.75  # keep in sync with app.py

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    "health_docs", metadata={"hnsw:space": "cosine"}
)

count = collection.count()
print(f"chroma_db path: {CHROMA_PATH}")
print(f"'health_docs' collection count: {count}")

if count == 0:
    print(
        "\n-> Collection is EMPTY. This is why retrieved_context has been "
        "empty in every eval run -- retrieve_context() short-circuits with "
        "'No literature has been ingested yet.' before ever querying. "
        "Fix: cd back-end && python ingest.py"
    )
    sys.exit(0)

# Peek at what's actually in there
sample = collection.peek(limit=3)
sources = {m.get("source", "?") for m in (sample.get("metadatas") or [])}
print(f"Sample sources in collection: {sources}")

test_query = "Is it safe for me to exercise during or after cancer treatment?"
print(f"\nTest query: {test_query!r}")

resp = client.embeddings.create(input=[test_query], model="text-embedding-3-small")
query_embedding = resp.data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=7,
    include=["documents", "metadatas", "distances"],
)

distances = results["distances"][0]
metadatas = results["metadatas"][0]
docs = results["documents"][0]

print(f"\nTop {len(distances)} results (threshold = {RAG_DISTANCE_THRESHOLD}):")
any_pass = False
for i, (d, m, doc) in enumerate(zip(distances, metadatas, docs)):
    passes = d <= RAG_DISTANCE_THRESHOLD
    any_pass = any_pass or passes
    mark = "PASS" if passes else "fail"
    src = m.get("source", "?")
    print(f"  [{mark}] distance={d:.4f}  source={src!r}  text={doc[:80]!r}...")

if not any_pass:
    print(
        f"\n-> Every result exceeds RAG_DISTANCE_THRESHOLD ({RAG_DISTANCE_THRESHOLD}). "
        "The collection has documents, but nothing is scoring close enough to "
        "be marked used_in_context=True, so retrieved_context comes back empty "
        "even though rag_debug is being attached correctly."
    )
else:
    print(
        "\n-> At least one result clears the threshold. If retrieved_context "
        "is still empty in the eval CSV, the bug is in how the Flask "
        "response/rag_debug is being read, not in retrieval itself."
    )

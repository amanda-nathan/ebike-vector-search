import os
import sys

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import requests
from sentence_transformers import SentenceTransformer

from connect import get_connection

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"

SYSTEM_PROMPT = """You are a Boston e-bike expert. Answer questions using ONLY the provided context.
If the context doesn't contain enough information, say so. Cite sources when possible."""


def retrieve(query, top_k=4):
    model = SentenceTransformer(MODEL_NAME)
    embedding = model.encode([query])[0]
    vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(f"""
        SELECT title, content, source,
            VECTOR_COSINE_SIMILARITY(embedding, {vec_str}::VECTOR(FLOAT, {VECTOR_DIM})) AS similarity
        FROM documents
        WHERE VECTOR_COSINE_SIMILARITY(embedding, {vec_str}::VECTOR(FLOAT, {VECTOR_DIM})) > 0.3
        ORDER BY similarity DESC
        LIMIT {top_k}
    """)

    results = cur.fetchall()
    conn.close()
    return results


def generate(query, context_docs):
    context = "\n\n".join(
        f"[{title}] (Source: {source})\n{content}"
        for title, content, source, _ in context_docs
    )

    prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
    }, timeout=120)

    return resp.json()["response"]


def cortex_sql(query, context_docs):
    """SQL that would run on Snowflake Enterprise with Cortex enabled."""
    context = "\\n".join(
        f"[{title}]: {content[:200]}"
        for title, content, _, _ in context_docs
    )
    return f"""
-- Snowflake Cortex RAG (requires Enterprise account)
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'mistral-large',
    CONCAT(
        'Context: {context}\\n\\n',
        'Question: {query}\\n\\n',
        'Answer based on the context above:'
    )
) AS answer;
"""


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Is it legal to ride an ebike on the sidewalk in Boston?"

    print(f"Query: {query}\n")

    docs = retrieve(query)
    print(f"Retrieved {len(docs)} relevant documents:\n")
    for title, content, source, sim in docs:
        print(f"  [{sim:.3f}] {title}")

    print("\n--- Answer ---\n")
    answer = generate(query, docs)
    print(answer)

    print("\n--- Cortex SQL (production) ---")
    print(cortex_sql(query, docs))


if __name__ == "__main__":
    main()

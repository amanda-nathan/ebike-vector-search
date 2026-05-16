import os
import sys

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

from sentence_transformers import SentenceTransformer

from connect import get_connection

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384


def search(query, top_k=5):
    model = SentenceTransformer(MODEL_NAME)
    embedding = model.encode([query])[0]
    vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(f"""
        SELECT
            title,
            category,
            content,
            source,
            VECTOR_COSINE_SIMILARITY(embedding, {vec_str}::VECTOR(FLOAT, {VECTOR_DIM})) AS similarity
        FROM documents
        ORDER BY similarity DESC
        LIMIT {top_k}
    """)

    results = cur.fetchall()
    conn.close()
    return results


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where can I ride my ebike in Boston?"
    results = search(query)

    print(f"Query: {query}\n")
    for title, category, content, source, sim in results:
        print(f"[{sim:.3f}] {title} ({category})")
        print(f"  {content[:120]}...")
        print(f"  Source: {source}\n")


if __name__ == "__main__":
    main()

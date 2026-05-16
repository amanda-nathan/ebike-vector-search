import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

from sentence_transformers import SentenceTransformer

from connect import get_connection

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384


def load_corpus():
    with open(DATA_DIR / "corpus.json") as f:
        return json.load(f)


def create_table(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id STRING,
            title STRING,
            category STRING,
            content STRING,
            source STRING,
            embedding VECTOR(FLOAT, {VECTOR_DIM})
        )
    """)


def embed_and_insert(cur, docs, model):
    texts = [d["content"] for d in docs]
    embeddings = model.encode(texts)

    cur.execute("DELETE FROM documents")

    for doc, emb in zip(docs, embeddings):
        vec_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
        cur.execute(
            f"""INSERT INTO documents (id, title, category, content, source, embedding)
                SELECT %s, %s, %s, %s, %s, {vec_str}::VECTOR(FLOAT, {VECTOR_DIM})""",
            (doc["id"], doc["title"], doc["category"], doc["content"], doc["source"]),
        )


def main():
    docs = load_corpus()
    model = SentenceTransformer(MODEL_NAME)

    conn = get_connection()
    cur = conn.cursor()

    create_table(cur)
    embed_and_insert(cur, docs, model)

    cur.execute("SELECT COUNT(*) FROM documents")
    count = cur.fetchone()[0]
    print(f"Ingested {count} documents into EBIKE_RAG.PUBLIC.documents")

    conn.close()


if __name__ == "__main__":
    main()

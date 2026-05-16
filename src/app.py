import os
import json
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

USE_SNOWFLAKE = os.environ.get("SNOWFLAKE_ACCOUNT_P") and Path.home().joinpath(".snowflake/rsa_key.p8").exists()

EXAMPLE_QUERIES = [
    "Can I ride my ebike on the sidewalk in Boston?",
    "Do I need a helmet?",
    "What bike infrastructure is planned for Hyde Park?",
    "What is the Ride Safe Act?",
    "Are there Bluebikes stations in Hyde Park?",
    "What are the speed limits for ebikes?",
    "How dangerous are ebikes for children?",
]


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_local_data():
    with open(DATA_DIR / "documents" / "corpus.json") as f:
        docs = json.load(f)
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    return docs, embeddings


def retrieve_local(query, model, top_k=5):
    docs, embeddings = load_local_data()
    query_vec = model.encode([query])[0]

    sims = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        d = docs[i]
        results.append((d["title"], d["category"], d["content"], d["source"], float(sims[i])))
    return results


def retrieve_snowflake(query, model, top_k=5):
    from connect import get_connection

    embedding = model.encode([query])[0]
    vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT title, category, content, source,
            VECTOR_COSINE_SIMILARITY(embedding, {vec_str}::VECTOR(FLOAT, {VECTOR_DIM})) AS similarity
        FROM documents
        ORDER BY similarity DESC
        LIMIT {top_k}
    """)
    results = cur.fetchall()
    conn.close()
    return results


def retrieve(query, model, top_k=5):
    if USE_SNOWFLAKE:
        return retrieve_snowflake(query, model, top_k)
    return retrieve_local(query, model, top_k)


st.set_page_config(page_title="Boston E-Bike Search", page_icon="⚡", layout="wide")

st.title("Boston E-Bike Vector Search")
st.caption("Semantic search over Boston e-bike regulations, safety data, and infrastructure plans")

if USE_SNOWFLAKE:
    st.sidebar.success("Connected to Snowflake")
else:
    st.sidebar.info("Running with local embeddings")

st.sidebar.markdown("---")
st.sidebar.subheader("Example queries")
for q in EXAMPLE_QUERIES:
    if st.sidebar.button(q, key=q):
        st.session_state["query"] = q

query = st.text_input("Ask a question about e-bikes in Boston",
                      value=st.session_state.get("query", ""))

if query:
    model = load_model()
    results = retrieve(query, model)

    st.markdown("### Retrieved documents")
    for title, category, content, source, sim in results:
        with st.expander(f"**{sim:.3f}** — {title} ({category})"):
            st.write(content)
            st.caption(f"Source: {source}")

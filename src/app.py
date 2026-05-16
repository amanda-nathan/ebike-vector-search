import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import streamlit as st
import requests
from sentence_transformers import SentenceTransformer

from connect import get_connection

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"

SYSTEM_PROMPT = """You are a Boston e-bike expert. Answer questions using ONLY the provided context.
If the context doesn't contain enough information, say so. Cite sources when possible."""

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


def retrieve(query, model, top_k=5):
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


def generate(query, docs):
    context = "\n\n".join(
        f"[{title}] (Source: {source})\n{content}"
        for title, _, content, source, _ in docs
    )

    prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
        }, timeout=120)
        return resp.json()["response"]
    except requests.ConnectionError:
        return "Ollama is not running. Start it with `ollama serve`."


st.set_page_config(page_title="Boston E-Bike Search", page_icon="⚡", layout="wide")

st.title("Boston E-Bike Vector Search")
st.caption("Semantic search over Boston e-bike regulations, safety data, and infrastructure plans")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Example queries")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=q):
            st.session_state["query"] = q

with col1:
    query = st.text_input("Ask a question about e-bikes in Boston",
                          value=st.session_state.get("query", ""))

    use_rag = st.toggle("Generate answer with LLM", value=True)

    if query:
        model = load_model()
        results = retrieve(query, model)

        if use_rag:
            with st.spinner("Generating answer..."):
                answer = generate(query, results)
            st.markdown("### Answer")
            st.write(answer)
            st.divider()

        st.markdown("### Retrieved documents")
        for title, category, content, source, sim in results:
            with st.expander(f"[{sim:.3f}] {title} ({category})"):
                st.write(content)
                st.caption(f"Source: {source}")

# E-Bike Vector Search

Semantic search and retrieval-augmented generation (RAG) over Boston e-bike regulations, safety data, and infrastructure plans.

**[Live demo](https://ebike-vector-search.streamlit.app)** — always works, no signup needed

> The Snowflake trial backend expires June 15, 2026. After that, the app automatically falls back to local embeddings (same search results, computed on Streamlit's servers instead of Snowflake). The live demo never goes down.

## Architecture

```
                         ┌─────────────────────────────────┐
                         │         Snowflake               │
corpus.json ──embed──►   │  VECTOR(FLOAT, 384) column      │
                         │  VECTOR_COSINE_SIMILARITY()      │
                         └────────────┬────────────────────┘
                                      │ top-k documents
user query ──embed──► similarity ─────┘
                                      │
                                      ▼
                              ┌────────────────┐
                              │  LLM (Mistral) │
                              │  context + query│
                              └───────┬────────┘
                                      │
                                      ▼
                                   answer
```

Two deployment modes:

| Component | Local (this repo) | Snowflake Enterprise |
|-----------|-------------------|---------------------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | `SNOWFLAKE.CORTEX.EMBED_TEXT_768('e2e-base', text)` |
| Vector storage | Snowflake `VECTOR(FLOAT, 384)` | Same |
| Similarity search | `VECTOR_COSINE_SIMILARITY()` | Same |
| LLM generation | Ollama (Mistral 7B, local) | `SNOWFLAKE.CORTEX.COMPLETE('mistral-large', prompt)` |

The local version runs the full RAG pipeline at zero cost. On a Snowflake Enterprise account, the entire pipeline collapses into SQL — no external services needed.

## What's in the corpus

21 documents sourced from Massachusetts state law, Boston city government, MassDOT, Vision Zero, CPSC, and news reporting:

| Category | Documents | Topics |
|----------|-----------|--------|
| Regulation | 5 | E-bike classification (Class 1/2/3), riding rules, speed limits, helmets |
| Safety | 3 | National injury statistics, child injuries, Boston crash data |
| Infrastructure | 7 | Hyde Park corridor, JP/Centre St, Better Bike Lanes, Go Boston 2030, Bluebikes, MassDOT plan |
| Legislation | 3 | Ride Safe Act (S.3077), city council ordinances |
| Policy | 1 | MassCEC e-bike incentive program |

## Setup options

### Option A: Live demo (no setup)

Visit the [Streamlit app](https://ebike-vector-search.streamlit.app). Runs semantic search over pre-computed embeddings. No accounts needed.

### Option B: Local with Snowflake

Full pipeline with vector storage in Snowflake and local LLM generation.

**1. Snowflake account**

Create a [free trial](https://signup.snowflake.com/) (30 days, $400 credit). The vector storage and similarity search work on trial accounts.

**2. Key pair authentication**

Snowflake trial accounts enforce MFA, which blocks password-based programmatic access. Key pair auth bypasses this.

```bash
mkdir -p ~/.snowflake
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -nocrypt -out ~/.snowflake/rsa_key.p8
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout -out ~/.snowflake/rsa_key.pub
```

Extract the public key body (no headers):

```bash
grep -v "^-" ~/.snowflake/rsa_key.pub | tr -d '\n'
```

Register it in the Snowflake web UI (Worksheets):

```sql
ALTER USER your_username SET RSA_PUBLIC_KEY='paste_key_here';
```

**3. Environment variables**

```bash
export SNOWFLAKE_ACCOUNT_P=ABCDEFG-XY12345   # from your Snowflake URL
export SNOWFLAKE_USER_P=your_username
```

**4. Install and run**

```bash
uv sync
cd src
uv run python ingest.py    # embed docs → store in Snowflake
uv run python search.py "Can I ride my ebike on the sidewalk?"
```

**5. RAG with Ollama (optional)**

```bash
brew install ollama
ollama pull mistral:latest
ollama serve &
uv run python rag.py "What is the Ride Safe Act?"
```

### Option C: Local without Snowflake

The Streamlit app falls back to numpy cosine similarity over pre-computed embeddings when no Snowflake credentials are found.

```bash
uv sync
uv run streamlit run streamlit_app.py
```

### Option D: Streamlit Cloud with Snowflake

Deploy to [Streamlit Community Cloud](https://share.streamlit.io) (free) with Snowflake backend:

1. Fork this repo
2. Deploy on share.streamlit.io (main file: `streamlit_app.py`)
3. Add secrets in app Settings → Secrets:

```toml
[snowflake]
account = "YOUR_ACCOUNT_ID"
user = "YOUR_USERNAME"
private_key = '''
-----BEGIN PRIVATE KEY-----
YOUR_KEY_HERE
-----END PRIVATE KEY-----
'''
```

## Project structure

```
ebike-vector-search/
├── streamlit_app.py              Web UI (works with or without Snowflake)
├── src/
│   ├── connect.py                Snowflake connection via RSA key pair
│   ├── ingest.py                 Embed corpus and load vectors into Snowflake
│   ├── search.py                 CLI semantic search
│   ├── rag.py                    CLI retrieve + generate (Ollama)
│   └── app.py                    Streamlit app (local dev version)
├── data/
│   ├── documents/
│   │   └── corpus.json           21 sourced documents
│   └── embeddings.npy            Pre-computed 384-dim vectors
├── .streamlit/
│   └── secrets.toml.example      Template for Snowflake credentials
├── RUNBOOK.md                    Step-by-step with expected output
├── .env.example
└── pyproject.toml
```

## Tech stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Vector database | Snowflake `VECTOR` type | Native vector storage + similarity search in SQL, no separate vector DB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 384-dim, fast, runs on CPU, Apache 2.0 license |
| LLM | Ollama + Mistral 7B | Local, free, same model family as Snowflake Cortex |
| Web UI | Streamlit | One-file deployment, native Snowflake integration |
| Auth | RSA key pair (PKCS8) | Bypasses MFA for programmatic access, standard Snowflake pattern |
| Package manager | uv | Fast, deterministic lockfile |
| Language | Python 3.13 | |

## How I built this

Step-by-step log of what I actually ran to create this project from scratch.

**1. Created Snowflake trial account**

Signed up at [signup.snowflake.com](https://signup.snowflake.com/). Got account ID `EVXCFCC-PX70116`. Trial gives 30 days and $400 compute credit.

**2. Generated RSA key pair for programmatic access**

Snowflake trial enforces MFA on password login, which blocks the Python connector. Key pair auth is the workaround.

```bash
mkdir -p ~/.snowflake
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -nocrypt -out ~/.snowflake/rsa_key.p8
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout -out ~/.snowflake/rsa_key.pub
```

Then registered the public key in the Snowflake web UI (Worksheets tab):

```sql
ALTER USER ahattaway SET RSA_PUBLIC_KEY='MIIBIjANBgkqh...';
```

**3. Set environment variables**

Added to `~/.zshrc`:

```bash
export SNOWFLAKE_ACCOUNT_P=EVXCFCC-PX70116
export SNOWFLAKE_USER_P=ahattaway
```

**4. Initialized the project**

```bash
mkdir ebike-vector-search && cd ebike-vector-search
uv init
uv add snowflake-connector-python sentence-transformers requests cryptography streamlit numpy
```

**5. Tested the Snowflake connection**

```python
conn = snowflake.connector.connect(account=..., user=..., private_key=pkb)
cur = conn.cursor()
cur.execute('SELECT CURRENT_VERSION()')
# → ('10.17.102',)
```

**6. Discovered Cortex is blocked on trial accounts**

```sql
SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('e2e-base', 'test');
-- ERROR: AI function EMBED_TEXT_768 is not available for trial accounts.
```

This meant I couldn't use Snowflake's built-in embedding or LLM functions. Solution: embed locally with sentence-transformers, generate locally with Ollama. The `VECTOR` type and `VECTOR_COSINE_SIMILARITY()` still work on trial.

**7. Verified vector operations work**

```sql
CREATE TABLE test_vec (id INT, embedding VECTOR(FLOAT, 3));
INSERT INTO test_vec SELECT 1, [1.0, 2.0, 3.0]::VECTOR(FLOAT, 3);
SELECT VECTOR_COSINE_SIMILARITY(embedding, [1.0, 0.0, 0.0]::VECTOR(FLOAT, 3)) FROM test_vec;
-- → 0.267
```

**8. Created the document corpus**

Researched Boston e-bike regulations, safety stats, infrastructure plans from official sources (Mass.gov, Boston.gov, CPSC, MassDOT). Wrote 21 documents into `data/documents/corpus.json` with title, category, content, and source fields.

**9. Built the ingestion pipeline**

```bash
cd src && uv run python ingest.py
# Ingested 21 documents into EBIKE_RAG.PUBLIC.documents
```

This runs each document through `all-MiniLM-L6-v2` (produces a 384-float vector per doc), then inserts into Snowflake using the `[...]::VECTOR(FLOAT, 384)` cast syntax.

**10. Tested semantic search**

```bash
uv run python search.py "Can I ride my ebike on the sidewalk?"
# [0.686] Where E-Bikes Can Ride in Boston (regulation)
# [0.564] Motorized Bicycle vs E-Bike Distinction (regulation)
# [0.515] Boston City Council E-Bike Ordinances 2026 (legislation)
```

Top result is correct — the document about where e-bikes can/can't ride, which includes the sidewalk ban.

**11. Added RAG with Ollama**

Already had Ollama installed with `mistral:latest` (same model family Snowflake Cortex uses).

```bash
uv run python rag.py "Is it legal to ride an ebike on the sidewalk in Boston?"
# Retrieved 4 relevant documents:
#   [0.748] Where E-Bikes Can Ride in Boston
#   [0.616] Boston City Council E-Bike Ordinances 2026
# --- Answer ---
# No, it is not legal to ride an e-bike on the sidewalk in Boston.
# The Boston.gov source states that e-bikes are prohibited on sidewalks.
```

**12. Built the Streamlit UI**

```bash
uv run streamlit run streamlit_app.py
# Local URL: http://localhost:8501
```

Added auto-detection: if Snowflake credentials exist, queries go to Snowflake. Otherwise falls back to numpy cosine similarity over pre-computed embeddings (`data/embeddings.npy`).

**13. Deployed to Streamlit Community Cloud**

- Connected GitHub repo at [share.streamlit.io](https://share.streamlit.io)
- Main file: `streamlit_app.py`
- Added Snowflake private key in Settings → Secrets (TOML format)
- Live at [ebike-vector-search.streamlit.app](https://ebike-vector-search.streamlit.app)

## How it works

### 1. Embedding

Each document $d_i$ is mapped to a dense vector $\mathbf{e}_i \in \mathbb{R}^{384}$ by a pre-trained transformer encoder:

$$\mathbf{e}_i = f_\theta(d_i)$$

The model (`all-MiniLM-L6-v2`) is a 6-layer distilled BERT trained on 1B+ sentence pairs with a contrastive objective. Documents with similar meaning cluster together in the embedding space regardless of word overlap.

### 2. Storage

Vectors are stored in a Snowflake column of type `VECTOR(FLOAT, 384)`:

```sql
CREATE TABLE documents (
    id STRING,
    title STRING,
    content STRING,
    embedding VECTOR(FLOAT, 384)
);
```

This is not an approximation (no HNSW, no IVF). Snowflake computes exact distances over the full vector set. For 21 documents this is instantaneous; the architecture scales to millions with Snowflake's compute elasticity.

### 3. Retrieval (cosine similarity)

Given a query $q$, embed it to $\mathbf{e}_q = f_\theta(q)$ and find the $k$ documents with highest cosine similarity:

$$\text{sim}(\mathbf{e}_q, \mathbf{e}_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_i}{\|\mathbf{e}_q\| \, \|\mathbf{e}_i\|}$$

Cosine similarity ranges from $-1$ (opposite meaning) to $1$ (identical meaning). It is invariant to vector magnitude — only the direction matters. Two documents about "helmet laws" will have $\text{sim} \approx 0.7$+ regardless of document length.

In SQL:

```sql
SELECT title, content,
    VECTOR_COSINE_SIMILARITY(embedding, <query_vector>::VECTOR(FLOAT, 384)) AS sim
FROM documents
ORDER BY sim DESC
LIMIT 5;
```

### 4. Generation (RAG)

The top-$k$ retrieved documents are concatenated into a context window and passed to an LLM with the original query:

$$\text{answer} = \text{LLM}\left(\text{system prompt},\ \bigoplus_{i=1}^{k} d_i,\ q\right)$$

where $\bigoplus$ denotes concatenation. The system prompt constrains the model to answer only from the provided context (grounding), which prevents hallucination.

The Cortex equivalent runs entirely in Snowflake:

```sql
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'mistral-large',
    CONCAT(context, '\n\nQuestion: ', query, '\n\nAnswer:')
) AS answer;
```

### Why RAG instead of fine-tuning

| | RAG | Fine-tuning |
|--|-----|------------|
| Data freshness | Instant (update corpus) | Requires retraining |
| Hallucination | Grounded in retrieved docs | Can still hallucinate |
| Cost | Embedding once + retrieval | GPU hours for training |
| Transparency | Can cite sources | Black box |
| Corpus size | Scales with vector DB | Limited by training budget |

For a domain like Boston e-bike regulations where facts change frequently (new legislation, infrastructure updates), RAG is strictly better than fine-tuning.

### Embedding geometry

The sentence-transformers model maps semantically similar texts to nearby points on a 384-dimensional hypersphere (after L2 normalization). The training objective (Multiple Negatives Ranking Loss) pulls positive pairs together and pushes negatives apart:

$$\mathcal{L} = -\log \frac{e^{\text{sim}(q, d^+) / \tau}}{\sum_{j} e^{\text{sim}(q, d_j) / \tau}}$$

where $\tau$ is a temperature parameter and $d^+$ is the positive (relevant) document. This is the same InfoNCE/contrastive loss used in CLIP and SimCLR.

The result: "Is it legal to ride an ebike on the sidewalk?" maps close to the document about "Where E-Bikes Can Ride in Boston" even though they share minimal lexical overlap. Traditional keyword search (TF-IDF, BM25) would fail here because the query says "sidewalk" and "legal" while the document says "prohibited" and "statewide ban."

## References

1. Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084). EMNLP 2019.

2. Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS 2020.

3. Snowflake Documentation. [Vector Data Type](https://docs.snowflake.com/en/sql-reference/data-types-vector). Snowflake Inc.

4. Snowflake Documentation. [Cortex LLM Functions](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions). Snowflake Inc.

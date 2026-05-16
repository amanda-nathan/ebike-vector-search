# E-Bike Vector Search

Semantic search and retrieval-augmented generation (RAG) over Boston e-bike regulations, safety data, and infrastructure plans. Uses Snowflake as the vector database with cosine similarity search, sentence-transformers for local embeddings, and Ollama for local LLM generation.

## Architecture

```
corpus.json → sentence-transformers (embed) → Snowflake VECTOR(FLOAT, 384)
                                                        ↓
user query → embed → VECTOR_COSINE_SIMILARITY → top-k docs → Ollama (generate) → answer
```

On a Snowflake Enterprise account, the embedding and generation steps can be replaced with Cortex functions (`EMBED_TEXT_768` and `COMPLETE`) so the entire pipeline runs inside Snowflake with no external dependencies.

## What's in the corpus

21 documents covering:
- Massachusetts e-bike classification law (Class 1, 2, 3 status)
- Boston riding rules and traffic laws
- Speed limits and helmet requirements
- Hyde Park Avenue Multimodal Corridor Project
- JP Centre/South Transportation Action Plan
- Better Bike Lanes program results (2022-2024)
- Go Boston 2030 and MassDOT Bicycle Plan
- National and Boston e-bike injury statistics
- The Ride Safe Act (S.3077, May 2026)
- Bluebikes e-bike share expansion
- MassCEC e-bike incentive program

## Setup

### 1. Snowflake account

Create a [Snowflake trial account](https://signup.snowflake.com/) or use an existing one.

### 2. Key pair authentication

```bash
mkdir -p ~/.snowflake
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -nocrypt -out ~/.snowflake/rsa_key.p8
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout -out ~/.snowflake/rsa_key.pub
```

Register the public key in Snowflake:

```sql
ALTER USER your_username SET RSA_PUBLIC_KEY='<contents of rsa_key.pub without headers>';
```

### 3. Environment variables

```bash
export SNOWFLAKE_ACCOUNT_P=YOUR_ACCOUNT_ID    # e.g. ABCDEFG-XY12345
export SNOWFLAKE_USER_P=YOUR_USERNAME
```

### 4. Install dependencies

```bash
uv sync
```

### 5. Install Ollama

```bash
brew install ollama
ollama pull mistral:latest
ollama serve
```

## Usage

```bash
cd src

uv run python ingest.py
uv run python search.py "Can I ride my ebike on the sidewalk?"
uv run python rag.py "What bike infrastructure is planned for Hyde Park?"
```

See [RUNBOOK.md](RUNBOOK.md) for detailed examples with expected output.

## Project structure

```
ebike-vector-search/
├── src/
│   ├── connect.py      Snowflake connection via key pair auth
│   ├── ingest.py       Embed documents and load into Snowflake
│   ├── search.py       Semantic similarity search
│   └── rag.py          Retrieve + generate with Ollama (shows Cortex SQL equivalent)
├── data/
│   └── documents/
│       └── corpus.json 21 e-bike documents with sources
├── RUNBOOK.md          Step-by-step with expected output
├── .env.example        Required environment variables
└── pyproject.toml
```

## Snowflake features demonstrated

- `VECTOR(FLOAT, 384)` column type for embedding storage
- `VECTOR_COSINE_SIMILARITY()` for nearest-neighbor retrieval
- Key pair authentication for programmatic access
- Database/schema/warehouse setup via Python connector
- Cortex `COMPLETE` and `EMBED_TEXT_768` SQL shown for Enterprise deployment

## Tech stack

Python 3.13, Snowflake, sentence-transformers (all-MiniLM-L6-v2), Ollama (Mistral 7B), uv

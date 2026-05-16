# Runbook

## Prerequisites

- Snowflake account with key pair authentication configured
- Environment variables: `SNOWFLAKE_ACCOUNT_P`, `SNOWFLAKE_USER_P`
- Private key at `~/.snowflake/rsa_key.p8`
- Ollama running locally with `mistral:latest`

## 1. Install dependencies

```bash
uv sync
```

## 2. Start Ollama (if not running)

```bash
ollama serve &
```

## 3. Ingest documents

Embeds all documents locally and stores vectors in Snowflake.

```bash
cd src && uv run python ingest.py
```

```
# Ingested 21 documents into EBIKE_RAG.PUBLIC.documents
```

## 4. Semantic search

```bash
uv run python search.py "Do I need a helmet on my ebike?"
```

```
# Query: Do I need a helmet on my ebike?
#
# [0.721] Massachusetts Helmet Requirements (safety)
#   Current Massachusetts law: riders under age 17 (16 and younger) MUST wear...
#   Source: Boston.gov - Bike Laws in Boston
#
# [0.543] The Ride Safe Act (S.3077) - Speed-Based Classification (legislation)
#   Filed May 4, 2026 by Governor Maura Healey. First-in-the-nation speed-based...
#   Source: Mass.gov - Governor Healey Files Ride Safe Act
# ...
```

## 5. Full RAG (retrieve + generate)

```bash
uv run python rag.py "What are the speed limits for ebikes in Massachusetts?"
```

```
# Query: What are the speed limits for ebikes in Massachusetts?
#
# Retrieved 4 relevant documents:
#   [0.712] E-Bike Speed Limits in Massachusetts
#   [0.583] Massachusetts E-Bike Classification Law
#   [0.521] The Ride Safe Act (S.3077) - Speed-Based Classification
#   [0.465] Motorized Bicycle vs E-Bike Distinction
#
# --- Answer ---
#
# Class 1 and Class 2 e-bikes in Massachusetts have motor assistance that
# ceases at 20 mph. Riders can exceed 20 mph under their own pedal power.
# Boston's default city speed limit is 25 mph, with most residential streets
# at 20 mph. The proposed Ride Safe Act (S.3077) would create speed-based
# tiers: Tier 0 up to 20 mph, Tier 1 21-30 mph, Tier 2 31-40 mph.
# (Source: Mass.gov - Massachusetts law about bicycles)
#
# --- Cortex SQL (production) ---
#
# SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', CONCAT(...)) AS answer;
```

## 6. Other example queries

```bash
uv run python rag.py "Is it legal to ride an ebike on the sidewalk in Boston?"
uv run python rag.py "What is the Ride Safe Act?"
uv run python rag.py "Are there Bluebikes stations in Hyde Park?"
uv run python search.py "bike lanes Jamaica Plain"
uv run python search.py "children ebike injuries"
```

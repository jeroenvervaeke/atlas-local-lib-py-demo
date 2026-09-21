# Plot Search — Atlas Local + LangChain

Semantic movie search over `sample_mflix.embedded_movies` running on a local
MongoDB Atlas (MongoD + Search) in Docker, wired via LangChain and Voyage AI.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Docker](https://docs.docker.com/) (running daemon)

## Run

```bash
uv run marimo edit ./demo.py
```

Dependencies (marimo, atlas-local-lib-py, pymongo, langchain…) are resolved
automatically from `../pyproject.toml` / this workspace; Docker pulls the
`mongodb-atlas-local` image on first boot.

Seeds all MongoDB sample datasets, including `sample_mflix` (3,483 movies).

## Configuration

Copy `.env` if absent and set your Voyage API key:

```bash
cp .env.example .env
# .env
VOYAGE_API_KEY=pa-...
```

Optional overrides: `VOYAGE_BASE_URL` (default `https://api.voyageai.com/v1`),
`VOYAGE_MODEL` (default `voyage-3-large`). Restart the marimo kernel after
editing `.env`.

## How it works

1. `LocalDeployment.get_or_create(load_sample_data=True)` boots MongoD + Atlas
   Search locally and loads the sample datasets.
2. The notebook creates a `vectorSearch` index on
   `plot_embedding_voyage_3_large` (2,048-d, cosine, L2-normalized).
3. Type a plot description → embedded with Voyage `voyage-3-large` at the
   2,048-d matryoshka slice → `$vectorSearch` top-5 → results table.

> ⚠️ The stored embeddings are BSON `binData` (2-byte header + 2,048 `float32`).
> Voyage's API defaults to 1,024-d, so the notebook requests
> `output_dimension: 2048` via a small LangChain embeddings shim.

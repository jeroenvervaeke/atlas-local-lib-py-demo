import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # 🎬 Plot Search — Atlas Local + LangChain

    Semantic search over **MongoDB Atlas Local** with the `sample_mflix.embedded_movies` dataset and **LangChain**.

    **The idea:** describe a movie plot in your own words (e.g. *"a shy teenager discovers he has superpowers"*)
    and get back the movies whose plots are *semantically* closest — no keyword matching.

    | piece | what it does |
    |---|---|
    | `atlas_local.LocalDeployment` | runs a real MongoD + Atlas **Search** locally (Docker) and seeds MongoDB's sample datasets |
    | `sample_mflix.embedded_movies` | 3,483 movies with `plot`, `fullplot` and pre-computed `plot_embedding_voyage_3_large` vectors |
    | Voyage AI `voyage-3-large` | embedding model (2,048-d, OpenAI-compatible endpoint) turning your query into a vector |
    | Atlas Vector Search | `$vectorSearch` — nearest-neighbour lookup inside the collection |
    | LangChain | `OpenAIEmbeddings` client + `MongoDBAtlasVectorSearch` retriever |

    > **Stored embeddings:** each `plot_embedding_voyage_3_large` is BSON `binData` = a 2-byte header
    > followed by 2,048 little-endian `float32` (2,048 × 4 + 2 = 8,194 bytes), **L2-normalized**, so cosine
    > similarity applies directly.
    """)
    return


@app.cell
def _():
    from atlas_local import LocalDeployment
    from dotenv import load_dotenv
    load_dotenv()

    import marimo as mo
    import os
    import struct
    from pymongo import MongoClient

    return LocalDeployment, MongoClient, mo, os


app._unparsable_cell(
    """
    mo.md(r\"\"\"
    ## 1 · Start the local Atlas cluster

    `LocalDeployment.get_or_create(...)` does the heavy lifting:

    * pulls / starts the `mongodb-atlas-local` Docker image (MongoD + Atlas Search)
    * with `load_sample_data=True` it seeds **all** of MongoDB's sample datasets (mflix among them)
    * returns a handle we use to resolve the connection string
    \"\"\")
    """,
    column=None, disabled=False, hide_code=True, name="md-start"
)


@app.cell
def _(LocalDeployment):
    deployment = LocalDeployment.get_or_create(name="rag-demo", load_sample_data=True)

    return (deployment,)


app._unparsable_cell(
    """
    mo.md(r\"\"\"
    ## 2 · Connect to the cluster

    `get_connection_string()` returns a standard `mongodb://` URI pointing at the container.
    We connect with **PyMongo** and cache a reference to the `sample_mflix.embedded_movies`
    collection — the mflix variant that ships with pre-computed `plot_embedding*` fields.

    `INDEX_NAME` is the Atlas Search index we build in step 4.
    \"\"\")
    """,
    column=None, disabled=False, hide_code=True, name="md-connect"
)


@app.cell
def connect(MongoClient, deployment, mo):
    uri = deployment.connection_string()

    client = MongoClient(uri)
    db = client["sample_mflix"]
    movies = db["embedded_movies"]
    INDEX_NAME = "plot_voyage3_idx"

    n_movies = movies.estimated_document_count()
    one = movies.find_one({}, {"title": 1})
    mo.vstack([
        mo.md(f"Connected to **`{db.name}.{movies.name}`** — `{n_movies}` movies"),
        mo.md(f"sample doc: **{one['title']}**"),
    ])
    return INDEX_NAME, movies


app._unparsable_cell(
    """
    mo.md(r\"\"\"## 3 · Embed the query with Voyage AI

    Vector search only compares vectors of the **same dimension**. Our index lives on the
    2,048-d embeddings of `plot_embedding_voyage_3_large`, so the query must be embedded
    to 2,048-d as well.

    Voyage's `voyage-3-large` is a **Matryoshka** model: can return different widths on request.

        POST /v1/embeddings
        { \"model\": \"voyage-3-large\", \"input\": \"…\", \"output_dimension\": 2048 }

    Default output is 1,024-d — we must pass **`output_dimension: 2048`** explicitly, which is
    why a small shim (`Voyage2048Embeddings`) wraps Voyage's OpenAI-compatible endpoint instead
    of stock `OpenAIEmbeddings`.

    | env var | default | purpose |
    |---|---|---|
    | `VOYAGE_API_KEY` | — | **required** API key |
    | `VOYAGE_BASE_URL` | `https://api.voyageai.com/v1` | endpoint (swap for a third-party proxy) |
    | `VOYAGE_MODEL` | `voyage-3-large` | Voyage model name (must support 2,048-d) |

    > Every stored vector is a 2-byte header + 2,048 × `float32`, L2-normalized ≈ cosine.
    \"\"\")
    """,
    column=None, disabled=False, hide_code=True, name="md-embeddings"
)


@app.cell
def embeddings(mo, os):
    from langchain_core.embeddings import Embeddings
    from openai import OpenAI

    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    embeddings = None
    if not VOYAGE_API_KEY:
        mo.stop("🔑 **Set `VOYAGE_API_KEY` in `.env`, restart the kernel, and run this cell.**")


    class Voyage2048Embeddings(Embeddings):
        """LangChain shim for Voyage's OpenAI-compatible endpoint.

        Requests the full 2,048-d matryoshka slice of voyage-3-large so query
        vectors match the stored plot_embedding_voyage_3_large (2,048-d).
        (The API default is 1,024-d, which would not match the vector index.)
        """

        def __init__(self, api_key, base_url="https://api.voyageai.com/v1", model="voyage-3-large"):
            self.model = model
            self._client = OpenAI(base_url=base_url, api_key=api_key)

        def _embed(self, texts):
            res = self._client.embeddings.create(
                model=self.model,
                input=texts,
                extra_body={"output_dimension": 2048},
            )
            return [d.embedding for d in res.data]

        def embed_query(self, text):
            return self._embed([text])[0]

        def embed_documents(self, texts):
            return self._embed(texts)


    embeddings = Voyage2048Embeddings(
        api_key=VOYAGE_API_KEY,
        base_url=os.getenv("VOYAGE_BASE_URL", "https://api.voyageai.com/v1"),
        model=os.getenv("VOYAGE_MODEL", "voyage-3-large"),
    )

    # sanity probe: dimension must equal the vector index dims
    probe = embeddings.embed_query("a robot learns to feel emotions")
    assert len(probe) == 2048, f"expected 2048-d output, got {len(probe)}"
    mo.md("✅ Query embedding: **2048-d** — matches the vector index")
    return (embeddings,)


app._unparsable_cell(
    """
    mo.md(r\"\"\"
    ## 4 · Ensure the Atlas Search vector index

    `$vectorSearch` needs an **Atlas Search `vectorSearch` index** mapping the field into vector space:

    * **numDimensions: 2048** — matches `Voyage 3 Large`
    * **similarity: cosine** — right for L2-normalized vectors

    Index creation is asynchronous (pending → ready); we poll until it is.
    \"\"\")
    """,
    column=None, disabled=False, hide_code=True, name="md-index"
)


app._unparsable_cell(
    """
    import time

    def ensure_vector_index(collection, name):
        \"\"\"Idempotent: create the vector index if missing, poll until READY.\"\"\"
        found = list(collection.list_search_indexes(name))
        if not any(i.get(\"status\") in (\"READY\", \"ACTIVE\") for i in found):
            if not found:
                collection.create_search_index({
                    \"name\": name,
                    \"type\": \"vectorSearch\",
                    \"definition\": {\"fields\": [
                        {\"type\": \"vector\",
                         \"path\": \"plot_embedding_voyage_3_large\",
                         \"numDimensions\": 2048,
                         \"similarity\": \"cosine\"}
                    ]},
                })
            for _ in range(60):
                ready = list(collection.list_search_indexes(name))
                if ready and ready[0].get(\"status\") in (\"READY\", \"ACTIVE\"):
                    break
                time.sleep(1)
            else:
                raise RuntimeError(\"vector index did not become READY\")

    index_status = ensure_vector_index(movies, INDEX_NAME)
    mo.md(f\"🟢 Vector index **`{INDEX_NAME}`** → `{index_status}`\")
    """,
    name="ensure-index"
)


app._unparsable_cell(
    """
    mo.md(r\"\"\"
    ## 5 · Wire it together with LangChain

    `MongoDBAtlasVectorSearch` (from `langchain-mongodb`) wraps the collection, the embedding
    client and the index into one retriever. It embeds your query, runs `$vectorSearch`,
    and hands back LangChain `Document`s — scored 0…1 (cosine).
    \"\"\")
    """,
    column=None, disabled=False, hide_code=True, name="md-retriever"
)


@app.cell
def retriever(INDEX_NAME, embeddings, mo, movies):
    from langchain_mongodb import MongoDBAtlasVectorSearch

    if embeddings is None:
        mo.stop("Run the embeddings cell first (step 3).")

    vector_store = MongoDBAtlasVectorSearch(
        collection=movies,
        embedding=embeddings,
        index_name=INDEX_NAME,
        text_key="plot",                        # page_content = the plot text
        embedding_key="plot_embedding_voyage_3_large",
    )

    hit = vector_store.similarity_search_with_score(
        "a lone survivor scavenges a post-apocalyptic wasteland", k=1
    )
    if hit:
        doc, score = hit[0]
        mo.md(f"Sanity probe → **{doc.metadata.get('title')}** (match **{score:.1%}**)")
    else:
        mo.status.warning("Sanity probe returned no hits — check index state.")
    return (vector_store,)


app._unparsable_cell(
    """
    mo.md(r\"\"\"## 6 · Search the movies — as you type

    Type a plot description; the cell below reacts to each (debounced) keystroke,
    embeds the text with Voyage, runs the top-5 `$vectorSearch`, and renders the
    closest matches as a table. No keywords, no button.
    \"\"\")
    """,
    column=None, disabled=False, hide_code=True, name="md-ui"
)


app._unparsable_cell(
    r"""
    query = mo.ui.text_area(
        label="Describe a movie plot",
        placeholder="A computer hacker discovers the world he lives in is a simulation…",
        debounce=300,
    )
    mo.hstack([query])
    """,
    name="search-ui"
)


@app.cell
def search_results(mo, query, vector_store):
    cards = []
    if query.value.strip():
        cards = vector_store.similarity_search_with_score(query.value, k=5)

    rows = [
        {
            "#": rank,
            "match": f"{score:.1%}",
            "title": doc.metadata.get("title", "…"),
            "year": doc.metadata.get("year"),
            "genres": ", ".join(doc.metadata.get("genres") or []),
            "imdb": (doc.metadata.get("imdb") or {}).get("rating"),
            "plot": (doc.page_content or "…")[:90],
        }
        for rank, (doc, score) in enumerate(cards, start=1)
    ]

    if rows:
        out = mo.ui.table(rows, page_size=5)
    elif query.value.strip():
        out = mo.md("😕 No close matches — try another plot description.")
    else:
        out = mo.md("👆 Type a plot description above — matches appear as you type.")
    out
    return


if __name__ == "__main__":
    app.run()

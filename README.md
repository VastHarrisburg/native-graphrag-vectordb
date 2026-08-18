# Native GraphRAG VectorDB

A working hybrid retrieval prototype that combines native Rust cosine-similarity
search with document-specific Python knowledge graphs. Documents are ingested
once, identified by their content hash, and queried through one CLI.

## What works

- Stable `doc_NNN` identifiers across chunks, vectors, search results, graphs,
  sources, and citations
- SHA-256 duplicate detection and update-in-place for changed documents
- Sentence-based chunks with overlap
- Sentence Transformer embeddings, plus a deterministic offline hash backend
- Cosine-similarity search in Rust
- One separately stored NetworkX-compatible `MultiDiGraph` per document
- Graph selection from each document's maximum semantic chunk score
- Query entity and acronym matching followed by bounded two-hop BFS
- Semantic/graph evidence merging and `chunk_id` deduplication
- Optional grounded OpenAI answer generation with validated citation IDs
- Safe extractive and insufficient-evidence fallback without an API key
- Document removal across every store
- Automated lifecycle/traversal tests and a semantic-versus-hybrid benchmark

## Architecture

```text
Ingest
document -> SHA-256 -> registry lookup -> chunks -> embeddings
         -> per-document entity/relationship graph -> registry

Query
question -> query embedding -> Rust vector search -> top document IDs
         -> entity matching -> each selected graph -> two-hop traversal
         -> deduplicate combined evidence -> grounded answer + citations
```

Rust owns vector scoring and the public CLI. Python owns chunking, embedding
generation, graph persistence/traversal, evidence assembly, and optional model
calls. JSON remains the storage layer so every stage is inspectable.

## Requirements

- Rust 1.85 or newer
- Python 3.10 or newer
- The dependencies in `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cargo build
```

For OpenAI-backed graph extraction or answer generation:

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env
```

`OPENAI_MODEL` is configurable in `.env`. The integration uses the Responses
API with a Pydantic structured output and validates model citations against the
evidence supplied to the model.

## Upload and ingest documents

The normal quality path uses Sentence Transformers for embeddings and the
local heuristic graph builder. Use `upload` to add a document; `ingest` remains
available as an equivalent command for pipeline-oriented workflows:

```bash
cargo run -- upload data/backup.txt
cargo run -- upload data/sample_client_rollout.txt
```

For a fast, deterministic offline run:

```bash
cargo run -- upload data/backup.txt --embedding-backend hash
```

To use model-backed entity and relationship extraction during ingestion:

```bash
cargo run -- upload data/backup.txt --graph-backend openai
```

Useful options are `--chunk-size 5`, `--overlap 1`,
`--embedding-backend auto|sentence-transformers|hash`, and
`--graph-backend heuristic|openai`.
Use `--force` to deliberately rebuild an unchanged document.

Ingesting identical content returns `"status": "reused"`. If the same source
path changes, its existing document ID is retained while its chunks, vectors,
and graph are rebuilt.

## Query

One command runs embedding, native Rust vector search, semantic graph
selection, two-hop traversal, evidence merging, and answer generation:

```bash
cargo run -- query "How did Helios reduce traffic to Orion?"
```

If `OPENAI_API_KEY` is absent, the command uses a source-grounded extractive
fallback. Force that behavior with `--no-llm`:

```bash
cargo run -- query \
  "How were the Atlas Gateway and Meridian Analytics Engine connected?" \
  --no-llm
```

When the index was ingested with the hash backend, use the same backend for a
fully explicit run:

```bash
cargo run -- query "How long is the Helios fetch timeout?" \
  --embedding-backend hash --no-llm
```

The JSON result contains `answer`, validated `citations`, retrieval counts,
semantic results, per-document graph results, deduplicated combined context,
sources, and vector/graph/total timing. Unsupported questions return an
insufficient-evidence answer with no citations.

## Storage

```text
data/
├── graph_registry.json
├── index_metadata.json
├── chunks.json
├── vectors.json
└── graphs/
    ├── doc_001/
    │   ├── nodes.json
    │   └── edges.json
    └── doc_002/
        ├── nodes.json
        └── edges.json
```

A chunk has one canonical shape:

```json
{
  "chunk_id": "doc_001_chunk_003",
  "document_id": "doc_001",
  "text": "...",
  "chunk_index": 3,
  "source": "data/backup.txt"
}
```

The registry maps content hashes and sources to graph files. Graph node and
edge arrays retain the prototype's `name`, `type`, `source`, `target`, and
`context` fields; evidence edges also carry `evidence_chunk_id`.

## Manage the index

```bash
cargo run -- stats
cargo run -- remove-doc doc_002
```

`remove-doc` removes the document's chunks, vectors, graph files, and registry
entry. It accepts a unique document ID, source path, or document name.

## Tests

```bash
cargo test
python -m unittest discover -s tests -v
```

The tests cover duplicate ingestion, changed documents, removal, stable IDs,
one-hop and two-hop BFS, graph limits, multiple edges, aliases, disconnected
and unknown entities, empty/missing graphs, document selection, evidence
deduplication, and valid source citations.

## Evaluation and benchmark

`evaluation/questions.json` contains direct, one-chunk, multi-chunk,
relationship, two-hop, multi-document, alias, no-entity, and unsupported
questions with expected evidence chunks.

```bash
cargo run -- benchmark evaluation/questions.json --embedding-backend hash
```

The benchmark compares semantic-only with semantic-plus-graph retrieval and
reports evidence recall, precision, vector latency, graph latency, total
latency, returned chunk IDs, and context size. On the checked-in two-document
hash index, the current nine-query fixture measured mean evidence recall of
`0.694` for semantic-only and `0.907` for hybrid retrieval. These small local
numbers demonstrate behavior, not production performance; rerun them on the
target machine and with Sentence Transformer embeddings before publishing
performance claims.

## Project structure

```text
src/          Rust CLI, storage models, vector math, and search
scripts/      Chunking, embeddings, ingestion, and query orchestration
graph/        Registry, extraction, loading, entity matching, and traversal
tests/        Python unit and integration tests
evaluation/   Benchmark questions and expected chunks
data/         Sample documents and generated inspectable stores
```

## Limitations

- JSON storage has no transactions or concurrent writer support.
- Rust search loads and sorts all vectors in memory.
- Heuristic graph extraction favors reproducibility over extraction quality;
  use `--graph-backend openai` for model-backed extraction.
- Alias matching uses exact names, acronyms, token overlap, and conservative
  fuzzy matching rather than an alias model.
- There is no clustering, classifier routing, cross-document graph merge,
  frontend, cloud deployment, or distributed processing in version 1.

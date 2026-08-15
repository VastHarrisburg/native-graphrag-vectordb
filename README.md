# Native GraphRAG VectorDB

An experimental hybrid retrieval project that combines a Rust command-line
interface and cosine-similarity search with Python-based embeddings and graph
extraction.

The repository currently uses JSON files as its storage layer. It is a
prototype for exploring the pieces of a GraphRAG pipeline rather than a
production-ready vector database.

## What is included

- Sentence-based document chunking with configurable overlap
- Embeddings from `sentence-transformers/all-MiniLM-L6-v2`
- JSON-backed chunks, vectors, entities, and relationships
- Cosine-similarity search implemented in Rust
- Experimental entity and relationship extraction with structured OpenAI
  responses
- NetworkX-based graph loading and traversal experiments
- Sample data describing a fictional cache-modernization project

## Project structure

```text
src/       Rust CLI, storage helpers, vector math, and similarity search
scripts/   Python chunking and sentence-transformer embedding pipeline
graph/     Experimental graph extraction, query analysis, and traversal code
data/      Sample documents and generated JSON artifacts
```

## Requirements

- A Rust toolchain that supports Rust 2024 edition (Rust 1.85 or newer)
- Python 3
- `sentence-transformers` for the vector pipeline
- Optional graph dependencies: `networkx`, `python-dotenv`, `pydantic`,
  `langchain`, and `langchain-openai`
- An OpenAI API key only when running the model-backed graph scripts

## Quick start

Clone the repository and enter the project directory:

```bash
git clone https://github.com/VastHarrisburg/native-graphrag-vectordb.git
cd native-graphrag-vectordb
```

Create a Python environment and install the embedding dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install sentence-transformers
```

Check that the Rust project builds:

```bash
cargo check
```

Display statistics for the checked-in sample artifacts:

```bash
cargo run -- stats
```

To try vector search, first place a source document at `data/read.txt`. The
included fictional document can be used as a sample:

```bash
cp data/backup.txt data/read.txt
cargo run -- search-vector \
  "How did the cache reduce database traffic?" 2 1 data/read.txt
```

The search command regenerates `data/chunks.json`, `data/vectors.json`, and
`data/query_vector.json`, then returns the three highest-scoring chunk IDs.
The first embedding run may take longer while the sentence-transformer model
is downloaded.

## Experimental graph pipeline

The files in `graph/` explore extracting entities and directed relationships,
merging them into JSON graph storage, deciding when graph retrieval is useful,
and analyzing a query for traversal direction.

Install the optional dependencies:

```bash
python -m pip install \
  networkx python-dotenv pydantic langchain langchain-openai
```

Create a local `.env` file (it is ignored by Git):

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The graph scripts are under active development and currently contain
prototype defaults, including local project paths and sample input filenames.
Review those defaults before running them in another checkout.

## Current limitations

- Persistence is file-based and does not provide transactions or concurrent
  access controls.
- Vector search loads all embeddings into memory and sorts every result.
- The CLI and graph components are not yet connected into one end-to-end
  GraphRAG query flow.
- Generated sample artifacts are committed for inspection and demonstration.

## Development

Run the Rust compiler checks with:

```bash
cargo check
```

Keep secrets in `.env`; the file is excluded by `.gitignore`.

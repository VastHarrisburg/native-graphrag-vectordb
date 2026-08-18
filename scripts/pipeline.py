from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graph.extraction import build_graph
from graph.registry import GraphRegistry
from graph.retrieval import retrieve_graphs
from scripts.processing import Processing


DATA_PATH = PROJECT_ROOT / "data"
CHUNKS_PATH = DATA_PATH / "chunks.json"
VECTORS_PATH = DATA_PATH / "vectors.json"
QUERY_VECTOR_PATH = DATA_PATH / "query_vector.json"
INDEX_METADATA_PATH = DATA_PATH / "index_metadata.json"


def load_environment():
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def read_json(path: Path, default):
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    Processing.write_json(path, value)


def resolve_source(path_value: str) -> Path:
    path = Path(path_value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"The document does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"The document path is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"The document is empty: {path}")

    return path


def ingest(args):
    started = time.perf_counter()
    source = resolve_source(args.path)
    registry_store = GraphRegistry(PROJECT_ROOT)
    registry = registry_store.load()
    source_name = registry_store.source_name(source)
    content_hash = registry_store.content_hash(source)
    duplicate = registry_store.find_by_hash(registry, content_hash)

    if duplicate and not args.force:
        document_id, entry = duplicate
        return {
            "status": "reused",
            "document_id": document_id,
            "source": entry["source_path"],
            "content_hash": content_hash,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    existing_source = registry_store.find_by_source(registry, source_name)
    document_id = (
        existing_source[0]
        if existing_source
        else registry_store.next_document_id(registry)
    )
    chunks = Processing.read_chunks(
        source,
        args.chunk_size,
        args.overlap,
        document_id=document_id,
        source=source_name,
    )

    stored_chunks = read_json(CHUNKS_PATH, [])
    stored_chunks = [
        chunk
        for chunk in stored_chunks
        if chunk.get("document_id") and chunk.get("document_id") != document_id
    ]
    stored_chunks.extend(chunks)

    vectors, embedding_model = Processing.embed_texts(
        [chunk["text"] for chunk in stored_chunks],
        backend=args.embedding_backend,
    )
    vector_records = [
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "vector": vector,
        }
        for chunk, vector in zip(stored_chunks, vectors)
    ]

    nodes, edges, graph_backend = build_graph(chunks, backend=args.graph_backend)
    graph_directory = DATA_PATH / "graphs" / document_id
    write_json(graph_directory / "nodes.json", nodes)
    write_json(graph_directory / "edges.json", edges)
    write_json(CHUNKS_PATH, stored_chunks)
    write_json(VECTORS_PATH, vector_records)
    write_json(
        INDEX_METADATA_PATH,
        {
            "embedding_model": embedding_model,
            "dimensions": len(vectors[0]) if vectors else 0,
        },
    )

    registry[document_id] = registry_store.make_entry(
        document_id,
        source_name,
        content_hash,
        embedding_model,
        graph_backend,
    )
    registry_store.save(registry)

    return {
        "status": "updated" if existing_source else "ingested",
        "document_id": document_id,
        "source": source_name,
        "content_hash": content_hash,
        "chunks": len(chunks),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "embedding_model": embedding_model,
        "graph_backend": graph_backend,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def embedding_backend_for_query(requested):
    if requested != "auto":
        return requested

    metadata = read_json(INDEX_METADATA_PATH, {})
    return "hash" if metadata.get("embedding_model") == "hash-384" else "auto"


def prepare_query(args):
    query = Processing.convert_query(
        args.query,
        backend=embedding_backend_for_query(args.embedding_backend),
    )
    write_json(QUERY_VECTOR_PATH, query)
    return {
        "query_vector_path": QUERY_VECTOR_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "embedding_model": query["embedding_model"],
    }


def combine_evidence(semantic_results, graph_results, max_context=8, max_characters=12000):
    chunks_by_id = {
        chunk.get("chunk_id", chunk.get("id")): chunk
        for chunk in read_json(CHUNKS_PATH, [])
    }
    evidence = {}

    for result in semantic_results:
        chunk_id = result["chunk_id"]
        evidence[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": result["document_id"],
            "text": result["text"],
            "source": result["source"],
            "semantic_score": result["similarity_score"],
            "graph_distance": None,
            "retrieved_by": ["semantic"],
            "relationships": [],
        }

    for graph_result in graph_results:
        distances = {
            node["name"]: node.get("distance", 2)
            for node in graph_result["nodes"]
            if node.get("type") == "chunk"
        }

        for chunk_id in graph_result["evidence_chunk_ids"]:
            chunk = chunks_by_id.get(chunk_id)

            if not chunk:
                continue

            item = evidence.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "document_id": chunk["document_id"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "semantic_score": None,
                    "graph_distance": distances.get(chunk_id, 2),
                    "retrieved_by": [],
                    "relationships": [],
                },
            )

            if "graph" not in item["retrieved_by"]:
                item["retrieved_by"].append("graph")

            item["graph_distance"] = min(
                item.get("graph_distance") or 2,
                distances.get(chunk_id, 2),
            )
            item["relationships"].extend(
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "context": edge.get("context", ""),
                    "distance": edge.get("distance"),
                }
                for edge in graph_result["edges"]
                if edge.get("evidence_chunk_id") == chunk_id
            )

    ranked = sorted(
        evidence.values(),
        key=lambda item: (
            len(item["retrieved_by"]) == 2,
            item["semantic_score"] is not None,
            item["semantic_score"] or -1.0,
            -(item["graph_distance"] or 99),
        ),
        reverse=True,
    )
    combined = []
    character_count = 0

    for item in ranked:
        if len(combined) >= max_context:
            break
        if combined and character_count + len(item["text"]) > max_characters:
            continue
        combined.append(item)
        character_count += len(item["text"])

    return combined


def extractive_answer(query, combined_context):
    ignored = {
        "and", "are", "did", "does", "for", "from", "how", "the", "this",
        "was", "were", "what", "when", "where", "which", "who", "why", "with",
    }
    query_words = {
        word for word in re.findall(r"[a-z0-9]+", query.casefold())
        if len(word) > 2 and word not in ignored
    }
    candidates = []

    for item in combined_context:
        for sentence in re.split(r"(?<=[.!?])\s+", item["text"]):
            sentence_words = set(re.findall(r"[a-z0-9]+", sentence.casefold()))
            score = len(query_words & sentence_words)

            if score:
                candidates.append((score, item["semantic_score"] or 0.0, sentence, item["chunk_id"]))

    candidates.sort(reverse=True)
    selected = []
    citations = []

    for _, _, sentence, chunk_id in candidates:
        if sentence not in selected:
            selected.append(sentence)
        if chunk_id not in citations:
            citations.append(chunk_id)
        if len(selected) == 3:
            break

    if not selected:
        return {
            "answer": "The stored documents do not contain enough evidence to answer this question.",
            "citations": [],
            "answer_mode": "insufficient-evidence",
        }

    return {
        "answer": " ".join(selected),
        "citations": citations,
        "answer_mode": "extractive-fallback",
    }


def llm_answer(query, combined_context):
    fallback = extractive_answer(query, combined_context)

    if not os.getenv("OPENAI_API_KEY"):
        return fallback

    try:
        from openai import OpenAI
        from pydantic import BaseModel, ConfigDict

        class GroundedAnswer(BaseModel):
            model_config = ConfigDict(extra="forbid")
            answer: str
            citations: list[str]

        client = OpenAI()
        response = client.responses.parse(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied evidence. Cite only supplied "
                        "chunk_id values. If evidence is insufficient, say so and return "
                        "an empty citations list."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": query, "evidence": combined_context},
                        ensure_ascii=False,
                    ),
                },
            ],
            text_format=GroundedAnswer,
        )
        answer = response.output_parsed

        if answer is None:
            raise RuntimeError("The model returned no structured answer")

        allowed = {item["chunk_id"] for item in combined_context}
        valid_citations = [citation for citation in answer.citations if citation in allowed]

        if answer.citations and not valid_citations:
            raise ValueError("The model returned no valid citation IDs")

        return {
            "answer": answer.answer,
            "citations": valid_citations,
            "answer_mode": "openai",
        }
    except Exception as error:
        fallback["llm_error"] = str(error)
        return fallback


def complete_query(args):
    started = time.perf_counter()
    semantic_results = read_json(Path(args.semantic_results), [])
    graph_results = []
    graph_ms = 0.0

    if not args.semantic_only:
        graph_results, graph_ms = retrieve_graphs(
            PROJECT_ROOT,
            args.query,
            semantic_results,
            max_documents=args.max_documents,
        )

    combined = combine_evidence(
        semantic_results,
        graph_results,
        max_context=args.max_context,
    )
    answer = (
        extractive_answer(args.query, combined)
        if args.no_llm
        else llm_answer(args.query, combined)
    )
    sources = []
    seen_sources = set()

    for item in combined:
        key = (item["document_id"], item["source"])
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({"document_id": key[0], "source": key[1]})

    result = {
        **answer,
        "retrieval": {
            "semantic_chunks": len(semantic_results),
            "graph_nodes": sum(len(item["nodes"]) for item in graph_results),
            "graph_edges": sum(len(item["edges"]) for item in graph_results),
            "combined_chunks": len(combined),
        },
        "query": args.query,
        "semantic_results": semantic_results,
        "graph_results": graph_results,
        "combined_context": combined,
        "sources": sources,
        "timing_ms": {
            "graph_traversal": round(graph_ms, 3),
            "post_vector_pipeline": round((time.perf_counter() - started) * 1000, 3),
        },
    }
    return result


def remove_document(args):
    registry_store = GraphRegistry(PROJECT_ROOT)
    registry = registry_store.load()
    document_id = args.document

    if document_id not in registry:
        matches = [
            key for key, entry in registry.items()
            if entry.get("source_path") == args.document
            or entry.get("document_name") == args.document
        ]

        if len(matches) != 1:
            raise KeyError(f"Unknown or ambiguous document: {args.document}")

        document_id = matches[0]

    entry = registry[document_id]
    chunks = [
        item for item in read_json(CHUNKS_PATH, [])
        if item.get("document_id") != document_id
    ]
    vectors = [
        item for item in read_json(VECTORS_PATH, [])
        if item.get("document_id") != document_id
    ]
    write_json(CHUNKS_PATH, chunks)
    write_json(VECTORS_PATH, vectors)

    removed_files = []
    for key in ("nodes_path", "edges_path"):
        path = PROJECT_ROOT / entry[key]
        if path.exists():
            path.unlink()
            removed_files.append(entry[key])

    graph_directory = (PROJECT_ROOT / entry["nodes_path"]).parent
    if graph_directory.exists() and not any(graph_directory.iterdir()):
        graph_directory.rmdir()

    del registry[document_id]
    registry_store.save(registry)
    return {
        "status": "removed",
        "document_id": document_id,
        "removed_graph_files": removed_files,
    }


def stats(_args):
    registry = GraphRegistry(PROJECT_ROOT).load()
    chunks = read_json(CHUNKS_PATH, [])
    vectors = read_json(VECTORS_PATH, [])
    return {
        "documents": len(registry),
        "document_ids": sorted(registry),
        "chunks": len(chunks),
        "vectors": len(vectors),
        "index_consistent": len(chunks) == len(vectors),
    }


def parser():
    root = argparse.ArgumentParser()
    subcommands = root.add_subparsers(dest="command", required=True)

    ingest_parser = subcommands.add_parser("ingest")
    ingest_parser.add_argument("path")
    ingest_parser.add_argument("--chunk-size", type=int, default=5)
    ingest_parser.add_argument("--overlap", type=int, default=1)
    ingest_parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence-transformers", "hash"],
        default="auto",
    )
    ingest_parser.add_argument(
        "--graph-backend", choices=["heuristic", "openai"], default="heuristic"
    )
    ingest_parser.add_argument("--force", action="store_true")
    ingest_parser.set_defaults(handler=ingest)

    prepare_parser = subcommands.add_parser("prepare-query")
    prepare_parser.add_argument("query")
    prepare_parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence-transformers", "hash"],
        default="auto",
    )
    prepare_parser.set_defaults(handler=prepare_query)

    complete_parser = subcommands.add_parser("complete-query")
    complete_parser.add_argument("query")
    complete_parser.add_argument("--semantic-results", required=True)
    complete_parser.add_argument("--max-documents", type=int, default=3)
    complete_parser.add_argument("--max-context", type=int, default=8)
    complete_parser.add_argument("--semantic-only", action="store_true")
    complete_parser.add_argument("--no-llm", action="store_true")
    complete_parser.set_defaults(handler=complete_query)

    remove_parser = subcommands.add_parser("remove")
    remove_parser.add_argument("document")
    remove_parser.set_defaults(handler=remove_document)

    stats_parser = subcommands.add_parser("stats")
    stats_parser.set_defaults(handler=stats)

    return root


def main():
    load_environment()
    arguments = parser().parse_args()

    try:
        result = arguments.handler(arguments)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

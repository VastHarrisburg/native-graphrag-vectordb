from __future__ import annotations

import os
import re
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict
except ImportError:
    BaseModel = None
    ConfigDict = None


if BaseModel:
    class ExtractedNode(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str


    class ExtractedEdge(BaseModel):
        model_config = ConfigDict(extra="forbid")
        source: str
        target: str
        context: str


    class GraphExtraction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        nodes: list[ExtractedNode]
        edges: list[ExtractedEdge]
        evidence_nodes: list[str]


COMMON_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "he",
    "her",
    "his",
    "if",
    "in",
    "it",
    "its",
    "she",
    "the",
    "their",
    "they",
    "this",
    "to",
    "when",
}


def normalize_name(name: str) -> str:
    name = re.sub(r"[_‐‑‒–—-]+", " ", name)
    name = re.sub(r"[^\w\s'-]", "", name.casefold())
    return re.sub(r"\s+", " ", name).strip(" -'")


def heuristic_entities(text: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:[A-Z][\w'-]*|[A-Z]{2,})"
        r"(?:\s+(?:(?:of|the|and|for)\s+)?(?:[A-Z][\w'-]*|[A-Z]{2,})){0,4}"
    )
    entities = []

    for match in pattern.finditer(text):
        value = normalize_name(match.group(0))
        words = value.split()

        while words and words[0] in {
            "a", "after", "although", "an", "before", "during", "every",
            "however", "if", "once", "the", "understanding", "whenever",
            "without",
        }:
            words.pop(0)

        value = " ".join(words)

        if not value or all(word in COMMON_WORDS for word in words):
            continue
        if value not in entities:
            entities.append(value)

    return entities[:20]


def _heuristic_graph(chunks: list[dict[str, Any]]):
    nodes = {}
    edges = []
    edge_keys = set()

    def add_node(name, node_type, **extra):
        key = normalize_name(name) if node_type == "entity" else name
        nodes.setdefault(key, {"name": key, "type": node_type, **extra})

    def add_edge(source, target, context, chunk_id):
        key = (source, target, context.casefold())

        if source == target or key in edge_keys:
            return

        edge_keys.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "context": context,
                "evidence_chunk_id": chunk_id,
            }
        )

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        add_node(
            chunk_id,
            "chunk",
            document_id=chunk["document_id"],
            content=text,
            source=chunk["source"],
        )
        entities = heuristic_entities(text)

        for entity in entities:
            add_node(entity, "entity", document_id=chunk["document_id"])
            add_edge(
                chunk_id,
                entity,
                f"{chunk_id} provides evidence about {entity}.",
                chunk_id,
            )

        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence_entities = [
                entity for entity in entities
                if re.search(rf"\b{re.escape(entity)}\b", sentence.casefold())
            ]

            for left, right in zip(sentence_entities, sentence_entities[1:]):
                add_edge(left, right, sentence.strip(), chunk_id)

    return list(nodes.values()), edges


def _openai_graph(chunks: list[dict[str, Any]]):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai and pydantic packages are required for OpenAI graph extraction"
        ) from error

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI graph extraction")
    if BaseModel is None:
        raise RuntimeError("pydantic is required for OpenAI graph extraction")

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    nodes = {}
    edges = []
    edge_keys = set()

    for chunk in chunks:
        response = client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Extract useful canonical entities and directed relationships "
                        "from only the supplied chunk. Keep relationship context factual."
                    ),
                },
                {"role": "user", "content": chunk["text"]},
            ],
            text_format=GraphExtraction,
        )
        extraction = response.output_parsed

        if extraction is None:
            raise RuntimeError(f"No graph extraction returned for {chunk['chunk_id']}")

        nodes[chunk["chunk_id"]] = {
            "name": chunk["chunk_id"],
            "type": "chunk",
            "document_id": chunk["document_id"],
            "content": chunk["text"],
            "source": chunk["source"],
        }

        names = [node.name for node in extraction.nodes]
        names.extend(edge.source for edge in extraction.edges)
        names.extend(edge.target for edge in extraction.edges)

        for name in names:
            normalized = normalize_name(name)
            if normalized:
                nodes.setdefault(
                    normalized,
                    {
                        "name": normalized,
                        "type": "entity",
                        "document_id": chunk["document_id"],
                    },
                )

        new_edges = [
            (edge.source, edge.target, edge.context)
            for edge in extraction.edges
        ]
        new_edges.extend(
            (
                chunk["chunk_id"],
                name,
                f"{chunk['chunk_id']} provides evidence about {normalize_name(name)}.",
            )
            for name in extraction.evidence_nodes[:4]
        )

        for source, target, context in new_edges:
            source = chunk["chunk_id"] if source == chunk["chunk_id"] else normalize_name(source)
            target = normalize_name(target)
            key = (source, target, context.casefold())

            if source not in nodes or target not in nodes or source == target or key in edge_keys:
                continue

            edge_keys.add(key)
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "context": re.sub(r"\s+", " ", context).strip(),
                    "evidence_chunk_id": chunk["chunk_id"],
                }
            )

    return list(nodes.values()), edges


def build_graph(chunks: list[dict[str, Any]], backend="heuristic"):
    if backend == "heuristic":
        return (*_heuristic_graph(chunks), "heuristic")
    if backend == "openai":
        return (*_openai_graph(chunks), "openai")
    raise ValueError(f"Unknown graph backend: {backend}")

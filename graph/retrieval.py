from __future__ import annotations

import difflib
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from graph.extraction import normalize_name
from graph.registry import GraphRegistry


def load_document_graph(registry: GraphRegistry, document_id: str):
    nodes_path, edges_path = registry.graph_paths(document_id)

    for path in (nodes_path, edges_path):
        if not path.exists():
            raise FileNotFoundError(f"Graph file is missing: {path}")

    nodes = json.loads(nodes_path.read_text(encoding="utf-8") or "[]")
    edges = json.loads(edges_path.read_text(encoding="utf-8") or "[]")

    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"Graph files for {document_id} must contain arrays")

    graph = nx.MultiDiGraph(document_id=document_id)

    for index, node in enumerate(nodes):
        name = node.get("name")

        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid node {index} in {nodes_path}")

        graph.add_node(name, **node)

    for index, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")

        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"Invalid edge {index} in {edges_path}")

        if source not in graph or target not in graph:
            continue

        graph.add_edge(source, target, **edge)

    return graph


def _acronym(name: str) -> str:
    ignored = {"a", "an", "and", "of", "the", "to"}
    return "".join(word[0] for word in name.split() if word not in ignored)


def match_query_entities(query: str, graph, limit=5) -> list[str]:
    normalized_query = normalize_name(query)
    ignored = {
        "a", "an", "and", "are", "did", "do", "does", "for", "from",
        "how", "in", "is", "of", "the", "to", "was", "were", "what",
        "when", "where", "which", "who", "why", "with",
    }
    query_words = set(normalized_query.split()) - ignored
    candidates = []

    for name, data in graph.nodes(data=True):
        if data.get("type") != "entity":
            continue

        normalized = normalize_name(name)
        score = 0.0

        if re.search(rf"\b{re.escape(normalized)}\b", normalized_query):
            score = 1.0
        elif len(_acronym(normalized)) > 1 and _acronym(normalized) in query_words:
            score = 0.95
        else:
            words = set(normalized.split()) - ignored
            overlap = len(words & query_words) / max(len(words), 1)
            fuzzy = difflib.SequenceMatcher(None, normalized, normalized_query).ratio()
            overlap_score = overlap * 0.85 if overlap >= 0.6 else 0.0
            score = max(overlap_score, fuzzy if fuzzy >= 0.72 else 0.0)

        if score:
            candidates.append((score, graph.degree(name), name))

    candidates.sort(reverse=True)
    return [name for _, _, name in candidates[:limit]]


def traverse_graph(
    graph,
    document_id: str,
    start_entities: list[str],
    depth=2,
    max_nodes=25,
    max_edges=40,
):
    distances = {}
    queue = deque()

    for entity in start_entities[:5]:
        if entity in graph and entity not in distances:
            distances[entity] = 0
            queue.append(entity)

    while queue and len(distances) < max_nodes:
        current = queue.popleft()
        current_distance = distances[current]

        if current_distance >= depth:
            continue

        neighbors = set(graph.successors(current)) | set(graph.predecessors(current))

        for neighbor in sorted(neighbors):
            if neighbor in distances:
                continue
            distances[neighbor] = current_distance + 1
            queue.append(neighbor)

            if len(distances) >= max_nodes:
                break

    returned_nodes = [
        {**graph.nodes[name], "name": name, "distance": distances[name]}
        for name in distances
    ]
    returned_edges = []

    for source, target, key, data in graph.edges(keys=True, data=True):
        if source not in distances or target not in distances:
            continue
        returned_edges.append(
            {
                **data,
                "source": source,
                "target": target,
                "key": key,
                "distance": max(distances[source], distances[target]),
            }
        )

        if len(returned_edges) >= max_edges:
            break

    evidence_ids = []

    for node in returned_nodes:
        if node.get("type") == "chunk":
            evidence_ids.append(node["name"])

    for edge in returned_edges:
        chunk_id = edge.get("evidence_chunk_id")
        if chunk_id and chunk_id not in evidence_ids:
            evidence_ids.append(chunk_id)

    return {
        "document_id": document_id,
        "start_entities": start_entities,
        "nodes": returned_nodes,
        "edges": returned_edges,
        "evidence_chunk_ids": evidence_ids,
    }


def select_documents(semantic_results, limit=3):
    best_scores = {}

    for result in semantic_results:
        document_id = result.get("document_id")

        if not document_id:
            continue

        best_scores[document_id] = max(
            best_scores.get(document_id, float("-inf")),
            float(result.get("similarity_score", result.get("score", 0.0))),
        )

    return [
        document_id
        for document_id, _ in sorted(
            best_scores.items(), key=lambda item: item[1], reverse=True
        )[:limit]
    ]


def retrieve_graphs(project_root, query, semantic_results, max_documents=3):
    registry = GraphRegistry(project_root)
    results = []
    started = time.perf_counter()

    for document_id in select_documents(semantic_results, max_documents):
        try:
            graph = load_document_graph(registry, document_id)
            entities = match_query_entities(query, graph)
            result = traverse_graph(graph, document_id, entities)
            result["error"] = None
        except (KeyError, FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            result = {
                "document_id": document_id,
                "start_entities": [],
                "nodes": [],
                "edges": [],
                "evidence_chunk_ids": [],
                "error": str(error),
            }

        results.append(result)

    return results, (time.perf_counter() - started) * 1000

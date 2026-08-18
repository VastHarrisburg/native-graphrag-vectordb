import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from graph.extraction import build_graph
from graph.registry import GraphRegistry
from graph.retrieval import (
    load_document_graph,
    match_query_entities,
    select_documents,
    traverse_graph,
)
from scripts import pipeline
from scripts.processing import Processing


class ProcessingTests(unittest.TestCase):
    def test_chunks_share_the_canonical_document_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "sample.txt"
            source.write_text("One sentence. Two sentence. Three sentence.")
            chunks = Processing.read_chunks(
                source,
                chunk_size=2,
                overlap=1,
                document_id="doc_007",
                source="documents/sample.txt",
            )

        self.assertEqual([item["chunk_id"] for item in chunks], [
            "doc_007_chunk_000",
            "doc_007_chunk_001",
        ])
        self.assertTrue(all(item["document_id"] == "doc_007" for item in chunks))

    def test_hash_embeddings_are_deterministic(self):
        first, model = Processing.embed_texts(["shared cache"], backend="hash")
        second, _ = Processing.embed_texts(["shared cache"], backend="hash")
        self.assertEqual(model, "hash-384")
        self.assertEqual(first, second)


class RegistryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.source = self.root / "documents" / "sample.txt"
        self.source.parent.mkdir()
        self.source.write_text("Atlas Gateway sends events to Aurora Event Log.")
        self.patchers = [
            patch.object(pipeline, "PROJECT_ROOT", self.root),
            patch.object(pipeline, "DATA_PATH", self.data),
            patch.object(pipeline, "CHUNKS_PATH", self.data / "chunks.json"),
            patch.object(pipeline, "VECTORS_PATH", self.data / "vectors.json"),
            patch.object(pipeline, "QUERY_VECTOR_PATH", self.data / "query_vector.json"),
            patch.object(pipeline, "INDEX_METADATA_PATH", self.data / "index_metadata.json"),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def arguments(self):
        return SimpleNamespace(
            path=str(self.source),
            chunk_size=2,
            overlap=0,
            embedding_backend="hash",
            graph_backend="heuristic",
            force=False,
        )

    def test_duplicate_ingestion_reuses_graph(self):
        first = pipeline.ingest(self.arguments())
        graph_path = self.data / "graphs" / first["document_id"] / "nodes.json"
        first_graph = graph_path.read_text()
        second = pipeline.ingest(self.arguments())

        self.assertEqual(first["document_id"], second["document_id"])
        self.assertEqual(second["status"], "reused")
        self.assertEqual(first_graph, graph_path.read_text())
        self.assertEqual(len(GraphRegistry(self.root).load()), 1)

    def test_changed_document_keeps_id_and_rebuilds_data(self):
        first = pipeline.ingest(self.arguments())
        self.source.write_text("Atlas Gateway sends events to Meridian Engine instead.")
        second = pipeline.ingest(self.arguments())

        self.assertEqual(second["status"], "updated")
        self.assertEqual(first["document_id"], second["document_id"])
        chunks = json.loads((self.data / "chunks.json").read_text())
        self.assertIn("Meridian", chunks[0]["text"])
        self.assertTrue(all(item["document_id"] == first["document_id"] for item in chunks))

    def test_remove_updates_all_stores(self):
        result = pipeline.ingest(self.arguments())
        removed = pipeline.remove_document(SimpleNamespace(document=result["document_id"]))

        self.assertEqual(removed["status"], "removed")
        self.assertEqual(json.loads((self.data / "chunks.json").read_text()), [])
        self.assertEqual(json.loads((self.data / "vectors.json").read_text()), [])
        self.assertEqual(GraphRegistry(self.root).load(), {})
        self.assertFalse((self.data / "graphs" / result["document_id"]).exists())


class GraphTraversalTests(unittest.TestCase):
    def graph(self):
        graph = nx.MultiDiGraph()
        graph.add_node("alpha", name="alpha", type="entity")
        graph.add_node("beta", name="beta", type="entity")
        graph.add_node("gamma", name="gamma", type="entity")
        graph.add_node("isolated", name="isolated", type="entity")
        graph.add_node(
            "doc_001_chunk_000",
            name="doc_001_chunk_000",
            type="chunk",
            content="evidence",
        )
        graph.add_edge("alpha", "beta", context="first")
        graph.add_edge("alpha", "beta", context="second")
        graph.add_edge("beta", "gamma", context="third")
        graph.add_edge(
            "gamma",
            "doc_001_chunk_000",
            context="evidence",
            evidence_chunk_id="doc_001_chunk_000",
        )
        return graph

    def test_one_and_two_hop_traversal(self):
        one = traverse_graph(self.graph(), "doc_001", ["alpha"], depth=1)
        two = traverse_graph(self.graph(), "doc_001", ["alpha"], depth=2)
        self.assertEqual({node["name"] for node in one["nodes"]}, {"alpha", "beta"})
        self.assertIn("gamma", {node["name"] for node in two["nodes"]})

    def test_multiple_edges_are_preserved(self):
        result = traverse_graph(self.graph(), "doc_001", ["alpha"], depth=1)
        contexts = [edge["context"] for edge in result["edges"]]
        self.assertEqual(contexts, ["first", "second"])

    def test_unknown_and_disconnected_entities_are_safe(self):
        unknown = traverse_graph(self.graph(), "doc_001", ["missing"])
        isolated = traverse_graph(self.graph(), "doc_001", ["isolated"])
        self.assertEqual(unknown["nodes"], [])
        self.assertEqual([node["name"] for node in isolated["nodes"]], ["isolated"])

    def test_alias_and_acronym_matching(self):
        graph = self.graph()
        graph.add_node(
            "atlas gateway", name="atlas gateway", type="entity"
        )
        self.assertEqual(
            match_query_entities("How does AG route traffic?", graph)[0],
            "atlas gateway",
        )

    def test_empty_graph_has_no_matches(self):
        graph = nx.MultiDiGraph()
        self.assertEqual(match_query_entities("anything", graph), [])
        self.assertEqual(traverse_graph(graph, "doc_001", [], depth=2)["nodes"], [])

    def test_highly_connected_entity_respects_limits(self):
        graph = nx.MultiDiGraph()
        graph.add_node("hub", name="hub", type="entity")
        for index in range(50):
            name = f"node-{index}"
            graph.add_node(name, name=name, type="entity")
            graph.add_edge("hub", name, context=str(index))

        result = traverse_graph(
            graph,
            "doc_001",
            ["hub"],
            max_nodes=15,
            max_edges=20,
        )
        self.assertLessEqual(len(result["nodes"]), 15)
        self.assertLessEqual(len(result["edges"]), 20)

    def test_document_selection_uses_maximum_chunk_score(self):
        results = [
            {"document_id": "long", "similarity_score": 0.2},
            {"document_id": "long", "similarity_score": 0.3},
            {"document_id": "short", "similarity_score": 0.9},
        ]
        self.assertEqual(select_documents(results, 2), ["short", "long"])

    def test_unknown_document_and_missing_graph_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = GraphRegistry(temporary)
            with self.assertRaises(KeyError):
                load_document_graph(registry, "doc_404")
            registry.save({
                "doc_001": {
                    "nodes_path": "data/graphs/doc_001/nodes.json",
                    "edges_path": "data/graphs/doc_001/edges.json",
                }
            })
            with self.assertRaises(FileNotFoundError):
                load_document_graph(registry, "doc_001")

    def test_empty_registered_graph_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "data/graphs/doc_001"
            directory.mkdir(parents=True)
            (directory / "nodes.json").write_text("[]")
            (directory / "edges.json").write_text("[]")
            registry = GraphRegistry(root)
            registry.save({
                "doc_001": {
                    "nodes_path": "data/graphs/doc_001/nodes.json",
                    "edges_path": "data/graphs/doc_001/edges.json",
                }
            })
            graph = load_document_graph(registry, "doc_001")
            self.assertEqual(graph.number_of_nodes(), 0)
            self.assertEqual(graph.number_of_edges(), 0)


class EvidenceTests(unittest.TestCase):
    def test_semantic_and_graph_evidence_is_deduplicated(self):
        chunks = [{
            "chunk_id": "doc_001_chunk_000",
            "document_id": "doc_001",
            "text": "Atlas reaches Aurora.",
            "source": "sample.txt",
        }]
        semantic = [{
            **chunks[0],
            "similarity_score": 0.8,
        }]
        graph = [{
            "document_id": "doc_001",
            "nodes": [{"name": "doc_001_chunk_000", "type": "chunk", "distance": 1}],
            "edges": [],
            "evidence_chunk_ids": ["doc_001_chunk_000"],
        }]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunks.json"
            path.write_text(json.dumps(chunks))
            with patch.object(pipeline, "CHUNKS_PATH", path):
                combined = pipeline.combine_evidence(semantic, graph)

        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["retrieved_by"], ["semantic", "graph"])

    def test_answer_citations_are_from_supplied_evidence(self):
        context = [{
            "chunk_id": "doc_001_chunk_000",
            "text": "Helios reduced database traffic.",
            "semantic_score": 0.8,
        }]
        answer = pipeline.extractive_answer("What reduced database traffic?", context)
        self.assertEqual(answer["citations"], ["doc_001_chunk_000"])


if __name__ == "__main__":
    unittest.main()

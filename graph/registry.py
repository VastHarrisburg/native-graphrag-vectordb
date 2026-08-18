from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class GraphRegistry:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.data_path = self.project_root / "data"
        self.registry_path = self.data_path / "graph_registry.json"

    @staticmethod
    def content_hash(path: Path) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}

        raw = self.registry_path.read_text(encoding="utf-8").strip()

        if not raw:
            return {}

        data = json.loads(raw)

        if not isinstance(data, dict):
            raise ValueError("graph_registry.json must contain an object")

        return data

    def save(self, registry: dict[str, dict[str, Any]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.registry_path)

    def source_name(self, path: Path) -> str:
        path = path.resolve()

        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def find_by_hash(registry, content_hash):
        for document_id, entry in registry.items():
            if entry.get("content_hash") == content_hash:
                return document_id, entry
        return None

    @staticmethod
    def find_by_source(registry, source_path):
        for document_id, entry in registry.items():
            if entry.get("source_path") == source_path:
                return document_id, entry
        return None

    @staticmethod
    def next_document_id(registry) -> str:
        numbers = []

        for document_id in registry:
            if document_id.startswith("doc_"):
                try:
                    numbers.append(int(document_id[4:]))
                except ValueError:
                    pass

        return f"doc_{max(numbers, default=0) + 1:03d}"

    def graph_paths(self, document_id: str) -> tuple[Path, Path]:
        entry = self.get(document_id)
        nodes_path = self.project_root / entry["nodes_path"]
        edges_path = self.project_root / entry["edges_path"]
        return nodes_path, edges_path

    def get(self, document_id: str) -> dict[str, Any]:
        registry = self.load()

        if document_id not in registry:
            raise KeyError(f"Unknown document ID: {document_id}")

        return registry[document_id]

    def make_entry(
        self,
        document_id: str,
        source_path: str,
        content_hash: str,
        embedding_model: str,
        graph_backend: str,
    ) -> dict[str, Any]:
        return {
            "document_name": Path(source_path).name,
            "source_path": source_path,
            "content_hash": content_hash,
            "nodes_path": f"data/graphs/{document_id}/nodes.json",
            "edges_path": f"data/graphs/{document_id}/edges.json",
            "embedding_model": embedding_model,
            "graph_backend": graph_backend,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

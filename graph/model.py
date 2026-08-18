from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field




class ExtractedNode(BaseModel):
    """
    One entity extracted from the current chunk.

    The model does not create chunk nodes. Python handles those.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        description="A concise lowercase canonical entity name.",
    )


class ExtractedEdge(BaseModel):
    """One directed semantic relationship."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        min_length=1,
        description="Lowercase canonical source node name.",
    )

    target: str = Field(
        min_length=1,
        description="Lowercase canonical target node name.",
    )

    context: str = Field(
        min_length=1,
        description="A complete sentence explaining the relationship.",
    )


class GraphExtraction(BaseModel):
    """
    The complete graph delta returned for one chunk.

    The prompt asks the model to include every edge endpoint in ``nodes``,
    but Python also creates any missing endpoint nodes as a safety fallback.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: list[ExtractedNode] = Field(
        description="Useful entity nodes from the current chunk.",
    )

    edges: list[ExtractedEdge] = Field(
        description="Useful directed relationships from the current chunk.",
    )

    evidence_nodes: list[str] = Field(
        description=(
            "Two to four important entity names that should connect "
            "back to the current chunk."
        ),
    )


# ============================================================
# Merge statistics
# ============================================================


@dataclass
class MergeStats:
    added_nodes: int = 0
    inferred_endpoint_nodes: int = 0
    duplicate_nodes: int = 0

    added_edges: int = 0
    duplicate_edges: int = 0

    skipped_new_edges: int = 0
    removed_invalid_existing_edges: int = 0

    total_nodes: int = 0
    total_edges: int = 0


# ============================================================
# Persistent graph storage
# ============================================================


class GraphStore:
    """
    Handles all deterministic graph operations.

    The language model never reads or writes the persistent graph.
    """

    MAX_EVIDENCE_EDGES = 4

    def __init__(
        self,
        project_root: Path,
        *,
        dedupe_edges_by_pair: bool = False,
    ) -> None:
        self.project_root = project_root

        self.node_file = project_root / "data/test.json"
        self.edge_file = project_root / "data/edges.json"

        # False:
        #   Keep separate edges between the same endpoints when their
        #   contexts differ.
        #
        # True:
        #   Keep only one edge for each source-target pair.
        self.dedupe_edges_by_pair = dedupe_edges_by_pair

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Perform conservative formatting normalization.

        This intentionally does not:
        - infer aliases
        - singularize words
        - use substring matching
        - use semantic similarity
        - merge related concepts
        """
        normalized = unicodedata.normalize("NFKC", name)

        normalized = normalized.strip().casefold()

        # Treat snake_case and common hyphen variants as spaces so simple
        # formatting differences collapse to the same key.
        normalized = normalized.replace("_", " ")
        normalized = re.sub(
            r"[-‐‑‒–—]+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        # Remove only punctuation at the beginning or end.
        normalized = normalized.strip(
            " \t\r\n.,;:!?\"'`"
        )

        return normalized

    @staticmethod
    def normalize_context(context: str) -> str:
        """
        Normalize context for exact duplicate comparison.

        The original readable context is still stored.
        """
        normalized = unicodedata.normalize(
            "NFKC",
            context,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized.strip(),
        )

        return normalized.casefold()

    # --------------------------------------------------------
    # JSON file handling
    # --------------------------------------------------------

    @staticmethod
    def read_json_array(
        path: Path,
    ) -> list[dict[str, Any]]:
        """
        Read a JSON array from disk.

        A missing or whitespace-only file is treated as an empty array.
        """

        if not path.exists():
            return []

        try:
            raw_content = path.read_text(
                encoding="utf-8"
            ).strip()
        except PermissionError as error:
            raise PermissionError(
                f"Permission denied while reading {path}"
            ) from error

        # An existing but empty file should behave like [].
        if not raw_content:
            return []

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed JSON in {path}: {error}"
            ) from error

        if not isinstance(data, list):
            raise ValueError(
                f"{path} must contain a JSON array"
            )

        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}[{index}] must be a JSON object"
                )

        return data

    @staticmethod
    def write_json_pair(
        node_file: Path,
        nodes: list[dict[str, Any]],
        edge_file: Path,
        edges: list[dict[str, Any]],
    ) -> None:
        """
        Fully serialize both files before replacing either destination.

        Temporary files prevent malformed partial JSON from being written
        if serialization fails.
        """
        node_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        edge_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_paths: list[Path] = []

        try:
            for destination, data in (
                (node_file, nodes),
                (edge_file, edges),
            ):
                file_descriptor, temporary_name = (
                    tempfile.mkstemp(
                        dir=destination.parent,
                        prefix=f".{destination.name}.",
                        suffix=".tmp",
                        text=True,
                    )
                )

                temporary_path = Path(temporary_name)
                temporary_paths.append(temporary_path)

                with os.fdopen(
                    file_descriptor,
                    "w",
                    encoding="utf-8",
                ) as temporary_file:
                    json.dump(
                        data,
                        temporary_file,
                        ensure_ascii=False,
                        indent=2,
                    )

                    temporary_file.write("\n")
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())

            os.replace(
                temporary_paths[0],
                node_file,
            )

            os.replace(
                temporary_paths[1],
                edge_file,
            )

            temporary_paths.clear()

        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(
                    missing_ok=True
                )

    # --------------------------------------------------------
    # Existing record validation
    # --------------------------------------------------------

    def clean_existing_node(
        self,
        record: dict[str, Any],
        index: int,
    ) -> tuple[str, dict[str, Any]]:
        document = record.get("document")
        name = record.get("name")
        node_type = record.get("type")

        if not isinstance(document, str):
            raise ValueError(
                f"{self.node_file}[{index}] "
                "has an invalid document"
            )

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"{self.node_file}[{index}] "
                "has an invalid name"
            )

        if node_type not in {"entity", "chunk"}:
            raise ValueError(
                f"{self.node_file}[{index}] "
                "has an invalid type"
            )

        normalized_name = self.normalize_name(name)

        if not normalized_name:
            raise ValueError(
                f"{self.node_file}[{index}] "
                "has an empty normalized name"
            )

        if node_type == "entity":
            cleaned = {
                "document": document,
                "name": normalized_name,
                "type": "entity",
            }

        else:
            content = record.get("content")

            if not isinstance(content, str):
                raise ValueError(
                    f"{self.node_file}[{index}] "
                    "chunk has invalid content"
                )

            cleaned = {
                "document": document,
                "name": name.strip(),
                "type": "chunk",
                "content": content,
            }

        return normalized_name, cleaned

    # --------------------------------------------------------
    # Edge deduplication
    # --------------------------------------------------------

    def edge_key(
        self,
        source_key: str,
        target_key: str,
        context: str,
    ) -> tuple[str, ...]:
        if self.dedupe_edges_by_pair:
            return (
                source_key,
                target_key,
            )

        return (
            source_key,
            target_key,
            self.normalize_context(context),
        )

    # --------------------------------------------------------
    # Main merge operation
    # --------------------------------------------------------

    def merge_and_write(
        self,
        extraction: GraphExtraction,
        *,
        chunk_name: str,
        document_path: str,
        chunk_content: str,
    ) -> MergeStats:
        """
        Read the persistent graph, merge one model extraction, validate
        everything, and write the result.
        """
        existing_nodes = self.read_json_array(
            self.node_file
        )

        existing_edges = self.read_json_array(
            self.edge_file
        )

        stats = MergeStats()

        # key -> complete persistent node record
        nodes_by_key: dict[
            str,
            dict[str, Any],
        ] = {}

        # ----------------------------------------------------
        # Normalize and deduplicate existing nodes
        # ----------------------------------------------------

        for index, record in enumerate(existing_nodes):
            key, cleaned_node = self.clean_existing_node(
                record,
                index,
            )

            if key in nodes_by_key:
                stats.duplicate_nodes += 1
                continue

            nodes_by_key[key] = cleaned_node

        # ----------------------------------------------------
        # Gather all entity names required by this extraction
        # ----------------------------------------------------

        # The model is instructed to list every edge endpoint in
        # extraction.nodes, but models can still omit some. Python therefore
        # treats edge endpoints and evidence anchors as node candidates too.
        #
        # Explicit model nodes are processed first so they remain the primary
        # source of node creation. Missing endpoint nodes are only a fallback.
        candidate_names: list[tuple[str, bool]] = [
            (node.name, False)
            for node in extraction.nodes
        ]

        candidate_names.extend(
            (edge.source, True)
            for edge in extraction.edges
        )
        candidate_names.extend(
            (edge.target, True)
            for edge in extraction.edges
        )
        candidate_names.extend(
            (name, True)
            for name in extraction.evidence_nodes
        )

        seen_current_keys: set[str] = set()
        explicit_node_keys = {
            self.normalize_name(node.name)
            for node in extraction.nodes
            if self.normalize_name(node.name)
        }

        for candidate_name, inferred in candidate_names:
            key = self.normalize_name(candidate_name)

            if not key or key in seen_current_keys:
                continue

            seen_current_keys.add(key)

            if key in nodes_by_key:
                stats.duplicate_nodes += 1
                continue

            nodes_by_key[key] = {
                "document": document_path,
                "name": key,
                "type": "entity",
            }

            stats.added_nodes += 1

            if inferred and key not in explicit_node_keys:
                stats.inferred_endpoint_nodes += 1

        # ----------------------------------------------------
        # Python creates the chunk node
        # ----------------------------------------------------

        has_graph_content = bool(
            extraction.nodes
            or extraction.edges
            or extraction.evidence_nodes
        )

        chunk_key = self.normalize_name(chunk_name)

        if has_graph_content:
            existing_chunk = nodes_by_key.get(
                chunk_key
            )

            if (
                existing_chunk is not None
                and existing_chunk.get("type") != "chunk"
            ):
                raise ValueError(
                    "Chunk name collides with an entity: "
                    f"{chunk_name}"
                )

            if existing_chunk is None:
                stats.added_nodes += 1

            # Reprocessing a chunk replaces its chunk record rather
            # than creating another copy.
            nodes_by_key[chunk_key] = {
                "document": document_path,
                "name": chunk_name,
                "type": "chunk",
                "content": chunk_content,
            }

        # ----------------------------------------------------
        # Merge edges
        # ----------------------------------------------------

        edges_by_key: dict[
            tuple[str, ...],
            dict[str, str],
        ] = {}

        def add_edge(
            source: str,
            target: str,
            context: str,
            *,
            existing: bool,
        ) -> None:
            source_key = self.normalize_name(source)
            target_key = self.normalize_name(target)

            cleaned_context = re.sub(
                r"\s+",
                " ",
                context.strip(),
            )

            invalid_fields = (
                not source_key
                or not target_key
                or not cleaned_context
            )

            missing_endpoint = (
                source_key not in nodes_by_key
                or target_key not in nodes_by_key
            )

            self_loop = source_key == target_key

            if (
                invalid_fields
                or missing_endpoint
                or self_loop
            ):
                if existing:
                    stats.removed_invalid_existing_edges += 1
                else:
                    stats.skipped_new_edges += 1

                return

            canonical_source = (
                nodes_by_key[source_key]["name"]
            )

            canonical_target = (
                nodes_by_key[target_key]["name"]
            )

            key = self.edge_key(
                source_key,
                target_key,
                cleaned_context,
            )

            if key in edges_by_key:
                stats.duplicate_edges += 1
                return

            edges_by_key[key] = {
                "source": canonical_source,
                "target": canonical_target,
                "context": cleaned_context,
            }

            if not existing:
                stats.added_edges += 1

        # ----------------------------------------------------
        # Normalize and validate existing edges
        # ----------------------------------------------------

        for index, record in enumerate(existing_edges):
            source = record.get("source")
            target = record.get("target")
            context = record.get("context")

            if not all(
                isinstance(value, str)
                for value in (
                    source,
                    target,
                    context,
                )
            ):
                raise ValueError(
                    f"{self.edge_file}[{index}] "
                    "has invalid fields"
                )

            add_edge(
                source,
                target,
                context,
                existing=True,
            )

        # ----------------------------------------------------
        # Add model-extracted semantic edges
        # ----------------------------------------------------

        for edge in extraction.edges:
            add_edge(
                edge.source,
                edge.target,
                edge.context,
                existing=False,
            )

        # ----------------------------------------------------
        # Add chunk evidence edges
        # ----------------------------------------------------

        if has_graph_content:
            seen_evidence: set[str] = set()

            for evidence_name in extraction.evidence_nodes:
                target_key = self.normalize_name(
                    evidence_name
                )

                if not target_key:
                    continue

                if target_key in seen_evidence:
                    continue

                if target_key not in nodes_by_key:
                    continue

                target_node = nodes_by_key[target_key]

                if target_node["type"] != "entity":
                    continue

                seen_evidence.add(target_key)

                target_name = target_node["name"]

                add_edge(
                    chunk_name,
                    target_name,
                    (
                        "The chunk provides important evidence "
                        f"about {target_name}."
                    ),
                    existing=False,
                )

                if (
                    len(seen_evidence)
                    >= self.MAX_EVIDENCE_EDGES
                ):
                    break

        final_nodes = list(
            nodes_by_key.values()
        )

        final_edges = list(
            edges_by_key.values()
        )

        self.write_json_pair(
            self.node_file,
            final_nodes,
            self.edge_file,
            final_edges,
        )

        stats.total_nodes = len(final_nodes)
        stats.total_edges = len(final_edges)

        return stats


# ============================================================
# Model and document processing
# ============================================================


class Model:
    CHUNK_SIZE = 15
    OVERLAP = 3

    def __init__(
        self,
        project_root: str | Path | None = None,
    ) -> None:
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root
            else Path(__file__).resolve().parents[1]
        )

        self.prompt_file = (
            self.project_root
            / "graph/System_Prompt.txt"
        )

        self.system_prompt: str | None = None
        self.extractor: Any | None = None

        self.graph_store = GraphStore(
            self.project_root,

            # Keep False unless you intentionally want only one
            # relationship for each source-target pair.
            dedupe_edges_by_pair=False,
        )

    async def initialize(self) -> None:
        """
        Load the prompt and model once.

        There is no MCP process, filesystem tool, or agent loop.
        """
        load_dotenv(
            self.project_root / ".env"
        )

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY was not found "
                "in the .env file"
            )

        try:
            self.system_prompt = (
                self.prompt_file.read_text(
                    encoding="utf-8"
                )
            )

        except FileNotFoundError as error:
            raise FileNotFoundError(
                "System prompt not found: "
                f"{self.prompt_file}"
            ) from error

        except PermissionError as error:
            raise PermissionError(
                "Permission denied while reading: "
                f"{self.prompt_file}"
            ) from error

        model = ChatOpenAI(
            model="gpt-5.4-mini",
        )

        self.extractor = model.with_structured_output(
            GraphExtraction,

            # Use provider-native structured JSON rather than asking
            # the model to manually format a JSON string.
            method="json_schema",

            # Gives access to usage metadata and parsing errors.
            include_raw=True,
        )

    async def run_model(
        self,
        chunk_name: str,
        document_path: str,
        chunk_content: str,
    ) -> GraphExtraction:
        """
        Return an in-memory graph delta.

        No persistent files are visible to the model.
        """
        if (
            self.extractor is None
            or self.system_prompt is None
        ):
            raise RuntimeError(
                "The model has not been initialized"
            )

        question = f"""Extract a graph delta from this chunk.

Chunk name: {chunk_name}
Document: {document_path}

Content:
<<<CURRENT_CHUNK_BEGIN>>>
{chunk_content}
<<<CURRENT_CHUNK_END>>>
"""

        result = await self.extractor.ainvoke(
            [
                SystemMessage(
                    content=self.system_prompt
                ),
                HumanMessage(
                    content=question
                ),
            ]
        )

        parsing_error = result.get(
            "parsing_error"
        )

        if parsing_error is not None:
            raise RuntimeError(
                "Structured-output parsing failed: "
                f"{parsing_error}"
            )

        parsed = result.get("parsed")

        if parsed is None:
            raise RuntimeError(
                "The model returned no graph extraction"
            )

        if not isinstance(
            parsed,
            GraphExtraction,
        ):
            parsed = GraphExtraction.model_validate(
                parsed
            )

        raw_message = result.get("raw")

        usage = getattr(
            raw_message,
            "usage_metadata",
            None,
        )

        if usage:
            print(
                f"Token usage for {chunk_name}: "
                f"{usage}"
            )

        print(
            f"\nModel extraction for {chunk_name}:"
        )

        print(
            parsed.model_dump_json(indent=2)
        )

        return parsed

    @staticmethod
    def split_sentences(
        content: str,
    ) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                content.strip(),
            )
            if sentence.strip()
        ]

    def document_name(
        self,
        file_path: Path,
    ) -> str:
        try:
            return file_path.relative_to(
                self.project_root
            ).as_posix()

        except ValueError:
            return file_path.as_posix()

    async def classify(
        self,
        file_path: str | Path,
    ) -> None:
        source_path = Path(
            file_path
        ).expanduser()

        if not source_path.is_absolute():
            source_path = (
                self.project_root
                / source_path
            )

        source_path = source_path.resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                "The file does not exist: "
                f"{source_path}"
            )

        if not source_path.is_file():
            raise ValueError(
                "The path is not a file: "
                f"{source_path}"
            )

        if source_path.stat().st_size == 0:
            raise ValueError(
                "The file is empty: "
                f"{source_path}"
            )

        if self.CHUNK_SIZE <= self.OVERLAP:
            raise ValueError(
                "CHUNK_SIZE must be greater "
                "than OVERLAP"
            )

        try:
            content = source_path.read_text(
                encoding="utf-8"
            )

        except PermissionError as error:
            raise PermissionError(
                "Permission denied while reading: "
                f"{source_path}"
            ) from error

        sentences = self.split_sentences(
            content
        )

        if not sentences:
            raise ValueError(
                "No sentences were found in: "
                f"{source_path}"
            )

        await self.initialize()

        document_path = self.document_name(
            source_path
        )

        step = (
            self.CHUNK_SIZE
            - self.OVERLAP
        )

        chunk_number = 0

        for start in range(
            0,
            len(sentences),
            step,
        ):
            chunk_sentences = sentences[
                start : start + self.CHUNK_SIZE
            ]

            if not chunk_sentences:
                break

            chunk_name = (
                f"{document_path}"
                f"_chunk_{chunk_number}.0"
            )

            chunk_content = "\n".join(
                chunk_sentences
            )

            # Step 1: model returns only a structured graph delta.
            extraction = await self.run_model(
                chunk_name=chunk_name,
                document_path=document_path,
                chunk_content=chunk_content,
            )

            # Step 2: Python validates, deduplicates, and saves it.
            stats = self.graph_store.merge_and_write(
                extraction,
                chunk_name=chunk_name,
                document_path=document_path,
                chunk_content=chunk_content,
            )

            print(
                f"\nMerge statistics for {chunk_name}:"
            )
            print(stats)

            chunk_number += 1

            if (
                start + self.CHUNK_SIZE
                >= len(sentences)
            ):
                break


async def main() -> None:
    model = Model()

    await model.classify(
        "data/backup.txt"
    )


if __name__ == "__main__":
    asyncio.run(main())

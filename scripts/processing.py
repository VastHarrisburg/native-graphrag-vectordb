import hashlib
import json
import math
import re
from pathlib import Path


class Processing:
    """Document chunking and embeddings used by the Rust CLI."""

    _models = {}

    @classmethod
    def read_chunks(
        cls,
        file_path,
        chunk_size,
        overlap,
        document_id=None,
        source=None,
    ):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"The file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"The path is not a file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"The file is empty: {path}")
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")

        content = path.read_text(encoding="utf-8")
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", content.strip())
            if sentence.strip()
        ]

        if not sentences:
            raise ValueError(f"No text was found in: {path}")

        document_id = document_id or path.stem
        source = source or path.as_posix()
        chunks = []
        step = chunk_size - overlap

        for chunk_index, start in enumerate(range(0, len(sentences), step)):
            selected = sentences[start : start + chunk_size]

            if not selected:
                break

            chunks.append(
                {
                    "chunk_id": f"{document_id}_chunk_{chunk_index:03d}",
                    "document_id": document_id,
                    "text": " ".join(selected),
                    "chunk_index": chunk_index,
                    "source": source,
                }
            )

            if start + chunk_size >= len(sentences):
                break

        return chunks

    @classmethod
    def write_chunks(
        cls,
        file_path,
        chunk_size,
        overlap,
        output_path="data/chunks.json",
        document_id=None,
        source=None,
    ):
        chunks = cls.read_chunks(
            file_path,
            chunk_size,
            overlap,
            document_id=document_id,
            source=source,
        )
        cls.write_json(Path(output_path), chunks)
        return chunks

    @staticmethod
    def _hash_embedding(text, dimensions=384):
        """Small deterministic fallback for tests and offline demos."""
        vector = [0.0] * dimensions
        words = re.findall(r"[a-z0-9]+", text.casefold())
        features = words + [
            f"{left}_{right}"
            for left, right in zip(words, words[1:])
        ]

        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))

        if magnitude:
            vector = [value / magnitude for value in vector]

        return vector

    @classmethod
    def embed_texts(cls, texts, backend="auto", model_name=None):
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"

        if backend not in {"auto", "sentence-transformers", "hash"}:
            raise ValueError(f"Unknown embedding backend: {backend}")

        if backend != "hash":
            try:
                from sentence_transformers import SentenceTransformer

                if model_name not in cls._models:
                    cls._models[model_name] = SentenceTransformer(model_name)

                values = cls._models[model_name].encode(texts)
                return [value.tolist() for value in values], model_name
            except (ImportError, OSError):
                if backend == "sentence-transformers":
                    raise

        return [cls._hash_embedding(text) for text in texts], "hash-384"

    @classmethod
    def create_embeddings(
        cls,
        file_path="data/vectors.json",
        chunks_path="data/chunks.json",
        backend="auto",
    ):
        chunks = cls.read_json(Path(chunks_path), default=[])
        vectors, model_name = cls.embed_texts(
            [chunk.get("text", chunk.get("content", "")) for chunk in chunks],
            backend=backend,
        )
        records = [
            {
                "chunk_id": chunk.get("chunk_id", chunk.get("id")),
                "document_id": chunk.get("document_id", ""),
                "vector": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        cls.write_json(Path(file_path), records)
        return records, model_name

    @classmethod
    def convert_query(cls, query, backend="auto"):
        if not query.strip():
            raise ValueError("The query cannot be empty")

        vectors, model_name = cls.embed_texts([query], backend=backend)
        return {"vector": vectors[0], "embedding_model": model_name}

    @classmethod
    def write_query(cls, query, output_path="data/query_vector.json"):
        cls.write_json(Path(output_path), query)

    @staticmethod
    def read_json(path, default=None):
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

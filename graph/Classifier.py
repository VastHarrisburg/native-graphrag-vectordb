import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from pydantic import BaseModel, ConfigDict, Field


class GraphRAGDecision(BaseModel):
    """Structured response returned by the classifier."""

    model_config = ConfigDict(extra="forbid")

    use_graph: bool = Field(
        description=(
            "True when GraphRAG is necessary to answer the query "
            "completely; otherwise false."
        )
    )


class Classifier:
    def __init__(
        self,
        project_root: str | Path = (
            "/Users/harishkrishnan/native-graphrag-vectordb"
        ),
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()

        # Keep the routing prompt separate from the graph-ingestion prompt.
        self.prompt_file = (
            self.project_root / "graph/Classifier_Prompt.txt"
        )

        self.agent: Any | None = None

    async def initialize(self) -> None:
        """Load the routing prompt and initialize the classifier once."""

        load_dotenv(self.project_root / ".env")

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY was not found in the .env file"
            )

        try:
            system_prompt = self.prompt_file.read_text(
                encoding="utf-8"
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Classifier prompt not found: {self.prompt_file}"
            ) from error
        except PermissionError as error:
            raise PermissionError(
                f"Permission denied while reading: {self.prompt_file}"
            ) from error

        self.agent = create_agent(
            model="gpt-5-nano",
            system_prompt=system_prompt,
            response_format=GraphRAGDecision,
        )

    def load_top_chunks(
        self,
        top_chunks_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Load and validate the vector-search results."""

        path = Path(top_chunks_path).expanduser()

        if not path.is_absolute():
            path = self.project_root / path

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Top-chunks file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Top-chunks path is not a file: {path}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                f"Top-chunks file is empty: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as source:
                data = json.load(source)
        except PermissionError as error:
            raise PermissionError(
                f"Permission denied while reading: {path}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path}: {error}"
            ) from error

        if not isinstance(data, list):
            raise ValueError(
                "The top-chunks JSON must contain an array"
            )

        for index, chunk in enumerate(data):
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Top-chunks item {index} must be a JSON object"
                )

        return data

    async def classify(
        self,
        query: str,
        top_chunks_path: str | Path,
    ) -> bool:
        """
        Return True when GraphRAG is necessary.

        Return False when the vector-search chunks are sufficient.
        """

        if self.agent is None:
            raise RuntimeError(
                "The classifier has not been initialized"
            )

        query = query.strip()

        if not query:
            raise ValueError("The user query cannot be empty")

        top_chunks = self.load_top_chunks(top_chunks_path)

        # json.dumps is preferable to inserting Python's list
        # representation into the prompt.
        chunks_json = json.dumps(
            top_chunks,
            ensure_ascii=False,
            indent=2,
        )

        question = f"""Classify whether GraphRAG is necessary.

User query:
<<<QUERY_BEGIN>>>
{query}
<<<QUERY_END>>>

Top vector-search results:
<<<CHUNKS_BEGIN>>>
{chunks_json}
<<<CHUNKS_END>>>
"""

        response = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        decision = response.get("structured_response")

        if decision is None:
            raise RuntimeError(
                "The classifier did not return a structured decision"
            )

        if not isinstance(decision, GraphRAGDecision):
            decision = GraphRAGDecision.model_validate(decision)

        print(
            "GraphRAG required:",
            str(decision.use_graph).lower(),
        )

        return decision.use_graph


async def main() -> None:
    classifier = Classifier()

    await classifier.initialize()

    use_graph = await classifier.classify(
        query=(
            "How did the local cache design affect "
            "customer dashboard performance?"
        ),
        top_chunks_path="data/top_chunks.json",
    )

    if use_graph:
        print("Run GraphRAG traversal")
    else:
        print("Use the vector-search chunks only")


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from pydantic import BaseModel, ConfigDict, Field


class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[str] = Field(
        description=(
            "Important lowercase entities explicitly mentioned "
            "in the query."
        )
    )

    direction: Literal[
        "incoming",
        "outgoing",
        "both",
    ] = Field(
        description=(
            "The direction to use during graph traversal."
        )
    )


class Analyzer:
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
            self.project_root / "graph/Query_Analyzer_Prompt.txt"
        )

        self.agent: Any | None = None

    async def initialize(self) -> None:
        """
        Load the query analyzer prompt and initialize the agent.
        """

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
                f"Query analyzer prompt not found: {self.prompt_file}"
            ) from error
        except PermissionError as error:
            raise PermissionError(
                f"Permission denied while reading: {self.prompt_file}"
            ) from error

        self.agent = create_agent(
            model="gpt-5-nano",
            system_prompt=system_prompt,
            response_format=QueryAnalysis,
        )

    async def classify(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Analyze the query and return its data as a dictionary.
        """

        if self.agent is None:
            raise RuntimeError(
                "The query analyzer has not been initialized"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "The user query cannot be empty"
            )

        question = f"""Analyze this query for graph traversal.

User query:
{query}
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

        structured_response = response.get(
            "structured_response"
        )

        if structured_response is None:
            raise RuntimeError(
                "The query analyzer did not return "
                "a structured response"
            )

        if not isinstance(
            structured_response,
            QueryAnalysis,
        ):
            structured_response = QueryAnalysis.model_validate(
                structured_response
            )

        return structured_response.model_dump()


async def main() -> None:
    analyzer = Analyzer()

    await analyzer.initialize()

    query_path = analyzer.project_root / "data/query.json"

    if not query_path.exists():
        raise FileNotFoundError(
            f"Query file does not exist: {query_path}"
        )

    if query_path.stat().st_size == 0:
        raise ValueError(
            f"Query file is empty: {query_path}"
        )

    try:
        with query_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            query_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {query_path}: {error}"
        ) from error

    query = query_data.get("query")

    if not isinstance(query, str):
        raise ValueError(
            "query.json must contain a string field named 'query'"
        )

    json_data = await analyzer.classify(
        query=query
    )

    print(
        json.dumps(
            json_data,
            indent=2,
            ensure_ascii=False,
        )
    )

    output_path = (
        analyzer.project_root
        / "data/query_analysis.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")


if __name__ == "__main__":
    asyncio.run(main())

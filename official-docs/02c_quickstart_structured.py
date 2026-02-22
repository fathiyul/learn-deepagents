import json
import os
from pydantic import BaseModel, Field
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


class SearchResult(BaseModel):
    """Result of a search."""

    summary: str = Field(description="A summary of the search result")
    source: str = Field(description="The source of the search result")
    category: str = Field(description="Classification of the search result")


class SearchResults(BaseModel):
    results: list[SearchResult]


# System prompt to steer the agent to be an expert researcher
websearch_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    model="gpt-5-mini",
    tools=[internet_search],
    system_prompt=websearch_instructions,
    response_format=SearchResults,  # type: ignore
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Know any popular retail analytics blog for me to learn from and replicate?",
            }
        ]
    }
)

structured_response = result["structured_response"]

# Export result to json
with open("output/output_02c_structured.json", "w") as f:
    json.dump(structured_response.model_dump(), f, indent=2)

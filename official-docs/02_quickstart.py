from langchain_core.messages import messages_to_dict
import json
import os
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


# System prompt to steer the agent to be an expert researcher
websearch_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    model="gpt-5-mini", tools=[internet_search], system_prompt=websearch_instructions
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Know any popular sport analytics blog for me to learn from and replicate?",
            }
        ]
    }
)

# Print the agent's response
print(result["messages"][-1].content)

serialized = {"messages": messages_to_dict(result["messages"])}

# Export result to json
with open("output/output_02.json", "w") as f:
    json.dump(serialized, f, indent=2)

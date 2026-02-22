from langchain_core.messages import messages_to_dict
import argparse
import json
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--prompt", default="What is trending today in the news?")
    parser.add_argument("--mode", type=int, default=1)
    args = parser.parse_args()

    if args.mode == 1:
        # -- Using CompiledSubAgent
        # Create a custom agent graph
        websearch_graph = create_agent(
            model=args.model,
            tools=[internet_search],
            system_prompt="Your task is to find relevant data on the internet",
        )

        # Use it as a custom subagent
        websearch_subagent = CompiledSubAgent(
            name="websearch-agent",
            description="Specialized agent to find relevant data on the internet",
            runnable=websearch_graph,
        )

        mode = "compiledsubagent"

    else:
        # -- Using SubAgent
        websearch_subagent = {
            "name": "websearch-agent",
            "description": "Specialized agent to find relevant data on the internet",
            "system_prompt": "Your task is to find relevant data on the internet",
            "tools": [internet_search],
            "model": args.model,  # Optional override, defaults to main agent model
        }

        mode = "subagent"

    subagents = [websearch_subagent]

    agent = create_deep_agent(model=args.model, subagents=subagents)  # type: ignore

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt,
                }
            ]
        }
    )

    # Print the agent's response
    print(result["messages"][-1].content)

    serialized = {"messages": messages_to_dict(result["messages"])}

    # Export result to json
    with open(f"output/output_03a_{mode}.json", "w") as f:
        json.dump(serialized, f, indent=2)

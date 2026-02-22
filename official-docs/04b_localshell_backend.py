from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.messages import messages_to_dict
from pathlib import Path
import json

from dotenv import load_dotenv

load_dotenv()

script_dir = Path(__file__).parent.resolve()

agent = create_deep_agent(
    model="gpt-5-mini",
    backend=LocalShellBackend(root_dir=script_dir, virtual_mode=True),
)

prompt = (
    "1. Write and run a Python script to generate 1000 random floats "
    "from a normal distribution (mean=2, std=3).\n"
    "2. Save these numbers to a file named 'distribution.txt' in your backend.\n"
)

# prompt = (
#     "1. Write and run a javascript code to generate 1000 random floats "
#     "from a normal distribution (mean=2, std=3).\n"
#     "2. Save these numbers to a file named 'distribution.txt' in your backend.\n"
# )

result = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config={"configurable": {"thread_id": "math_session_1"}},
)

print(result["messages"][-1].content)

serialized = {"messages": messages_to_dict(result["messages"])}

# Export result to json
with open("output/output_04b_localshell_backend.json", "w") as f:
    json.dump(serialized, f, indent=2)

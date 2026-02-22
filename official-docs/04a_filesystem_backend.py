from pathlib import Path
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import messages_to_dict
import json

from dotenv import load_dotenv

load_dotenv()

# Gets the absolute path to the directory containing THIS script
script_dir = Path(__file__).parent.resolve()

agent = create_deep_agent(
    model="gpt-5-mini",
    backend=FilesystemBackend(root_dir=script_dir, virtual_mode=True),
)

# Now this prompt will actually work:
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "List the files in this project and tell me what the README says.",
            }
        ]
    }
)

# Print the agent's response
print(result["messages"][-1].content)

serialized = {"messages": messages_to_dict(result["messages"])}

# Export result to json
with open("output/output_04a_filesystem_backend.json", "w") as f:
    json.dump(serialized, f, indent=2)

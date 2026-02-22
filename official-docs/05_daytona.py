from daytona import Daytona

from deepagents import create_deep_agent
from langchain_daytona import DaytonaSandbox
import json
from langchain_core.messages import messages_to_dict

from dotenv import load_dotenv

load_dotenv()

sandbox = Daytona().create()
backend = DaytonaSandbox(sandbox=sandbox)

agent = create_deep_agent(
    model="gpt-5-mini",
    system_prompt="You are a Python coding assistant with sandbox access.",
    backend=backend,
)

prompt = (
    "1. Write and run a Python script to generate 1000 random floats "
    "from a normal distribution (mean=2, std=3).\n"
    "2. Save these numbers to a file named 'distribution.txt' in your backend.\n"
    "3. Tell me the actual mean and std of generated distribution.txt"
)

# prompt = "Create a small Python package and run pytest"

try:
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    # Move print INSIDE the try so it runs before the cleanup
    print("\n--- Agent Response ---")
    print(result["messages"][-1].content)
except Exception as e:
    print(f"Error during agent execution: {e}")
finally:
    try:
        # 1. Give it more time (120s)
        # 2. Wrap it so a timeout here doesn't crash the script
        print("Stopping sandbox...")
        sandbox.stop(timeout=120)
    except Exception as timeout_err:  # noqa: F841
        print("Sandbox took too long to stop, but it will auto-terminate. Moving on...")


serialized = {"messages": messages_to_dict(result["messages"])}

# Export result to json
with open("output/output_05_daytona.json", "w") as f:
    json.dump(serialized, f, indent=2)

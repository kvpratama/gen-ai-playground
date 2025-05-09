import os
# import asyncio
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters
# import nest_asyncio
# nest_asyncio.apply()

from smolagents import LiteLLMModel

model = LiteLLMModel(
        model_id="ollama_chat/qwen2.5-coder:3b",  # Or try other Ollama-supported models
        api_base="http://127.0.0.1:11434",  # Default Ollama local server
        num_ctx=8192,
)

# Set the Windows Proactor event loop policy
# if os.name == 'nt':  # Only do this on Windows
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
    agent.run("Please find a remedy for hangover.")
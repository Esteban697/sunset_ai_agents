from typing import Any, Callable, List
import json

from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient


MCP_CLIENT = MultiServerMCPClient(
    {
        "local_server": {
            "transport": "stdio",
            "command": "node",
            "args": [
                r"C:\Users\esteb\Videos\LangGraph\windy-webcams-mcp-server\build\index.js"
            ],
            "cwd": r"C:\Users\esteb\Videos\LangGraph\windy-webcams-mcp-server",
        }
    },
)

_MCP_TOOLS: dict[str, BaseTool] | None = None


async def get_cached_mcp_tools() -> dict[str, BaseTool]:
    global _MCP_TOOLS
    if _MCP_TOOLS is None:
        tools = await MCP_CLIENT.get_tools()
        _MCP_TOOLS = {t.name: t for t in tools}
    return _MCP_TOOLS


# @tool
# async def list_mcp_tools() -> str:
#     """List currently available tool calls from the local MCP server."""
#     tool_map = await get_cached_mcp_tools()
#     payload = [
#         {"name": t.name, "description": t.description}
#         for t in tool_map.values()
#     ]
#     return json.dumps(payload, ensure_ascii=False)


@tool
async def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Call a specific tool exposed by the local MCP server to get the requested Webcam access
    
    Argument need to be 2 character country code like ES for Spain, IT for Italy, etc

    """
    tool_map = await get_cached_mcp_tools()

    if tool_name not in tool_map:
        return json.dumps(
            {
                "error": f"Tool '{tool_name}' is not available.",
                "available_tools": sorted(tool_map.keys()),
            },
            ensure_ascii=False,
        )

    result = await tool_map[tool_name].ainvoke(arguments)

    if isinstance(result, str):
        return result

    return json.dumps(result, ensure_ascii=False, default=str)


TOOLS: List[Callable[..., Any]] = [call_mcp_tool]
# TOOLS: List[Callable[..., Any]] = [list_mcp_tools, call_mcp_tool]
import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any, Callable, Dict, List, Optional, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from react_agent.context import Context

logger = logging.getLogger(__name__)


def _log_server_message(params: Any) -> None:
    logger.warning("MCP server log: %s", params)


@tool
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


def pick_best_webcam(webcams: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    def score(cam: Dict[str, Any]) -> int:
        s = 0
        player = cam.get("player") or {}
        images = cam.get("images") or {}
        location = cam.get("location") or {}

        if player.get("live"):
            s += 50
        if player.get("day"):
            s += 20
        if images.get("current") or images.get("daylight"):
            s += 20
        if location.get("city"):
            s += 5
        if cam.get("status") == "active":
            s += 5

        return s

    ranked = sorted(webcams, key=score, reverse=True)
    return ranked[0] if ranked else None


class WindyMCPClient:
    def __init__(
        self,
        server_script_path: str,
        command: str = "node",
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.server_script_path = server_script_path
        self.command = command
        self.env = env or os.environ.copy()

        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        async with self._lock:
            if self._connected and self._session is not None:
                return

            logger.info("Connecting Windy MCP client...")

            stack = AsyncExitStack()
            try:
                server = StdioServerParameters(
                    command=self.command,
                    args=[self.server_script_path],
                    env=self.env,
                )

                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(server)
                )

                session = await stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        logging_callback=_log_server_message,
                    )
                )

                await asyncio.wait_for(session.initialize(), timeout=20)
                await asyncio.wait_for(session.send_ping(), timeout=10)

                tools_response = await asyncio.wait_for(session.list_tools(), timeout=10)
                tool_names = [tool.name for tool in tools_response.tools]
                logger.info("Available MCP tools: %s", tool_names)

                required = {"get_webcams_by_location", "get_webcam"}
                missing = required.difference(tool_names)
                if missing:
                    raise RuntimeError(
                        f"Missing MCP tools: {sorted(missing)}. Available tools={tool_names}"
                    )

                self._exit_stack = stack
                self._session = session
                self._connected = True
                logger.info("Windy MCP client connected")
            except Exception:
                await stack.aclose()
                raise

    async def close(self) -> None:
        async with self._lock:
            if self._exit_stack is not None:
                logger.info("Closing Windy MCP client...")
                await self._exit_stack.aclose()

            self._exit_stack = None
            self._session = None
            self._connected = False

    async def _ensure_connected(self) -> ClientSession:
        if not self._connected or self._session is None:
            await self.connect()

        assert self._session is not None
        return self._session

    async def call_tool_with_timeout(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_seconds: int = 30,
    ) -> Any:
        session = await self._ensure_connected()
        logger.info("Calling tool=%s args=%s", tool_name, arguments)
        started = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=timeout_seconds,
            )
            elapsed = time.perf_counter() - started
            logger.info("Tool=%s completed in %.2fs", tool_name, elapsed)
            return result
        except asyncio.TimeoutError as e:
            elapsed = time.perf_counter() - started
            logger.exception("Tool=%s timed out after %.2fs", tool_name, elapsed)
            raise RuntimeError(
                f"MCP tool '{tool_name}' timed out after {timeout_seconds}s"
            ) from e

    async def get_country_live_feed(
        self,
        country_code: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        logger.info("Starting get_country_live_feed country=%s limit=%s",
            country_code,
            limit,
        )

        result = await self.call_tool_with_timeout(
            "get_webcams_by_location",
            {
                "location_type": "country",
                "location_code": country_code.upper(),
                "limit": limit,
                "order": "popularity",
            },
            timeout_seconds=45,
        )

        if not result.content:
            raise RuntimeError("get_webcams_by_location returned no content")
        
        logger.info("get_webcams_by_location returned result: %s", result)

        payload = getattr(result.content, "text", None)
        if not payload:
            raise RuntimeError(f"Expected text content, got: {result.content!r}")

        try:
            data = json.loads(payload)
        except Exception as e:
            logger.exception("Failed to parse location payload: %r", payload[:1000])
            raise RuntimeError("Location payload was not valid JSON") from e

        webcams = ((data.get("data") or {}).get("webcams")) or []
        best = pick_best_webcam(webcams)

        if not best:
            return {
                "country_code": country_code.upper(),
                "found": False,
                "message": "No webcams found for that country.",
            }

        webcam_id = best.get("id")
        if not webcam_id:
            raise RuntimeError(f"Best webcam missing id: {best!r}")

        details = await self.call_tool_with_timeout(
            "get_webcam",
            {
                "webcam_id": webcam_id,
                "lang": "en",
            },
            timeout_seconds=45,
        )

        if not details.content:
            raise RuntimeError("get_webcam returned no content")

        detail_payload = getattr(details.content, "text", None)
        if not detail_payload:
            raise RuntimeError(f"Expected text detail content, got: {details.content!r}")

        try:
            detail_data = json.loads(detail_payload)
        except Exception as e:
            logger.exception("Failed to parse detail payload: %r", detail_payload[:1000])
            raise RuntimeError("Detail payload was not valid JSON") from e

        webcam = detail_data.get("data") or {}
        images = webcam.get("images") or {}
        player = webcam.get("player") or {}
        location = webcam.get("location") or {}

        return {
            "country_code": country_code.upper(),
            "found": True,
            "webcam": {
                "id": webcam.get("id"),
                "title": webcam.get("title"),
                "city": location.get("city"),
                "region": (location.get("region") or {}).get("name"),
                "country": (location.get("country") or {}).get("name"),
                "current_image": (images.get("current") or {}).get("preview"),
                "daylight_image": (images.get("daylight") or {}).get("preview"),
                "live_player": player.get("live"),
                "day_player": player.get("day"),
                "web_url": (webcam.get("urls") or {}).get("web"),
            },
        }


WINDY_MCP_CLIENT = WindyMCPClient(
    server_script_path=r"C:\Users\esteb\Videos\LangGraph\windy-webcams-mcp-server\build\index.js",
    command="node",
    env=os.environ.copy(),
)


@tool
async def get_live_country_webcam(country_code: str) -> str:
    """Retrieve a current webcam feed for a selected country code.

    Input must be a 2-letter ISO country code such as ES, JP, US, FR, or DE.
    Returns JSON with the selected webcam's live player URL and current image URL when available.
    """
    result = await WINDY_MCP_CLIENT.get_country_live_feed(country_code)
    return json.dumps(result)


TOOLS: List[Callable[..., Any]] = [search, get_live_country_webcam]
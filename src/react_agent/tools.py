import json
import asyncio
from typing import Any, Dict, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime
from langchain_core.tools import tool

from react_agent.context import Context
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_SERVER_URL = "http://localhost:6274/mcp"

#OPTIONAL CLASS: WindyWebcamSessionManager
class WindyWebcamSessionManager:
    def __init__(self) -> None:
        self._session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None
        self._stdio_cm = None
        self._session_cm = None
        self._lock = asyncio.Lock()

    async def get_session(self) -> ClientSession:
        if self._session is not None:
            return self._session

        async with self._lock:
            if self._session is not None:
                return self._session

            server = StdioServerParameters(
                command="npx",
                #args=["windy-webcam-mcp-server"],
                args=["/c", "npx", "-y", "windy-webcam-mcp-server"]
            )

            self._stdio_cm = stdio_client(server)
            self._read_stream, self._write_stream = await self._stdio_cm.__aenter__()

            self._session_cm = ClientSession(self._read_stream, self._write_stream)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()

            return self._session

    async def close(self) -> None:
        async with self._lock:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
                self._session_cm = None
                self._session = None

            if self._stdio_cm is not None:
                await self._stdio_cm.__aexit__(None, None, None)
                self._stdio_cm = None
                self._read_stream = None
                self._write_stream = None



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


@tool
async def get_live_country_webcam(country_code: str) -> str:
    """Retrieve a current webcam feed for a selected country code.

    Input must be a 2-letter ISO country code such as ES, JP, US, FR, or DE.
    Returns JSON with the selected webcam's live player URL and current image URL when available.
    """
    result = await _get_country_live_feed_async(country_code)
    return json.dumps(result)


TOOLS: List[Callable[..., Any]] = [search, get_live_country_webcam]


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


async def _get_country_live_feed_async(country_code: str, limit: int = 15) -> Dict[str, Any]:
    async with streamable_http_client(MCP_SERVER_URL) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                "get_webcams_by_location",
                {
                    "location_type": "country",
                    "location_code": country_code.upper(),
                    "limit": limit,
                    "order": "popularity",
                },
            )

            payload = result.content[0].text
            data = json.loads(payload)
            webcams = data["data"]["webcams"]

            best = pick_best_webcam(webcams)
            if not best:
                return {
                    "country_code": country_code.upper(),
                    "found": False,
                    "message": "No webcams found for that country.",
                }

            webcam_id = best["id"]

            details = await session.call_tool(
                "get_webcam",
                {
                    "webcam_id": webcam_id,
                    "lang": "en",
                },
            )

            detail_payload = details.content[0].text
            detail_data = json.loads(detail_payload)
            webcam = detail_data["data"]

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
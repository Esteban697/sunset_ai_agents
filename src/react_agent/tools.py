import os
import asyncio
import json
import logging
import time
from contextlib import AsyncExitStack
from typing import Any, Callable, Dict, List, Optional, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from react_agent.context import Context


logger = logging.getLogger(__name__)


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
    def __init__(self, server_url: str = "http://localhost:6277/mcp") -> None:
        self.server_url = server_url.rstrip("/")
        #self.session_token = "b8875922da1d0148ecb962c5c0ba477b0d0a1005a1ed64f473dc2a91ac1e4157"

    async def _call_tool_once(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_seconds: int = 30,
    ) -> Any:
        logger.info("Connecting MCP over HTTP to %s", self.server_url)
        started = time.perf_counter()

        async with AsyncExitStack() as stack:
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamablehttp_client(
                    self.server_url,
                    # headers={
                    #     "Authorization": f"Bearer {self.session_token}"
                    # },
                )
            )

            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await asyncio.wait_for(session.initialize(), timeout=20)

            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=timeout_seconds,
            )

            elapsed = time.perf_counter() - started
            logger.info("Tool=%s completed in %.2fs", tool_name, elapsed)
            return result

    async def call_tool_with_timeout(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_seconds: int = 30,
    ) -> Any:
        try:
            return await self._call_tool_once(tool_name, arguments, timeout_seconds)
        except asyncio.CancelledError:
            logger.exception("Tool=%s was cancelled by the runtime", tool_name)
            raise RuntimeError(f"MCP tool '{tool_name}' was cancelled") from None
        except asyncio.TimeoutError as e:
            logger.exception("Tool=%s timed out after %ss", tool_name, timeout_seconds)
            raise RuntimeError(
                f"MCP tool '{tool_name}' timed out after {timeout_seconds}s"
            ) from e

    async def get_country_live_feed(
        self,
        country_code: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
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

        content = getattr(result, "content", None)
        if not content:
            raise RuntimeError("get_webcams_by_location returned no content")

        first_item = content[0] if isinstance(content, list) and content else None
        payload = getattr(first_item, "text", None) if first_item else None
        if not payload:
            raise RuntimeError(f"Expected text content, got: {content!r}")

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

        detail_content = getattr(details, "content", None)
        if not detail_content:
            raise RuntimeError("get_webcam returned no content")

        first_detail_item = (
            detail_content[0] if isinstance(detail_content, list) and detail_content else None
        )
        detail_payload = getattr(first_detail_item, "text", None) if first_detail_item else None
        if not detail_payload:
            raise RuntimeError(f"Expected text detail content, got: {detail_content!r}")

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


WINDY_MCP_CLIENT = WindyMCPClient(server_url="http://localhost:6277/mcp")


@tool
async def get_live_country_webcam(country_code: str) -> str:
    """Retrieve a current webcam feed for a selected country code.

    Input must be a 2-letter ISO country code such as ES, JP, US, FR, or DE.
    Returns JSON with the selected webcam's live player URL and current image URL when available.
    """
    result = await WINDY_MCP_CLIENT.get_country_live_feed(country_code)
    return json.dumps(result)


TOOLS: List[Callable[..., Any]] = [search, get_live_country_webcam]
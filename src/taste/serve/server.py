"""MCP server: a thin wrapper over `tools.py`.

No `from __future__ import annotations` here: the SDK inspects the tool functions'
annotations to build their input schemas.

stdout carries MCP protocol traffic only. Anything a tool (or a model library it loads)
prints is redirected to stderr.
"""

import contextlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from taste.config import load_settings
from taste.serve import tools
from taste.store import Store

log = logging.getLogger("taste.serve")


def _sdk():
    """(server class, Image helper, ToolError). mcp 2.x renamed FastMCP to MCPServer."""
    try:
        from mcp.server.mcpserver import Image, MCPServer
        from mcp.server.mcpserver.exceptions import ToolError

        return MCPServer, Image, ToolError
    except ImportError:
        from mcp.server.fastmcp import FastMCP, Image
        from mcp.server.fastmcp.exceptions import ToolError

        return FastMCP, Image, ToolError


def build_server(home: str | Path | None = None):
    """Build the server object. Nothing heavy is loaded: the embedder comes on first use."""
    server_cls, image_cls, tool_error = _sdk()
    store = Store(load_settings(home).home)
    server = server_cls("taste", instructions=tools.SERVER_INSTRUCTIONS)

    extra: dict[str, Any] = {}
    if "structured_output" in inspect.signature(server.tool).parameters:
        extra["structured_output"] = False  # replies are text and images for the agent to read

    @contextlib.contextmanager
    def guard():
        """Keep stdout clean and turn bad-input errors into readable tool errors."""
        try:
            with contextlib.redirect_stdout(sys.stderr):
                yield
        except tools.ToolInputError as exc:
            raise tool_error(str(exc)) from exc

    def tool(name: str):
        return server.tool(name=name, description=tools.TOOL_DESCRIPTIONS[name], **extra)

    @tool("get_taste")
    def get_taste(domain: Optional[str] = None,
                  detail: Literal["summary", "full"] = "summary") -> str:
        with guard():
            return tools.get_taste(store, domain, detail)

    @tool("get_exemplars")
    def get_exemplars(query: str, domain: Optional[str] = None, k: int = 6) -> list:
        with guard():
            reply = tools.get_exemplars(store, query, domain, k)
        return [reply.text, *[image_cls(data=t.data, format=t.format) for t in reply.thumbnails]]

    @tool("critique")
    def critique(image_path: str) -> str:
        with guard():
            return tools.critique(store, image_path).text

    @tool("log_feedback")
    def log_feedback(verdict: Literal["like", "dislike", "rule"], note: str,
                     image_path: Optional[str] = None, domain: Optional[str] = None,
                     context: Optional[str] = None) -> str:
        with guard():
            fb = tools.log_feedback(store, verdict, note, image_path, domain, context)
            return tools.feedback_confirmation(store, fb)

    return server


def run_stdio(home: str | Path | None = None) -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    server = build_server(home)
    log.info("taste MCP server starting on stdio")
    server.run("stdio")

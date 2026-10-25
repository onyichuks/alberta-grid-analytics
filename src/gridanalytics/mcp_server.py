"""Model Context Protocol server: lets an AI assistant query the Alberta grid analysis through tools.

Run:     python -m gridanalytics.mcp_server        (from the repository, with src on PYTHONPATH)
Connect: add it to any MCP client (see README, "Ask the analysis with an AI agent").

The server only reads the published result files. Every answer carries the table it came from, so a person
can check it, and the assistant cannot change data or results.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from gridanalytics import agent_tools as T

mcp = FastMCP("alberta-grid-analytics")
for fn in T.TOOLS:
    mcp.tool()(fn)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

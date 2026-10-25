"""Start the MCP server as a client would (over stdio), list its tools and ask three questions.

Usage:  python scripts/try_mcp.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    params = StdioServerParameters(command=sys.executable, args=["-m", "gridanalytics.mcp_server"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])
            for name, args in [("constrained_energy", {"year": 2025, "planning_area": "Medicine Hat"}),
                               ("forecast_at", {"local_time": "2024-01-13 18:00"}),
                               ("asset_value", {"asset": "Travers"})]:
                r = await s.call_tool(name, args)
                print(f"\n{name}({args}) ->")
                print(json.dumps(json.loads(r.content[0].text), indent=1)[:700])


if __name__ == "__main__":
    asyncio.run(main())

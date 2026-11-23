# show_tools.py
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from wedding_rag.agents.budget_agent import clean_schema


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "wedding_rag.mcp_server.server"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            for t in (await s.list_tools()).tools:
                print(f"\n== {t.name}\n{t.description}\n-- raw schema:\n{json.dumps(t.inputSchema, indent=1)}"
                      f"\n-- cleaned for Gemini:\n{json.dumps(clean_schema(t.inputSchema), indent=1)}")


asyncio.run(main())
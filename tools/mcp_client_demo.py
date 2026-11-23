# mcp_client_demo.py
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "wedding_rag.mcp_server.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

            res = await session.call_tool("calculate_budget", {
                "total": 30000, "allocations": [{"category": "venue", "amount": 12000},
                                                {"category": "catering", "amount": 9000},
                                                {"category": "photography", "amount": 3000}]})

            print("budget:", res.content[0].text)

            res = await session.call_tool("search_threads", {"query": "how much does a celebrant cost",
                                                 "category": "celebrant", "k": 3})

            for block in res.content:
                print("search:", block.text[:300])

            res = await session.call_tool("calculate_budget", {"total": -5, "allocations": [{"category": "a", "amount": 1}]})
            print("bad input -> isError:", res.isError)


asyncio.run(main())
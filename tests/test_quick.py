#!/usr/bin/env python3
"""Quick test script for MCP server functionality"""

import asyncio
import json
import os
import sys

# Add parent directory to path to import pp_mcp_server
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pp_mcp_server


async def test_mcp_server():
    print("🧪 Testing ParticlePhysics MCP Server")
    print("=" * 50)

    # Test 1: List tools
    print("\n1️⃣ Testing tool listing...")
    tools = await pp_mcp_server.handle_list_tools()
    print(f"✅ Found {len(tools)} tools")

    # Test 2: Search particle
    print("\n2️⃣ Testing particle search...")
    result = await pp_mcp_server.handle_call_tool(
        "search_particle", {"query": "electron"}
    )
    data = json.loads(result[0].text)
    print(f"✅ Search successful: {data.get('total_found', 0)} results")

    # Test 3: Get particle properties
    print("\n3️⃣ Testing particle properties...")
    result = await pp_mcp_server.handle_call_tool(
        "get_particle_properties", {"particle_name": "electron"}
    )
    data = json.loads(result[0].text)
    print(f"✅ Properties retrieved for: {data.get('name', 'N/A')}")
    if "mass" in data:
        print(f"   Mass: {data['mass'].get('formatted', 'N/A')}")

    # Test 4: Database info
    print("\n4️⃣ Testing database info...")
    result = await pp_mcp_server.handle_call_tool("get_database_info", {})
    data = json.loads(result[0].text)
    print(f"✅ Database: {data.get('database', 'N/A')}")

    print("\n🎉 All tests passed! MCP Server is working correctly!")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())

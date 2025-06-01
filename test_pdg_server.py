#!/usr/bin/env python3
"""
Test script for PDG MCP Server

This script tests the basic functionality of the PDG MCP server
to ensure all tools work correctly.
"""

import asyncio
import json
import sys
import traceback
from typing import Any, Dict


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import mcp.server
        import mcp.types as types

        print("✓ MCP packages imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MCP: {e}")
        return False

    try:
        import pdg

        api = pdg.connect()
        print("✓ PDG package imported and connected successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import PDG: {e}")
        print("  Install with: pip install pdg")
        return False
    except Exception as e:
        print(f"✗ Failed to connect to PDG database: {e}")
        return False


def test_basic_pdg_functionality():
    """Test basic PDG API functionality."""
    print("\nTesting basic PDG functionality...")

    try:
        import pdg

        api = pdg.connect()

        # Test getting a simple particle
        particle = api.get_particle_by_name("e-")
        print(f"✓ Retrieved electron: {particle.name}")

        # Test getting particle by MCID
        proton = api.get_particle_by_mcid(2212)
        print(f"✓ Retrieved proton by MCID: {proton.name}")

        # Test getting particle properties
        mass = particle.mass
        print(f"✓ Electron mass: {mass:.6f} GeV")

        return True
    except Exception as e:
        print(f"✗ Basic PDG functionality test failed: {e}")
        traceback.print_exc()
        return False


async def test_mcp_server_tools():
    """Test that the MCP server tools are properly defined."""
    print("\nTesting MCP server tool definitions...")

    try:
        # Import the server module
        # Create a mock server to test tool definitions
        from mcp import server

        import pdg_mcp_server as mcp_module

        # Test that we can access the handle_list_tools function
        tools = await mcp_module.handle_list_tools()

        expected_tools = [
            "search_particle",
            "get_particle_properties",
            "get_branching_fractions",
            "list_particles",
            "get_particle_by_mcid",
            "compare_particles",
            "get_database_info",
        ]

        tool_names = [tool.name for tool in tools]

        for expected_tool in expected_tools:
            if expected_tool in tool_names:
                print(f"✓ Tool '{expected_tool}' is properly defined")
            else:
                print(f"✗ Tool '{expected_tool}' is missing")
                return False

        print(f"✓ All {len(expected_tools)} tools are properly defined")
        return True

    except Exception as e:
        print(f"✗ MCP server tool test failed: {e}")
        traceback.print_exc()
        return False


async def simulate_tool_call(tool_name: str, arguments: Dict[str, Any]):
    """Simulate a tool call to test functionality."""
    try:
        import pdg_mcp_server as mcp_module

        # Create a simple test by calling the tool handler
        result = await mcp_module.handle_call_tool(tool_name, arguments)

        if result and len(result) > 0:
            response_text = result[0].text
            response_data = json.loads(response_text)

            if "error" in response_data:
                print(f"✗ Tool '{tool_name}' returned error: {response_data['error']}")
                return False
            else:
                print(f"✓ Tool '{tool_name}' executed successfully")
                return True
        else:
            print(f"✗ Tool '{tool_name}' returned no result")
            return False

    except Exception as e:
        print(f"✗ Tool '{tool_name}' failed: {e}")
        return False


async def test_tool_functionality():
    """Test actual tool functionality with sample calls."""
    print("\nTesting tool functionality...")

    test_cases = [
        {
            "tool": "search_particle",
            "args": {"query": "e-"},
            "description": "Search for electron",
        },
        {
            "tool": "get_particle_properties",
            "args": {"particle_name": "p"},
            "description": "Get proton properties",
        },
        {
            "tool": "get_particle_by_mcid",
            "args": {"mcid": 211},
            "description": "Get pi+ by Monte Carlo ID",
        },
        {
            "tool": "list_particles",
            "args": {"particle_type": "lepton", "limit": 5},
            "description": "List 5 leptons",
        },
        {
            "tool": "compare_particles",
            "args": {"particle_names": ["e-", "mu-"], "properties": ["mass", "charge"]},
            "description": "Compare electron and muon",
        },
        {
            "tool": "get_database_info",
            "args": {},
            "description": "Get database information",
        },
    ]

    success_count = 0

    for test_case in test_cases:
        print(f"\n  Testing: {test_case['description']}")
        if await simulate_tool_call(test_case["tool"], test_case["args"]):
            success_count += 1

    print(f"\n✓ {success_count}/{len(test_cases)} tool tests passed")
    return success_count == len(test_cases)


async def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTesting error handling...")

    error_test_cases = [
        {
            "tool": "search_particle",
            "args": {"query": "nonexistent_particle_xyz"},
            "description": "Search for non-existent particle",
        },
        {
            "tool": "get_particle_properties",
            "args": {"particle_name": "invalid_particle"},
            "description": "Get properties of invalid particle",
        },
        {
            "tool": "get_particle_by_mcid",
            "args": {"mcid": 999999},
            "description": "Get particle by invalid MCID",
        },
    ]

    success_count = 0

    for test_case in error_test_cases:
        print(f"\n  Testing error handling: {test_case['description']}")
        try:
            import pdg_mcp_server as mcp_module

            result = await mcp_module.handle_call_tool(
                test_case["tool"], test_case["args"]
            )

            if result and len(result) > 0:
                response_text = result[0].text
                response_data = json.loads(response_text)

                # For error cases, we expect either an error field or a "No particles found" message
                if (
                    isinstance(response_data, list)
                    and len(response_data) > 0
                    and "error" in response_data[0]
                ) or (
                    "error" in response_data
                    or "No particles found" in str(response_data)
                    or "message" in response_data
                ):
                    print(f"✓ Error properly handled")
                    success_count += 1
                else:
                    print(f"✗ Error not properly handled: {response_data}")
            else:
                print(f"✗ No error response returned")

        except Exception as e:
            print(f"✓ Exception properly caught: {type(e).__name__}")
            success_count += 1

    print(f"\n✓ {success_count}/{len(error_test_cases)} error handling tests passed")
    return success_count == len(error_test_cases)


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<40} {status}")

    print("-" * 60)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The PDG MCP server is ready to use.")
        return True
    else:
        print(
            f"\n⚠️  {total_tests - passed_tests} tests failed. Please check the errors above."
        )
        return False


async def main():
    """Run all tests."""
    print("PDG MCP Server Test Suite")
    print("=" * 60)

    # Run all tests
    results = {}

    results["Import Test"] = test_imports()
    if not results["Import Test"]:
        print("\n❌ Critical failure: Cannot proceed without proper imports")
        print_summary(results)
        return

    results["Basic PDG Functionality"] = test_basic_pdg_functionality()
    results["MCP Server Tools"] = await test_mcp_server_tools()
    results["Tool Functionality"] = await test_tool_functionality()
    results["Error Handling"] = await test_error_handling()

    # Print final summary
    success = print_summary(results)

    if success:
        print("\n🚀 Server is ready! You can now:")
        print("   1. Run the server: python mcp.py")
        print("   2. Configure it in your MCP client")
        print("   3. Start exploring particle physics data!")
    else:
        print("\n🔧 Please fix the failing tests before using the server.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)

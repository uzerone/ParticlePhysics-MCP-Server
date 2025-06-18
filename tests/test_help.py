#!/usr/bin/env python3
"""
Test the help system for ParticlePhysics MCP Server
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pp_mcp_help


def test_help_generation():
    """Test all help generation formats."""
    print("Testing ParticlePhysics MCP Server Help System")
    print("=" * 60)

    # Test simple format
    print("\n1. Testing simple format...")
    try:
        simple_output = pp_mcp_help.generate_simple_list()
        assert "PARTICLEPHYSICS MCP SERVER - TOOL LIST" in simple_output
        assert "TOTAL TOOLS:" in simple_output
        print("✅ Simple format works")
    except Exception as e:
        print(f"❌ Simple format failed: {e}")
        return False

    # Test LLM format
    print("\n2. Testing LLM format...")
    try:
        llm_output = pp_mcp_help.generate_llm_prompt_helper()
        assert "ParticlePhysics MCP Server Tool Reference for LLMs" in llm_output
        assert "Example Tool Calls" in llm_output
        print("✅ LLM format works")
    except Exception as e:
        print(f"❌ LLM format failed: {e}")
        return False

    # Test markdown format
    print("\n3. Testing markdown format...")
    try:
        md_output = pp_mcp_help.generate_markdown_help()
        assert "# ParticlePhysics MCP Server - Available Tools" in md_output
        assert "## Table of Contents" in md_output
        print("✅ Markdown format works")
    except Exception as e:
        print(f"❌ Markdown format failed: {e}")
        return False

    # Test JSON format
    print("\n4. Testing JSON format...")
    try:
        json_data = pp_mcp_help.generate_json_help()
        assert json_data["server"] == "ParticlePhysics MCP Server"
        assert "modules" in json_data
        assert json_data["total_tools"] > 0

        # Verify JSON is serializable
        json_str = json.dumps(json_data, indent=2)
        assert len(json_str) > 0

        print(f"✅ JSON format works (found {json_data['total_tools']} tools)")
    except Exception as e:
        print(f"❌ JSON format failed: {e}")
        return False

    # Test tool counting
    print("\n5. Verifying tool counts...")
    total_tools = 0
    for module_name, module_data in json_data["modules"].items():
        tool_count = module_data["tool_count"]
        print(f"   - {module_name}: {tool_count} tools")
        total_tools += tool_count

    assert total_tools == json_data["total_tools"]
    print(f"✅ Total tools verified: {total_tools}")

    print("\n" + "=" * 60)
    print("🎉 All help system tests passed!")
    return True


if __name__ == "__main__":
    success = test_help_generation()
    sys.exit(0 if success else 1)

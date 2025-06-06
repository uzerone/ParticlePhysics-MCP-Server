#!/usr/bin/env python3
"""
Basic test suite for ParticlePhysics MCP Server
This is a minimal test to ensure the CI workflow runs successfully.
"""

import sys
import traceback


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import mcp.types

        print("✅ MCP types imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MCP types: {e}")
        return False

    try:
        import pp_mcp_server

        print("✅ PP MCP server imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PP MCP server: {e}")
        return False

    try:
        from modules import (
            api,
            data,
            decay,
            errors,
            measurement,
            particle,
            units,
            utils,
        )

        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return False

    return True


def test_server_functions():
    """Test that server functions exist."""
    print("Testing server functions...")

    try:
        import pp_mcp_server

        if hasattr(pp_mcp_server, "run_server"):
            print("✅ run_server function exists")
        else:
            print("❌ run_server function not found")
            return False

        if hasattr(pp_mcp_server, "ensure_pdg_connection"):
            print("✅ ensure_pdg_connection function exists")
        else:
            print("❌ ensure_pdg_connection function not found")
            return False

    except Exception as e:
        print(f"❌ Error testing server functions: {e}")
        return False

    return True


def test_module_tools():
    """Test that module tools can be loaded."""
    print("Testing module tools...")

    try:
        from modules.api import get_api_tools
        from modules.data import get_data_tools
        from modules.decay import get_decay_tools
        from modules.errors import get_error_tools
        from modules.measurement import get_measurement_tools
        from modules.particle import get_particle_tools
        from modules.units import get_units_tools
        from modules.utils import get_utils_tools

        api_tools = get_api_tools()
        data_tools = get_data_tools()
        decay_tools = get_decay_tools()
        error_tools = get_error_tools()
        measurement_tools = get_measurement_tools()
        particle_tools = get_particle_tools()
        units_tools = get_units_tools()
        utils_tools = get_utils_tools()

        total_tools = (
            len(api_tools)
            + len(data_tools)
            + len(decay_tools)
            + len(error_tools)
            + len(measurement_tools)
            + len(particle_tools)
            + len(units_tools)
            + len(utils_tools)
        )

        print(f"✅ Loaded {total_tools} tools across 8 modules")
        print(f"   - API tools: {len(api_tools)}")
        print(f"   - Data tools: {len(data_tools)}")
        print(f"   - Decay tools: {len(decay_tools)}")
        print(f"   - Error tools: {len(error_tools)}")
        print(f"   - Measurement tools: {len(measurement_tools)}")
        print(f"   - Particle tools: {len(particle_tools)}")
        print(f"   - Units tools: {len(units_tools)}")
        print(f"   - Utils tools: {len(utils_tools)}")

    except Exception as e:
        print(f"❌ Error loading module tools: {e}")
        traceback.print_exc()
        return False

    return True


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")

    try:
        from modules.config import config

        print(f"✅ Configuration loaded: environment={config.environment}")
        print(f"   - Cache enabled: {config.cache.enabled}")
        print(f"   - Rate limiting enabled: {config.rate_limit.enabled}")
        print(f"   - Security validation: {config.security.input_validation}")

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

    return True


def test_json_config():
    """Test that the MCP JSON config is valid."""
    print("Testing JSON configuration...")

    try:
        import json
        import os

        config_file = "mcp-server.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                data = json.load(f)

            assert "mcpServers" in data, "Missing mcpServers key"
            assert (
                "particlephysics" in data["mcpServers"]
            ), "Missing particlephysics server"
            server = data["mcpServers"]["particlephysics"]
            assert "command" in server, "Missing command"
            assert "args" in server, "Missing args"

            print("✅ MCP JSON configuration is valid")
        else:
            print("⚠️  MCP JSON configuration file not found (optional)")

    except Exception as e:
        print(f"❌ Error testing JSON config: {e}")
        return False

    return True


def run_all_tests():
    """Run all tests and return success status."""
    print("=" * 60)
    print("ParticlePhysics MCP Server - Basic Test Suite")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Server Function Tests", test_server_functions),
        ("Module Tools Tests", test_module_tools),
        ("Configuration Tests", test_configuration),
        ("JSON Config Tests", test_json_config),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: FAILED with exception: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

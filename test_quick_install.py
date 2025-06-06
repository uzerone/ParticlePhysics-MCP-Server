#!/usr/bin/env python3
"""
Test script to verify the ParticlePhysics MCP Server quick installation.
"""

import subprocess
import sys
import json
import tempfile
import os


def test_uvx_installation():
    """Test installation via uvx."""
    print("Testing uvx installation method...")
    
    try:
        # Test if uvx is available
        result = subprocess.run(["uvx", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ uvx is not installed. Install it with: pip install uvx")
            return False
        print("✅ uvx is available")
        
        # Test running the server with uvx
        cmd = [
            "uvx",
            "--from",
            "git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git",
            "pp-mcp-server",
            "--help"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Server can be run via uvx")
            return True
        else:
            print(f"❌ Failed to run server: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_pipx_installation():
    """Test installation via pipx."""
    print("\nTesting pipx installation method...")
    
    try:
        # Test if pipx is available
        result = subprocess.run(["pipx", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ pipx is not installed. Install it with: pip install pipx")
            return False
        print("✅ pipx is available")
        
        # Test running the server with pipx
        cmd = [
            "pipx",
            "run",
            "--spec",
            "git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git",
            "pp-mcp-server",
            "--help"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Server can be run via pipx")
            return True
        else:
            print(f"❌ Failed to run server: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_json_config():
    """Test the JSON configuration files."""
    print("\nTesting JSON configuration files...")
    
    config_files = [
        "mcp-server.json",
        "mcp-config-easy.json",
        "claude-desktop-quick.json"
    ]
    
    all_valid = True
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
                print(f"✅ {config_file} is valid JSON")
            except json.JSONDecodeError as e:
                print(f"❌ {config_file} has invalid JSON: {e}")
                all_valid = False
        else:
            print(f"⚠️  {config_file} not found")
    
    return all_valid


def main():
    """Run all tests."""
    print("🧪 Testing ParticlePhysics MCP Server Quick Installation")
    print("=" * 60)
    
    results = {
        "JSON Config": test_json_config(),
        "uvx Method": test_uvx_installation(),
        "pipx Method": test_pipx_installation(),
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("-" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! The quick installation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
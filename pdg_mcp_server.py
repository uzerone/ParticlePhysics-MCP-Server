#!/usr/bin/env python3
"""
PDG (Particle Data Group) MCP Server

This MCP server provides access to particle physics data from the Particle Data Group
through their Python API. It allows users to search for particles, get their properties,
branching fractions, and other physics data in a user-friendly way.

Organized into 8 specialized modules:
- api: Core API functionality (search, properties, etc.)
- data: Data handling and measurements
- decay: Decay analysis and branching fractions  
- errors: Error handling and diagnostics
- measurement: PDG measurement objects and analysis
- particle: PDG particle objects and quantum numbers
- units: Unit conversions and physics constants
- utils: PDG utility functions and data processing

Features:
- 64 MCP tools across 8 specialized modules
- Search particles by name, Monte Carlo ID, or PDG ID
- Get particle properties (mass, lifetime, quantum numbers, etc.)
- Access branching fractions and decay information
- Get measurements and references
- List all available particles
- Advanced data handling and measurements
- Decay analysis with subdecay support
- Comprehensive error handling and diagnostics
- Physics unit conversions and constants
- PDG utility functions and data processing
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import modular components
from pdg_modules import api, data, decay, errors, measurement, particle, units, utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdg-mcp-server")

# Global variable to store the PDG API connection
pdg_api = None


def ensure_pdg_connection():
    """Ensure PDG API is connected, with helpful error handling."""
    global pdg_api
    if pdg_api is None:
        try:
            import pdg

            pdg_api = pdg.connect()
            logger.info("Successfully connected to PDG database")
        except ImportError:
            raise Exception(
                "PDG package not installed. Please install it using: pip install pdg"
            )
        except Exception as e:
            raise Exception(f"Failed to connect to PDG database: {str(e)}")
    return pdg_api


# Create the MCP server
server = Server("pdg-server")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available PDG tools from all modules."""
    all_tools = []
    
    # Collect tools from all modules
    all_tools.extend(api.get_api_tools())
    all_tools.extend(data.get_data_tools())
    all_tools.extend(decay.get_decay_tools())
    all_tools.extend(errors.get_error_tools())
    all_tools.extend(measurement.get_measurement_tools())
    all_tools.extend(particle.get_particle_tools())
    all_tools.extend(units.get_units_tools())
    all_tools.extend(utils.get_utils_tools())
    
    logger.info(f"Loaded {len(all_tools)} tools from modular structure")
    return all_tools


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls by routing to appropriate module handlers."""
    try:
        # Ensure PDG API connection
        api_instance = ensure_pdg_connection()
        
        # Get all tool names from each module for routing
        api_tool_names = {tool.name for tool in api.get_api_tools()}
        data_tool_names = {tool.name for tool in data.get_data_tools()}
        decay_tool_names = {tool.name for tool in decay.get_decay_tools()}
        error_tool_names = {tool.name for tool in errors.get_error_tools()}
        measurement_tool_names = {tool.name for tool in measurement.get_measurement_tools()}
        particle_tool_names = {tool.name for tool in particle.get_particle_tools()}
        units_tool_names = {tool.name for tool in units.get_units_tools()}
        utils_tool_names = {tool.name for tool in utils.get_utils_tools()}
        
        # Route to appropriate module handler
        if name in api_tool_names:
            return await api.handle_api_tools(name, arguments, api_instance)
        elif name in data_tool_names:
            return await data.handle_data_tools(name, arguments, api_instance)
        elif name in decay_tool_names:
            return await decay.handle_decay_tools(name, arguments, api_instance)
        elif name in error_tool_names:
            return await errors.handle_error_tools(name, arguments, api_instance)
        elif name in measurement_tool_names:
            return await measurement.handle_measurement_tools(name, arguments, api_instance)
        elif name in particle_tool_names:
            return await particle.handle_particle_tools(name, arguments, api_instance)
        elif name in units_tool_names:
            return await units.handle_units_tools(name, arguments, api_instance)
        elif name in utils_tool_names:
            return await utils.handle_utils_tools(name, arguments, api_instance)
        else:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2),
                )
            ]

    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"Tool execution failed: {str(e)}",
                        "tool": name,
                        "arguments": arguments,
                    },
                    indent=2,
                ),
            )
        ]


async def main():
    """Main server entry point."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pdg-server",
                server_version="2.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def run_server():
    """Run the server with proper error handling."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting PDG MCP Server...")
    logger.info("Available modules: api, data, decay, errors, measurement, particle, units, utils")
    logger.info("Total tools available: 64 across 8 modules")
    run_server()
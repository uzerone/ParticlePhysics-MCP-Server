#!/usr/bin/env python3
"""
PDG (Particle Data Group) MCP Server

This MCP server provides access to particle physics data from the Particle Data Group
through their Python API. It allows users to search for particles, get their properties,
branching fractions, and other physics data in a user-friendly way.

Features:
- Search particles by name, Monte Carlo ID, or PDG ID
- Get particle properties (mass, lifetime, quantum numbers, etc.)
- Access branching fractions and decay information
- Get measurements and references
- List all available particles
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

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
    """List all available PDG tools."""
    return [
        types.Tool(
            name="search_particle",
            description="Search for a particle by name, Monte Carlo ID, or PDG ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Particle name (e.g., 'pi+', 'proton'), Monte Carlo ID (e.g., '211'), or PDG ID (e.g., 'S008')",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["name", "mcid", "pdgid", "auto"],
                        "default": "auto",
                        "description": "Type of search to perform. 'auto' will try to determine the best method.",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_particle_properties",
            description="Get detailed properties of a particle including mass, quantum numbers, lifetime, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle (e.g., 'pi+', 'proton', 'H')",
                    },
                    "include_measurements": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include individual measurements and references",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_branching_fractions",
            description="Get branching fractions and decay modes for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle (e.g., 'B+', 'tau-')",
                    },
                    "decay_type": {
                        "type": "string",
                        "enum": ["exclusive", "inclusive", "all"],
                        "default": "exclusive",
                        "description": "Type of branching fractions to retrieve",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of decay modes to return",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="list_particles",
            description="List all available particles or filter by type",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_type": {
                        "type": "string",
                        "enum": ["all", "baryon", "meson", "lepton", "boson", "quark"],
                        "default": "all",
                        "description": "Filter particles by type",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of particles to return",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_particle_by_mcid",
            description="Get particle information using Monte Carlo particle ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "mcid": {
                        "type": "integer",
                        "description": "Monte Carlo particle ID (e.g., 211 for pi+, 2212 for proton)",
                    }
                },
                "required": ["mcid"],
            },
        ),
        types.Tool(
            name="compare_particles",
            description="Compare properties of multiple particles",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of particle names to compare",
                    },
                    "properties": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "mass",
                                "lifetime",
                                "charge",
                                "spin",
                                "quantum_numbers",
                            ],
                        },
                        "default": ["mass", "lifetime", "charge"],
                        "description": "Properties to compare",
                    },
                },
                "required": ["particle_names"],
            },
        ),
        types.Tool(
            name="get_database_info",
            description="Get information about the PDG database being used",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


def format_particle_info(particle, include_basic=True, include_measurements=False):
    """Format particle information in a readable way."""
    try:
        info = {
            "name": particle.name,
            "mcid": getattr(particle, "mcid", "N/A"),
            "charge": getattr(particle, "charge", "N/A"),
        }

        if include_basic:
            # Basic properties
            try:
                info["mass"] = f"{particle.mass:.6f} GeV" if particle.mass else "N/A"
                info["mass_error"] = (
                    f"±{particle.mass_error:.6f} GeV" if particle.mass_error else "N/A"
                )
            except:
                info["mass"] = "N/A"
                info["mass_error"] = "N/A"

            try:
                info["lifetime"] = (
                    f"{particle.lifetime:.2e} s" if particle.lifetime else "N/A"
                )
            except:
                info["lifetime"] = "N/A"

            try:
                info["width"] = f"{particle.width:.6f} GeV" if particle.width else "N/A"
            except:
                info["width"] = "N/A"

            # Quantum numbers
            quantum_numbers = {}
            for qn in ["quantum_J", "quantum_P", "quantum_C", "quantum_G", "quantum_I"]:
                try:
                    value = getattr(particle, qn, None)
                    if value is not None:
                        quantum_numbers[qn.replace("quantum_", "")] = str(value)
                except:
                    pass
            if quantum_numbers:
                info["quantum_numbers"] = quantum_numbers

            # Particle type
            particle_types = []
            for ptype in ["is_baryon", "is_meson", "is_lepton", "is_boson", "is_quark"]:
                try:
                    if getattr(particle, ptype, False):
                        particle_types.append(ptype.replace("is_", ""))
                except:
                    pass
            if particle_types:
                info["particle_type"] = particle_types

        if include_measurements:
            # Get measurements for mass
            try:
                measurements = []
                for mass_prop in particle.masses():
                    for measurement in mass_prop.get_measurements():
                        if hasattr(measurement, "reference") and measurement.reference:
                            measurements.append(
                                {
                                    "property": "mass",
                                    "value": (
                                        measurement.get_value().value_text
                                        if measurement.get_value()
                                        else "N/A"
                                    ),
                                    "reference": {
                                        "title": getattr(
                                            measurement.reference, "title", "N/A"
                                        ),
                                        "doi": getattr(
                                            measurement.reference, "doi", "N/A"
                                        ),
                                        "year": getattr(
                                            measurement.reference,
                                            "publication_year",
                                            "N/A",
                                        ),
                                    },
                                }
                            )
                if measurements:
                    info["measurements"] = measurements[:5]  # Limit to first 5
            except:
                pass

        return info
    except Exception as e:
        return {
            "name": str(particle),
            "error": f"Error formatting particle info: {str(e)}",
        }


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        api = ensure_pdg_connection()

        if name == "search_particle":
            query = arguments["query"]
            search_type = arguments.get("search_type", "auto")

            results = []

            if search_type == "auto":
                # Try to determine the best search method
                if query.isdigit():
                    search_type = "mcid"
                elif query.upper().startswith("S") and any(c.isdigit() for c in query):
                    search_type = "pdgid"
                else:
                    search_type = "name"

            try:
                if search_type == "name":
                    particle = api.get_particle_by_name(query)
                    if particle:
                        results.append(format_particle_info(particle))
                elif search_type == "mcid":
                    particle = api.get_particle_by_mcid(int(query))
                    if particle:
                        results.append(format_particle_info(particle))
                elif search_type == "pdgid":
                    items = api.get(query)
                    if items:
                        for item in items[:5]:  # Limit results
                            if hasattr(item, "name"):
                                results.append(format_particle_info(item))
                            else:
                                results.append(
                                    {"pdgid": query, "description": str(item)}
                                )
            except Exception as e:
                results.append({"error": f"Search failed: {str(e)}"})

            if not results:
                results.append({"message": f"No particles found for query: {query}"})

            return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

        elif name == "get_particle_properties":
            particle_name = arguments["particle_name"]
            include_measurements = arguments.get("include_measurements", False)

            try:
                particle = api.get_particle_by_name(particle_name)
                info = format_particle_info(
                    particle,
                    include_basic=True,
                    include_measurements=include_measurements,
                )

                # Add additional properties
                try:
                    properties = []
                    for prop in particle.properties():
                        properties.append(
                            {
                                "pdgid": prop.pdgid,
                                "description": prop.description,
                                "value": prop.display_value_text,
                                "units": getattr(prop, "units", "N/A"),
                            }
                        )
                    if properties:
                        info["all_properties"] = properties[:10]  # Limit to first 10
                except:
                    pass

                return [types.TextContent(type="text", text=json.dumps(info, indent=2))]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Failed to get particle properties: {str(e)}"},
                            indent=2,
                        ),
                    )
                ]

        elif name == "get_branching_fractions":
            particle_name = arguments["particle_name"]
            decay_type = arguments.get("decay_type", "exclusive")
            limit = arguments.get("limit", 20)

            try:
                particle = api.get_particle_by_name(particle_name)
                decays = []

                if decay_type in ["exclusive", "all"]:
                    for bf in particle.exclusive_branching_fractions():
                        decays.append(
                            {
                                "description": bf.description,
                                "branching_fraction": bf.value,
                                "display_value": bf.display_value_text,
                                "is_limit": bf.is_limit,
                                "type": "exclusive",
                            }
                        )
                        if len(decays) >= limit:
                            break

                if decay_type in ["inclusive", "all"] and len(decays) < limit:
                    try:
                        for bf in particle.inclusive_branching_fractions():
                            decays.append(
                                {
                                    "description": bf.description,
                                    "branching_fraction": bf.value,
                                    "display_value": bf.display_value_text,
                                    "is_limit": bf.is_limit,
                                    "type": "inclusive",
                                }
                            )
                            if len(decays) >= limit:
                                break
                    except:
                        pass  # Some particles don't have inclusive branching fractions

                result = {
                    "particle": particle_name,
                    "decay_modes": decays,
                    "total_found": len(decays),
                }

                return [
                    types.TextContent(type="text", text=json.dumps(result, indent=2))
                ]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Failed to get branching fractions: {str(e)}"},
                            indent=2,
                        ),
                    )
                ]

        elif name == "list_particles":
            particle_type = arguments.get("particle_type", "all")
            limit = arguments.get("limit", 50)

            try:
                particles = []
                count = 0

                # Get particles list and convert to individual particles
                all_particles = api.get_particles()

                # If it's a PdgParticleList, convert to list
                if hasattr(all_particles, "__iter__") and not isinstance(
                    all_particles, str
                ):
                    particle_list = list(all_particles)
                else:
                    particle_list = [all_particles]

                for particle in particle_list:
                    if count >= limit:
                        break

                    # Skip if particle doesn't have basic attributes
                    if not hasattr(particle, "name"):
                        continue

                    # Filter by type if specified
                    if particle_type != "all":
                        type_method = f"is_{particle_type}"
                        if hasattr(particle, type_method) and not getattr(
                            particle, type_method
                        ):
                            continue

                    info = {
                        "name": getattr(particle, "name", "Unknown"),
                        "mcid": getattr(particle, "mcid", "N/A"),
                        "charge": getattr(particle, "charge", "N/A"),
                    }

                    # Add mass if available
                    try:
                        if particle.mass:
                            info["mass"] = f"{particle.mass:.6f} GeV"
                    except:
                        pass

                    particles.append(info)
                    count += 1

                result = {
                    "particles": particles,
                    "filter": particle_type,
                    "count": len(particles),
                    "limited_to": limit,
                }

                return [
                    types.TextContent(type="text", text=json.dumps(result, indent=2))
                ]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Failed to list particles: {str(e)}"}, indent=2
                        ),
                    )
                ]

        elif name == "get_particle_by_mcid":
            mcid = arguments["mcid"]

            try:
                particle = api.get_particle_by_mcid(mcid)
                info = format_particle_info(particle, include_basic=True)

                return [types.TextContent(type="text", text=json.dumps(info, indent=2))]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Failed to get particle by MCID: {str(e)}"},
                            indent=2,
                        ),
                    )
                ]

        elif name == "compare_particles":
            particle_names = arguments["particle_names"]
            properties = arguments.get("properties", ["mass", "lifetime", "charge"])

            try:
                comparison = {"particles": []}

                for name in particle_names:
                    try:
                        particle = api.get_particle_by_name(name)
                        particle_info = {"name": name}

                        for prop in properties:
                            if prop == "mass":
                                try:
                                    particle_info["mass"] = (
                                        f"{particle.mass:.6f} GeV"
                                        if particle.mass
                                        else "N/A"
                                    )
                                except:
                                    particle_info["mass"] = "N/A"
                            elif prop == "lifetime":
                                try:
                                    particle_info["lifetime"] = (
                                        f"{particle.lifetime:.2e} s"
                                        if particle.lifetime
                                        else "N/A"
                                    )
                                except:
                                    particle_info["lifetime"] = "N/A"
                            elif prop == "charge":
                                particle_info["charge"] = getattr(
                                    particle, "charge", "N/A"
                                )
                            elif prop == "spin":
                                try:
                                    particle_info["spin"] = str(
                                        getattr(particle, "quantum_J", "N/A")
                                    )
                                except:
                                    particle_info["spin"] = "N/A"
                            elif prop == "quantum_numbers":
                                qn = {}
                                for q in [
                                    "quantum_J",
                                    "quantum_P",
                                    "quantum_C",
                                    "quantum_G",
                                    "quantum_I",
                                ]:
                                    try:
                                        value = getattr(particle, q, None)
                                        if value is not None:
                                            qn[q.replace("quantum_", "")] = str(value)
                                    except:
                                        pass
                                particle_info["quantum_numbers"] = qn

                        comparison["particles"].append(particle_info)
                    except Exception as e:
                        comparison["particles"].append(
                            {"name": name, "error": f"Failed to get particle: {str(e)}"}
                        )

                return [
                    types.TextContent(
                        type="text", text=json.dumps(comparison, indent=2)
                    )
                ]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Failed to compare particles: {str(e)}"},
                            indent=2,
                        ),
                    )
                ]

        elif name == "get_database_info":
            try:
                info = {
                    "edition": getattr(api, "edition", "N/A"),
                    "info_keys": api.info_keys() if hasattr(api, "info_keys") else [],
                }

                # Get additional info
                try:
                    for key in api.info_keys()[:10]:  # Limit to first 10 keys
                        try:
                            info[key] = api.info(key)
                        except:
                            pass
                except:
                    pass

                return [types.TextContent(type="text", text=json.dumps(info, indent=2))]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Failed to get database info: {str(e)}"},
                            indent=2,
                        ),
                    )
                ]

        else:
            return [
                types.TextContent(
                    type="text", text=json.dumps({"error": f"Unknown tool: {name}"})
                )
            ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Tool execution failed: {str(e)}"}),
            )
        ]


async def main():
    """Main function to run the MCP server."""
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pdg-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def run_server():
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()

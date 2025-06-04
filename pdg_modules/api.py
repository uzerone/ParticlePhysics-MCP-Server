"""
PDG API Core Functionality Module

This module contains tools that mirror the core PDG API functionality:
- Particle search and lookup
- Basic particle properties
- Particle comparison
- Database information
- Particle listing by type
"""

import json
from typing import Any, Dict, List

import mcp.types as types


def get_api_tools() -> List[types.Tool]:
    """Return all API-related MCP tools."""
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
        types.Tool(
            name="get_canonical_name",
            description="Get the canonical PDG name for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Particle name (can be non-canonical)",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="get_particles_by_name",
            description="Get all particles matching a name (supports partial/generic matches)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Particle name or partial name to search for",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether the search should be case-sensitive",
                    },
                    "edition": {
                        "type": "string",
                        "description": "Specific PDG edition to search in",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of particles to return",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="get_editions",
            description="Get list of all available PDG Review editions and current default",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get_pdg_by_identifier",
            description="Get PDG data object by its identifier (PDG ID)",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdgid": {
                        "type": "string",
                        "description": "PDG identifier (e.g., 'S008', 'M100', particle names)",
                    },
                    "edition": {
                        "type": "string",
                        "description": "Specific PDG edition to retrieve from",
                    },
                },
                "required": ["pdgid"],
            },
        ),
        types.Tool(
            name="get_all_pdg_identifiers",
            description="Get all PDG identifiers and quantities, optionally filtered by data type",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_type_key": {
                        "type": "string",
                        "description": "Filter by specific data type (e.g., 'M' for mass, 'G' for width, 'T' for lifetime)",
                    },
                    "edition": {
                        "type": "string",
                        "description": "Specific PDG edition to retrieve data from",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of identifiers to return",
                    },
                },
                "required": [],
            },
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


async def handle_api_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Handle API-related tool calls."""

    if name == "search_particle":
        query = arguments["query"]
        search_type = arguments.get("search_type", "auto")
        if search_type is None:
            search_type = "auto"

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
                            results.append({"pdgid": query, "description": str(item)})
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

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
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
                            particle_info["charge"] = getattr(particle, "charge", "N/A")
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
                types.TextContent(type="text", text=json.dumps(comparison, indent=2))
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

    elif name == "get_canonical_name":
        name = arguments["name"]

        try:
            canonical_name = api.get_canonical_name(name)

            result = {
                "input_name": name,
                "canonical_name": canonical_name,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get canonical name: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_particles_by_name":
        name = arguments["name"]
        case_sensitive = arguments.get("case_sensitive", True)
        edition = arguments.get("edition")
        limit = arguments.get("limit", 20)

        try:
            particles = []
            count = 0

            for particle_list in api.get_particles_by_name(
                name, case_sensitive=case_sensitive, edition=edition
            ):
                if count >= limit:
                    break

                # Handle both single particles and particle lists
                if hasattr(particle_list, "__iter__") and not isinstance(
                    particle_list, str
                ):
                    for particle in particle_list:
                        if count >= limit:
                            break
                        particles.append(
                            format_particle_info(particle, include_basic=True)
                        )
                        count += 1
                else:
                    particles.append(
                        format_particle_info(particle_list, include_basic=True)
                    )
                    count += 1

            result = {
                "search_name": name,
                "case_sensitive": case_sensitive,
                "edition": edition,
                "particles": particles,
                "count": len(particles),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get particles by name: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_editions":
        try:
            editions_list = api.editions
            default_edition = api.default_edition

            result = {
                "available_editions": editions_list,
                "default_edition": default_edition,
                "total_editions": len(editions_list),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get editions: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_pdg_by_identifier":
        pdgid = arguments["pdgid"]
        edition = arguments.get("edition")

        try:
            pdg_obj = api.get(pdgid, edition=edition)

            # Format the object based on its type
            result = {
                "pdgid": pdgid,
                "edition": edition,
                "object_type": type(pdg_obj).__name__,
            }

            # Try to get common attributes
            try:
                result["description"] = pdg_obj.description
            except:
                pass

            try:
                result["data_type"] = pdg_obj.data_type
            except:
                pass

            try:
                result["data_flags"] = pdg_obj.data_flags
            except:
                pass

            # Handle different object types
            if hasattr(pdg_obj, "__iter__") and not isinstance(pdg_obj, str):
                # It's a list/iterator of objects
                items = []
                for item in list(pdg_obj)[:10]:  # Limit to first 10 items
                    try:
                        if hasattr(item, "name"):
                            # It's a particle
                            items.append(format_particle_info(item, include_basic=True))
                        else:
                            # Other object types
                            items.append(
                                {
                                    "description": getattr(
                                        item, "description", str(item)
                                    ),
                                    "type": type(item).__name__,
                                }
                            )
                    except:
                        items.append({"error": "Could not format item"})
                result["items"] = items
                result["items_count"] = len(items)
            else:
                # Single object
                try:
                    if hasattr(pdg_obj, "name"):
                        # It's a particle
                        result["particle"] = format_particle_info(
                            pdg_obj, include_basic=True
                        )
                    elif hasattr(pdg_obj, "value"):
                        # It's a measurement/property
                        result["value"] = pdg_obj.value
                        result["display_value"] = getattr(
                            pdg_obj, "display_value_text", str(pdg_obj)
                        )
                    else:
                        # Other object type
                        result["content"] = str(pdg_obj)
                except Exception as format_error:
                    result["raw_content"] = str(pdg_obj)
                    result["format_error"] = str(format_error)

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get PDG object by identifier: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_all_pdg_identifiers":
        data_type_key = arguments.get("data_type_key")
        edition = arguments.get("edition")
        limit = arguments.get("limit", 50)

        try:
            identifiers = []
            count = 0

            for item in api.get_all(data_type_key=data_type_key, edition=edition):
                if count >= limit:
                    break

                try:
                    identifier_info = {
                        "pdgid": item.pdgid,
                        "description": item.description,
                        "data_type": item.data_type,
                        "edition": getattr(item, "edition", edition),
                    }

                    # Try to get additional info
                    try:
                        identifier_info["data_flags"] = item.data_flags
                    except:
                        pass

                    identifiers.append(identifier_info)
                    count += 1
                except Exception as e:
                    # Skip items that can't be processed
                    continue

            result = {
                "data_type_filter": data_type_key,
                "edition": edition,
                "identifiers": identifiers,
                "count": len(identifiers),
                "limited_to": limit,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get PDG identifiers: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": f"Unknown API tool: {name}"})
            )
        ]

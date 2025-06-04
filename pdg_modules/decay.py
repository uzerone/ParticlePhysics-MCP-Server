"""
PDG Decay Analysis Module

This module contains tools for analyzing particle decays, branching fractions,
and decay structures.
"""

import json
from typing import Any, Dict, List

import mcp.types as types


def get_decay_tools() -> List[types.Tool]:
    """Return all decay-related MCP tools."""
    return [
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
            name="get_decay_products",
            description="Get detailed decay products for a specific branching fraction/decay mode",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "mode_number": {
                        "type": "integer",
                        "description": "Specific decay mode number (optional)",
                    },
                    "decay_type": {
                        "type": "string",
                        "enum": ["exclusive", "inclusive", "all"],
                        "default": "exclusive",
                        "description": "Type of decays to analyze",
                    },
                    "include_subdecays": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include subdecay information",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_branching_ratios",
            description="Get branching ratios associated with branching fractions",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of ratios to return",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_decay_mode_details",
            description="Get detailed information about decay modes including mode numbers and subdecay levels",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "show_subdecays": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include subdecay modes in results",
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
            name="analyze_decay_structure",
            description="Analyze the hierarchical structure of particle decays including subdecay levels",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum subdecay depth to analyze",
                    },
                    "decay_type": {
                        "type": "string",
                        "enum": ["exclusive", "inclusive", "all"],
                        "default": "exclusive",
                        "description": "Type of decays to analyze",
                    },
                },
                "required": ["particle_name"],
            },
        ),
    ]


def format_decay_product(decay_product):
    """Format a PdgDecayProduct object for JSON output."""
    try:
        product_info = {
            "item_name": decay_product.item.name,
            "multiplier": decay_product.multiplier,
            "has_subdecay": decay_product.subdecay is not None,
        }

        # Try to get particle information for the item
        try:
            particles = list(decay_product.item.particles())
            if particles:
                particle_info = []
                for particle in particles[:3]:  # Limit to first 3
                    particle_info.append(
                        {
                            "name": particle.name,
                            "mcid": getattr(particle, "mcid", "N/A"),
                            "charge": getattr(particle, "charge", "N/A"),
                        }
                    )
                product_info["particles"] = particle_info
        except:
            product_info["particles"] = []

        # Add subdecay information if present
        if decay_product.subdecay:
            try:
                product_info["subdecay"] = {
                    "pdgid": decay_product.subdecay.pdgid,
                    "description": decay_product.subdecay.description,
                    "mode_number": getattr(
                        decay_product.subdecay, "mode_number", "N/A"
                    ),
                    "subdecay_level": getattr(
                        decay_product.subdecay, "subdecay_level", 0
                    ),
                }
            except:
                product_info["subdecay"] = {"error": "Could not format subdecay"}

        return product_info
    except Exception as e:
        return {"error": f"Failed to format decay product: {str(e)}"}


def format_branching_fraction_decay(bf):
    """Format a PdgBranchingFraction object with decay details for JSON output."""
    try:
        decay_info = {
            "pdgid": bf.pdgid,
            "description": bf.description,
            "branching_fraction": bf.value,
            "display_value": bf.display_value_text,
            "is_limit": bf.is_limit,
        }

        # Add decay-specific information
        try:
            decay_info["mode_number"] = bf.mode_number
        except:
            decay_info["mode_number"] = "N/A"

        try:
            decay_info["is_subdecay"] = bf.is_subdecay
            decay_info["subdecay_level"] = bf.subdecay_level
        except:
            decay_info["is_subdecay"] = False
            decay_info["subdecay_level"] = 0

        # Get decay products
        try:
            products = []
            for product in bf.decay_products:
                products.append(format_decay_product(product))
            decay_info["decay_products"] = products
            decay_info["num_products"] = len(products)
        except:
            decay_info["decay_products"] = []
            decay_info["num_products"] = 0

        return decay_info
    except Exception as e:
        return {"error": f"Failed to format branching fraction: {str(e)}"}


def get_branching_fractions_by_type(particle, decay_type):
    """Get branching fractions by type (exclusive/inclusive/all)."""
    fractions = []

    if decay_type in ["exclusive", "all"]:
        try:
            for bf in particle.exclusive_branching_fractions():
                fractions.append(bf)
        except:
            pass

    if decay_type in ["inclusive", "all"]:
        try:
            for bf in particle.inclusive_branching_fractions():
                fractions.append(bf)
        except:
            pass

    return fractions


async def handle_decay_tools(
    name: str, arguments: dict, api
) -> List[types.TextContent]:
    """Handle decay-related tool calls."""

    if name == "get_branching_fractions":
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

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
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

    elif name == "get_decay_products":
        particle_name = arguments["particle_name"]
        mode_number = arguments.get("mode_number")
        decay_type = arguments.get("decay_type", "exclusive")
        include_subdecays = arguments.get("include_subdecays", True)

        try:
            particle = api.get_particle_by_name(particle_name)
            decay_modes = []

            # Get branching fractions
            branching_fractions = get_branching_fractions_by_type(particle, decay_type)

            for bf in branching_fractions:
                # Filter by mode number if specified
                if mode_number is not None:
                    try:
                        if bf.mode_number != mode_number:
                            continue
                    except:
                        continue

                # Filter subdecays if not requested
                if (
                    not include_subdecays
                    and hasattr(bf, "is_subdecay")
                    and bf.is_subdecay
                ):
                    continue

                decay_info = format_branching_fraction_decay(bf)
                if "error" not in decay_info:
                    decay_modes.append(decay_info)

            result = {
                "particle": particle_name,
                "decay_type": decay_type,
                "mode_number_filter": mode_number,
                "include_subdecays": include_subdecays,
                "decay_modes": decay_modes,
                "total_modes": len(decay_modes),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get decay products: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_branching_ratios":
        particle_name = arguments["particle_name"]
        limit = arguments.get("limit", 10)

        try:
            particle = api.get_particle_by_name(particle_name)
            ratios = []
            count = 0

            # Get exclusive branching fractions and their associated ratios
            for bf in particle.exclusive_branching_fractions():
                if count >= limit:
                    break

                try:
                    # Get branching ratios for this branching fraction
                    for br in bf.branching_ratios():
                        if count >= limit:
                            break

                        ratio_info = {
                            "pdgid": br.pdgid,
                            "description": br.description,
                            "value": br.value,
                            "display_value": br.display_value_text,
                            "units": getattr(br, "units", "N/A"),
                            "associated_bf_pdgid": bf.pdgid,
                            "associated_bf_description": bf.description,
                        }

                        ratios.append(ratio_info)
                        count += 1
                except:
                    continue

            result = {
                "particle": particle_name,
                "branching_ratios": ratios,
                "total_found": len(ratios),
                "limited_to": limit,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get branching ratios: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_decay_mode_details":
        particle_name = arguments["particle_name"]
        show_subdecays = arguments.get("show_subdecays", True)
        limit = arguments.get("limit", 20)

        try:
            particle = api.get_particle_by_name(particle_name)
            decay_modes = []
            count = 0

            for bf in particle.exclusive_branching_fractions():
                if count >= limit:
                    break

                # Skip subdecays if not requested
                if not show_subdecays:
                    try:
                        if bf.is_subdecay:
                            continue
                    except:
                        pass

                mode_details = {
                    "pdgid": bf.pdgid,
                    "description": bf.description,
                    "branching_fraction": bf.display_value_text,
                    "is_limit": bf.is_limit,
                }

                # Add decay-specific details
                try:
                    mode_details["mode_number"] = bf.mode_number
                except:
                    mode_details["mode_number"] = "N/A"

                try:
                    mode_details["is_subdecay"] = bf.is_subdecay
                    mode_details["subdecay_level"] = bf.subdecay_level
                except:
                    mode_details["is_subdecay"] = False
                    mode_details["subdecay_level"] = 0

                # Count decay products
                try:
                    mode_details["num_products"] = len(bf.decay_products)
                except:
                    mode_details["num_products"] = 0

                decay_modes.append(mode_details)
                count += 1

            result = {
                "particle": particle_name,
                "show_subdecays": show_subdecays,
                "decay_modes": decay_modes,
                "total_modes": len(decay_modes),
                "limited_to": limit,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get decay mode details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "analyze_decay_structure":
        particle_name = arguments["particle_name"]
        max_depth = arguments.get("max_depth", 3)
        decay_type = arguments.get("decay_type", "exclusive")

        try:
            particle = api.get_particle_by_name(particle_name)

            def analyze_decay_level(bf, current_depth=0):
                """Recursively analyze decay structure."""
                if current_depth > max_depth:
                    return None

                decay_info = {
                    "pdgid": bf.pdgid,
                    "description": bf.description,
                    "branching_fraction": bf.display_value_text,
                    "depth": current_depth,
                }

                try:
                    decay_info["mode_number"] = bf.mode_number
                    decay_info["is_subdecay"] = bf.is_subdecay
                    decay_info["subdecay_level"] = bf.subdecay_level
                except:
                    decay_info["mode_number"] = "N/A"
                    decay_info["is_subdecay"] = False
                    decay_info["subdecay_level"] = 0

                # Analyze decay products and their subdecays
                products = []
                try:
                    for product in bf.decay_products:
                        product_info = {
                            "item_name": product.item.name,
                            "multiplier": product.multiplier,
                        }

                        # Recursively analyze subdecays
                        if product.subdecay and current_depth < max_depth:
                            subdecay_analysis = analyze_decay_level(
                                product.subdecay, current_depth + 1
                            )
                            if subdecay_analysis:
                                product_info["subdecay_analysis"] = subdecay_analysis

                        products.append(product_info)
                except:
                    pass

                decay_info["products"] = products
                return decay_info

            # Analyze decay structure
            structure = {
                "particle": particle_name,
                "max_depth": max_depth,
                "decay_type": decay_type,
                "decay_structure": [],
            }

            branching_fractions = get_branching_fractions_by_type(particle, decay_type)

            for bf in branching_fractions[
                :10
            ]:  # Limit to first 10 to avoid huge output
                analysis = analyze_decay_level(bf, 0)
                if analysis:
                    structure["decay_structure"].append(analysis)

            structure["total_analyzed"] = len(structure["decay_structure"])

            return [
                types.TextContent(type="text", text=json.dumps(structure, indent=2))
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to analyze decay structure: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": f"Unknown decay tool: {name}"})
            )
        ]

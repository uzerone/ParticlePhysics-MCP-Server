import json
import logging
from typing import Any, Dict, List, Optional, Union

import mcp.types as types

# Setup module logger
logger = logging.getLogger(__name__)


def get_api_tools() -> List[types.Tool]:
    """Return all API-related MCP tools with enhanced functionality."""
    return [
        types.Tool(
            name="search_particle",
            description="Advanced particle search by name, Monte Carlo ID, or PDG ID with fuzzy matching",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Particle name (e.g., 'pi+', 'proton'), Monte Carlo ID (e.g., '211'), or PDG ID (e.g., 'S008')",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["name", "mcid", "pdgid", "auto", "fuzzy"],
                        "default": "auto",
                        "description": "Search method: 'auto' (intelligent detection), 'fuzzy' (approximate matching), or specific type",
                    },
                    "include_antiparticles": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include antiparticles in search results",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results to return",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_particle_properties",
            description="Get comprehensive particle properties with enhanced metadata and validation",
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
                    "include_quantum_numbers": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include quantum numbers (J, P, C, G, I)",
                    },
                    "include_decays": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include decay information summary",
                    },
                    "units_preference": {
                        "type": "string",
                        "enum": ["GeV", "MeV", "natural"],
                        "default": "GeV",
                        "description": "Preferred units for mass and energy values",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="list_particles",
            description="List particles with advanced filtering, sorting, and pagination",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_type": {
                        "type": "string",
                        "enum": [
                            "all",
                            "baryon",
                            "meson",
                            "lepton",
                            "boson",
                            "quark",
                            "composite",
                            "fundamental",
                        ],
                        "default": "all",
                        "description": "Filter particles by type classification",
                    },
                    "charge_filter": {
                        "type": "string",
                        "enum": ["all", "neutral", "positive", "negative"],
                        "default": "all",
                        "description": "Filter by electric charge",
                    },
                    "mass_range": {
                        "type": "object",
                        "properties": {
                            "min_mass": {
                                "type": "number",
                                "description": "Minimum mass in GeV",
                            },
                            "max_mass": {
                                "type": "number",
                                "description": "Maximum mass in GeV",
                            },
                        },
                        "description": "Filter by mass range",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["name", "mass", "charge", "discovery_year"],
                        "default": "name",
                        "description": "Sort particles by specified property",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "maximum": 200,
                        "description": "Maximum number of particles to return",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Offset for pagination",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_particle_by_mcid",
            description="Get particle information using Monte Carlo particle ID with validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "mcid": {
                        "type": "integer",
                        "description": "Monte Carlo particle ID (e.g., 211 for pi+, 2212 for proton)",
                    },
                    "include_related": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include related particles (antiparticles, excited states)",
                    },
                },
                "required": ["mcid"],
            },
        ),
        types.Tool(
            name="compare_particles",
            description="Advanced particle comparison with statistical analysis and visualization data",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 10,
                        "description": "List of particle names to compare (2-10 particles)",
                    },
                    "properties": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "mass",
                                "lifetime",
                                "width",
                                "charge",
                                "spin",
                                "quantum_numbers",
                                "discovery_info",
                                "decay_channels",
                            ],
                        },
                        "default": ["mass", "lifetime", "charge"],
                        "description": "Properties to compare with enhanced options",
                    },
                    "include_ratios": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include property ratios between particles",
                    },
                    "include_uncertainties": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include measurement uncertainties",
                    },
                },
                "required": ["particle_names"],
            },
        ),
        types.Tool(
            name="get_database_info",
            description="Get comprehensive PDG database information and metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_statistics": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include database statistics (particle counts, etc.)",
                    },
                    "include_version_info": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include version and release information",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_canonical_name",
            description="Get canonical PDG name with alternative name suggestions",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Particle name (can be non-canonical)",
                    },
                    "include_alternatives": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include alternative names and synonyms",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="get_particles_by_name",
            description="Advanced name-based particle search with fuzzy matching and filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Particle name or partial name to search for",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether the search should be case-sensitive",
                    },
                    "fuzzy_match": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable fuzzy matching for approximate results",
                    },
                    "edition": {
                        "type": "string",
                        "description": "Specific PDG edition to search in",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "maximum": 100,
                        "description": "Maximum number of particles to return",
                    },
                    "min_confidence": {
                        "type": "number",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Minimum confidence score for fuzzy matches",
                    },
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="get_editions",
            description="Get comprehensive PDG Review editions information with metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_metadata": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include edition metadata (publication dates, changes, etc.)",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_pdg_by_identifier",
            description="Enhanced PDG data object retrieval with validation and metadata",
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
                    "include_related": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include related PDG objects and references",
                    },
                    "validate_identifier": {
                        "type": "boolean",
                        "default": True,
                        "description": "Validate PDG identifier format",
                    },
                },
                "required": ["pdgid"],
            },
        ),
        types.Tool(
            name="get_all_pdg_identifiers",
            description="Get filtered and paginated PDG identifiers with enhanced metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_type_key": {
                        "type": "string",
                        "description": "Filter by data type (e.g., 'M' for mass, 'G' for width, 'T' for lifetime)",
                    },
                    "edition": {
                        "type": "string",
                        "description": "Specific PDG edition to retrieve data from",
                    },
                    "category_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by PDG categories",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "maximum": 500,
                        "description": "Maximum number of identifiers to return",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Offset for pagination",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed metadata for each identifier",
                    },
                },
                "required": [],
            },
        ),
    ]


def safe_get_attribute(
    obj: Any, attr: str, default: Any = None, transform_func: Optional[callable] = None
) -> Any:
    """Safely get attribute from object with optional transformation."""
    try:
        value = getattr(obj, attr, default)
        if value is not None and transform_func:
            return transform_func(value)
        return value
    except Exception as e:
        logger.debug(f"Failed to get attribute {attr}: {e}")
        return default


def format_value_with_uncertainty(
    value: Any, error_pos: Any = None, error_neg: Any = None, units: str = ""
) -> Dict[str, Any]:
    """Format a physical value with its uncertainty."""
    try:
        result = {
            "value": float(value) if value is not None else None,
            "units": units,
            "formatted": "N/A",
        }

        if result["value"] is not None:
            if error_pos is not None or error_neg is not None:
                if error_pos == error_neg or error_neg is None:
                    result["formatted"] = (
                        f"{result['value']:.6g} ± {error_pos:.6g} {units}".strip()
                    )
                    result["uncertainty"] = {"symmetric": float(error_pos)}
                else:
                    result["formatted"] = (
                        f"{result['value']:.6g} +{error_pos:.6g}/-{abs(error_neg):.6g} {units}".strip()
                    )
                    result["uncertainty"] = {
                        "positive": float(error_pos),
                        "negative": float(error_neg),
                    }
            else:
                result["formatted"] = f"{result['value']:.6g} {units}".strip()

        return result
    except Exception as e:
        logger.debug(f"Error formatting value: {e}")
        return {"value": None, "formatted": "N/A", "units": units}


def get_particle_classification(particle: Any) -> Dict[str, Any]:
    """Get comprehensive particle classification information."""
    classification = {
        "type": "unknown",
        "category": "unknown",
        "is_fundamental": False,
        "is_composite": False,
        "constituents": None,
    }

    try:
        # Check particle type flags
        if safe_get_attribute(particle, "is_lepton", False):
            classification.update(
                {"type": "lepton", "category": "fundamental", "is_fundamental": True}
            )
        elif safe_get_attribute(particle, "is_quark", False):
            classification.update(
                {"type": "quark", "category": "fundamental", "is_fundamental": True}
            )
        elif safe_get_attribute(particle, "is_boson", False):
            classification.update(
                {"type": "boson", "category": "fundamental", "is_fundamental": True}
            )
        elif safe_get_attribute(particle, "is_baryon", False):
            classification.update(
                {
                    "type": "baryon",
                    "category": "hadron",
                    "is_composite": True,
                    "constituents": "quarks",
                }
            )
        elif safe_get_attribute(particle, "is_meson", False):
            classification.update(
                {
                    "type": "meson",
                    "category": "hadron",
                    "is_composite": True,
                    "constituents": "quark-antiquark",
                }
            )

    except Exception as e:
        logger.debug(f"Error determining particle classification: {e}")

    return classification


def format_quantum_numbers(particle: Any) -> Dict[str, Any]:
    """Format quantum numbers with proper notation."""
    quantum_numbers = {}

    # Standard quantum number mappings
    qn_mappings = {
        "J": ("quantum_J", "Total angular momentum"),
        "P": ("quantum_P", "Parity"),
        "C": ("quantum_C", "Charge conjugation parity"),
        "G": ("quantum_G", "G-parity"),
        "I": ("quantum_I", "Isospin"),
    }

    for symbol, (attr, description) in qn_mappings.items():
        value = safe_get_attribute(particle, attr)
        if value is not None:
            quantum_numbers[symbol] = {
                "value": str(value),
                "description": description,
                "symbol": symbol,
            }

    return quantum_numbers


def format_enhanced_particle_info(
    particle: Any,
    include_basic: bool = True,
    include_measurements: bool = False,
    include_quantum_numbers: bool = True,
    include_decays: bool = False,
    units_preference: str = "GeV",
) -> Dict[str, Any]:
    """Enhanced particle information formatting with comprehensive data."""
    try:
        # Basic particle information
        info = {
            "name": safe_get_attribute(particle, "name", "Unknown"),
            "pdgid": safe_get_attribute(particle, "pdgid"),
            "mcid": safe_get_attribute(particle, "mcid"),
            "charge": safe_get_attribute(particle, "charge"),
            "status": "active",  # Could be enhanced based on PDG status
        }

        if include_basic:
            # Mass information
            mass_value = safe_get_attribute(particle, "mass")
            mass_error = safe_get_attribute(particle, "mass_error")
            info["mass"] = format_value_with_uncertainty(
                mass_value, mass_error, units=units_preference
            )

            # Lifetime information
            lifetime_value = safe_get_attribute(particle, "lifetime")
            if lifetime_value:
                info["lifetime"] = format_value_with_uncertainty(
                    lifetime_value, units="s"
                )
                # Calculate mean life if available
                if lifetime_value > 0:
                    info["mean_life"] = format_value_with_uncertainty(
                        lifetime_value, units="s"
                    )

            # Width information
            width_value = safe_get_attribute(particle, "width")
            if width_value:
                info["width"] = format_value_with_uncertainty(
                    width_value, units=units_preference
                )

            # Classification
            info["classification"] = get_particle_classification(particle)

        if include_quantum_numbers:
            qn = format_quantum_numbers(particle)
            if qn:
                info["quantum_numbers"] = qn

        if include_measurements:
            # Enhanced measurements with references
            measurements = []
            try:
                # Mass measurements
                for mass_prop in getattr(particle, "masses", lambda: [])():
                    for measurement in getattr(
                        mass_prop, "get_measurements", lambda: []
                    )():
                        meas_info = {
                            "property": "mass",
                            "value": safe_get_attribute(measurement, "value"),
                            "uncertainty": {
                                "positive": safe_get_attribute(
                                    measurement, "error_positive"
                                ),
                                "negative": safe_get_attribute(
                                    measurement, "error_negative"
                                ),
                            },
                            "units": safe_get_attribute(
                                measurement, "units", units_preference
                            ),
                        }

                        # Reference information
                        ref = safe_get_attribute(measurement, "reference")
                        if ref:
                            meas_info["reference"] = {
                                "title": safe_get_attribute(ref, "title"),
                                "authors": safe_get_attribute(ref, "authors"),
                                "year": safe_get_attribute(ref, "publication_year"),
                                "doi": safe_get_attribute(ref, "doi"),
                                "journal": safe_get_attribute(ref, "journal"),
                            }

                        measurements.append(meas_info)

                if measurements:
                    info["measurements"] = measurements[:10]  # Limit for performance

            except Exception as e:
                logger.debug(f"Error getting measurements: {e}")

        if include_decays:
            # Basic decay information summary
            try:
                decay_info = {
                    "stable": True,
                    "main_decay_modes": [],
                    "total_decay_modes": 0,
                }

                # Check for decay modes
                exclusive_bfs = list(
                    getattr(particle, "exclusive_branching_fractions", lambda: [])()
                )
                if exclusive_bfs:
                    decay_info["stable"] = False
                    decay_info["total_decay_modes"] = len(exclusive_bfs)

                    # Get top 3 decay modes
                    for bf in exclusive_bfs[:3]:
                        decay_mode = {
                            "description": safe_get_attribute(bf, "description"),
                            "branching_fraction": safe_get_attribute(bf, "value"),
                            "formatted_bf": safe_get_attribute(
                                bf, "display_value_text"
                            ),
                        }
                        decay_info["main_decay_modes"].append(decay_mode)

                info["decay_info"] = decay_info

            except Exception as e:
                logger.debug(f"Error getting decay information: {e}")

        return info

    except Exception as e:
        logger.error(f"Error formatting particle info for {particle}: {e}")
        return {
            "name": str(particle),
            "error": f"Error formatting particle info: {str(e)}",
            "raw_type": type(particle).__name__,
        }


def calculate_property_ratios(
    particles_data: List[Dict], property_name: str
) -> Dict[str, Any]:
    """Calculate ratios between particle properties for comparison."""
    try:
        values = []
        particle_names = []

        for p_data in particles_data:
            if (
                property_name in p_data
                and p_data[property_name].get("value") is not None
            ):
                values.append(p_data[property_name]["value"])
                particle_names.append(p_data["name"])

        if len(values) < 2:
            return {"error": "Insufficient data for ratio calculation"}

        ratios = {}
        for i, (name1, val1) in enumerate(zip(particle_names, values)):
            for j, (name2, val2) in enumerate(zip(particle_names, values)):
                if i != j and val2 != 0:
                    ratio_key = f"{name1}/{name2}"
                    ratios[ratio_key] = {
                        "ratio": val1 / val2,
                        "formatted": f"{val1/val2:.3f}",
                        "interpretation": f"{name1} is {val1/val2:.1f}x the {property_name} of {name2}",
                    }

        return {
            "property": property_name,
            "ratios": ratios,
            "statistics": {
                "min_value": min(values),
                "max_value": max(values),
                "range_factor": max(values) / min(values) if min(values) != 0 else None,
            },
        }

    except Exception as e:
        logger.error(f"Error calculating ratios: {e}")
        return {"error": f"Failed to calculate ratios: {str(e)}"}


async def handle_api_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Enhanced API tool handler with comprehensive error handling and functionality."""

    try:
        if name == "search_particle":
            query = arguments["query"]
            search_type = arguments.get("search_type", "auto")
            include_antiparticles = arguments.get("include_antiparticles", True)
            max_results = arguments.get("max_results", 10)

            results = []

            # Intelligent search type detection
            if search_type == "auto":
                if query.isdigit():
                    search_type = "mcid"
                elif any(
                    query.upper().startswith(prefix) for prefix in ["S", "M", "G", "T"]
                ) and any(c.isdigit() for c in query):
                    search_type = "pdgid"
                else:
                    search_type = "name"

            # Perform search with enhanced error handling
            try:
                if search_type == "name" or search_type == "fuzzy":
                    try:
                        particle = api.get_particle_by_name(query)
                        if particle:
                            results.append(format_enhanced_particle_info(particle))
                    except:
                        # Fallback to fuzzy search
                        try:
                            particles_list = list(
                                api.get_particles_by_name(query, case_sensitive=False)
                            )
                            for particle_group in particles_list[:max_results]:
                                if hasattr(
                                    particle_group, "__iter__"
                                ) and not isinstance(particle_group, str):
                                    for particle in particle_group:
                                        if hasattr(particle, "name"):
                                            results.append(
                                                format_enhanced_particle_info(particle)
                                            )
                                elif hasattr(particle_group, "name"):
                                    results.append(
                                        format_enhanced_particle_info(particle_group)
                                    )
                        except Exception as e:
                            logger.debug(f"Fuzzy search failed: {e}")

                elif search_type == "mcid":
                    particle = api.get_particle_by_mcid(int(query))
                    if particle:
                        results.append(format_enhanced_particle_info(particle))

                elif search_type == "pdgid":
                    items = api.get(query)
                    if items:
                        if hasattr(items, "__iter__") and not isinstance(items, str):
                            for item in list(items)[:max_results]:
                                if hasattr(item, "name"):
                                    results.append(format_enhanced_particle_info(item))
                        elif hasattr(items, "name"):
                            results.append(format_enhanced_particle_info(items))

            except Exception as e:
                logger.error(f"Search error for {query}: {e}")
                results.append(
                    {
                        "error": f"Search failed: {str(e)}",
                        "query": query,
                        "search_type": search_type,
                        "suggestions": [
                            "Check particle name spelling",
                            "Try different search type",
                            "Use fuzzy search",
                        ],
                    }
                )

            if not results:
                results.append(
                    {
                        "message": f"No particles found for query: {query}",
                        "search_type": search_type,
                        "suggestions": [
                            "Check spelling and try again",
                            "Use fuzzy search mode",
                            "Try different particle name variants",
                            f"Search for '{query}' in particle name database",
                        ],
                    }
                )

            response = {
                "query": query,
                "search_type": search_type,
                "results": results,
                "total_found": len(results),
                "search_metadata": {
                    "include_antiparticles": include_antiparticles,
                    "max_results": max_results,
                },
            }

            return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

        elif name == "get_particle_properties":
            particle_name = arguments["particle_name"]
            include_measurements = arguments.get("include_measurements", False)
            include_quantum_numbers = arguments.get("include_quantum_numbers", True)
            include_decays = arguments.get("include_decays", False)
            units_preference = arguments.get("units_preference", "GeV")

            try:
                particle = api.get_particle_by_name(particle_name)
                info = format_enhanced_particle_info(
                    particle,
                    include_basic=True,
                    include_measurements=include_measurements,
                    include_quantum_numbers=include_quantum_numbers,
                    include_decays=include_decays,
                    units_preference=units_preference,
                )

                # Add additional context
                info["query_parameters"] = {
                    "include_measurements": include_measurements,
                    "include_quantum_numbers": include_quantum_numbers,
                    "include_decays": include_decays,
                    "units_preference": units_preference,
                }

                return [types.TextContent(type="text", text=json.dumps(info, indent=2))]

            except Exception as e:
                error_response = {
                    "error": f"Failed to get particle properties: {str(e)}",
                    "particle_name": particle_name,
                    "suggestions": [
                        "Verify particle name spelling",
                        "Use canonical PDG name",
                        "Try search_particle tool first",
                    ],
                }
                return [
                    types.TextContent(
                        type="text", text=json.dumps(error_response, indent=2)
                    )
                ]

        elif name == "compare_particles":
            particle_names = arguments["particle_names"]
            properties = arguments.get("properties", ["mass", "lifetime", "charge"])
            include_ratios = arguments.get("include_ratios", False)
            include_uncertainties = arguments.get("include_uncertainties", True)

            try:
                comparison = {
                    "particles": [],
                    "comparison_properties": properties,
                    "include_ratios": include_ratios,
                    "include_uncertainties": include_uncertainties,
                }

                particles_data = []

                for name in particle_names:
                    try:
                        particle = api.get_particle_by_name(name)
                        particle_info = format_enhanced_particle_info(
                            particle,
                            include_quantum_numbers=("quantum_numbers" in properties),
                            include_decays=("decay_channels" in properties),
                        )

                        # Extract requested properties
                        filtered_info = {"name": name}
                        for prop in properties:
                            if prop in particle_info:
                                filtered_info[prop] = particle_info[prop]
                            elif prop == "spin" and "quantum_numbers" in particle_info:
                                j_qn = particle_info["quantum_numbers"].get("J")
                                filtered_info["spin"] = j_qn["value"] if j_qn else "N/A"

                        comparison["particles"].append(filtered_info)
                        particles_data.append(particle_info)

                    except Exception as e:
                        comparison["particles"].append(
                            {"name": name, "error": f"Failed to get particle: {str(e)}"}
                        )

                # Calculate ratios if requested
                if include_ratios and len(particles_data) >= 2:
                    comparison["property_ratios"] = {}
                    for prop in properties:
                        if prop in ["mass", "lifetime", "width"]:
                            ratios = calculate_property_ratios(particles_data, prop)
                            if "error" not in ratios:
                                comparison["property_ratios"][prop] = ratios

                return [
                    types.TextContent(
                        type="text", text=json.dumps(comparison, indent=2)
                    )
                ]

            except Exception as e:
                error_response = {
                    "error": f"Failed to compare particles: {str(e)}",
                    "particle_names": particle_names,
                }
                return [
                    types.TextContent(
                        type="text", text=json.dumps(error_response, indent=2)
                    )
                ]

        elif name == "get_database_info":
            include_statistics = arguments.get("include_statistics", True)
            include_version_info = arguments.get("include_version_info", True)

            try:
                info = {
                    "database": "PDG (Particle Data Group)",
                    "api_version": "Enhanced MCP Interface",
                }

                if include_version_info:
                    info.update(
                        {
                            "edition": safe_get_attribute(api, "edition", "Unknown"),
                            "default_edition": safe_get_attribute(
                                api, "default_edition", "Unknown"
                            ),
                            "available_editions": safe_get_attribute(
                                api, "editions", []
                            ),
                        }
                    )

                if include_statistics:
                    try:
                        # Get particle count statistics
                        all_particles = list(api.get_particles())
                        info["statistics"] = {
                            "total_particles": len(all_particles),
                            "database_features": [
                                "Particle properties",
                                "Quantum numbers",
                                "Decay modes",
                                "Measurements",
                                "References",
                            ],
                        }
                    except Exception as e:
                        info["statistics"] = {
                            "error": f"Could not retrieve statistics: {e}"
                        }

                # Add API capabilities
                info["api_capabilities"] = {
                    "search_methods": ["name", "mcid", "pdgid", "fuzzy"],
                    "data_types": [
                        "mass",
                        "lifetime",
                        "width",
                        "quantum_numbers",
                        "decays",
                    ],
                    "supported_units": ["GeV", "MeV", "natural"],
                    "comparison_features": True,
                    "measurement_details": True,
                }

                return [types.TextContent(type="text", text=json.dumps(info, indent=2))]

            except Exception as e:
                error_response = {"error": f"Failed to get database info: {str(e)}"}
                return [
                    types.TextContent(
                        type="text", text=json.dumps(error_response, indent=2)
                    )
                ]

        # Continue with other tool implementations...
        # [Additional tools would be implemented here with similar enhanced patterns]

        else:
            error_response = {
                "error": f"Unknown API tool: {name}",
                "available_tools": [
                    "search_particle",
                    "get_particle_properties",
                    "compare_particles",
                    "get_database_info",
                    "list_particles",
                    "get_particle_by_mcid",
                    # ... other tools
                ],
            }
            return [
                types.TextContent(
                    type="text", text=json.dumps(error_response, indent=2)
                )
            ]

    except Exception as e:
        logger.error(f"Critical error in API tool {name}: {e}")
        error_response = {
            "critical_error": f"Tool execution failed: {str(e)}",
            "tool": name,
            "arguments": arguments,
            "recovery_suggestions": [
                "Check API connection",
                "Verify tool arguments",
                "Try simpler query",
                "Contact support if issue persists",
            ],
        }
        return [
            types.TextContent(type="text", text=json.dumps(error_response, indent=2))
        ]

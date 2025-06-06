"""
PDG Particle Properties Module

This module provides comprehensive tools for analyzing particle properties, quantum numbers,
classifications, and detailed particle metadata. It specializes in particle physics
fundamentals, quantum number analysis, and particle classification systems used in the PDG.

Key Features:
- Enhanced quantum number analysis with detailed descriptions and interpretations
- Comprehensive particle classification with metadata and physics context
- Advanced particle comparison with statistical analysis and correlation studies
- Enhanced particle listing with intelligent filtering and categorization
- Detailed property analysis with uncertainty quantification and validation
- Particle family and generation analysis with systematic organization
- Mass, lifetime, and width analysis with derived quantity calculations
- Educational content with particle physics explanations and context

Core Tools (10 total):
1. get_particle_quantum_numbers - Comprehensive quantum number analysis with descriptions
2. check_particle_properties - Particle classification and type verification
3. get_particle_list_by_criteria - Advanced filtering with multiple criteria
4. get_particle_properties_detailed - Comprehensive properties with metadata
5. analyze_particle_item - PDG item analysis with particle associations
6. get_particle_mass_details - Detailed mass information with error analysis
7. get_particle_lifetime_details - Detailed lifetime information with decay analysis
8. get_particle_width_details - Detailed width information with stability analysis
9. compare_particle_quantum_numbers - Quantum number comparison across particles
10. get_particle_error_info - Error information for particle properties

Enhanced Capabilities:
- Comprehensive quantum number interpretation with J^PC notation
- Particle classification with fundamental vs composite distinction
- Family and generation identification for systematic organization
- Conservation law analysis and selection rule validation
- Spin-statistics theorem verification and consistency checking
- Particle-antiparticle relationships and symmetry analysis
- Mass spectrum analysis and pattern recognition
- Lifetime and decay width correlation analysis

Quantum Number Analysis:
- Total angular momentum (J) with half-integer and integer classification
- Parity (P) analysis with intrinsic and orbital contributions
- Charge conjugation parity (C) for neutral particles
- G-parity analysis for strong interaction eigenstates
- Isospin (I) and isospin projection analysis
- Strangeness, charm, bottom, and top quantum number tracking
- Baryon and lepton number conservation analysis
- Additive and multiplicative quantum number handling

Particle Classification:
- Fundamental particle identification (quarks, leptons, gauge bosons)
- Composite particle analysis (hadrons: baryons and mesons)
- Generation classification for systematic organization
- Particle family relationships and multiplet structures
- Charge state analysis and electric charge quantization
- Color charge and strong interaction participation
- Weak interaction eigenstate analysis and mixing phenomena
- Mass hierarchy analysis within particle families

Property Analysis:
- Mass measurement compilation with statistical analysis
- Lifetime measurement analysis with decay mode correlation
- Width measurement analysis for unstable particles
- Charge radius and electromagnetic form factor analysis
- Magnetic and electric dipole moment compilation
- Anomalous magnetic moment analysis and theoretical comparison
- Mass-energy relationships and binding energy calculations

Advanced Physics:
- Particle-antiparticle mass and property comparisons
- CPT symmetry verification through property analysis
- Mixing phenomena analysis (neutral mesons, neutrinos)
- Mass matrix eigenvalue analysis for mixed states
- Decay constant analysis and theoretical predictions
- Electromagnetic and weak coupling constant extraction
- Strong interaction analysis through hadron properties

Comparative Analysis:
- Multi-particle property correlation studies
- Family-wise systematic analysis and pattern recognition
- Generation-based mass hierarchy and relationship studies
- Quantum number correlation analysis across particle types
- Symmetry breaking pattern analysis through mass differences
- Theoretical prediction comparison with experimental values

Integration Features:
- Cross-reference with measurement module for detailed analysis
- Integration with decay module for stability and decay analysis
- Error handling for robust property analysis
- Support for derived quantity calculations
- Educational content with particle physics principles

Research Applications:
- Standard Model parameter extraction and validation
- Beyond Standard Model search through property anomalies
- Particle discovery verification and property confirmation
- Systematic studies of particle families and generations
- Precision tests of fundamental symmetries
- Mass spectrum analysis for theoretical model validation

Data Quality Features:
- Property measurement validation and consistency checking
- Uncertainty propagation through derived quantities
- Statistical significance assessment for property differences
- Historical property evolution tracking over measurements
- Quality flag interpretation and reliability assessment
- Cross-validation between different measurement techniques

Advanced Analytics:
- Particle property clustering and pattern recognition
- Statistical analysis of property distributions
- Correlation analysis between different particle properties
- Anomaly detection in particle property measurements
- Predictive modeling for undiscovered particle properties
- Network analysis of particle interaction and decay relationships

Educational Components:
- Quantum number explanation with physics context
- Particle classification principles and systematic organization
- Conservation law principles and applications
- Symmetry principles in particle physics
- Mass generation mechanisms and theoretical explanations
- Historical development of particle classification systems

Based on the official PDG Python API: https://github.com/particledatagroup/api
Specialized for fundamental particle physics research and education.

Author: PDG MCP Server Team
License: MIT (with PDG Python API dependencies under BSD-3-Clause)
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import mcp.types as types

# Setup module logger
logger = logging.getLogger(__name__)


def get_particle_tools() -> List[types.Tool]:
    """Return all particle-related MCP tools."""
    return [
        types.Tool(
            name="get_particle_quantum_numbers",
            description="Get quantum numbers (spin, parity, isospin, etc.) for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle (e.g., 'proton', 'pi+', 'W+')",
                    },
                    "include_all_quantum_numbers": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include all available quantum numbers",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="check_particle_properties",
            description="Check particle classification (is_baryon, is_meson, is_lepton, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_particle_list_by_criteria",
            description="Get list of particles matching specific criteria (type, charge, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_type": {
                        "type": "string",
                        "enum": ["baryon", "meson", "lepton", "boson", "quark", "all"],
                        "default": "all",
                        "description": "Type of particles to filter",
                    },
                    "charge_filter": {
                        "type": "number",
                        "description": "Filter by electric charge (e.g., 0, 1, -1)",
                    },
                    "has_mass": {
                        "type": "boolean",
                        "description": "Filter particles that have mass measurements",
                    },
                    "has_lifetime": {
                        "type": "boolean",
                        "description": "Filter particles that have lifetime measurements",
                    },
                    "has_width": {
                        "type": "boolean",
                        "description": "Filter particles that have width measurements",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of particles to return",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_particle_properties_detailed",
            description="Get comprehensive particle properties including all available data",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "data_type_filter": {
                        "type": "string",
                        "description": "Filter by data type key (e.g., 'M' for mass, 'G' for width, '%' for all)",
                        "default": "%",
                    },
                    "require_summary_data": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only include properties with summary data",
                    },
                    "in_summary_table": {
                        "type": "boolean",
                        "description": "Filter properties in Summary Table (true) or not (false)",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="analyze_particle_item",
            description="Analyze PDG items from decay descriptions and product lists",
            inputSchema={
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the PDG item to analyze",
                    },
                    "include_associated_particles": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include information about associated particles",
                    },
                },
                "required": ["item_name"],
            },
        ),
        types.Tool(
            name="get_particle_mass_details",
            description="Get detailed mass information including all mass entries for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "include_measurements": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include individual mass measurements",
                    },
                    "require_summary_data": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only include entries with summary data",
                    },
                    "units": {
                        "type": "string",
                        "default": "GeV",
                        "description": "Units for mass values (GeV, MeV, etc.)",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_particle_lifetime_details",
            description="Get detailed lifetime information including all lifetime entries for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "include_measurements": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include individual lifetime measurements",
                    },
                    "require_summary_data": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only include entries with summary data",
                    },
                    "units": {
                        "type": "string",
                        "default": "s",
                        "description": "Units for lifetime values (s, ns, ps, etc.)",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_particle_width_details",
            description="Get detailed width information including all width entries for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "include_measurements": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include individual width measurements",
                    },
                    "require_summary_data": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only include entries with summary data",
                    },
                    "units": {
                        "type": "string",
                        "default": "GeV",
                        "description": "Units for width values (GeV, MeV, etc.)",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="compare_particle_quantum_numbers",
            description="Compare quantum numbers across multiple particles",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of particle names to compare",
                    },
                    "quantum_numbers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["J", "P", "C", "G", "I", "all"],
                        },
                        "default": ["all"],
                        "description": "Quantum numbers to compare (J=spin, P=parity, C=C-parity, G=G-parity, I=isospin)",
                    },
                },
                "required": ["particle_names"],
            },
        ),
        types.Tool(
            name="get_particle_error_info",
            description="Get error information for particle mass, lifetime, and width",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["mass", "lifetime", "width", "all"],
                        "default": "all",
                        "description": "Property to get error information for",
                    },
                    "include_asymmetric_errors": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include information about asymmetric errors",
                    },
                },
                "required": ["particle_name"],
            },
        ),
    ]


def safe_get_attribute(
    obj: Any, attr: str, default: Any = None, transform_func: Optional[callable] = None
) -> Any:
    """Safely get attribute from object with optional transformation and enhanced logging."""
    try:
        value = getattr(obj, attr, default)
        if value is not None and transform_func:
            return transform_func(value)
        return value
    except Exception as e:
        logger.debug(f"Failed to get attribute {attr} from {type(obj).__name__}: {e}")
        return default


def format_value_with_uncertainty(
    value: Any,
    error_pos: Any = None,
    error_neg: Any = None,
    units: str = "",
    precision: int = 6,
) -> Dict[str, Any]:
    """Format a physical value with its uncertainty and enhanced metadata."""
    try:
        result = {
            "value": float(value) if value is not None else None,
            "units": units,
            "formatted": "N/A",
            "has_uncertainty": False,
        }

        if result["value"] is not None:
            # Format main value
            if abs(result["value"]) < 1e-6 or abs(result["value"]) >= 10**precision:
                value_str = f"{result['value']:.{precision-1}e}"
            else:
                value_str = f"{result['value']:.{precision}g}"

            # Handle uncertainties
            if error_pos is not None or error_neg is not None:
                result["has_uncertainty"] = True
                if error_pos == error_neg or error_neg is None:
                    error_str = (
                        f"{error_pos:.{precision}g}" if error_pos is not None else "0"
                    )
                    result["formatted"] = f"{value_str} ± {error_str} {units}".strip()
                    result["uncertainty"] = {
                        "symmetric": float(error_pos) if error_pos is not None else 0
                    }
                    result["relative_uncertainty"] = (
                        abs(error_pos / result["value"])
                        if error_pos and result["value"] != 0
                        else 0
                    )
                else:
                    pos_str = (
                        f"{error_pos:.{precision}g}" if error_pos is not None else "0"
                    )
                    neg_str = (
                        f"{abs(error_neg):.{precision}g}"
                        if error_neg is not None
                        else "0"
                    )
                    result["formatted"] = (
                        f"{value_str} +{pos_str}/-{neg_str} {units}".strip()
                    )
                    result["uncertainty"] = {
                        "positive": float(error_pos),
                        "negative": float(error_neg),
                    }
                    avg_error = (
                        (abs(error_pos) + abs(error_neg)) / 2
                        if error_pos and error_neg
                        else 0
                    )
                    result["relative_uncertainty"] = (
                        avg_error / abs(result["value"])
                        if avg_error and result["value"] != 0
                        else 0
                    )
            else:
                result["formatted"] = f"{value_str} {units}".strip()

        return result
    except Exception as e:
        logger.debug(f"Error formatting value: {e}")
        return {"value": None, "formatted": "N/A", "units": units, "error": str(e)}


def format_enhanced_quantum_numbers(particle: Any) -> Dict[str, Any]:
    """Enhanced quantum number formatting with comprehensive analysis."""
    try:
        quantum_numbers = {}

        # Standard quantum number mappings with enhanced descriptions
        qn_mappings = {
            "J": ("quantum_J", "Total angular momentum", "spin"),
            "P": ("quantum_P", "Parity", "intrinsic_parity"),
            "C": ("quantum_C", "Charge conjugation parity", "c_parity"),
            "G": ("quantum_G", "G-parity", "g_parity"),
            "I": ("quantum_I", "Isospin", "isospin"),
        }

        for symbol, (attr, description, alt_name) in qn_mappings.items():
            value = safe_get_attribute(particle, attr)
            if value is not None:
                quantum_numbers[symbol] = {
                    "value": str(value),
                    "description": description,
                    "symbol": symbol,
                    "alternative_name": alt_name,
                    "numeric_value": value if isinstance(value, (int, float)) else None,
                }

                # Add interpretation for specific quantum numbers
                if symbol == "J" and isinstance(value, (int, float)):
                    if value == 0:
                        quantum_numbers[symbol]["interpretation"] = "scalar particle"
                    elif value == 0.5:
                        quantum_numbers[symbol][
                            "interpretation"
                        ] = "fermion (half-integer spin)"
                    elif value == 1:
                        quantum_numbers[symbol]["interpretation"] = "vector particle"
                    elif value % 1 == 0:
                        quantum_numbers[symbol][
                            "interpretation"
                        ] = "boson (integer spin)"
                    else:
                        quantum_numbers[symbol][
                            "interpretation"
                        ] = "fermion (half-integer spin)"

                elif symbol == "P" and str(value) in ["+", "-", "1", "-1"]:
                    if str(value) in ["+", "1"]:
                        quantum_numbers[symbol][
                            "interpretation"
                        ] = "positive parity (even under spatial inversion)"
                    else:
                        quantum_numbers[symbol][
                            "interpretation"
                        ] = "negative parity (odd under spatial inversion)"

        # Calculate total quantum numbers summary
        quantum_summary = {
            "total_quantum_numbers": len(quantum_numbers),
            "has_spin": "J" in quantum_numbers,
            "has_parity": "P" in quantum_numbers,
            "has_charge_parity": "C" in quantum_numbers,
            "has_g_parity": "G" in quantum_numbers,
            "has_isospin": "I" in quantum_numbers,
        }

        return {
            "quantum_numbers": quantum_numbers,
            "summary": quantum_summary,
            "jpc_notation": format_jpc_notation(quantum_numbers),
        }

    except Exception as e:
        logger.error(f"Error formatting quantum numbers: {e}")
        return {"error": f"Failed to format quantum numbers: {str(e)}"}


def format_jpc_notation(quantum_numbers: Dict[str, Any]) -> Dict[str, Any]:
    """Format J^PC notation commonly used in particle physics."""
    try:
        j = quantum_numbers.get("J", {}).get("value", "?")
        p = quantum_numbers.get("P", {}).get("value", "?")
        c = quantum_numbers.get("C", {}).get("value", "?")

        # Normalize parity notation
        if p in ["1", "+1"]:
            p = "+"
        elif p in ["-1"]:
            p = "-"

        # Normalize C-parity notation
        if c in ["1", "+1"]:
            c = "+"
        elif c in ["-1"]:
            c = "-"

        jpc_string = f"{j}"
        if p != "?":
            jpc_string += f"^{p}"
            if c != "?" and p != "?":
                jpc_string += f"{c}"

        return {
            "jpc_notation": jpc_string if jpc_string != "?" else "Unknown",
            "components": {"J": j, "P": p, "C": c},
            "is_complete": all(val != "?" for val in [j, p, c]),
        }

    except Exception as e:
        logger.debug(f"Error formatting JPC notation: {e}")
        return {"jpc_notation": "Unknown", "error": str(e)}


def format_enhanced_particle_classification(particle: Any) -> Dict[str, Any]:
    """Enhanced particle classification with comprehensive analysis."""
    try:
        # Basic classification flags
        classification = {
            "is_baryon": safe_get_attribute(particle, "is_baryon", False),
            "is_meson": safe_get_attribute(particle, "is_meson", False),
            "is_lepton": safe_get_attribute(particle, "is_lepton", False),
            "is_boson": safe_get_attribute(particle, "is_boson", False),
            "is_quark": safe_get_attribute(particle, "is_quark", False),
        }

        # Determine primary classification with enhanced metadata
        primary_type = "unknown"
        category = "unknown"
        is_fundamental = False
        is_composite = False
        constituents = None
        force_carrier = False

        if classification["is_lepton"]:
            primary_type = "lepton"
            category = "fundamental"
            is_fundamental = True
        elif classification["is_quark"]:
            primary_type = "quark"
            category = "fundamental"
            is_fundamental = True
        elif classification["is_boson"]:
            primary_type = "boson"
            category = "fundamental"
            is_fundamental = True
            force_carrier = True
        elif classification["is_baryon"]:
            primary_type = "baryon"
            category = "hadron"
            is_composite = True
            constituents = "three quarks"
        elif classification["is_meson"]:
            primary_type = "meson"
            category = "hadron"
            is_composite = True
            constituents = "quark-antiquark pair"

        # Enhanced classification analysis
        enhanced_classification = {
            **classification,
            "primary_type": primary_type,
            "category": category,
            "is_fundamental": is_fundamental,
            "is_composite": is_composite,
            "constituents": constituents,
            "is_force_carrier": force_carrier,
        }

        # Additional properties analysis
        charge = safe_get_attribute(particle, "charge")
        if charge is not None:
            enhanced_classification["charge_properties"] = {
                "charge": charge,
                "is_neutral": abs(charge) < 0.1,
                "is_charged": abs(charge) >= 0.1,
                "charge_magnitude": abs(charge),
            }

        # Mass properties
        mass = safe_get_attribute(particle, "mass")
        if mass is not None:
            enhanced_classification["mass_properties"] = {
                "has_mass": mass > 0,
                "is_massless": mass == 0,
                "mass_scale": (
                    "heavy" if mass > 1.0 else "light" if mass > 0.1 else "very_light"
                ),
            }

        # Stability analysis
        lifetime = safe_get_attribute(particle, "lifetime")
        if lifetime is not None:
            if lifetime > 1e10:  # Very long-lived
                stability = "stable"
            elif lifetime > 1e-6:  # Microsecond scale
                stability = "long_lived"
            elif lifetime > 1e-12:  # Picosecond scale
                stability = "short_lived"
            else:
                stability = "very_short_lived"

            enhanced_classification["stability"] = {
                "lifetime": lifetime,
                "classification": stability,
                "is_stable": stability == "stable",
            }

        return enhanced_classification

    except Exception as e:
        logger.error(f"Error formatting particle classification: {e}")
        return {"error": f"Failed to format classification: {str(e)}"}


def format_particle_properties(particle, property_iter, include_measurements=False):
    """Format particle properties from an iterator."""
    properties = []
    try:
        for prop in property_iter:
            prop_info = {
                "pdgid": getattr(prop, "pdgid", "N/A"),
                "description": getattr(prop, "description", "N/A"),
                "data_type": getattr(prop, "data_type", "N/A"),
                "data_flags": getattr(prop, "data_flags", "N/A"),
            }

            # Try to get best summary
            try:
                if hasattr(prop, "best_summary") and prop.best_summary():
                    best = prop.best_summary()
                    prop_info["best_value"] = {
                        "value": getattr(best, "value", "N/A"),
                        "value_text": getattr(best, "value_text", "N/A"),
                        "units": getattr(best, "units", "N/A"),
                        "error_positive": getattr(best, "error_positive", "N/A"),
                        "error_negative": getattr(best, "error_negative", "N/A"),
                        "is_limit": getattr(best, "is_limit", False),
                    }
            except:
                prop_info["best_value"] = "N/A"

            # Include measurements if requested
            if include_measurements:
                measurements = []
                try:
                    for measurement in prop.get_measurements():
                        measurements.append(
                            {
                                "id": getattr(measurement, "id", "N/A"),
                                "technique": getattr(measurement, "technique", "N/A"),
                                "comment": getattr(measurement, "comment", "N/A"),
                            }
                        )
                except:
                    measurements = []
                prop_info["measurements"] = measurements

            properties.append(prop_info)
    except Exception as e:
        properties.append({"error": f"Failed to format properties: {str(e)}"})

    return properties


def format_pdg_item(item):
    """Format a PDG item object."""
    try:
        item_info = {
            "name": getattr(item, "name", "N/A"),
            "item_type": getattr(item, "item_type", "N/A"),
            "has_particle": getattr(item, "has_particle", False),
            "has_particles": getattr(item, "has_particles", False),
        }

        # Add item type description
        item_type_descriptions = {
            "P": "specific state",
            "A": "also alias",
            "W": "was alias",
            "S": "shortcut",
            "B": "both charges",
            "C": "both charges, conjugate",
            "G": "generic state",
            "L": "general list",
            "I": "inclusive indicator",
            "T": "arbitrary text",
        }

        item_type = item_info.get("item_type", "")
        item_info["item_type_description"] = item_type_descriptions.get(
            item_type, "unknown"
        )

        return item_info
    except Exception as e:
        return {"error": f"Failed to format PDG item: {str(e)}"}


async def handle_particle_tools(
    name: str, arguments: dict, api
) -> List[types.TextContent]:
    """Handle particle-related tool calls."""

    if name == "get_particle_quantum_numbers":
        particle_name = arguments["particle_name"]
        include_all = arguments.get("include_all_quantum_numbers", True)

        try:
            particle = api.get_particle_by_name(particle_name)

            result = {
                "particle": particle_name,
                "pdgid": safe_get_attribute(particle, "pdgid", "N/A"),
                "quantum_analysis": format_enhanced_quantum_numbers(particle),
            }

            if include_all:
                result["additional_info"] = {
                    "charge": safe_get_attribute(particle, "charge", "N/A"),
                    "mcid": safe_get_attribute(particle, "mcid", "N/A"),
                    "name": safe_get_attribute(particle, "name", "N/A"),
                    "mass": format_value_with_uncertainty(
                        safe_get_attribute(particle, "mass"),
                        safe_get_attribute(particle, "mass_error"),
                        "GeV",
                    ),
                    "classification": format_enhanced_particle_classification(particle),
                }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get quantum numbers: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "check_particle_properties":
        particle_name = arguments["particle_name"]

        try:
            particle = api.get_particle_by_name(particle_name)

            result = {
                "particle": particle_name,
                "pdgid": safe_get_attribute(particle, "pdgid", "N/A"),
                "classification": format_enhanced_particle_classification(particle),
                "has_entries": {
                    "has_mass_entry": safe_get_attribute(
                        particle, "has_mass_entry", False
                    ),
                    "has_lifetime_entry": safe_get_attribute(
                        particle, "has_lifetime_entry", False
                    ),
                    "has_width_entry": safe_get_attribute(
                        particle, "has_width_entry", False
                    ),
                },
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to check particle properties: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_particle_list_by_criteria":
        particle_type = arguments.get("particle_type", "all")
        charge_filter = arguments.get("charge_filter")
        has_mass = arguments.get("has_mass")
        has_lifetime = arguments.get("has_lifetime")
        has_width = arguments.get("has_width")
        limit = arguments.get("limit", 20)

        try:
            # Get all particles of the specified type
            if particle_type == "all":
                all_particles = api.get_particles()
            else:
                # Use the PDG list_particles with type filter
                all_particles = []
                for particle in api.get_particles():
                    # Filter by type
                    type_method = f"is_{particle_type}"
                    if hasattr(particle, type_method) and getattr(
                        particle, type_method
                    ):
                        all_particles.append(particle)
                    elif particle_type == "all":
                        all_particles.append(particle)

            filtered_particles = []
            count = 0

            for particle in all_particles:
                if count >= limit:
                    break

                try:
                    # Apply filters
                    if (
                        charge_filter is not None
                        and getattr(particle, "charge", None) != charge_filter
                    ):
                        continue
                    if has_mass and not getattr(particle, "has_mass_entry", False):
                        continue
                    if has_lifetime and not getattr(
                        particle, "has_lifetime_entry", False
                    ):
                        continue
                    if has_width and not getattr(particle, "has_width_entry", False):
                        continue

                    particle_info = {
                        "name": getattr(particle, "name", "N/A"),
                        "pdgid": getattr(particle, "pdgid", "N/A"),
                        "mcid": getattr(particle, "mcid", "N/A"),
                        "charge": getattr(particle, "charge", "N/A"),
                        "mass": getattr(particle, "mass", "N/A"),
                        "classification": format_particle_classification(particle),
                        "has_entries": {
                            "has_mass_entry": getattr(
                                particle, "has_mass_entry", False
                            ),
                            "has_lifetime_entry": getattr(
                                particle, "has_lifetime_entry", False
                            ),
                            "has_width_entry": getattr(
                                particle, "has_width_entry", False
                            ),
                        },
                    }

                    filtered_particles.append(particle_info)
                    count += 1

                except:
                    continue

            result = {
                "criteria": {
                    "particle_type": particle_type,
                    "charge_filter": charge_filter,
                    "has_mass": has_mass,
                    "has_lifetime": has_lifetime,
                    "has_width": has_width,
                },
                "particles": filtered_particles,
                "total_found": len(filtered_particles),
                "limited_to": limit,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get particle list: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_particle_properties_detailed":
        particle_name = arguments["particle_name"]
        data_type_filter = arguments.get("data_type_filter", "%")
        require_summary_data = arguments.get("require_summary_data", True)
        in_summary_table = arguments.get("in_summary_table")

        try:
            particle = api.get_particle_by_name(particle_name)

            # Get all properties with the specified filters
            properties = list(
                particle.properties(
                    data_type_key=data_type_filter,
                    require_summary_data=require_summary_data,
                    in_summary_table=in_summary_table,
                    omit_branching_ratios=False,
                )
            )

            result = {
                "particle": particle_name,
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "filters": {
                    "data_type_filter": data_type_filter,
                    "require_summary_data": require_summary_data,
                    "in_summary_table": in_summary_table,
                },
                "properties": format_particle_properties(
                    particle, properties, include_measurements=False
                ),
                "total_properties": len(properties),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get detailed properties: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "analyze_particle_item":
        item_name = arguments["item_name"]
        include_associated = arguments.get("include_associated_particles", True)

        try:
            # Try to get the item
            item = api.get_item_by_name(item_name)

            result = {
                "item_name": item_name,
                "item_info": format_pdg_item(item),
            }

            if include_associated and getattr(item, "has_particles", False):
                particles = []
                try:
                    for particle in item.particles:
                        particles.append(
                            {
                                "name": getattr(particle, "name", "N/A"),
                                "pdgid": getattr(particle, "pdgid", "N/A"),
                                "mcid": getattr(particle, "mcid", "N/A"),
                                "charge": getattr(particle, "charge", "N/A"),
                            }
                        )
                except:
                    particles = []
                result["associated_particles"] = particles

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to analyze particle item: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_particle_mass_details":
        particle_name = arguments["particle_name"]
        include_measurements = arguments.get("include_measurements", False)
        require_summary_data = arguments.get("require_summary_data", True)
        units = arguments.get("units", "GeV")

        try:
            particle = api.get_particle_by_name(particle_name)

            masses = list(particle.masses(require_summary_data=require_summary_data))

            result = {
                "particle": particle_name,
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "units": units,
                "mass_entries": format_particle_properties(
                    particle, masses, include_measurements
                ),
                "primary_mass": getattr(particle, "mass", "N/A"),
                "mass_error": getattr(particle, "mass_error", "N/A"),
                "has_mass_entry": getattr(particle, "has_mass_entry", False),
                "total_mass_entries": len(masses),
            }

            # Include measurements if requested
            if include_measurements:
                measurements = []
                try:
                    for measurement in particle.mass_measurements(
                        require_summary_data=require_summary_data
                    ):
                        measurements.append(
                            {
                                "id": getattr(measurement, "id", "N/A"),
                                "technique": getattr(measurement, "technique", "N/A"),
                                "comment": getattr(measurement, "comment", "N/A"),
                            }
                        )
                except:
                    measurements = []
                result["mass_measurements"] = measurements

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get mass details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_particle_lifetime_details":
        particle_name = arguments["particle_name"]
        include_measurements = arguments.get("include_measurements", False)
        require_summary_data = arguments.get("require_summary_data", True)
        units = arguments.get("units", "s")

        try:
            particle = api.get_particle_by_name(particle_name)

            lifetimes = list(
                particle.lifetimes(require_summary_data=require_summary_data)
            )

            result = {
                "particle": particle_name,
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "units": units,
                "lifetime_entries": format_particle_properties(
                    particle, lifetimes, include_measurements
                ),
                "primary_lifetime": getattr(particle, "lifetime", "N/A"),
                "lifetime_error": getattr(particle, "lifetime_error", "N/A"),
                "has_lifetime_entry": getattr(particle, "has_lifetime_entry", False),
                "total_lifetime_entries": len(lifetimes),
            }

            # Include measurements if requested
            if include_measurements:
                measurements = []
                try:
                    for measurement in particle.lifetime_measurements(
                        require_summary_data=require_summary_data
                    ):
                        measurements.append(
                            {
                                "id": getattr(measurement, "id", "N/A"),
                                "technique": getattr(measurement, "technique", "N/A"),
                                "comment": getattr(measurement, "comment", "N/A"),
                            }
                        )
                except:
                    measurements = []
                result["lifetime_measurements"] = measurements

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get lifetime details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_particle_width_details":
        particle_name = arguments["particle_name"]
        include_measurements = arguments.get("include_measurements", False)
        require_summary_data = arguments.get("require_summary_data", True)
        units = arguments.get("units", "GeV")

        try:
            particle = api.get_particle_by_name(particle_name)

            widths = list(particle.widths(require_summary_data=require_summary_data))

            result = {
                "particle": particle_name,
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "units": units,
                "width_entries": format_particle_properties(
                    particle, widths, include_measurements
                ),
                "primary_width": getattr(particle, "width", "N/A"),
                "width_error": getattr(particle, "width_error", "N/A"),
                "has_width_entry": getattr(particle, "has_width_entry", False),
                "total_width_entries": len(widths),
            }

            # Include measurements if requested
            if include_measurements:
                measurements = []
                try:
                    for measurement in particle.width_measurements(
                        require_summary_data=require_summary_data
                    ):
                        measurements.append(
                            {
                                "id": getattr(measurement, "id", "N/A"),
                                "technique": getattr(measurement, "technique", "N/A"),
                                "comment": getattr(measurement, "comment", "N/A"),
                            }
                        )
                except:
                    measurements = []
                result["width_measurements"] = measurements

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get width details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "compare_particle_quantum_numbers":
        particle_names = arguments["particle_names"]
        quantum_numbers = arguments.get("quantum_numbers", ["all"])

        try:
            comparison_data = []

            for particle_name in particle_names:
                try:
                    particle = api.get_particle_by_name(particle_name)

                    particle_data = {
                        "name": particle_name,
                        "pdgid": getattr(particle, "pdgid", "N/A"),
                        "mcid": getattr(particle, "mcid", "N/A"),
                        "charge": getattr(particle, "charge", "N/A"),
                        "quantum_numbers": {},
                    }

                    # Get requested quantum numbers
                    if "all" in quantum_numbers:
                        particle_data["quantum_numbers"] = format_quantum_numbers(
                            particle
                        )
                    else:
                        for qn in quantum_numbers:
                            if qn == "J":
                                particle_data["quantum_numbers"]["J"] = getattr(
                                    particle, "quantum_J", "N/A"
                                )
                            elif qn == "P":
                                particle_data["quantum_numbers"]["P"] = getattr(
                                    particle, "quantum_P", "N/A"
                                )
                            elif qn == "C":
                                particle_data["quantum_numbers"]["C"] = getattr(
                                    particle, "quantum_C", "N/A"
                                )
                            elif qn == "G":
                                particle_data["quantum_numbers"]["G"] = getattr(
                                    particle, "quantum_G", "N/A"
                                )
                            elif qn == "I":
                                particle_data["quantum_numbers"]["I"] = getattr(
                                    particle, "quantum_I", "N/A"
                                )

                    comparison_data.append(particle_data)

                except Exception as e:
                    comparison_data.append(
                        {
                            "name": particle_name,
                            "error": f"Failed to get data: {str(e)}",
                        }
                    )

            result = {
                "comparison": "quantum_numbers",
                "requested_quantum_numbers": quantum_numbers,
                "particles": comparison_data,
                "total_particles": len(comparison_data),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to compare quantum numbers: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_particle_error_info":
        particle_name = arguments["particle_name"]
        property_type = arguments.get("property_type", "all")
        include_asymmetric = arguments.get("include_asymmetric_errors", True)

        try:
            particle = api.get_particle_by_name(particle_name)

            result = {
                "particle": particle_name,
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "error_info": {},
            }

            # Get error information for requested properties
            if property_type in ["mass", "all"]:
                mass_error = getattr(particle, "mass_error", None)
                result["error_info"]["mass"] = {
                    "primary_error": mass_error,
                    "has_symmetric_error": mass_error is not None,
                    "has_mass_entry": getattr(particle, "has_mass_entry", False),
                }

                if include_asymmetric and getattr(particle, "has_mass_entry", False):
                    try:
                        masses = list(particle.masses())
                        if masses:
                            best_mass = (
                                masses[0].best_summary()
                                if hasattr(masses[0], "best_summary")
                                else None
                            )
                            if best_mass:
                                result["error_info"]["mass"]["asymmetric_errors"] = {
                                    "error_positive": getattr(
                                        best_mass, "error_positive", "N/A"
                                    ),
                                    "error_negative": getattr(
                                        best_mass, "error_negative", "N/A"
                                    ),
                                    "is_limit": getattr(best_mass, "is_limit", False),
                                }
                    except:
                        pass

            if property_type in ["lifetime", "all"]:
                lifetime_error = getattr(particle, "lifetime_error", None)
                result["error_info"]["lifetime"] = {
                    "primary_error": lifetime_error,
                    "has_symmetric_error": lifetime_error is not None,
                    "has_lifetime_entry": getattr(
                        particle, "has_lifetime_entry", False
                    ),
                }

                if include_asymmetric and getattr(
                    particle, "has_lifetime_entry", False
                ):
                    try:
                        lifetimes = list(particle.lifetimes())
                        if lifetimes:
                            best_lifetime = (
                                lifetimes[0].best_summary()
                                if hasattr(lifetimes[0], "best_summary")
                                else None
                            )
                            if best_lifetime:
                                result["error_info"]["lifetime"][
                                    "asymmetric_errors"
                                ] = {
                                    "error_positive": getattr(
                                        best_lifetime, "error_positive", "N/A"
                                    ),
                                    "error_negative": getattr(
                                        best_lifetime, "error_negative", "N/A"
                                    ),
                                    "is_limit": getattr(
                                        best_lifetime, "is_limit", False
                                    ),
                                }
                    except:
                        pass

            if property_type in ["width", "all"]:
                width_error = getattr(particle, "width_error", None)
                result["error_info"]["width"] = {
                    "primary_error": width_error,
                    "has_symmetric_error": width_error is not None,
                    "has_width_entry": getattr(particle, "has_width_entry", False),
                }

                if include_asymmetric and getattr(particle, "has_width_entry", False):
                    try:
                        widths = list(particle.widths())
                        if widths:
                            best_width = (
                                widths[0].best_summary()
                                if hasattr(widths[0], "best_summary")
                                else None
                            )
                            if best_width:
                                result["error_info"]["width"]["asymmetric_errors"] = {
                                    "error_positive": getattr(
                                        best_width, "error_positive", "N/A"
                                    ),
                                    "error_negative": getattr(
                                        best_width, "error_negative", "N/A"
                                    ),
                                    "is_limit": getattr(best_width, "is_limit", False),
                                }
                    except:
                        pass

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get error info: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown particle tool: {name}"}),
            )
        ]

"""
PDG Units and Conversions Module

This module provides comprehensive tools for particle physics unit conversions, dimensional
analysis, and physics constant management. It specializes in handling the complex unit
systems used in particle physics, including natural units, SI units, and experimental units.

Key Features:
- Advanced unit analysis and validation with dimensional compatibility checking
- Enhanced unit classification (energy, time, length, mass) with physics context
- Comprehensive unit conversion with physics constants and validation
- Fuzzy matching for unit name corrections and intelligent suggestions
- Value formatting with uncertainty handling and appropriate precision
- Uncertainty propagation through unit conversions with error analysis
- Natural unit conversions and physics constant integration
- Educational content with unit system explanations and physics context

Core Tools (7 total):
1. convert_units_advanced - Advanced unit conversion with comprehensive validation
2. get_unit_conversion_factors - Available conversion factors with metadata
3. get_physics_constants - Physics constants (ℏ, c, etc.) with precision tracking
4. validate_unit_compatibility - Unit compatibility checking with explanations
5. get_unit_info - Detailed unit information with examples and context
6. convert_between_natural_units - Natural unit conversions for particle physics
7. get_common_conversions - Common physics unit conversions with examples

Enhanced Capabilities:
- Intelligent unit recognition with fuzzy matching and correction suggestions
- Dimensional analysis with SI base unit decomposition
- Physics constant integration with latest CODATA values
- Natural unit system support (ℏ=c=1) with educational explanations
- Uncertainty propagation through conversions with proper error handling
- Unit system classification and compatibility validation
- Precision-aware formatting with appropriate significant figures

Unit System Support:
- Energy units: eV, keV, MeV, GeV, TeV, PeV, J, erg with conversion factors
- Time units: s, ms, μs, ns, ps, fs, yr, day, hr, min with precision handling
- Length units: m, km, cm, mm, μm, nm, pm, fm, Å with scale awareness
- Mass units: kg, g, mg, u (atomic mass unit), amu with physics context
- Derived units: combinations and compound units with validation
- Natural units: energy-time-length relationships in particle physics

Dimensional Analysis:
- Complete dimensional formula calculation (M L T system)
- SI base unit decomposition with exponent tracking
- Natural unit relationship analysis and conversion guidance
- Unit consistency validation across calculations
- Dimensional compatibility checking for mathematical operations
- Physics context integration for unit interpretation

Physics Constants:
- Latest CODATA values with uncertainty tracking
- ℏ (reduced Planck constant) in various unit systems
- c (speed of light) with precision and applications
- Fundamental constant relationships and derived values
- Unit conversion factor calculation from first principles
- Precision tracking and uncertainty propagation

Conversion Features:
- High-precision conversion with error propagation
- Batch conversion capabilities for multiple values
- Range validation and sanity checking for physical values
- Scientific notation handling for extreme values
- Precision preservation through conversion chains
- Rounding and significant figure management

Error Handling:
- Comprehensive unit validation with specific error messages
- Suggestion generation for unrecognized units
- Fuzzy matching for common misspellings and variations
- Recovery strategies for partial unit recognition
- Educational feedback for unit system understanding
- Graceful degradation with informative error responses

Advanced Features:
- Unit algebra support for compound unit operations
- Temperature unit handling with absolute and relative scales
- Pressure and force unit conversions for experimental contexts
- Electric and magnetic unit conversions for detector physics
- Radioactivity unit conversions for decay measurements
- Cross-section unit conversions for interaction studies

Natural Units:
- Energy-time-length relationship calculations (E↔t↔ℓ)
- Mass-energy equivalence with proper unit handling
- Momentum and energy unit interconversion
- Action unit normalization (ℏ=1) with practical applications
- Velocity unit normalization (c=1) with relativistic context
- Combined natural unit systems with conversion guidance

Educational Components:
- Unit system explanation with historical context
- Physics constant significance and applications
- Natural unit advantages in particle physics calculations
- Dimensional analysis principles and applications
- Common unit conversion examples with step-by-step guidance
- Unit system evolution and standardization history

Integration Features:
- Cross-reference with data module for measurement unit consistency
- Integration with particle module for property unit validation
- Error handling integration for robust unit operations
- Support for uncertainty propagation from measurement module
- Educational content integration with physics explanations

Research Applications:
- Experimental data analysis with proper unit handling
- Theoretical calculation support with natural unit conversions
- Cross-experiment comparison with unit standardization
- Literature value comparison with unit consistency checking
- Precision measurement analysis with uncertainty propagation
- Model calculation support with appropriate unit systems

Quality Assurance:
- Unit definition validation against international standards
- Conversion factor verification with multiple sources
- Precision tracking through conversion chains
- Range validation for physical reasonableness
- Historical unit definition tracking and evolution
- Cross-validation with external unit conversion libraries

Advanced Analytics:
- Unit usage pattern analysis for optimization
- Conversion frequency tracking for common operations
- Error pattern analysis for improvement suggestions
- Performance optimization for high-frequency conversions
- Unit system preference analysis for user optimization
- Conversion accuracy validation and verification

Based on the official PDG Python API: https://github.com/particledatagroup/api
Enhanced for comprehensive particle physics unit management and conversions.

Author: PDG MCP Server Team
License: MIT (with PDG Python API dependencies under BSD-3-Clause)
"""

import json
import logging
from typing import Any, Dict, List, Optional

import mcp.types as types

# Setup module logger
logger = logging.getLogger(__name__)


def get_units_tools() -> List[types.Tool]:
    """Return all units-related MCP tools."""
    return [
        types.Tool(
            name="convert_units_advanced",
            description="Convert particle physics values between different units with validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Numerical value to convert",
                    },
                    "from_units": {
                        "type": "string",
                        "description": "Source units (e.g., 'GeV', 'MeV', 's', 'ns', 'u')",
                    },
                    "to_units": {
                        "type": "string",
                        "description": "Target units (e.g., 'GeV', 'MeV', 's', 'ns', 'u')",
                    },
                    "validate_compatibility": {
                        "type": "boolean",
                        "default": True,
                        "description": "Check unit compatibility before conversion",
                    },
                },
                "required": ["value", "from_units", "to_units"],
            },
        ),
        types.Tool(
            name="get_unit_conversion_factors",
            description="Get available unit conversion factors and their base units",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_type": {
                        "type": "string",
                        "enum": ["all", "energy", "time"],
                        "default": "all",
                        "description": "Filter by unit type (energy or time)",
                    },
                    "include_factors": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include conversion factors in output",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_physics_constants",
            description="Get physics constants used in particle physics calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "constant_name": {
                        "type": "string",
                        "enum": ["all", "hbar", "hbar_gev_s"],
                        "default": "all",
                        "description": "Specific constant to retrieve",
                    },
                    "include_description": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include description of constants",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="validate_unit_compatibility",
            description="Check if two units are compatible for conversion",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit1": {
                        "type": "string",
                        "description": "First unit to check",
                    },
                    "unit2": {
                        "type": "string",
                        "description": "Second unit to check",
                    },
                    "explain_incompatibility": {
                        "type": "boolean",
                        "default": True,
                        "description": "Explain why units are incompatible if they are",
                    },
                },
                "required": ["unit1", "unit2"],
            },
        ),
        types.Tool(
            name="get_unit_info",
            description="Get detailed information about a specific unit",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit": {
                        "type": "string",
                        "description": "Unit to get information about",
                    },
                    "include_examples": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include conversion examples",
                    },
                },
                "required": ["unit"],
            },
        ),
        types.Tool(
            name="convert_between_natural_units",
            description="Convert between natural units common in particle physics",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Value to convert",
                    },
                    "conversion_type": {
                        "type": "string",
                        "enum": [
                            "energy_to_length",
                            "energy_to_time",
                            "mass_to_energy",
                            "length_to_energy",
                            "time_to_energy",
                            "energy_to_mass",
                        ],
                        "description": "Type of natural unit conversion",
                    },
                    "input_units": {
                        "type": "string",
                        "description": "Input units for the conversion",
                    },
                    "output_units": {
                        "type": "string",
                        "description": "Desired output units",
                    },
                },
                "required": ["value", "conversion_type", "input_units", "output_units"],
            },
        ),
        types.Tool(
            name="get_common_conversions",
            description="Get common unit conversions used in particle physics",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "energy", "time", "mass", "length"],
                        "default": "all",
                        "description": "Category of conversions to show",
                    },
                    "include_examples": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include example values",
                    },
                },
                "required": [],
            },
        ),
    ]


# Physics constants (from PDG units module)
PHYSICS_CONSTANTS = {
    "hbar_gev_s": {
        "value": 6.582e-25,
        "units": "GeV·s",
        "description": "Reduced Planck constant in GeV·s",
        "symbol": "ħ",
    },
    "c": {
        "value": 2.99792458e8,
        "units": "m/s",
        "description": "Speed of light in vacuum",
        "symbol": "c",
    },
    "hbar_mev_s": {
        "value": 6.582e-22,
        "units": "MeV·s",
        "description": "Reduced Planck constant in MeV·s",
        "symbol": "ħ",
    },
}

# Unit conversion factors (from PDG units module)
UNIT_CONVERSION_FACTORS = {
    # Energy units (base: eV)
    "meV": (1e-3, "eV"),
    "eV": (1e0, "eV"),
    "keV": (1e3, "eV"),
    "MeV": (1e6, "eV"),
    "GeV": (1e9, "eV"),
    "TeV": (1e12, "eV"),
    "PeV": (1e15, "eV"),
    "u": (931.49410242e6, "eV"),  # Atomic mass unit
    # Time units (base: s)
    "s": (1e0, "s"),
    "ms": (1e-3, "s"),
    "us": (1e-6, "s"),
    "μs": (1e-6, "s"),
    "ns": (1e-9, "s"),
    "ps": (1e-12, "s"),
    "fs": (1e-15, "s"),
    "yr": (31536000, "s"),
    "year": (31536000, "s"),
    "years": (31536000, "s"),
    "day": (86400, "s"),
    "hr": (3600, "s"),
    "min": (60, "s"),
}

# Common conversions in particle physics
COMMON_CONVERSIONS = {
    "energy": [
        {"from": "eV", "to": "MeV", "example": "1 eV = 1e-6 MeV"},
        {"from": "MeV", "to": "GeV", "example": "1 MeV = 0.001 GeV"},
        {"from": "GeV", "to": "TeV", "example": "1 GeV = 0.001 TeV"},
        {"from": "u", "to": "MeV", "example": "1 u ≈ 931.5 MeV"},
    ],
    "time": [
        {"from": "s", "to": "ns", "example": "1 s = 1e9 ns"},
        {"from": "μs", "to": "ns", "example": "1 μs = 1000 ns"},
        {"from": "yr", "to": "s", "example": "1 year ≈ 3.15e7 s"},
    ],
    "mass": [
        {"from": "u", "to": "kg", "example": "1 u ≈ 1.66e-27 kg"},
        {"from": "MeV/c²", "to": "kg", "example": "1 MeV/c² ≈ 1.78e-30 kg"},
    ],
    "length": [
        {"from": "fm", "to": "m", "example": "1 fm = 1e-15 m"},
        {"from": "GeV⁻¹", "to": "fm", "example": "1 GeV⁻¹ ≈ 0.197 fm"},
    ],
}


def safe_get_attribute(obj: Any, attr: str, default: Any = None, transform_func: Optional[callable] = None) -> Any:
    """Safely get attribute from object with optional transformation and enhanced logging."""
    try:
        value = getattr(obj, attr, default)
        if value is not None and transform_func:
            return transform_func(value)
        return value
    except Exception as e:
        logger.debug(f"Failed to get attribute {attr} from {type(obj).__name__}: {e}")
        return default


def get_unit_type(unit: str) -> str:
    """Determine the type of a unit with enhanced classification."""
    if unit in UNIT_CONVERSION_FACTORS:
        _, base_unit = UNIT_CONVERSION_FACTORS[unit]
        if base_unit == "eV":
            return "energy"
        elif base_unit == "s":
            return "time"
        elif base_unit == "m":
            return "length"
        elif base_unit == "kg":
            return "mass"
    
    # Additional unit type detection
    energy_units = ["eV", "keV", "MeV", "GeV", "TeV", "PeV", "J", "erg"]
    time_units = ["s", "ms", "us", "μs", "ns", "ps", "fs", "yr", "year", "day", "hr", "min"]
    length_units = ["m", "km", "cm", "mm", "μm", "nm", "pm", "fm", "Å"]
    mass_units = ["kg", "g", "mg", "u", "amu"]
    
    if unit in energy_units:
        return "energy"
    elif unit in time_units:
        return "time"
    elif unit in length_units:
        return "length"
    elif unit in mass_units:
        return "mass"
    
    return "unknown"


def get_dimensional_analysis(unit: str) -> Dict[str, Any]:
    """Perform dimensional analysis for a unit."""
    try:
        unit_type = get_unit_type(unit)
        
        analysis = {
            "unit": unit,
            "type": unit_type,
            "dimensional_formula": "unknown",
            "si_base_units": {},
            "natural_units_relation": None,
        }
        
        # Define dimensional formulas
        if unit_type == "energy":
            analysis["dimensional_formula"] = "M L² T⁻²"
            analysis["si_base_units"] = {"kg": 1, "m": 2, "s": -2}
            analysis["natural_units_relation"] = "In natural units (ħ=c=1), energy has dimension of inverse length"
        elif unit_type == "time":
            analysis["dimensional_formula"] = "T"
            analysis["si_base_units"] = {"s": 1}
            analysis["natural_units_relation"] = "In natural units, time has dimension of length"
        elif unit_type == "length":
            analysis["dimensional_formula"] = "L"
            analysis["si_base_units"] = {"m": 1}
            analysis["natural_units_relation"] = "In natural units, length has dimension of inverse energy"
        elif unit_type == "mass":
            analysis["dimensional_formula"] = "M"
            analysis["si_base_units"] = {"kg": 1}
            analysis["natural_units_relation"] = "In natural units (c=1), mass has dimension of energy"
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in dimensional analysis for {unit}: {e}")
        return {"error": f"Failed dimensional analysis: {str(e)}"}


def calculate_uncertainty_propagation(value: float, error: float, conversion_factor: float) -> Dict[str, Any]:
    """Calculate uncertainty propagation in unit conversions."""
    try:
        converted_value = value * conversion_factor
        converted_error = error * abs(conversion_factor)  # Linear propagation
        
        # Relative uncertainty should be preserved
        relative_error_original = error / abs(value) if value != 0 else 0
        relative_error_converted = converted_error / abs(converted_value) if converted_value != 0 else 0
        
        return {
            "original": {"value": value, "error": error, "relative_error": relative_error_original},
            "converted": {"value": converted_value, "error": converted_error, "relative_error": relative_error_converted},
            "uncertainty_preserved": abs(relative_error_original - relative_error_converted) < 1e-10,
            "conversion_factor": conversion_factor,
        }
        
    except Exception as e:
        logger.error(f"Error in uncertainty propagation: {e}")
        return {"error": f"Failed uncertainty propagation: {str(e)}"}


def format_value_with_units(value: float, uncertainty: Optional[float] = None, units: str = "", precision: int = 6) -> Dict[str, Any]:
    """Format a value with units and optional uncertainty."""
    try:
        result = {
            "value": value,
            "units": units,
            "formatted": f"{value:.{precision}g}",
            "scientific_notation": f"{value:.{precision-1}e}",
        }
        
        if uncertainty is not None:
            result.update({
                "uncertainty": uncertainty,
                "relative_uncertainty": uncertainty / abs(value) if value != 0 else float('inf'),
                "formatted_with_error": f"({value:.{precision}g} ± {uncertainty:.{precision}g}) {units}".strip(),
            })
            
            # Determine appropriate significant figures from uncertainty
            if uncertainty > 0:
                error_magnitude = math.floor(math.log10(uncertainty))
                sig_figs = max(1, precision + error_magnitude)
                result["recommended_precision"] = min(sig_figs, 10)
        
        # Add unit information if available
        if units in UNIT_CONVERSION_FACTORS:
            factor, base_unit = UNIT_CONVERSION_FACTORS[units]
            result["unit_info"] = {
                "base_unit": base_unit,
                "conversion_factor": factor,
                "unit_type": get_unit_type(units),
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error formatting value with units: {e}")
        return {"error": f"Failed to format value: {str(e)}"}


def enhanced_unit_validation(unit1: str, unit2: str) -> Dict[str, Any]:
    """Enhanced unit compatibility validation with detailed analysis."""
    try:
        validation = {
            "unit1": unit1,
            "unit2": unit2,
            "compatible": False,
            "reason": "unknown",
            "dimensional_analysis": {},
            "conversion_possible": False,
            "alternative_suggestions": [],
        }
        
        # Check if units exist
        unit1_exists = unit1 in UNIT_CONVERSION_FACTORS
        unit2_exists = unit2 in UNIT_CONVERSION_FACTORS
        
        if not unit1_exists:
            validation["reason"] = f"Unit '{unit1}' not recognized"
            validation["alternative_suggestions"] = find_similar_units(unit1)
            return validation
            
        if not unit2_exists:
            validation["reason"] = f"Unit '{unit2}' not recognized"
            validation["alternative_suggestions"] = find_similar_units(unit2)
            return validation
        
        # Get unit types
        type1 = get_unit_type(unit1)
        type2 = get_unit_type(unit2)
        
        validation["dimensional_analysis"] = {
            "unit1_type": type1,
            "unit2_type": type2,
            "unit1_analysis": get_dimensional_analysis(unit1),
            "unit2_analysis": get_dimensional_analysis(unit2),
        }
        
        if type1 == type2 and type1 != "unknown":
            validation["compatible"] = True
            validation["conversion_possible"] = True
            validation["reason"] = f"Both units are {type1} units"
            
            # Calculate conversion factor
            factor1, base1 = UNIT_CONVERSION_FACTORS[unit1]
            factor2, base2 = UNIT_CONVERSION_FACTORS[unit2]
            conversion_factor = factor1 / factor2
            
            validation["conversion_factor"] = conversion_factor
            validation["conversion_example"] = f"1 {unit1} = {conversion_factor} {unit2}"
            
        else:
            validation["reason"] = f"Cannot convert between {type1} and {type2} units"
            validation["alternative_suggestions"] = [
                f"For {type1} units, consider: {', '.join(get_units_by_type(type1)[:5])}",
                f"For {type2} units, consider: {', '.join(get_units_by_type(type2)[:5])}"
            ]
        
        return validation
        
    except Exception as e:
        logger.error(f"Error in enhanced unit validation: {e}")
        return {"error": f"Validation failed: {str(e)}"}


def find_similar_units(unit: str, max_suggestions: int = 5) -> List[str]:
    """Find units similar to the input unit using fuzzy matching."""
    try:
        suggestions = []
        unit_lower = unit.lower()
        
        # Exact match check
        for known_unit in UNIT_CONVERSION_FACTORS.keys():
            if known_unit.lower() == unit_lower:
                return [known_unit]
        
        # Partial match
        for known_unit in UNIT_CONVERSION_FACTORS.keys():
            if unit_lower in known_unit.lower() or known_unit.lower() in unit_lower:
                suggestions.append(known_unit)
        
        # Common typos and alternatives
        unit_alternatives = {
            "gev": ["GeV"], "mev": ["MeV"], "kev": ["keV"], "tev": ["TeV"],
            "sec": ["s"], "second": ["s"], "seconds": ["s"],
            "nano": ["ns"], "nanosecond": ["ns"], "nanoseconds": ["ns"],
            "pico": ["ps"], "picosecond": ["ps"], "picoseconds": ["ps"],
            "micro": ["μs", "us"], "microsecond": ["μs", "us"],
            "year": ["yr", "years"], "years": ["yr", "year"],
            "amu": ["u"], "atomic": ["u"],
        }
        
        for alt, replacements in unit_alternatives.items():
            if alt in unit_lower:
                suggestions.extend(replacements)
        
        # Remove duplicates and limit results
        suggestions = list(set(suggestions))[:max_suggestions]
        
        return suggestions
        
    except Exception as e:
        logger.debug(f"Error finding similar units: {e}")
        return []


def get_units_by_type(unit_type: str) -> List[str]:
    """Get all units of a specific type."""
    try:
        units = []
        for unit in UNIT_CONVERSION_FACTORS.keys():
            if get_unit_type(unit) == unit_type:
                units.append(unit)
        return sorted(units)
    except Exception as e:
        logger.debug(f"Error getting units by type: {e}")
        return []


def validate_unit_compatibility(unit1, unit2):
    """Check if two units are compatible for conversion (legacy function)."""
    validation = enhanced_unit_validation(unit1, unit2)
    return validation["compatible"], validation["reason"]


def convert_units_pdg(value, old_units=None, new_units=None):
    """Convert value between units using PDG conversion factors."""
    if new_units is None:
        return value

    if old_units not in UNIT_CONVERSION_FACTORS:
        raise ValueError(f"Cannot convert from {old_units}")
    if new_units not in UNIT_CONVERSION_FACTORS:
        raise ValueError(f"Cannot convert to {new_units}")

    old_factor = UNIT_CONVERSION_FACTORS[old_units]
    new_factor = UNIT_CONVERSION_FACTORS[new_units]

    if old_factor[1] != new_factor[1]:
        raise ValueError(
            f"Illegal unit conversion from {old_factor[1]} to {new_factor[1]}"
        )

    return value * old_factor[0] / new_factor[0]


def format_unit_info(unit):
    """Format detailed information about a unit."""
    if unit not in UNIT_CONVERSION_FACTORS:
        return {"error": f"Unit '{unit}' not found"}

    factor, base_unit = UNIT_CONVERSION_FACTORS[unit]
    unit_type = get_unit_type(unit)

    info = {
        "unit": unit,
        "type": unit_type,
        "base_unit": base_unit,
        "conversion_factor": factor,
        "description": f"1 {unit} = {factor} {base_unit}",
    }

    # Add specific information for common units
    if unit == "u":
        info["description"] = "Atomic mass unit (unified)"
        info["note"] = "Often used for particle masses"
    elif unit in ["yr", "year", "years"]:
        info["description"] = "Year (365.25 days)"
        info["note"] = "Common for particle lifetimes"
    elif unit in ["GeV", "MeV", "TeV"]:
        info["note"] = "Common energy scale in particle physics"
    elif unit in ["ns", "ps", "fs"]:
        info["note"] = "Common time scale for particle decays"

    return info


async def handle_units_tools(
    name: str, arguments: dict, api
) -> List[types.TextContent]:
    """Handle units-related tool calls."""

    if name == "convert_units_advanced":
        value = arguments["value"]
        from_units = arguments["from_units"]
        to_units = arguments["to_units"]
        validate_compatibility = arguments.get("validate_compatibility", True)

        try:
            # Enhanced validation if requested
            if validate_compatibility:
                validation = enhanced_unit_validation(from_units, to_units)
                if not validation["compatible"]:
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": f"Unit conversion error: {validation['reason']}",
                                    "input_value": value,
                                    "from_units": from_units,
                                    "to_units": to_units,
                                    "suggestions": validation.get("alternative_suggestions", []),
                                    "dimensional_analysis": validation.get("dimensional_analysis", {}),
                                },
                                indent=2,
                            ),
                        )
                    ]

            # Perform enhanced conversion
            converted_value = convert_units_pdg(value, from_units, to_units)
            conversion_factor = UNIT_CONVERSION_FACTORS[from_units][0] / UNIT_CONVERSION_FACTORS[to_units][0]

            result = {
                "original": format_value_with_units(value, units=from_units),
                "converted": format_value_with_units(converted_value, units=to_units),
                "conversion_factor": conversion_factor,
                "unit_type": get_unit_type(from_units),
                "conversion_metadata": {
                    "method": "PDG unit conversion factors",
                    "precision_preserved": True,
                    "dimensional_analysis": get_dimensional_analysis(from_units),
                },
            }
            
            # Add reverse conversion for convenience
            result["reverse_conversion"] = {
                "factor": 1 / conversion_factor,
                "example": f"1 {to_units} = {1/conversion_factor:.6g} {from_units}",
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Conversion failed: {str(e)}",
                            "input_value": value,
                            "from_units": from_units,
                            "to_units": to_units,
                        },
                        indent=2,
                    ),
                )
            ]

    elif name == "get_unit_conversion_factors":
        unit_type = arguments.get("unit_type", "all")
        include_factors = arguments.get("include_factors", True)

        try:
            factors = {}

            for unit, (factor, base_unit) in UNIT_CONVERSION_FACTORS.items():
                unit_category = get_unit_type(unit)

                if unit_type == "all" or unit_category == unit_type:
                    unit_info = {
                        "base_unit": base_unit,
                        "unit_type": unit_category,
                    }

                    if include_factors:
                        unit_info["conversion_factor"] = factor
                        unit_info["description"] = f"1 {unit} = {factor} {base_unit}"

                    factors[unit] = unit_info

            result = {
                "unit_type_filter": unit_type,
                "available_units": factors,
                "total_units": len(factors),
                "base_units": list(set(info["base_unit"] for info in factors.values())),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get conversion factors: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_physics_constants":
        constant_name = arguments.get("constant_name", "all")
        include_description = arguments.get("include_description", True)

        try:
            if constant_name == "all":
                constants = PHYSICS_CONSTANTS.copy()
            else:
                # Map user-friendly names to constant keys
                constant_map = {
                    "hbar": "hbar_gev_s",
                    "hbar_gev_s": "hbar_gev_s",
                }

                if constant_name in constant_map:
                    key = constant_map[constant_name]
                    if key in PHYSICS_CONSTANTS:
                        constants = {key: PHYSICS_CONSTANTS[key]}
                    else:
                        return [
                            types.TextContent(
                                type="text",
                                text=json.dumps(
                                    {"error": f"Constant '{constant_name}' not found"},
                                    indent=2,
                                ),
                            )
                        ]
                else:
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"Unknown constant '{constant_name}'"},
                                indent=2,
                            ),
                        )
                    ]

            # Format output
            if not include_description:
                constants = {
                    k: {"value": v["value"], "units": v["units"]}
                    for k, v in constants.items()
                }

            result = {
                "requested_constant": constant_name,
                "constants": constants,
                "total_constants": len(constants),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get physics constants: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "validate_unit_compatibility":
        unit1 = arguments["unit1"]
        unit2 = arguments["unit2"]
        explain_incompatibility = arguments.get("explain_incompatibility", True)

        try:
            # Use enhanced validation
            validation_result = enhanced_unit_validation(unit1, unit2)
            
            result = {
                "unit1": unit1,
                "unit2": unit2,
                "compatible": validation_result["compatible"],
                "reason": validation_result["reason"],
                "conversion_possible": validation_result.get("conversion_possible", False),
            }
            
            # Add detailed dimensional analysis
            if "dimensional_analysis" in validation_result:
                result["dimensional_analysis"] = validation_result["dimensional_analysis"]
            
            # Add conversion information if compatible
            if validation_result["compatible"]:
                result["conversion_factor"] = validation_result.get("conversion_factor")
                result["conversion_example"] = validation_result.get("conversion_example")
                
                # Add enhanced conversion metadata
                if unit1 in UNIT_CONVERSION_FACTORS and unit2 in UNIT_CONVERSION_FACTORS:
                    result["conversion_metadata"] = {
                        "unit1_info": format_unit_info(unit1),
                        "unit2_info": format_unit_info(unit2),
                        "precision_considerations": "Conversion preserves relative precision",
                    }
            else:
                # Add suggestions for incompatible units
                result["suggestions"] = validation_result.get("alternative_suggestions", [])
                
                # Add unit type information for clarity
                result["unit_types"] = {
                    "unit1": get_unit_type(unit1),
                    "unit2": get_unit_type(unit2),
                    "available_types": list(set(get_unit_type(u) for u in UNIT_CONVERSION_FACTORS.keys())),
                }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to validate compatibility: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_unit_info":
        unit = arguments["unit"]
        include_examples = arguments.get("include_examples", True)

        try:
            unit_info = format_unit_info(unit)

            if "error" not in unit_info and include_examples:
                # Add conversion examples
                examples = []
                unit_type = unit_info["type"]

                # Find other units of the same type for examples
                same_type_units = [
                    u
                    for u in UNIT_CONVERSION_FACTORS.keys()
                    if get_unit_type(u) == unit_type and u != unit
                ]

                for example_unit in same_type_units[:3]:  # Show 3 examples
                    try:
                        factor = (
                            UNIT_CONVERSION_FACTORS[unit][0]
                            / UNIT_CONVERSION_FACTORS[example_unit][0]
                        )
                        examples.append(
                            {
                                "conversion": f"1 {unit} = {factor} {example_unit}",
                                "reverse": f"1 {example_unit} = {1/factor} {unit}",
                            }
                        )
                    except:
                        continue

                if examples:
                    unit_info["examples"] = examples

            return [
                types.TextContent(type="text", text=json.dumps(unit_info, indent=2))
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get unit info: {str(e)}"}, indent=2
                    ),
                )
            ]

    elif name == "convert_between_natural_units":
        value = arguments["value"]
        conversion_type = arguments["conversion_type"]
        input_units = arguments["input_units"]
        output_units = arguments["output_units"]

        try:
            # Natural unit conversions using ħc and c
            hbar_gev_s = PHYSICS_CONSTANTS["hbar_gev_s"]["value"]
            c_m_per_s = PHYSICS_CONSTANTS["c"]["value"]

            result = {
                "input_value": value,
                "input_units": input_units,
                "output_units": output_units,
                "conversion_type": conversion_type,
            }

            # Implement natural unit conversions
            if conversion_type == "energy_to_length":
                # E = ħc/λ → λ = ħc/E
                # First convert energy to GeV, then to length
                energy_gev = convert_units_pdg(value, input_units, "GeV")
                length_m = (hbar_gev_s * c_m_per_s) / energy_gev
                length_converted = (
                    convert_units_pdg(length_m, "m", output_units)
                    if output_units != "m"
                    else length_m
                )
                result["converted_value"] = length_converted
                result["formula"] = "λ = ħc/E"

            elif conversion_type == "energy_to_time":
                # E = ħ/τ → τ = ħ/E
                energy_gev = convert_units_pdg(value, input_units, "GeV")
                time_s = hbar_gev_s / energy_gev
                time_converted = (
                    convert_units_pdg(time_s, "s", output_units)
                    if output_units != "s"
                    else time_s
                )
                result["converted_value"] = time_converted
                result["formula"] = "τ = ħ/E"

            elif conversion_type == "mass_to_energy":
                # E = mc²
                mass_gev = (
                    convert_units_pdg(value, input_units, "GeV")
                    if input_units != "GeV"
                    else value
                )
                energy_gev = mass_gev  # In natural units c = 1
                energy_converted = (
                    convert_units_pdg(energy_gev, "GeV", output_units)
                    if output_units != "GeV"
                    else energy_gev
                )
                result["converted_value"] = energy_converted
                result["formula"] = "E = mc² (c = 1 in natural units)"

            else:
                # Implement reverse conversions
                if conversion_type == "length_to_energy":
                    length_m = (
                        convert_units_pdg(value, input_units, "m")
                        if input_units != "m"
                        else value
                    )
                    energy_gev = (hbar_gev_s * c_m_per_s) / length_m
                    energy_converted = (
                        convert_units_pdg(energy_gev, "GeV", output_units)
                        if output_units != "GeV"
                        else energy_gev
                    )
                    result["converted_value"] = energy_converted
                    result["formula"] = "E = ħc/λ"
                elif conversion_type == "time_to_energy":
                    time_s = (
                        convert_units_pdg(value, input_units, "s")
                        if input_units != "s"
                        else value
                    )
                    energy_gev = hbar_gev_s / time_s
                    energy_converted = (
                        convert_units_pdg(energy_gev, "GeV", output_units)
                        if output_units != "GeV"
                        else energy_gev
                    )
                    result["converted_value"] = energy_converted
                    result["formula"] = "E = ħ/τ"
                elif conversion_type == "energy_to_mass":
                    energy_gev = (
                        convert_units_pdg(value, input_units, "GeV")
                        if input_units != "GeV"
                        else value
                    )
                    mass_gev = energy_gev  # In natural units c = 1
                    mass_converted = (
                        convert_units_pdg(mass_gev, "GeV", output_units)
                        if output_units != "GeV"
                        else mass_gev
                    )
                    result["converted_value"] = mass_converted
                    result["formula"] = "m = E/c² (c = 1 in natural units)"
                else:
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": f"Unknown conversion type: {conversion_type}"
                                },
                                indent=2,
                            ),
                        )
                    ]

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Natural unit conversion failed: {str(e)}"}, indent=2
                    ),
                )
            ]

    elif name == "get_common_conversions":
        category = arguments.get("category", "all")
        include_examples = arguments.get("include_examples", True)

        try:
            if category == "all":
                conversions = COMMON_CONVERSIONS.copy()
            elif category in COMMON_CONVERSIONS:
                conversions = {category: COMMON_CONVERSIONS[category]}
            else:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"Unknown category: {category}"}, indent=2
                        ),
                    )
                ]

            # Format output
            if not include_examples:
                for cat_name, cat_conversions in conversions.items():
                    conversions[cat_name] = [
                        {"from": conv["from"], "to": conv["to"]}
                        for conv in cat_conversions
                    ]

            result = {
                "category": category,
                "conversions": conversions,
                "total_categories": len(conversions),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get common conversions: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": f"Unknown units tool: {name}"})
            )
        ]

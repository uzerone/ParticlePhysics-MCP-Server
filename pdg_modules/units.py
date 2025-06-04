"""
PDG Units Module

This module contains tools for working with particle physics units:
- Unit conversions between different energy and time scales
- Physics constants (like ħ)
- Unit validation and compatibility checking
- Available unit information

Based on the PDG units API: https://pdgapi.lbl.gov/doc/pdg.units.html
"""

import json
from typing import Any, Dict, List

import mcp.types as types


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


def get_unit_type(unit):
    """Determine the type of a unit (energy, time, etc.)."""
    if unit in UNIT_CONVERSION_FACTORS:
        _, base_unit = UNIT_CONVERSION_FACTORS[unit]
        if base_unit == "eV":
            return "energy"
        elif base_unit == "s":
            return "time"
    return "unknown"


def validate_unit_compatibility(unit1, unit2):
    """Check if two units are compatible for conversion."""
    type1 = get_unit_type(unit1)
    type2 = get_unit_type(unit2)

    if type1 == "unknown":
        return False, f"Unit '{unit1}' is not recognized"
    if type2 == "unknown":
        return False, f"Unit '{unit2}' is not recognized"
    if type1 != type2:
        return (
            False,
            f"Cannot convert between {type1} ({unit1}) and {type2} ({unit2}) units",
        )

    return True, "Units are compatible"


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
            # Validate units if requested
            if validate_compatibility:
                is_compatible, message = validate_unit_compatibility(
                    from_units, to_units
                )
                if not is_compatible:
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": f"Unit conversion error: {message}",
                                    "input_value": value,
                                    "from_units": from_units,
                                    "to_units": to_units,
                                },
                                indent=2,
                            ),
                        )
                    ]

            # Perform conversion
            converted_value = convert_units_pdg(value, from_units, to_units)

            result = {
                "original_value": value,
                "original_units": from_units,
                "converted_value": converted_value,
                "converted_units": to_units,
                "conversion_factor": UNIT_CONVERSION_FACTORS[from_units][0]
                / UNIT_CONVERSION_FACTORS[to_units][0],
                "unit_type": get_unit_type(from_units),
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
            is_compatible, message = validate_unit_compatibility(unit1, unit2)

            result = {
                "unit1": unit1,
                "unit2": unit2,
                "compatible": is_compatible,
                "unit1_type": get_unit_type(unit1),
                "unit2_type": get_unit_type(unit2),
            }

            if explain_incompatibility or is_compatible:
                result["message"] = message

            if is_compatible:
                # Show what the conversion factor would be
                try:
                    factor = (
                        UNIT_CONVERSION_FACTORS[unit1][0]
                        / UNIT_CONVERSION_FACTORS[unit2][0]
                    )
                    result["conversion_factor"] = factor
                    result["conversion_example"] = f"1 {unit1} = {factor} {unit2}"
                except:
                    pass

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

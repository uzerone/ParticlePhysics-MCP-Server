"""
PDG Data Handling Module

This module contains tools for handling PDG data measurements, summary values,
unit conversions, and property details.
"""

import json
from typing import Any, Dict, List

import mcp.types as types


def get_data_tools() -> List[types.Tool]:
    """Return all data-related MCP tools."""
    return [
        types.Tool(
            name="get_mass_measurements",
            description="Get detailed mass measurements and summary values for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle (e.g., 'pi+', 'proton')",
                    },
                    "include_summary_values": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include summary values from PDG tables",
                    },
                    "include_measurements": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include individual measurements",
                    },
                    "units": {
                        "type": "string",
                        "default": "GeV",
                        "description": "Units for mass values (e.g., 'GeV', 'MeV', 'kg')",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_lifetime_measurements",
            description="Get detailed lifetime measurements and summary values for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle (e.g., 'tau-', 'mu-')",
                    },
                    "include_summary_values": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include summary values from PDG tables",
                    },
                    "include_measurements": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include individual measurements",
                    },
                    "units": {
                        "type": "string",
                        "default": "s",
                        "description": "Units for lifetime values (e.g., 's', 'ns', 'ps')",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_width_measurements",
            description="Get detailed width measurements and summary values for a particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle (e.g., 'Z0', 'W+')",
                    },
                    "include_summary_values": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include summary values from PDG tables",
                    },
                    "include_measurements": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include individual measurements",
                    },
                    "units": {
                        "type": "string",
                        "default": "GeV",
                        "description": "Units for width values (e.g., 'GeV', 'MeV')",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_summary_values",
            description="Get summary values for any particle property with detailed information",
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
                        "description": "Type of property to get summary values for",
                    },
                    "summary_table_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "Only include values from Summary Tables (not Particle Listings)",
                    },
                    "units": {
                        "type": "string",
                        "description": "Convert values to specified units",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_measurements_by_property",
            description="Get individual measurements for a specific particle property",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["mass", "lifetime", "width"],
                        "description": "Type of property to get measurements for",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of measurements to return",
                    },
                    "include_references": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include reference information for measurements",
                    },
                },
                "required": ["particle_name", "property_type"],
            },
        ),
        types.Tool(
            name="convert_units",
            description="Convert particle physics values between different units",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Numerical value to convert",
                    },
                    "from_units": {
                        "type": "string",
                        "description": "Source units (e.g., 'GeV', 'MeV', 's', 'ns')",
                    },
                    "to_units": {
                        "type": "string",
                        "description": "Target units (e.g., 'GeV', 'MeV', 's', 'ns')",
                    },
                },
                "required": ["value", "from_units", "to_units"],
            },
        ),
        types.Tool(
            name="get_particle_text",
            description="Get text information and descriptions for particles or physics concepts",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdgid": {
                        "type": "string",
                        "description": "PDG ID (e.g., 'S008' for Standard Model, particle names also accepted)",
                    },
                    "text_type": {
                        "type": "string",
                        "enum": ["description", "review", "note", "all"],
                        "default": "all",
                        "description": "Type of text information to retrieve",
                    },
                },
                "required": ["pdgid"],
            },
        ),
        types.Tool(
            name="get_property_details",
            description="Get detailed information about a specific particle property including data flags and type info",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle",
                    },
                    "property_pdgid": {
                        "type": "string",
                        "description": "PDG ID of the specific property (optional, will search if not provided)",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["mass", "lifetime", "width", "branching_fraction"],
                        "description": "Type of property to get details for",
                    },
                },
                "required": ["particle_name", "property_type"],
            },
        ),
        types.Tool(
            name="get_data_type_keys",
            description="Get list of PDG data type keys with descriptions",
            inputSchema={
                "type": "object",
                "properties": {
                    "as_text": {
                        "type": "boolean",
                        "default": True,
                        "description": "Return as formatted text (true) or structured data (false)",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_value_type_keys",
            description="Get list of PDG summary value type keys with descriptions",
            inputSchema={
                "type": "object",
                "properties": {
                    "as_text": {
                        "type": "boolean",
                        "default": True,
                        "description": "Return as formatted text (true) or structured data (false)",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_key_documentation",
            description="Get documentation for specific key values or flags used in PDG API",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the database table",
                    },
                    "column_name": {
                        "type": "string",
                        "description": "Name of the database column",
                    },
                    "key": {
                        "type": "string",
                        "description": "Key value to get documentation for",
                    },
                },
                "required": ["table_name", "column_name", "key"],
            },
        ),
    ]


def format_summary_value(summary_value):
    """Format a PdgSummaryValue object for JSON output."""
    try:
        result = {}

        # Core value information with safe access
        result["value"] = getattr(summary_value, "value", None)

        # Try different attribute names for value text
        for attr in ["value_text", "display_value_text", "text", "display_text"]:
            if hasattr(summary_value, attr):
                result["value_text"] = getattr(summary_value, attr)
                break
        else:
            result["value_text"] = str(getattr(summary_value, "value", "N/A"))

        # Display value text with fallback
        result["display_value_text"] = getattr(
            summary_value, "display_value_text", result["value_text"]
        )

        # Error information
        result["error_positive"] = getattr(summary_value, "error_positive", None)
        result["error_negative"] = getattr(summary_value, "error_negative", None)
        result["error"] = getattr(summary_value, "error", None)

        # Units and properties
        result["units"] = getattr(summary_value, "units", "N/A")
        result["is_limit"] = getattr(summary_value, "is_limit", False)
        result["is_lower_limit"] = getattr(summary_value, "is_lower_limit", False)
        result["is_upper_limit"] = getattr(summary_value, "is_upper_limit", False)
        result["confidence_level"] = getattr(summary_value, "confidence_level", None)
        result["scale_factor"] = getattr(summary_value, "scale_factor", None)
        result["comment"] = getattr(summary_value, "comment", None)

        # Summary table information
        result["in_summary_table"] = getattr(summary_value, "in_summary_table", False)
        result["value_type"] = getattr(summary_value, "value_type", "N/A")
        result["value_type_key"] = getattr(summary_value, "value_type_key", "N/A")
        result["pdgid"] = getattr(summary_value, "pdgid", "N/A")

        return result
    except Exception as e:
        return {"error": f"Failed to format summary value: {str(e)}"}


def format_measurement(measurement):
    """Format a PdgMeasurement object for JSON output."""
    try:
        formatted = {
            "id": getattr(measurement, "id", "N/A"),
            "pdgid": getattr(measurement, "pdgid", "N/A"),
        }

        # Try to get measurement value
        try:
            if hasattr(measurement, "get_value") and measurement.get_value():
                value = measurement.get_value()
                formatted["value"] = {
                    "value": getattr(value, "value", "N/A"),
                    "value_text": getattr(value, "value_text", "N/A"),
                    "units": getattr(value, "units", "N/A"),
                    "error_positive": getattr(value, "error_positive", "N/A"),
                    "error_negative": getattr(value, "error_negative", "N/A"),
                }
        except:
            formatted["value"] = "N/A"

        # Try to get reference information
        try:
            if hasattr(measurement, "reference") and measurement.reference:
                ref = measurement.reference
                formatted["reference"] = {
                    "title": getattr(ref, "title", "N/A"),
                    "authors": getattr(ref, "authors", "N/A"),
                    "doi": getattr(ref, "doi", "N/A"),
                    "publication_year": getattr(ref, "publication_year", "N/A"),
                    "journal": getattr(ref, "journal", "N/A"),
                }
        except:
            formatted["reference"] = "N/A"

        return formatted
    except Exception as e:
        return {"error": f"Failed to format measurement: {str(e)}"}


def format_property_details(prop):
    """Format detailed property information from PdgProperty objects."""
    try:
        details = {
            "pdgid": prop.pdgid,
            "description": prop.description,
            "data_type": prop.data_type,
            "data_flags": prop.data_flags,
            "edition": prop.edition,
        }

        # Get parent information
        try:
            details["parent_pdgid"] = prop.get_parent_pdgid()
        except:
            details["parent_pdgid"] = "N/A"

        # Get best summary if available
        try:
            if prop.has_best_summary():
                best = prop.best_summary()
                details["best_summary"] = format_summary_value(best)
        except:
            pass

        # Get number of summary values
        try:
            details["n_summary_values"] = prop.n_summary_table_values()
        except:
            details["n_summary_values"] = 0

        return details
    except Exception as e:
        return {"error": f"Failed to format property details: {str(e)}"}


def get_property_by_type(particle, property_type):
    """Get specific property objects from a particle."""
    try:
        if property_type == "mass":
            return list(particle.masses())
        elif property_type == "lifetime":
            return list(particle.lifetimes())
        elif property_type == "width":
            return list(particle.widths())
        else:
            return []
    except Exception as e:
        return []


async def handle_data_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Handle data-related tool calls."""

    if name == "get_mass_measurements":
        particle_name = arguments["particle_name"]
        include_summary_values = arguments.get("include_summary_values", True)
        include_measurements = arguments.get("include_measurements", False)
        units = arguments.get("units", "GeV")

        try:
            particle = api.get_particle_by_name(particle_name)
            mass_data = {"particle": particle_name, "property": "mass", "units": units}

            if include_summary_values:
                summary_values = []
                for mass_prop in particle.masses():
                    for summary in mass_prop.summary_values():
                        sv_data = format_summary_value(summary)
                        # Convert units if requested
                        if (
                            units != "GeV"
                            and "value" in sv_data
                            and sv_data["value"] is not None
                        ):
                            try:
                                converted_value = summary.get_value(units)
                                sv_data["converted_value"] = converted_value
                                sv_data["converted_units"] = units
                            except:
                                pass
                        summary_values.append(sv_data)
                mass_data["summary_values"] = summary_values

            if include_measurements:
                measurements = []
                for mass_prop in particle.masses():
                    for measurement in mass_prop.get_measurements():
                        measurements.append(format_measurement(measurement))
                mass_data["measurements"] = measurements

            return [
                types.TextContent(type="text", text=json.dumps(mass_data, indent=2))
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get mass measurements: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_lifetime_measurements":
        particle_name = arguments["particle_name"]
        include_summary_values = arguments.get("include_summary_values", True)
        include_measurements = arguments.get("include_measurements", False)
        units = arguments.get("units", "s")

        try:
            particle = api.get_particle_by_name(particle_name)
            lifetime_data = {
                "particle": particle_name,
                "property": "lifetime",
                "units": units,
            }

            if include_summary_values:
                summary_values = []
                for lifetime_prop in particle.lifetimes():
                    for summary in lifetime_prop.summary_values():
                        sv_data = format_summary_value(summary)
                        # Convert units if requested
                        if (
                            units != "s"
                            and "value" in sv_data
                            and sv_data["value"] is not None
                        ):
                            try:
                                converted_value = summary.get_value(units)
                                sv_data["converted_value"] = converted_value
                                sv_data["converted_units"] = units
                            except:
                                pass
                        summary_values.append(sv_data)
                lifetime_data["summary_values"] = summary_values

            if include_measurements:
                measurements = []
                for lifetime_prop in particle.lifetimes():
                    for measurement in lifetime_prop.get_measurements():
                        measurements.append(format_measurement(measurement))
                lifetime_data["measurements"] = measurements

            return [
                types.TextContent(type="text", text=json.dumps(lifetime_data, indent=2))
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get lifetime measurements: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_width_measurements":
        particle_name = arguments["particle_name"]
        include_summary_values = arguments.get("include_summary_values", True)
        include_measurements = arguments.get("include_measurements", False)
        units = arguments.get("units", "GeV")

        try:
            particle = api.get_particle_by_name(particle_name)
            width_data = {
                "particle": particle_name,
                "property": "width",
                "units": units,
            }

            if include_summary_values:
                summary_values = []
                for width_prop in particle.widths():
                    for summary in width_prop.summary_values():
                        sv_data = format_summary_value(summary)
                        # Convert units if requested
                        if (
                            units != "GeV"
                            and "value" in sv_data
                            and sv_data["value"] is not None
                        ):
                            try:
                                converted_value = summary.get_value(units)
                                sv_data["converted_value"] = converted_value
                                sv_data["converted_units"] = units
                            except:
                                pass
                        summary_values.append(sv_data)
                width_data["summary_values"] = summary_values

            if include_measurements:
                measurements = []
                for width_prop in particle.widths():
                    for measurement in width_prop.get_measurements():
                        measurements.append(format_measurement(measurement))
                width_data["measurements"] = measurements

            return [
                types.TextContent(type="text", text=json.dumps(width_data, indent=2))
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get width measurements: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_summary_values":
        particle_name = arguments["particle_name"]
        property_type = arguments.get("property_type", "all")
        summary_table_only = arguments.get("summary_table_only", False)
        units = arguments.get("units")

        try:
            particle = api.get_particle_by_name(particle_name)
            result = {"particle": particle_name, "summary_values": {}}

            property_types = (
                ["mass", "lifetime", "width"]
                if property_type == "all"
                else [property_type]
            )

            for prop_type in property_types:
                result["summary_values"][prop_type] = []
                for prop in get_property_by_type(particle, prop_type):
                    for summary in prop.summary_values(
                        summary_table_only=summary_table_only
                    ):
                        sv_data = format_summary_value(summary)
                        # Convert units if requested
                        if (
                            units
                            and "value" in sv_data
                            and sv_data["value"] is not None
                        ):
                            try:
                                converted_value = summary.get_value(units)
                                sv_data["converted_value"] = converted_value
                                sv_data["converted_units"] = units
                            except:
                                pass
                        result["summary_values"][prop_type].append(sv_data)

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get summary values: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_measurements_by_property":
        particle_name = arguments["particle_name"]
        property_type = arguments["property_type"]
        limit = arguments.get("limit", 10)
        include_references = arguments.get("include_references", True)

        try:
            particle = api.get_particle_by_name(particle_name)
            measurements = []

            for prop in get_property_by_type(particle, property_type):
                for measurement in prop.get_measurements():
                    if len(measurements) >= limit:
                        break
                    meas_data = format_measurement(measurement)
                    if not include_references and "reference" in meas_data:
                        del meas_data["reference"]
                    measurements.append(meas_data)

            result = {
                "particle": particle_name,
                "property": property_type,
                "measurements": measurements,
                "total_found": len(measurements),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get measurements: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "convert_units":
        value = arguments["value"]
        from_units = arguments["from_units"]
        to_units = arguments["to_units"]

        try:
            # Use PDG units module for conversion
            from pdg.units import convert

            converted_value = convert(value, from_units, to_units)

            result = {
                "original_value": value,
                "original_units": from_units,
                "converted_value": converted_value,
                "converted_units": to_units,
                "conversion_factor": converted_value / value if value != 0 else None,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to convert units: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_particle_text":
        pdgid = arguments["pdgid"]
        text_type = arguments.get("text_type", "all")

        try:
            # Try to get as text object first
            try:
                text_obj = api.get(pdgid)
                if hasattr(text_obj, "__iter__"):
                    text_obj = list(text_obj)[0]
            except:
                # Try as particle name
                try:
                    particle = api.get_particle_by_name(pdgid)
                    pdgid = particle.pdgid
                    text_obj = api.get(pdgid)
                except:
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"Could not find text for: {pdgid}"}, indent=2
                            ),
                        )
                    ]

            result = {
                "pdgid": pdgid,
                "description": getattr(text_obj, "description", "N/A"),
                "data_type": getattr(text_obj, "data_type", "N/A"),
            }

            # Try to get text content
            try:
                if hasattr(text_obj, "text"):
                    result["text"] = str(text_obj.text)
                elif hasattr(text_obj, "__str__"):
                    result["text"] = str(text_obj)
            except:
                result["text"] = "N/A"

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get particle text: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_property_details":
        particle_name = arguments["particle_name"]
        property_type = arguments["property_type"]
        property_pdgid = arguments.get("property_pdgid")

        try:
            particle = api.get_particle_by_name(particle_name)
            properties = []

            for prop in get_property_by_type(particle, property_type):
                if property_pdgid and prop.pdgid != property_pdgid:
                    continue
                properties.append(format_property_details(prop))

            result = {
                "particle": particle_name,
                "property_type": property_type,
                "properties": properties,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get property details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_data_type_keys":
        as_text = arguments.get("as_text", True)

        try:
            data_types = api.doc_data_type_keys(as_text=as_text)

            if as_text:
                result = {"data_type_keys": data_types}
            else:
                # Convert RowMapping objects to dict for JSON serialization
                serializable_types = []
                for dt in data_types:
                    if hasattr(dt, "_mapping"):
                        serializable_types.append(dict(dt._mapping))
                    elif hasattr(dt, "keys"):
                        serializable_types.append(dict(dt))
                    else:
                        serializable_types.append(str(dt))
                result = {
                    "data_type_keys": serializable_types,
                    "count": len(serializable_types),
                }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get data type keys: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_value_type_keys":
        as_text = arguments.get("as_text", True)

        try:
            value_types = api.doc_value_type_keys(as_text=as_text)

            if as_text:
                result = {"value_type_keys": value_types}
            else:
                # Convert RowMapping objects to dict for JSON serialization
                serializable_types = []
                for vt in value_types:
                    if hasattr(vt, "_mapping"):
                        serializable_types.append(dict(vt._mapping))
                    elif hasattr(vt, "keys"):
                        serializable_types.append(dict(vt))
                    else:
                        serializable_types.append(str(vt))
                result = {
                    "value_type_keys": serializable_types,
                    "count": len(serializable_types),
                }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get value type keys: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_key_documentation":
        table_name = arguments["table_name"]
        column_name = arguments["column_name"]
        key = arguments["key"]

        try:
            doc = api.doc_key_value(table_name, column_name, key)

            result = {
                "table_name": table_name,
                "column_name": column_name,
                "key": key,
                "documentation": dict(doc) if doc else None,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get key documentation: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": f"Unknown data tool: {name}"})
            )
        ]

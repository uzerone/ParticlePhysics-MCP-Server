"""
PDG Measurement Module

This module contains tools for working with PDG measurement objects:
- PdgMeasurement: Individual measurements from PDG Listings
- PdgValue: Numerical values associated with measurements
- PdgReference: Literature references for measurements
- PdgFootnote: Footnotes associated with measurements

Based on the PDG measurement API: https://pdgapi.lbl.gov/doc/pdg.measurement.html
"""

import json
import mcp.types as types
from typing import Any, Dict, List


def get_measurement_tools() -> List[types.Tool]:
    """Return all measurement-related MCP tools."""
    return [
        types.Tool(
            name="get_measurement_details",
            description="Get detailed information about a specific PDG measurement including values and references",
            inputSchema={
                "type": "object",
                "properties": {
                    "measurement_id": {
                        "type": "integer",
                        "description": "PDG measurement ID to retrieve details for",
                    },
                    "include_values": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include all values associated with this measurement",
                    },
                    "include_reference": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include reference/publication information",
                    },
                    "include_footnotes": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include footnotes associated with the measurement",
                    },
                },
                "required": ["measurement_id"],
            },
        ),
        types.Tool(
            name="get_measurement_value_details",
            description="Get detailed information about a specific measurement value including errors and units",
            inputSchema={
                "type": "object",
                "properties": {
                    "value_id": {
                        "type": "integer",
                        "description": "PDG value ID to retrieve details for",
                    },
                    "include_error_breakdown": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include statistical and systematic error breakdown",
                    },
                },
                "required": ["value_id"],
            },
        ),
        types.Tool(
            name="get_reference_details",
            description="Get detailed publication information for a PDG reference",
            inputSchema={
                "type": "object",
                "properties": {
                    "reference_id": {
                        "type": "integer",
                        "description": "PDG reference ID to retrieve details for",
                    },
                    "include_doi": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include DOI and external identifiers",
                    },
                },
                "required": ["reference_id"],
            },
        ),
        types.Tool(
            name="search_measurements_by_reference",
            description="Search for measurements by reference properties (year, DOI, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to search measurements for",
                    },
                    "publication_year": {
                        "type": "integer",
                        "description": "Publication year to filter by",
                    },
                    "doi": {
                        "type": "string",
                        "description": "DOI to search for",
                    },
                    "author": {
                        "type": "string",
                        "description": "Author name to search for",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of measurements to return",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_footnote_details",
            description="Get footnote text and associated references",
            inputSchema={
                "type": "object",
                "properties": {
                    "footnote_id": {
                        "type": "integer",
                        "description": "PDG footnote ID to retrieve details for",
                    },
                    "include_references": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include references that use this footnote",
                    },
                },
                "required": ["footnote_id"],
            },
        ),
        types.Tool(
            name="analyze_measurement_errors",
            description="Analyze error components (statistical vs systematic) across multiple measurements",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to analyze measurements for",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["mass", "lifetime", "width"],
                        "description": "Type of property to analyze",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of measurements to analyze",
                    },
                },
                "required": ["particle_name", "property_type"],
            },
        ),
        types.Tool(
            name="get_measurements_for_particle",
            description="Get all measurements for a specific particle with detailed breakdown",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to get measurements for",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["mass", "lifetime", "width", "all"],
                        "default": "all",
                        "description": "Type of property to get measurements for",
                    },
                    "include_references": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include reference information for each measurement",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of measurements to return",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="compare_measurement_techniques",
            description="Compare different measurement techniques for a particle property",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to analyze",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["mass", "lifetime", "width"],
                        "description": "Type of property to analyze techniques for",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of measurements to analyze",
                    },
                },
                "required": ["particle_name", "property_type"],
            },
        ),
    ]


def format_pdg_measurement(measurement):
    """Format a PdgMeasurement object for JSON output."""
    try:
        formatted = {
            "id": getattr(measurement, "id", "N/A"),
            "pdgid": getattr(measurement, "pdgid", "N/A"),
            "event_count": getattr(measurement, "event_count", "N/A"),
            "confidence_level": getattr(measurement, "confidence_level", "N/A"),
            "technique": getattr(measurement, "technique", "N/A"),
            "charge": getattr(measurement, "charge", "N/A"),
            "changebar": getattr(measurement, "changebar", False),
            "comment": getattr(measurement, "comment", "N/A"),
        }
        return formatted
    except Exception as e:
        return {"error": f"Failed to format measurement: {str(e)}"}


def format_pdg_value(value):
    """Format a PdgValue object for JSON output."""
    try:
        formatted = {
            "id": getattr(value, "id", "N/A"),
            "column_name": getattr(value, "column_name", "N/A"),
            "column_name_tex": getattr(value, "column_name_tex", "N/A"),
            "unit_text": getattr(value, "unit_text", "N/A"),
            "value": getattr(value, "value", "N/A"),
            "value_text": getattr(value, "value_text", "N/A"),
            "display_value_text": getattr(value, "display_value_text", "N/A"),
            "display_power_of_ten": getattr(value, "display_power_of_ten", "N/A"),
            "display_in_percent": getattr(value, "display_in_percent", False),
            "is_limit": getattr(value, "is_limit", False),
            "is_upper_limit": getattr(value, "is_upper_limit", False),
            "is_lower_limit": getattr(value, "is_lower_limit", False),
            "used_in_average": getattr(value, "used_in_average", False),
            "used_in_fit": getattr(value, "used_in_fit", False),
            "error_positive": getattr(value, "error_positive", "N/A"),
            "error_negative": getattr(value, "error_negative", "N/A"),
            "stat_error_positive": getattr(value, "stat_error_positive", "N/A"),
            "stat_error_negative": getattr(value, "stat_error_negative", "N/A"),
            "syst_error_positive": getattr(value, "syst_error_positive", "N/A"),
            "syst_error_negative": getattr(value, "syst_error_negative", "N/A"),
        }
        
        # Calculate symmetric errors safely
        try:
            formatted["error"] = getattr(value, "error", "N/A")
        except:
            formatted["error"] = "N/A"
            
        try:
            formatted["stat_error"] = getattr(value, "stat_error", "N/A")
        except:
            formatted["stat_error"] = "N/A"
            
        try:
            formatted["syst_error"] = getattr(value, "syst_error", "N/A")
        except:
            formatted["syst_error"] = "N/A"
            
        return formatted
    except Exception as e:
        return {"error": f"Failed to format value: {str(e)}"}


def format_pdg_reference(reference):
    """Format a PdgReference object for JSON output."""
    try:
        formatted = {
            "id": getattr(reference, "id", "N/A"),
            "publication_name": getattr(reference, "publication_name", "N/A"),
            "publication_year": getattr(reference, "publication_year", "N/A"),
            "title": getattr(reference, "title", "N/A"),
            "doi": getattr(reference, "doi", "N/A"),
            "inspire_id": getattr(reference, "inspire_id", "N/A"),
            "document_id": getattr(reference, "document_id", "N/A"),
        }
        return formatted
    except Exception as e:
        return {"error": f"Failed to format reference: {str(e)}"}


def format_pdg_footnote(footnote):
    """Format a PdgFootnote object for JSON output."""
    try:
        formatted = {
            "id": getattr(footnote, "id", "N/A"),
            "text": getattr(footnote, "text", "N/A"),
        }
        return formatted
    except Exception as e:
        return {"error": f"Failed to format footnote: {str(e)}"}


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


async def handle_measurement_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Handle measurement-related tool calls."""
    
    if name == "get_measurement_details":
        measurement_id = arguments["measurement_id"]
        include_values = arguments.get("include_values", True)
        include_reference = arguments.get("include_reference", True)
        include_footnotes = arguments.get("include_footnotes", True)

        try:
            # Import PDG measurement module
            from pdg.measurement import PdgMeasurement
            
            measurement = PdgMeasurement(api, measurement_id)
            result = format_pdg_measurement(measurement)
            
            if include_values:
                values = []
                try:
                    for value in measurement.values():
                        values.append(format_pdg_value(value))
                except:
                    values = []
                result["values"] = values
                
            if include_reference:
                try:
                    reference = measurement.reference
                    result["reference"] = format_pdg_reference(reference)
                except:
                    result["reference"] = "N/A"
                    
            if include_footnotes:
                footnotes = []
                try:
                    for footnote in measurement.footnotes():
                        footnotes.append(format_pdg_footnote(footnote))
                except:
                    footnotes = []
                result["footnotes"] = footnotes

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get measurement details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_measurement_value_details":
        value_id = arguments["value_id"]
        include_error_breakdown = arguments.get("include_error_breakdown", True)

        try:
            from pdg.measurement import PdgValue
            
            value = PdgValue(api, value_id)
            result = format_pdg_value(value)
            
            if include_error_breakdown:
                # Add detailed error analysis
                try:
                    error_analysis = {
                        "total_error_positive": value.error_positive,
                        "total_error_negative": value.error_negative,
                        "statistical_error_positive": value.stat_error_positive,
                        "statistical_error_negative": value.stat_error_negative,
                        "systematic_error_positive": value.syst_error_positive,
                        "systematic_error_negative": value.syst_error_negative,
                        "has_asymmetric_errors": value.error_positive != value.error_negative,
                        "error_dominance": "systematic" if (value.syst_error_positive or 0) > (value.stat_error_positive or 0) else "statistical"
                    }
                    result["error_analysis"] = error_analysis
                except:
                    result["error_analysis"] = "N/A"

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get value details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_reference_details":
        reference_id = arguments["reference_id"]
        include_doi = arguments.get("include_doi", True)

        try:
            from pdg.measurement import PdgReference
            
            reference = PdgReference(api, reference_id)
            result = format_pdg_reference(reference)
            
            if include_doi:
                # Add additional identifiers and links
                result["external_links"] = {
                    "doi_url": f"https://doi.org/{reference.doi}" if reference.doi else None,
                    "inspire_url": f"https://inspirehep.net/literature/{reference.inspire_id}" if reference.inspire_id else None,
                }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get reference details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "search_measurements_by_reference":
        particle_name = arguments["particle_name"]
        publication_year = arguments.get("publication_year")
        doi = arguments.get("doi")
        author = arguments.get("author")
        limit = arguments.get("limit", 10)

        try:
            particle = api.get_particle_by_name(particle_name)
            measurements = []
            count = 0

            # Get all measurements for the particle and filter by reference criteria
            for prop_type in ["mass", "lifetime", "width"]:
                if count >= limit:
                    break
                    
                for prop in get_property_by_type(particle, prop_type):
                    if count >= limit:
                        break
                        
                    for measurement in prop.get_measurements():
                        if count >= limit:
                            break
                            
                        try:
                            reference = measurement.reference
                            
                            # Apply filters
                            if publication_year and reference.publication_year != publication_year:
                                continue
                            if doi and reference.doi != doi:
                                continue
                            if author and author.lower() not in (reference.document_id or "").lower():
                                continue
                                
                            # Format measurement with reference
                            meas_data = format_pdg_measurement(measurement)
                            meas_data["reference"] = format_pdg_reference(reference)
                            meas_data["property_type"] = prop_type
                            
                            measurements.append(meas_data)
                            count += 1
                            
                        except:
                            continue

            result = {
                "particle": particle_name,
                "search_criteria": {
                    "publication_year": publication_year,
                    "doi": doi,
                    "author": author,
                },
                "measurements": measurements,
                "total_found": len(measurements),
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to search measurements by reference: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_footnote_details":
        footnote_id = arguments["footnote_id"]
        include_references = arguments.get("include_references", True)

        try:
            from pdg.measurement import PdgFootnote
            
            footnote = PdgFootnote(api, footnote_id)
            result = format_pdg_footnote(footnote)
            
            if include_references:
                references = []
                try:
                    for ref_measurement in footnote.references():
                        references.append(format_pdg_measurement(ref_measurement))
                except:
                    references = []
                result["references"] = references

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get footnote details: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "analyze_measurement_errors":
        particle_name = arguments["particle_name"]
        property_type = arguments["property_type"]
        limit = arguments.get("limit", 20)

        try:
            particle = api.get_particle_by_name(particle_name)
            analysis = {
                "particle": particle_name,
                "property_type": property_type,
                "measurements_analyzed": [],
                "error_statistics": {
                    "total_measurements": 0,
                    "with_statistical_errors": 0,
                    "with_systematic_errors": 0,
                    "with_asymmetric_errors": 0,
                    "average_relative_stat_error": 0,
                    "average_relative_syst_error": 0,
                },
            }

            stat_errors = []
            syst_errors = []
            count = 0

            for prop in get_property_by_type(particle, property_type):
                if count >= limit:
                    break
                    
                for measurement in prop.get_measurements():
                    if count >= limit:
                        break
                        
                    try:
                        # Get the primary value for this measurement
                        value = measurement.get_value()
                        
                        meas_analysis = {
                            "measurement_id": measurement.id,
                            "value": value.value,
                            "has_stat_error": value.stat_error_positive is not None,
                            "has_syst_error": value.syst_error_positive is not None,
                            "is_asymmetric": value.error_positive != value.error_negative,
                            "relative_stat_error": None,
                            "relative_syst_error": None,
                        }
                        
                        # Calculate relative errors
                        if value.value and value.stat_error_positive:
                            rel_stat = abs(value.stat_error_positive / value.value)
                            meas_analysis["relative_stat_error"] = rel_stat
                            stat_errors.append(rel_stat)
                            
                        if value.value and value.syst_error_positive:
                            rel_syst = abs(value.syst_error_positive / value.value)
                            meas_analysis["relative_syst_error"] = rel_syst
                            syst_errors.append(rel_syst)
                        
                        analysis["measurements_analyzed"].append(meas_analysis)
                        count += 1
                        
                        # Update statistics
                        analysis["error_statistics"]["total_measurements"] += 1
                        if value.stat_error_positive is not None:
                            analysis["error_statistics"]["with_statistical_errors"] += 1
                        if value.syst_error_positive is not None:
                            analysis["error_statistics"]["with_systematic_errors"] += 1
                        if value.error_positive != value.error_negative:
                            analysis["error_statistics"]["with_asymmetric_errors"] += 1
                            
                    except:
                        continue

            # Calculate averages
            if stat_errors:
                analysis["error_statistics"]["average_relative_stat_error"] = sum(stat_errors) / len(stat_errors)
            if syst_errors:
                analysis["error_statistics"]["average_relative_syst_error"] = sum(syst_errors) / len(syst_errors)

            return [types.TextContent(type="text", text=json.dumps(analysis, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to analyze measurement errors: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_measurements_for_particle":
        particle_name = arguments["particle_name"]
        property_type = arguments.get("property_type", "all")
        include_references = arguments.get("include_references", True)
        limit = arguments.get("limit", 50)

        try:
            particle = api.get_particle_by_name(particle_name)
            all_measurements = []
            count = 0

            # Determine which property types to analyze
            prop_types = ["mass", "lifetime", "width"] if property_type == "all" else [property_type]

            for prop_type in prop_types:
                if count >= limit:
                    break
                    
                for prop in get_property_by_type(particle, prop_type):
                    if count >= limit:
                        break
                        
                    for measurement in prop.get_measurements():
                        if count >= limit:
                            break
                            
                        try:
                            meas_data = format_pdg_measurement(measurement)
                            meas_data["property_type"] = prop_type
                            
                            # Add values
                            values = []
                            for value in measurement.values():
                                values.append(format_pdg_value(value))
                            meas_data["values"] = values
                            
                            # Add reference if requested
                            if include_references:
                                try:
                                    reference = measurement.reference
                                    meas_data["reference"] = format_pdg_reference(reference)
                                except:
                                    meas_data["reference"] = "N/A"
                            
                            all_measurements.append(meas_data)
                            count += 1
                            
                        except:
                            continue

            result = {
                "particle": particle_name,
                "property_type": property_type,
                "measurements": all_measurements,
                "total_found": len(all_measurements),
                "limited_to": limit,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get measurements for particle: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "compare_measurement_techniques":
        particle_name = arguments["particle_name"]
        property_type = arguments["property_type"]
        limit = arguments.get("limit", 20)

        try:
            particle = api.get_particle_by_name(particle_name)
            techniques = {}
            count = 0

            for prop in get_property_by_type(particle, property_type):
                if count >= limit:
                    break
                    
                for measurement in prop.get_measurements():
                    if count >= limit:
                        break
                        
                    try:
                        technique = measurement.technique or "Unknown"
                        
                        if technique not in techniques:
                            techniques[technique] = {
                                "technique": technique,
                                "measurement_count": 0,
                                "measurements": [],
                                "average_precision": None,
                                "year_range": {"earliest": None, "latest": None},
                            }
                        
                        # Add measurement info
                        meas_info = {
                            "measurement_id": measurement.id,
                            "year": getattr(measurement.reference, "publication_year", "N/A"),
                            "value": None,
                            "relative_error": None,
                        }
                        
                        # Get primary value and calculate precision
                        try:
                            value = measurement.get_value()
                            meas_info["value"] = value.value
                            if value.value and value.error_positive:
                                meas_info["relative_error"] = abs(value.error_positive / value.value)
                        except:
                            pass
                        
                        techniques[technique]["measurements"].append(meas_info)
                        techniques[technique]["measurement_count"] += 1
                        
                        # Update year range
                        year = meas_info["year"]
                        if year != "N/A":
                            if techniques[technique]["year_range"]["earliest"] is None or year < techniques[technique]["year_range"]["earliest"]:
                                techniques[technique]["year_range"]["earliest"] = year
                            if techniques[technique]["year_range"]["latest"] is None or year > techniques[technique]["year_range"]["latest"]:
                                techniques[technique]["year_range"]["latest"] = year
                        
                        count += 1
                        
                    except:
                        continue

            # Calculate average precision for each technique
            for technique_data in techniques.values():
                relative_errors = [m["relative_error"] for m in technique_data["measurements"] if m["relative_error"] is not None]
                if relative_errors:
                    technique_data["average_precision"] = sum(relative_errors) / len(relative_errors)

            result = {
                "particle": particle_name,
                "property_type": property_type,
                "techniques": list(techniques.values()),
                "total_techniques": len(techniques),
                "total_measurements_analyzed": count,
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to compare measurement techniques: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": f"Unknown measurement tool: {name}"})
            )
        ] 
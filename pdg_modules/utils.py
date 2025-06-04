"""
PDG Utils Module

This module contains utility tools for PDG data manipulation:
- PDG identifier parsing and manipulation
- Property selection and ranking
- Data processing utilities
- PDG rounding rules

Based on the PDG utils API: https://pdgapi.lbl.gov/doc/pdg.utils.html
"""

import json
import math
import mcp.types as types
from typing import Any, Dict, List


def get_utils_tools() -> List[types.Tool]:
    """Return all utils-related MCP tools."""
    return [
        types.Tool(
            name="parse_pdg_identifier",
            description="Parse PDG Identifier and return base identifier and edition",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdgid": {
                        "type": "string",
                        "description": "PDG identifier to parse (e.g., 'S008', 'M100/2024')",
                    },
                },
                "required": ["pdgid"],
            },
        ),
        types.Tool(
            name="get_base_pdg_id",
            description="Get the normalized base part of a PDG identifier",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdgid": {
                        "type": "string",
                        "description": "PDG identifier (e.g., 'S008/2024', 'M100')",
                    },
                },
                "required": ["pdgid"],
            },
        ),
        types.Tool(
            name="make_pdg_identifier",
            description="Create a normalized full PDG identifier with optional edition",
            inputSchema={
                "type": "object",
                "properties": {
                    "baseid": {
                        "type": "string",
                        "description": "Base PDG identifier (e.g., 'S008', 'M100')",
                    },
                    "edition": {
                        "type": "string",
                        "description": "PDG edition (optional, e.g., '2024')",
                    },
                },
                "required": ["baseid"],
            },
        ),
        types.Tool(
            name="find_best_property",
            description="Find the 'best' property from a list based on PDG criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to get properties for",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["mass", "lifetime", "width", "all"],
                        "description": "Type of property to find best value for",
                    },
                    "pedantic": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use strict criteria (may raise ambiguity errors)",
                    },
                },
                "required": ["particle_name", "property_type"],
            },
        ),
        types.Tool(
            name="apply_pdg_rounding",
            description="Apply PDG rounding rules to value and error",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Value to round",
                    },
                    "error": {
                        "type": "number",
                        "description": "Error/uncertainty (must be > 0)",
                    },
                },
                "required": ["value", "error"],
            },
        ),
        types.Tool(
            name="get_linked_data",
            description="Get linked data from PDG database tables",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Particle name to get linked data for",
                    },
                    "link_type": {
                        "type": "string",
                        "enum": ["measurements", "references", "footnotes", "values"],
                        "description": "Type of linked data to retrieve",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of linked items to return",
                    },
                },
                "required": ["particle_name", "link_type"],
            },
        ),
        types.Tool(
            name="normalize_pdg_data",
            description="Normalize and validate PDG data structures",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_input": {
                        "type": "string",
                        "description": "PDG data to normalize (particle name, ID, etc.)",
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["particle_name", "pdg_id", "value", "measurement"],
                        "description": "Type of data being normalized",
                    },
                    "strict": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use strict normalization rules",
                    },
                },
                "required": ["data_input", "data_type"],
            },
        ),
        types.Tool(
            name="get_pdg_table_data",
            description="Get raw data from PDG database tables",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the PDG database table",
                    },
                    "row_id": {
                        "type": "integer",
                        "description": "ID of the row to retrieve",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include table metadata in response",
                    },
                },
                "required": ["table_name", "row_id"],
            },
        ),
    ]


def pdg_round_utils(value, error):
    """Apply PDG rounding rules to value and error (from PDG utils)."""
    if error <= 0.:
        raise ValueError('PDG rounding requires error larger than zero')
    
    log = math.log10(abs(error))
    if abs(error) < 1.0 and int(log) != log:
        power = int(log)
    else:
        power = int(log) + 1
    
    reduced_error = error * 10 ** (-power)
    if reduced_error < 0.355:
        n_digits = 2
    elif reduced_error < 0.950:
        n_digits = 1
    else:
        reduced_error = 0.1
        power += 1
        n_digits = 2
    
    new_error = round(reduced_error, n_digits) * 10 ** power
    new_value = round(value * 10 ** (-power), n_digits) * 10 ** power
    return new_value, new_error


def parse_pdg_id_utils(pdgid):
    """Parse PDG Identifier and return (base identifier, edition)."""
    try:
        baseid, edition = pdgid.split('/')
    except ValueError:
        baseid = pdgid
        edition = None
    return baseid.upper(), edition


def base_pdg_id_utils(pdgid):
    """Return normalized base part of PDG Identifier."""
    return parse_pdg_id_utils(pdgid)[0]


def make_pdg_id_utils(baseid, edition=None):
    """Return normalized full PDG Identifier, possibly including edition."""
    if baseid is None:
        return None
    if edition is None:
        return baseid.upper()
    else:
        return ('%s/%s' % (baseid, edition)).upper()


def find_best_property_utils(properties, pedantic=False, quantity=None):
    """Find the 'best' property from an iterable based on PDG criteria."""
    for_what = ' for %s' % quantity if quantity else ''
    
    # Filter out alternates
    props_without_alternates = []
    for p in properties:
        try:
            if hasattr(p, 'data_flags') and 'A' not in p.data_flags:
                props_without_alternates.append(p)
            elif not hasattr(p, 'data_flags'):
                props_without_alternates.append(p)
        except:
            props_without_alternates.append(p)
    
    # In non-pedantic mode, filter out "special" values
    if not pedantic:
        filtered_props = []
        for p in props_without_alternates:
            try:
                if hasattr(p, 'data_flags') and 's' not in p.data_flags:
                    filtered_props.append(p)
                elif not hasattr(p, 'data_flags'):
                    filtered_props.append(p)
            except:
                filtered_props.append(p)
        props_without_alternates = filtered_props
    
    if len(props_without_alternates) == 0:
        return None, f'No best property found{for_what}'
    elif len(props_without_alternates) == 1:
        return props_without_alternates[0], "success"
    else:
        if pedantic:
            return None, f'Ambiguous best property{for_what}'
        else:
            # Look for properties with 'D' flag (default/recommended)
            props_best = []
            for p in props_without_alternates:
                try:
                    if hasattr(p, 'data_flags') and 'D' in p.data_flags:
                        props_best.append(p)
                except:
                    pass
            
            if len(props_best) >= 1:
                return props_best[0], "success"
            else:
                return props_without_alternates[0], "success"


def get_property_by_type(particle, property_type):
    """Get properties of a specific type from a particle."""
    properties = []
    
    try:
        if property_type == "mass" or property_type == "all":
            for mass_prop in particle.masses():
                properties.append(mass_prop)
    except:
        pass
    
    try:
        if property_type == "lifetime" or property_type == "all":
            for lifetime_prop in particle.lifetimes():
                properties.append(lifetime_prop)
    except:
        pass
    
    try:
        if property_type == "width" or property_type == "all":
            for width_prop in particle.widths():
                properties.append(width_prop)
    except:
        pass
    
    return properties


async def handle_utils_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Handle utils-related tool calls."""
    
    if name == "parse_pdg_identifier":
        pdgid = arguments["pdgid"]
        
        try:
            base_id, edition = parse_pdg_id_utils(pdgid)
            
            result = {
                "original_pdgid": pdgid,
                "base_identifier": base_id,
                "edition": edition,
                "normalized": make_pdg_id_utils(base_id, edition),
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to parse PDG identifier: {str(e)}"}, indent=2)
            )]
    
    elif name == "get_base_pdg_id":
        pdgid = arguments["pdgid"]
        
        try:
            base_id = base_pdg_id_utils(pdgid)
            
            result = {
                "original_pdgid": pdgid,
                "base_identifier": base_id,
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to get base PDG ID: {str(e)}"}, indent=2)
            )]
    
    elif name == "make_pdg_identifier":
        baseid = arguments["baseid"]
        edition = arguments.get("edition")
        
        try:
            normalized_id = make_pdg_id_utils(baseid, edition)
            
            result = {
                "base_identifier": baseid,
                "edition": edition,
                "normalized_pdgid": normalized_id,
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to make PDG identifier: {str(e)}"}, indent=2)
            )]
    
    elif name == "find_best_property":
        particle_name = arguments["particle_name"]
        property_type = arguments["property_type"]
        pedantic = arguments.get("pedantic", False)
        
        try:
            particle = api.get_particle_by_name(particle_name)
            properties = get_property_by_type(particle, property_type)
            
            if not properties:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"No {property_type} properties found for {particle_name}"}, indent=2)
                )]
            
            best_prop, status = find_best_property_utils(properties, pedantic, property_type)
            
            if best_prop is None:
                result = {
                    "particle_name": particle_name,
                    "property_type": property_type,
                    "status": "error",
                    "message": status,
                    "total_properties": len(properties),
                }
            else:
                result = {
                    "particle_name": particle_name,
                    "property_type": property_type,
                    "status": "success",
                    "best_property": {
                        "pdgid": getattr(best_prop, "pdgid", "N/A"),
                        "description": getattr(best_prop, "description", "N/A"),
                        "display_value": getattr(best_prop, "display_value_text", "N/A"),
                        "value": getattr(best_prop, "value", "N/A"),
                        "units": getattr(best_prop, "units", "N/A"),
                        "data_flags": getattr(best_prop, "data_flags", "N/A"),
                    },
                    "total_properties": len(properties),
                    "pedantic_mode": pedantic,
                }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to find best property: {str(e)}"}, indent=2)
            )]
    
    elif name == "apply_pdg_rounding":
        value = arguments["value"]
        error = arguments["error"]
        
        try:
            rounded_value, rounded_error = pdg_round_utils(value, error)
            
            result = {
                "original_value": value,
                "original_error": error,
                "rounded_value": rounded_value,
                "rounded_error": rounded_error,
                "rounding_applied": True,
                "change_in_value": abs(rounded_value - value),
                "change_in_error": abs(rounded_error - error),
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to apply PDG rounding: {str(e)}"}, indent=2)
            )]
    
    elif name == "get_linked_data":
        particle_name = arguments["particle_name"]
        link_type = arguments["link_type"]
        limit = arguments.get("limit", 10)
        
        try:
            particle = api.get_particle_by_name(particle_name)
            linked_data = []
            
            if link_type == "measurements":
                # Get measurements from properties
                count = 0
                for prop in particle.properties():
                    if count >= limit:
                        break
                    try:
                        for measurement in prop.get_measurements():
                            if count >= limit:
                                break
                            linked_data.append({
                                "type": "measurement",
                                "id": getattr(measurement, "id", "N/A"),
                                "property_pdgid": getattr(prop, "pdgid", "N/A"),
                                "reference_id": getattr(measurement, "reference_id", "N/A") if hasattr(measurement, "reference_id") else "N/A",
                                "value": getattr(measurement.get_value(), "value_text", "N/A") if hasattr(measurement, "get_value") and measurement.get_value() else "N/A",
                            })
                            count += 1
                    except:
                        continue
            
            elif link_type == "references":
                # Get references from measurements
                count = 0
                ref_ids = set()
                for prop in particle.properties():
                    if count >= limit:
                        break
                    try:
                        for measurement in prop.get_measurements():
                            if count >= limit:
                                break
                            if hasattr(measurement, "reference") and measurement.reference:
                                ref = measurement.reference
                                ref_id = getattr(ref, "id", "N/A")
                                if ref_id not in ref_ids:
                                    ref_ids.add(ref_id)
                                    linked_data.append({
                                        "type": "reference",
                                        "id": ref_id,
                                        "title": getattr(ref, "title", "N/A"),
                                        "year": getattr(ref, "publication_year", "N/A"),
                                        "doi": getattr(ref, "doi", "N/A"),
                                    })
                                    count += 1
                    except:
                        continue
            
            elif link_type == "values":
                # Get values from measurements
                count = 0
                for prop in particle.properties():
                    if count >= limit:
                        break
                    try:
                        for measurement in prop.get_measurements():
                            if count >= limit:
                                break
                            value = measurement.get_value()
                            if value:
                                linked_data.append({
                                    "type": "value",
                                    "id": getattr(value, "id", "N/A"),
                                    "measurement_id": getattr(measurement, "id", "N/A"),
                                    "value_text": getattr(value, "value_text", "N/A"),
                                    "error_positive": getattr(value, "error_positive", "N/A"),
                                    "error_negative": getattr(value, "error_negative", "N/A"),
                                })
                                count += 1
                    except:
                        continue
            
            result = {
                "particle_name": particle_name,
                "link_type": link_type,
                "linked_data": linked_data,
                "count": len(linked_data),
                "limited_to": limit,
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to get linked data: {str(e)}"}, indent=2)
            )]
    
    elif name == "normalize_pdg_data":
        data_input = arguments["data_input"]
        data_type = arguments["data_type"]
        strict = arguments.get("strict", False)
        
        try:
            result = {
                "original_input": data_input,
                "data_type": data_type,
                "strict_mode": strict,
            }
            
            if data_type == "particle_name":
                # Normalize particle name
                try:
                    canonical_name = api.get_canonical_name(data_input)
                    result["normalized"] = canonical_name
                    result["is_valid"] = True
                except:
                    result["normalized"] = data_input.strip().lower()
                    result["is_valid"] = False
                    result["note"] = "Could not find canonical name"
            
            elif data_type == "pdg_id":
                # Normalize PDG ID
                base_id, edition = parse_pdg_id_utils(data_input)
                result["normalized"] = make_pdg_id_utils(base_id, edition)
                result["base_id"] = base_id
                result["edition"] = edition
                result["is_valid"] = True
            
            elif data_type == "value":
                # Normalize numerical value
                try:
                    numeric_value = float(data_input)
                    result["normalized"] = numeric_value
                    result["is_valid"] = True
                    result["type"] = "numeric"
                except ValueError:
                    result["normalized"] = data_input.strip()
                    result["is_valid"] = False
                    result["type"] = "text"
                    result["note"] = "Could not convert to numeric value"
            
            elif data_type == "measurement":
                # Normalize measurement identifier
                try:
                    measurement_id = int(data_input)
                    result["normalized"] = measurement_id
                    result["is_valid"] = True
                    result["type"] = "integer"
                except ValueError:
                    result["normalized"] = data_input.strip()
                    result["is_valid"] = False
                    result["type"] = "text"
                    result["note"] = "Could not convert to measurement ID"
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to normalize PDG data: {str(e)}"}, indent=2)
            )]
    
    elif name == "get_pdg_table_data":
        table_name = arguments["table_name"]
        row_id = arguments["row_id"]
        include_metadata = arguments.get("include_metadata", True)
        
        try:
            # This is a simplified implementation as direct table access 
            # depends on the specific PDG API implementation
            result = {
                "table_name": table_name,
                "row_id": row_id,
                "status": "simulated",
                "note": "Direct table access requires low-level PDG API access",
            }
            
            if include_metadata:
                result["metadata"] = {
                    "available_tables": ["particles", "measurements", "values", "references", "footnotes"],
                    "access_method": "via PDG API",
                    "implementation": "This would require direct database access through the PDG utils API",
                }
            
            # Attempt to get some data through available API methods
            try:
                if table_name == "particles":
                    # Try to get particle data
                    particles = api.get_particles()
                    result["sample_data"] = f"Table contains particle data, example: {list(particles)[0].name if particles else 'N/A'}"
                elif table_name == "measurements":
                    result["sample_data"] = "Measurements table contains PDG measurement data"
                else:
                    result["sample_data"] = f"Table '{table_name}' data access not implemented in this simulation"
            except Exception as inner_e:
                result["sample_data"] = f"Could not access table data: {str(inner_e)}"
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to get PDG table data: {str(e)}"}, indent=2)
            )]
    
    else:
        return [types.TextContent(
            type="text", text=json.dumps({"error": f"Unknown utils tool: {name}"})
        )] 
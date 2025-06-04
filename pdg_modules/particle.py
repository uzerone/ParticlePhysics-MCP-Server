"""
PDG Particle Module

This module contains tools for working with PDG particle objects:
- PdgParticle: Container class for all particle information
- PdgItem: Items in decay descriptions and product lists
- PdgParticleList: Lists of particles with advanced filtering

Based on the PDG particle API: https://pdgapi.lbl.gov/doc/pdg.particle.html
"""

import json
import mcp.types as types
from typing import Any, Dict, List


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
                            "enum": ["J", "P", "C", "G", "I", "all"]
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


def format_quantum_numbers(particle):
    """Format quantum numbers for a particle."""
    try:
        quantum_info = {
            "J": getattr(particle, "quantum_J", "N/A"),  # Spin
            "P": getattr(particle, "quantum_P", "N/A"),  # Parity
            "C": getattr(particle, "quantum_C", "N/A"),  # C-parity
            "G": getattr(particle, "quantum_G", "N/A"),  # G-parity
            "I": getattr(particle, "quantum_I", "N/A"),  # Isospin
        }
        return quantum_info
    except Exception as e:
        return {"error": f"Failed to format quantum numbers: {str(e)}"}


def format_particle_classification(particle):
    """Format particle classification information."""
    try:
        classification = {
            "is_baryon": getattr(particle, "is_baryon", False),
            "is_meson": getattr(particle, "is_meson", False),
            "is_lepton": getattr(particle, "is_lepton", False),
            "is_boson": getattr(particle, "is_boson", False),
            "is_quark": getattr(particle, "is_quark", False),
        }
        
        # Determine primary classification
        primary_type = "unknown"
        if classification["is_baryon"]:
            primary_type = "baryon"
        elif classification["is_meson"]:
            primary_type = "meson"
        elif classification["is_lepton"]:
            primary_type = "lepton"
        elif classification["is_boson"]:
            primary_type = "boson"
        elif classification["is_quark"]:
            primary_type = "quark"
            
        classification["primary_type"] = primary_type
        return classification
    except Exception as e:
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
                        measurements.append({
                            "id": getattr(measurement, "id", "N/A"),
                            "technique": getattr(measurement, "technique", "N/A"),
                            "comment": getattr(measurement, "comment", "N/A"),
                        })
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
            'P': "specific state",
            'A': "also alias", 
            'W': "was alias",
            'S': "shortcut",
            'B': "both charges",
            'C': "both charges, conjugate",
            'G': "generic state",
            'L': "general list",
            'I': "inclusive indicator",
            'T': "arbitrary text"
        }
        
        item_type = item_info.get("item_type", "")
        item_info["item_type_description"] = item_type_descriptions.get(item_type, "unknown")
        
        return item_info
    except Exception as e:
        return {"error": f"Failed to format PDG item: {str(e)}"}


async def handle_particle_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Handle particle-related tool calls."""
    
    if name == "get_particle_quantum_numbers":
        particle_name = arguments["particle_name"]
        include_all = arguments.get("include_all_quantum_numbers", True)

        try:
            particle = api.get_particle_by_name(particle_name)
            
            result = {
                "particle": particle_name,
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "quantum_numbers": format_quantum_numbers(particle)
            }
            
            if include_all:
                result["additional_info"] = {
                    "charge": getattr(particle, "charge", "N/A"),
                    "mcid": getattr(particle, "mcid", "N/A"),
                    "name": getattr(particle, "name", "N/A"),
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
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "classification": format_particle_classification(particle),
                "has_entries": {
                    "has_mass_entry": getattr(particle, "has_mass_entry", False),
                    "has_lifetime_entry": getattr(particle, "has_lifetime_entry", False),
                    "has_width_entry": getattr(particle, "has_width_entry", False),
                }
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
                    if hasattr(particle, type_method) and getattr(particle, type_method):
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
                    if charge_filter is not None and getattr(particle, "charge", None) != charge_filter:
                        continue
                    if has_mass and not getattr(particle, "has_mass_entry", False):
                        continue  
                    if has_lifetime and not getattr(particle, "has_lifetime_entry", False):
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
                            "has_mass_entry": getattr(particle, "has_mass_entry", False),
                            "has_lifetime_entry": getattr(particle, "has_lifetime_entry", False),
                            "has_width_entry": getattr(particle, "has_width_entry", False),
                        }
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
            properties = list(particle.properties(
                data_type_key=data_type_filter,
                require_summary_data=require_summary_data,
                in_summary_table=in_summary_table,
                omit_branching_ratios=False
            ))
            
            result = {
                "particle": particle_name,
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "filters": {
                    "data_type_filter": data_type_filter,
                    "require_summary_data": require_summary_data,
                    "in_summary_table": in_summary_table,
                },
                "properties": format_particle_properties(particle, properties, include_measurements=False),
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
                        particles.append({
                            "name": getattr(particle, "name", "N/A"),
                            "pdgid": getattr(particle, "pdgid", "N/A"),
                            "mcid": getattr(particle, "mcid", "N/A"),
                            "charge": getattr(particle, "charge", "N/A"),
                        })
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
                "mass_entries": format_particle_properties(particle, masses, include_measurements),
                "primary_mass": getattr(particle, "mass", "N/A"),
                "mass_error": getattr(particle, "mass_error", "N/A"),
                "has_mass_entry": getattr(particle, "has_mass_entry", False),
                "total_mass_entries": len(masses),
            }
            
            # Include measurements if requested
            if include_measurements:
                measurements = []
                try:
                    for measurement in particle.mass_measurements(require_summary_data=require_summary_data):
                        measurements.append({
                            "id": getattr(measurement, "id", "N/A"),
                            "technique": getattr(measurement, "technique", "N/A"),
                            "comment": getattr(measurement, "comment", "N/A"),
                        })
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
            
            lifetimes = list(particle.lifetimes(require_summary_data=require_summary_data))
            
            result = {
                "particle": particle_name,
                "pdgid": getattr(particle, "pdgid", "N/A"),
                "units": units,
                "lifetime_entries": format_particle_properties(particle, lifetimes, include_measurements),
                "primary_lifetime": getattr(particle, "lifetime", "N/A"),
                "lifetime_error": getattr(particle, "lifetime_error", "N/A"),
                "has_lifetime_entry": getattr(particle, "has_lifetime_entry", False),
                "total_lifetime_entries": len(lifetimes),
            }
            
            # Include measurements if requested
            if include_measurements:
                measurements = []
                try:
                    for measurement in particle.lifetime_measurements(require_summary_data=require_summary_data):
                        measurements.append({
                            "id": getattr(measurement, "id", "N/A"),
                            "technique": getattr(measurement, "technique", "N/A"),
                            "comment": getattr(measurement, "comment", "N/A"),
                        })
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
                "width_entries": format_particle_properties(particle, widths, include_measurements),
                "primary_width": getattr(particle, "width", "N/A"),
                "width_error": getattr(particle, "width_error", "N/A"),
                "has_width_entry": getattr(particle, "has_width_entry", False),
                "total_width_entries": len(widths),
            }
            
            # Include measurements if requested
            if include_measurements:
                measurements = []
                try:
                    for measurement in particle.width_measurements(require_summary_data=require_summary_data):
                        measurements.append({
                            "id": getattr(measurement, "id", "N/A"),
                            "technique": getattr(measurement, "technique", "N/A"),
                            "comment": getattr(measurement, "comment", "N/A"),
                        })
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
                        particle_data["quantum_numbers"] = format_quantum_numbers(particle)
                    else:
                        for qn in quantum_numbers:
                            if qn == "J":
                                particle_data["quantum_numbers"]["J"] = getattr(particle, "quantum_J", "N/A")
                            elif qn == "P":
                                particle_data["quantum_numbers"]["P"] = getattr(particle, "quantum_P", "N/A")
                            elif qn == "C":
                                particle_data["quantum_numbers"]["C"] = getattr(particle, "quantum_C", "N/A")
                            elif qn == "G":
                                particle_data["quantum_numbers"]["G"] = getattr(particle, "quantum_G", "N/A")
                            elif qn == "I":
                                particle_data["quantum_numbers"]["I"] = getattr(particle, "quantum_I", "N/A")
                    
                    comparison_data.append(particle_data)
                    
                except Exception as e:
                    comparison_data.append({
                        "name": particle_name,
                        "error": f"Failed to get data: {str(e)}"
                    })
            
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
                            best_mass = masses[0].best_summary() if hasattr(masses[0], "best_summary") else None
                            if best_mass:
                                result["error_info"]["mass"]["asymmetric_errors"] = {
                                    "error_positive": getattr(best_mass, "error_positive", "N/A"),
                                    "error_negative": getattr(best_mass, "error_negative", "N/A"),
                                    "is_limit": getattr(best_mass, "is_limit", False),
                                }
                    except:
                        pass
                        
            if property_type in ["lifetime", "all"]:
                lifetime_error = getattr(particle, "lifetime_error", None)
                result["error_info"]["lifetime"] = {
                    "primary_error": lifetime_error,
                    "has_symmetric_error": lifetime_error is not None,
                    "has_lifetime_entry": getattr(particle, "has_lifetime_entry", False),
                }
                
                if include_asymmetric and getattr(particle, "has_lifetime_entry", False):
                    try:
                        lifetimes = list(particle.lifetimes())
                        if lifetimes:
                            best_lifetime = lifetimes[0].best_summary() if hasattr(lifetimes[0], "best_summary") else None
                            if best_lifetime:
                                result["error_info"]["lifetime"]["asymmetric_errors"] = {
                                    "error_positive": getattr(best_lifetime, "error_positive", "N/A"),
                                    "error_negative": getattr(best_lifetime, "error_negative", "N/A"),
                                    "is_limit": getattr(best_lifetime, "is_limit", False),
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
                            best_width = widths[0].best_summary() if hasattr(widths[0], "best_summary") else None
                            if best_width:
                                result["error_info"]["width"]["asymmetric_errors"] = {
                                    "error_positive": getattr(best_width, "error_positive", "N/A"),
                                    "error_negative": getattr(best_width, "error_negative", "N/A"),
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
                type="text", text=json.dumps({"error": f"Unknown particle tool: {name}"})
            )
        ] 
"""
PDG Error Handling Module

This module contains tools for error handling, validation, and diagnostics
when working with PDG data and APIs.
"""

import json
import mcp.types as types
from typing import Any, Dict, List


def get_error_tools() -> List[types.Tool]:
    """Return all error-related MCP tools."""
    return [
        types.Tool(
            name="validate_pdg_identifier",
            description="Validate PDG identifiers and diagnose common errors",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdgid": {
                        "type": "string",
                        "description": "PDG identifier to validate",
                    },
                    "check_data_availability": {
                        "type": "boolean",
                        "default": True,
                        "description": "Check if data is available for this identifier",
                    },
                    "suggest_alternatives": {
                        "type": "boolean",
                        "default": True,
                        "description": "Suggest alternative identifiers if invalid",
                    },
                },
                "required": ["pdgid"],
            },
        ),
        types.Tool(
            name="get_error_info",
            description="Get information about PDG API error types and their meanings",
            inputSchema={
                "type": "object",
                "properties": {
                    "error_type": {
                        "type": "string",
                        "enum": ["all", "PdgApiError", "PdgInvalidPdgIdError", "PdgNoDataError", "PdgAmbiguousValueError", "PdgRoundingError"],
                        "default": "all",
                        "description": "Specific error type to get information about",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="diagnose_lookup_issues",
            description="Diagnose common issues with particle or data lookups",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that's causing issues (particle name, PDG ID, etc.)",
                    },
                    "lookup_type": {
                        "type": "string",
                        "enum": ["particle_name", "pdg_id", "mcid", "property"],
                        "default": "particle_name",
                        "description": "Type of lookup being attempted",
                    },
                    "include_suggestions": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include suggestions for fixing the issue",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="safe_particle_lookup",
            description="Safely lookup particle with comprehensive error handling and alternatives",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Particle name, PDG ID, or Monte Carlo ID",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["auto", "name", "pdgid", "mcid"],
                        "default": "auto",
                        "description": "Type of search to perform",
                    },
                    "return_alternatives": {
                        "type": "boolean",
                        "default": True,
                        "description": "Return alternative matches if exact match fails",
                    },
                    "include_error_details": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed error information",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


def get_pdg_error_info():
    """Get information about PDG error types and their meanings."""
    return {
        "PdgApiError": {
            "description": "PDG API base exception",
            "when_raised": "General PDG API errors and base class for other PDG exceptions",
            "typical_causes": [
                "Database connection issues",
                "API initialization problems",
                "General PDG library errors"
            ],
            "solutions": [
                "Check PDG API installation",
                "Verify database connectivity",
                "Restart the API connection"
            ]
        },
        "PdgInvalidPdgIdError": {
            "description": "Exception raised when encountering an invalid PDG Identifier",
            "when_raised": "When a PDG ID format is invalid or doesn't exist in the database",
            "typical_causes": [
                "Malformed PDG identifier",
                "Non-existent PDG ID",
                "Typos in particle names or IDs",
                "Using outdated PDG identifiers"
            ],
            "solutions": [
                "Check PDG ID format (e.g., 'S008', 'M100')",
                "Verify particle name spelling",
                "Use canonical particle names",
                "Check available identifiers with get_all_pdg_identifiers"
            ]
        },
        "PdgNoDataError": {
            "description": "Exception raised if no data is found",
            "when_raised": "When a valid identifier exists but has no associated data",
            "typical_causes": [
                "Particle exists but has no measured properties",
                "Data not available in current PDG edition",
                "Requesting specific property that doesn't exist",
                "Database incomplete for certain particles"
            ],
            "solutions": [
                "Check if particle has the requested property",
                "Try a different PDG edition",
                "Use summary values instead of measurements",
                "Check particle existence with basic search first"
            ]
        },
        "PdgAmbiguousValueError": {
            "description": "Exception raised when choice of value is ambiguous and there is no single best value",
            "when_raised": "When multiple values exist and system cannot determine which to use",
            "typical_causes": [
                "Multiple measurements with no clear best value",
                "Conflicting data sources",
                "Properties with multiple representations",
                "Summary values with different criteria"
            ],
            "solutions": [
                "Specify value type or criteria",
                "Use get_summary_values to see all options",
                "Request measurements instead of summary",
                "Filter by specific value types"
            ]
        },
        "PdgRoundingError": {
            "description": "Exception raised when PDG rounding is undefined",
            "when_raised": "When attempting to round values using undefined PDG conventions",
            "typical_causes": [
                "Rounding rules not defined for specific values",
                "Precision issues with very small/large numbers",
                "Undefined significant figures",
                "Custom rounding operations"
            ],
            "solutions": [
                "Use raw values without rounding",
                "Apply manual rounding if needed",
                "Check value precision requirements",
                "Use display_value_text instead of numeric values"
            ]
        }
    }


def diagnose_query_issues(query, lookup_type="particle_name"):
    """Diagnose common issues with particle/data queries."""
    suggestions = []
    potential_issues = []
    
    # Common particle name issues
    if lookup_type == "particle_name":
        # Check for common misspellings or variations
        common_corrections = {
            "electron": ["e-", "e+"],
            "muon": ["mu-", "mu+"],
            "proton": ["p", "p+"],
            "neutron": ["n", "n0"],
            "pion": ["pi+", "pi-", "pi0"],
            "kaon": ["K+", "K-", "K0"],
            "tau": ["tau-", "tau+"],
            "neutrino": ["nu_e", "nu_mu", "nu_tau"],
            "photon": ["gamma"],
            "w boson": ["W+", "W-"],
            "z boson": ["Z0"],
            "higgs": ["H"],
        }
        
        query_lower = query.lower()
        for common_name, pdg_names in common_corrections.items():
            if common_name in query_lower:
                suggestions.extend([f"Try '{name}' instead of '{query}'" for name in pdg_names])
        
        # Check for common formatting issues
        if " " in query:
            potential_issues.append("Particle names usually don't contain spaces")
            suggestions.append(f"Try removing spaces: '{query.replace(' ', '')}'")
        
        if query.endswith("_"):
            potential_issues.append("Particle names ending with underscore might need specific formatting")
            suggestions.append(f"Try without underscore: '{query[:-1]}'")
    
    # PDG ID format issues
    elif lookup_type == "pdg_id":
        if not any(c.isdigit() for c in query):
            potential_issues.append("PDG IDs usually contain numbers")
            suggestions.append("PDG IDs typically follow patterns like 'S008', 'M100', etc.")
        
        if len(query) < 3:
            potential_issues.append("PDG IDs are usually longer")
            suggestions.append("Most PDG IDs are 3+ characters (e.g., 'S008', 'M100')")
    
    # Monte Carlo ID issues
    elif lookup_type == "mcid":
        try:
            mcid_num = int(query)
            if mcid_num < 0:
                potential_issues.append("Monte Carlo IDs are typically positive")
            if mcid_num > 99999:
                potential_issues.append("Very large Monte Carlo IDs are uncommon")
        except ValueError:
            potential_issues.append("Monte Carlo IDs should be numeric")
            suggestions.append("Monte Carlo IDs are integers (e.g., 211, 2212)")
    
    # General suggestions
    if not suggestions:
        suggestions = [
            "Check particle name spelling",
            "Try canonical PDG names (e.g., 'e-' instead of 'electron')",
            "Use get_canonical_name to find correct naming",
            "Search with get_particles_by_name for partial matches"
        ]
    
    return {
        "query": query,
        "lookup_type": lookup_type,
        "potential_issues": potential_issues if potential_issues else ["No obvious issues detected"],
        "suggestions": suggestions
    }


def safe_pdg_operation(operation_func, *args, **kwargs):
    """Safely execute a PDG operation with comprehensive error handling."""
    try:
        return operation_func(*args, **kwargs), None
    except Exception as e:
        # Try to import PDG errors to check specific types
        try:
            import pdg.errors as pdg_errors
            
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "is_pdg_error": isinstance(e, (
                    pdg_errors.PdgApiError,
                    pdg_errors.PdgInvalidPdgIdError,
                    pdg_errors.PdgNoDataError,
                    pdg_errors.PdgAmbiguousValueError,
                    pdg_errors.PdgRoundingError
                ))
            }
            
            # Add specific guidance based on error type
            if isinstance(e, pdg_errors.PdgInvalidPdgIdError):
                error_info["guidance"] = "Check PDG identifier format and spelling"
            elif isinstance(e, pdg_errors.PdgNoDataError):
                error_info["guidance"] = "No data available for this identifier"
            elif isinstance(e, pdg_errors.PdgAmbiguousValueError):
                error_info["guidance"] = "Multiple values found, be more specific"
            elif isinstance(e, pdg_errors.PdgRoundingError):
                error_info["guidance"] = "Use raw values or display_value_text"
            else:
                error_info["guidance"] = "General PDG API error"
                
        except ImportError:
            # Fallback if PDG errors module not available
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "is_pdg_error": "PDG" in type(e).__name__ or "Pdg" in str(e),
                "guidance": "Check query format and try alternatives"
            }
        
        return None, error_info


def format_particle_info_safe(particle, include_basic=True, include_measurements=False):
    """Format particle information in a readable way (safe version)."""
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

        return info
    except Exception as e:
        return {
            "name": str(particle),
            "error": f"Error formatting particle info: {str(e)}",
        }


async def handle_error_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Handle error-related tool calls."""
    
    if name == "validate_pdg_identifier":
        pdgid = arguments["pdgid"]
        check_data_availability = arguments.get("check_data_availability", True)
        suggest_alternatives = arguments.get("suggest_alternatives", True)

        try:
            validation_result = {
                "pdgid": pdgid,
                "is_valid": False,
                "has_data": False,
                "error_details": None,
                "alternatives": [],
                "suggestions": []
            }

            # Try to validate the PDG ID
            def validate_operation():
                return api.get(pdgid)

            result, error_info = safe_pdg_operation(validate_operation)

            if error_info:
                validation_result["error_details"] = error_info
                validation_result["is_valid"] = False
                
                # Generate suggestions based on error type
                if "Invalid" in error_info.get("error_type", ""):
                    validation_result["suggestions"].extend([
                        "Check PDG identifier format",
                        "Verify spelling and capitalization",
                        "Use standard PDG naming conventions"
                    ])
            else:
                validation_result["is_valid"] = True
                
                # Check if data is available
                if check_data_availability and result:
                    try:
                        # Try to get some basic properties
                        if hasattr(result, 'description'):
                            validation_result["has_data"] = True
                            validation_result["description"] = result.description
                        elif hasattr(result, '__iter__'):
                            validation_result["has_data"] = len(list(result)) > 0
                    except:
                        validation_result["has_data"] = False

            # Suggest alternatives if requested and needed
            if suggest_alternatives and (not validation_result["is_valid"] or not validation_result["has_data"]):
                # Try to find similar identifiers
                try:
                    # Look for similar particle names
                    diagnosis = diagnose_query_issues(pdgid, "pdg_id")
                    validation_result["alternatives"].extend(diagnosis["suggestions"])
                    
                    # Try to find canonical name
                    try:
                        canonical = api.get_canonical_name(pdgid)
                        if canonical != pdgid:
                            validation_result["alternatives"].append(f"Try canonical name: '{canonical}'")
                    except:
                        pass
                        
                except:
                    pass

            return [types.TextContent(type="text", text=json.dumps(validation_result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to validate PDG identifier: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "get_error_info":
        error_type = arguments.get("error_type", "all")

        try:
            error_info_data = get_pdg_error_info()
            
            if error_type == "all":
                result = {
                    "error_types": error_info_data,
                    "total_types": len(error_info_data),
                    "summary": "PDG API provides specific exception types for different error conditions"
                }
            else:
                if error_type in error_info_data:
                    result = {
                        "error_type": error_type,
                        "details": error_info_data[error_type]
                    }
                else:
                    result = {
                        "error": f"Unknown error type: {error_type}",
                        "available_types": list(error_info_data.keys())
                    }

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

    elif name == "diagnose_lookup_issues":
        query = arguments["query"]
        lookup_type = arguments.get("lookup_type", "particle_name")
        include_suggestions = arguments.get("include_suggestions", True)

        try:
            diagnosis = diagnose_query_issues(query, lookup_type)
            
            # Try actual lookup to get real error information
            def test_lookup():
                if lookup_type == "particle_name":
                    return api.get_particle_by_name(query)
                elif lookup_type == "pdg_id":
                    return api.get(query)
                elif lookup_type == "mcid":
                    return api.get_particle_by_mcid(int(query))
                else:
                    return api.get_particle_by_name(query)

            result, error_info = safe_pdg_operation(test_lookup)
            
            diagnosis_result = {
                "query": query,
                "lookup_type": lookup_type,
                "diagnosis": diagnosis,
                "actual_error": error_info,
                "lookup_successful": result is not None
            }

            if result is not None:
                diagnosis_result["success_info"] = "Lookup succeeded - no issues detected"
                if hasattr(result, 'name'):
                    diagnosis_result["found_particle"] = result.name

            if include_suggestions and error_info:
                # Add specific suggestions based on actual error
                if "Invalid" in error_info.get("error_type", ""):
                    diagnosis_result["specific_suggestions"] = [
                        "Try get_canonical_name to find correct naming",
                        "Use get_particles_by_name for partial matching",
                        "Check get_all_pdg_identifiers for valid IDs"
                    ]

            return [types.TextContent(type="text", text=json.dumps(diagnosis_result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to diagnose lookup issues: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    elif name == "safe_particle_lookup":
        query = arguments["query"]
        search_type = arguments.get("search_type", "auto")
        return_alternatives = arguments.get("return_alternatives", True)
        include_error_details = arguments.get("include_error_details", True)

        try:
            lookup_result = {
                "query": query,
                "search_type": search_type,
                "success": False,
                "particle": None,
                "alternatives": [],
                "error_details": None
            }

            # Determine search type
            if search_type == "auto":
                if query.isdigit():
                    search_type = "mcid"
                elif query.upper().startswith(("S", "M", "G", "T")) and any(c.isdigit() for c in query):
                    search_type = "pdgid"
                else:
                    search_type = "name"

            # Try primary lookup
            def primary_lookup():
                if search_type == "name":
                    return api.get_particle_by_name(query)
                elif search_type == "mcid":
                    return api.get_particle_by_mcid(int(query))
                elif search_type == "pdgid":
                    items = api.get(query)
                    if hasattr(items, '__iter__'):
                        particles = []
                        for item in items:
                            if hasattr(item, 'name'):
                                particles.append(item)
                        return particles[0] if particles else None
                    return items if hasattr(items, 'name') else None
                else:
                    return api.get_particle_by_name(query)

            result, error_info = safe_pdg_operation(primary_lookup)

            if result:
                lookup_result["success"] = True
                lookup_result["particle"] = format_particle_info_safe(result)
                lookup_result["search_type_used"] = search_type
            else:
                lookup_result["error_details"] = error_info if include_error_details else None

                # Try alternatives if requested
                if return_alternatives:
                    alternatives = []
                    
                    # Try different search types
                    for alt_type in ["name", "mcid", "pdgid"]:
                        if alt_type != search_type:
                            try:
                                def alt_lookup():
                                    if alt_type == "name":
                                        return api.get_particle_by_name(query)
                                    elif alt_type == "mcid" and query.isdigit():
                                        return api.get_particle_by_mcid(int(query))
                                    elif alt_type == "pdgid":
                                        return api.get(query)
                                    return None

                                alt_result, _ = safe_pdg_operation(alt_lookup)
                                if alt_result and hasattr(alt_result, 'name'):
                                    alternatives.append({
                                        "method": alt_type,
                                        "particle": format_particle_info_safe(alt_result)
                                    })
                            except:
                                continue

                    # Try partial name matching
                    try:
                        def partial_lookup():
                            return list(api.get_particles_by_name(query, case_sensitive=False))[:3]

                        partial_results, _ = safe_pdg_operation(partial_lookup)
                        if partial_results:
                            for particle_list in partial_results:
                                if hasattr(particle_list, '__iter__'):
                                    for particle in particle_list:
                                        if hasattr(particle, 'name'):
                                            alternatives.append({
                                                "method": "partial_match",
                                                "particle": format_particle_info_safe(particle)
                                            })
                                elif hasattr(particle_list, 'name'):
                                    alternatives.append({
                                        "method": "partial_match",
                                        "particle": format_particle_info_safe(particle_list)
                                    })
                    except:
                        pass

                    lookup_result["alternatives"] = alternatives[:5]  # Limit to 5 alternatives

            return [types.TextContent(type="text", text=json.dumps(lookup_result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to perform safe particle lookup: {str(e)}"},
                        indent=2,
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": f"Unknown error tool: {name}"})
            )
        ] 
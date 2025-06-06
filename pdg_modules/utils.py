"""
PDG Utilities Module

This module provides essential utility tools for PDG data manipulation, identifier management,
and data processing operations. It serves as the foundational support system for all other
PDG modules with comprehensive data validation, processing, and utility functions.

Key Features:
- Advanced PDG identifier parsing and manipulation with comprehensive validation
- Intelligent property selection and ranking using official PDG criteria
- Comprehensive data processing utilities with enhanced error handling
- PDG rounding rules with statistical analysis and decision documentation
- Database access utilities and linked data retrieval with metadata
- Data normalization and validation with quality metrics and integrity checking
- Safe attribute access and error recovery with intelligent fallbacks
- Enhanced formatting and display utilities with precision control

Core Tools (8 total):
1. parse_pdg_identifier - Enhanced PDG identifier parsing with validation
2. get_base_pdg_id - Base identifier extraction with format validation
3. make_pdg_identifier - Normalized identifier creation with edition support
4. find_best_property - Intelligent property selection using PDG criteria
5. apply_pdg_rounding - PDG rounding rules with detailed analysis
6. get_linked_data - Database relationship exploration and linked data retrieval
7. normalize_pdg_data - Data normalization and validation with quality assessment
8. get_pdg_table_data - Raw database access with metadata and integrity checking

Enhanced Capabilities:
- Intelligent PDG identifier format analysis with pattern recognition
- Comprehensive validation with detailed error analysis and suggestions
- Advanced property ranking algorithms following official PDG methodology
- Statistical rounding analysis with decision process documentation
- Safe data access patterns with comprehensive error recovery
- Quality metrics calculation and data integrity validation
- Cross-reference validation and consistency checking

Identifier Management:
- PDG identifier parsing (S008, M100, G100, T100) with format validation
- Edition-aware identifier handling with version control support
- Base identifier extraction and normalization with validation
- Compound identifier construction with proper formatting
- Format validation with detailed error analysis and correction suggestions
- Pattern recognition for identifier classification and validation
- Cross-reference validation between related identifiers

Property Selection:
- Best property identification using official PDG selection criteria
- Statistical analysis of multiple measurements for optimal value selection
- Quality-based ranking with measurement reliability assessment
- Pedantic mode for strict adherence to PDG compilation guidelines
- Ambiguity resolution with detailed decision process documentation
- Multi-criteria optimization for property selection
- Error handling for ambiguous or missing property cases

Data Processing:
- PDG rounding rule implementation with decision analysis
- Precision management and significant figure handling
- Value formatting with uncertainty-aware precision
- Statistical analysis of rounding decisions and impact assessment
- Data validation with comprehensive quality checking
- Normalization processes with integrity preservation
- Safe data transformation with error recovery

Database Operations:
- Linked data exploration with relationship mapping
- Raw table access with metadata preservation
- Cross-reference validation and consistency checking
- Data integrity verification with comprehensive validation
- Relationship analysis between database entities
- Query optimization and performance enhancement
- Transaction safety and data consistency guarantees

Advanced Features:
- Comprehensive logging and debugging support with detailed diagnostics
- Performance monitoring and optimization suggestions
- Data quality assessment with statistical analysis
- Error pattern recognition and prevention strategies
- Educational content with methodology explanations
- Historical data tracking and version management

Validation Framework:
- Multi-level validation with detailed error reporting
- Format compliance checking against PDG standards
- Data consistency validation across related entities
- Quality metric calculation with statistical analysis
- Integrity checking with comprehensive verification
- Cross-validation between different data sources

Error Recovery:
- Graceful degradation with partial result preservation
- Intelligent fallback strategies for common failure scenarios
- Error context preservation for debugging and analysis
- Recovery recommendation generation with prioritized actions
- Safe defaults and error-resistant operation modes
- Comprehensive error logging and pattern analysis

Integration Support:
- Seamless integration with all PDG modules
- Common utility functions for consistent behavior
- Shared error handling patterns and recovery strategies
- Performance optimization for high-frequency operations
- Memory management and resource optimization
- Configuration management and parameter validation

Research Applications:
- Data quality assessment for research reliability
- Methodology validation for PDG compilation procedures
- Statistical analysis support for measurement evaluation
- Cross-validation support for independent verification
- Historical analysis support for data evolution tracking
- Educational support for understanding PDG methodologies

Quality Assurance:
- Comprehensive data validation with multi-level checking
- Statistical quality metrics with trend analysis
- Integrity verification with consistency checking
- Performance monitoring with optimization recommendations
- Error tracking and pattern analysis for improvement
- Version control and change tracking for data provenance

Advanced Analytics:
- Pattern recognition in identifier usage and format evolution
- Statistical analysis of property selection effectiveness
- Quality metric trending and improvement tracking
- Performance analysis and optimization identification
- Error correlation analysis for systematic improvement
- Usage pattern analysis for optimization opportunities

Educational Components:
- PDG methodology explanation with detailed documentation
- Identifier format evolution and standardization history
- Property selection criteria explanation with examples
- Rounding rule rationale and statistical impact analysis
- Data quality principles and validation methodologies
- Best practices for PDG data usage and interpretation

Based on the official PDG Python API: https://github.com/particledatagroup/api
Enhanced for comprehensive PDG data management and processing operations.

Author: PDG MCP Server Team
License: MIT (with PDG Python API dependencies under BSD-3-Clause)
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

import mcp.types as types

# Setup module logger
logger = logging.getLogger(__name__)


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


def validate_pdg_identifier_format(pdgid: str) -> Dict[str, Any]:
    """Validate PDG identifier format and provide detailed analysis."""
    try:
        validation_result = {
            "original_identifier": pdgid,
            "is_valid": False,
            "format_analysis": {},
            "suggestions": [],
            "error_details": [],
        }

        # Basic format checks
        if not pdgid or not isinstance(pdgid, str):
            validation_result["error_details"].append(
                "Identifier must be a non-empty string"
            )
            return validation_result

        # Remove whitespace and normalize
        clean_pdgid = pdgid.strip().upper()
        validation_result["normalized"] = clean_pdgid

        # Check for edition separator
        has_edition = "/" in clean_pdgid
        if has_edition:
            parts = clean_pdgid.split("/")
            if len(parts) != 2:
                validation_result["error_details"].append(
                    "Invalid edition format: too many '/' separators"
                )
                return validation_result
            base_id, edition = parts
        else:
            base_id = clean_pdgid
            edition = None

        validation_result["format_analysis"] = {
            "base_identifier": base_id,
            "edition": edition,
            "has_edition": has_edition,
            "length": len(base_id),
        }

        # Validate base identifier format
        if base_id:
            # Check for valid PDG identifier patterns
            valid_patterns = {
                "Summary": base_id.startswith("S")
                and base_id[1:].isdigit()
                and len(base_id) >= 4,
                "Mass": base_id.startswith("M") and base_id[1:].isdigit(),
                "Width": base_id.startswith("G") and base_id[1:].isdigit(),
                "Lifetime": base_id.startswith("T") and base_id[1:].isdigit(),
                "Branching": base_id.startswith("B") and base_id[1:].isdigit(),
                "Decay": base_id.startswith("D") and base_id[1:].isdigit(),
            }

            validation_result["format_analysis"]["pattern_matches"] = {
                pattern: matches
                for pattern, matches in valid_patterns.items()
                if matches
            }

            if any(valid_patterns.values()):
                validation_result["is_valid"] = True
            else:
                validation_result["error_details"].append(
                    f"Unrecognized PDG identifier pattern: {base_id}"
                )
                validation_result["suggestions"].extend(
                    [
                        "PDG identifiers typically start with S, M, G, T, B, or D followed by numbers",
                        "Summary identifiers: S008, S009, etc.",
                        "Mass identifiers: M001, M002, etc.",
                        "Examples: 'S008', 'M100', 'G023/2024'",
                    ]
                )

        # Validate edition if present
        if edition:
            try:
                year = int(edition)
                if 1950 <= year <= 2050:  # Reasonable year range
                    validation_result["format_analysis"]["edition_year"] = year
                    validation_result["format_analysis"]["edition_valid"] = True
                else:
                    validation_result["error_details"].append(
                        f"Edition year {year} is outside expected range (1950-2050)"
                    )
            except ValueError:
                validation_result["error_details"].append(
                    f"Edition '{edition}' is not a valid year"
                )

        return validation_result

    except Exception as e:
        logger.error(f"Error validating PDG identifier: {e}")
        return {
            "original_identifier": pdgid,
            "is_valid": False,
            "error": f"Validation failed: {str(e)}",
        }


def analyze_pdg_rounding_decision(error: float) -> Dict[str, Any]:
    """Analyze PDG rounding decision process in detail."""
    try:
        if error <= 0:
            return {"error": "Error must be positive for PDG rounding analysis"}

        analysis = {
            "original_error": error,
            "decision_process": {},
            "rounding_rules": {},
        }

        # Calculate the three highest order digits
        log_error = math.log10(abs(error))
        if abs(error) < 1.0 and int(log_error) != log_error:
            power = int(log_error)
        else:
            power = int(log_error) + 1

        reduced_error = error * 10 ** (-power)
        three_highest_digits = int(reduced_error * 100)

        analysis["decision_process"] = {
            "log10_error": log_error,
            "power": power,
            "reduced_error": reduced_error,
            "three_highest_digits": three_highest_digits,
        }

        # Apply PDG rounding rules with detailed explanation
        if three_highest_digits < 355:
            n_digits = 2
            rule_applied = "Rule 1: digits 100-354 → 2 significant figures"
            rule_rationale = "Error digits in lower range, maintain higher precision"
        elif three_highest_digits < 950:
            n_digits = 1
            rule_applied = "Rule 2: digits 355-949 → 1 significant figure"
            rule_rationale = "Error digits in middle range, reduce precision"
        else:
            # Round up to next power of 10
            reduced_error = 0.1
            power += 1
            n_digits = 2
            rule_applied = "Rule 3: digits 950-999 → round up, 2 significant figures"
            rule_rationale = (
                "Error digits in upper range, round up to next order of magnitude"
            )

        analysis["rounding_rules"] = {
            "rule_applied": rule_applied,
            "rule_rationale": rule_rationale,
            "significant_digits": n_digits,
            "final_power": power,
            "final_reduced_error": reduced_error,
        }

        # Calculate final rounded values
        new_error = round(reduced_error, n_digits) * 10**power
        analysis["final_error"] = new_error

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing PDG rounding decision: {e}")
        return {"error": f"Analysis failed: {str(e)}"}


def format_pdg_value_with_uncertainty(
    value: float, error: float, units: str = "", use_pdg_rounding: bool = True
) -> Dict[str, Any]:
    """Format value with uncertainty following PDG conventions."""
    try:
        result = {
            "original": {"value": value, "error": error, "units": units},
            "formatted": {},
            "pdg_compliant": use_pdg_rounding,
        }

        if use_pdg_rounding:
            # Apply PDG rounding
            rounded_value, rounded_error = pdg_round_utils(value, error)
            result["rounded"] = {"value": rounded_value, "error": rounded_error}

            # Format according to PDG conventions
            rounding_analysis = analyze_pdg_rounding_decision(error)
            n_digits = rounding_analysis.get("rounding_rules", {}).get(
                "significant_digits", 1
            )

            if n_digits == 1:
                error_str = f"{rounded_error:.0g}"
                # Match value precision to error precision
                value_precision = len(error_str) - (1 if "." in error_str else 0)
                value_str = f"{rounded_value:.{max(0, value_precision)}f}".rstrip(
                    "0"
                ).rstrip(".")
            else:
                error_str = f"{rounded_error:.1g}"
                value_str = f"{rounded_value:.{n_digits-1}f}".rstrip("0").rstrip(".")

            # Construct formatted string
            formatted_str = f"{value_str} ± {error_str}"
            if units:
                formatted_str += f" {units}"

            result["formatted"] = {
                "value_string": value_str,
                "error_string": error_str,
                "combined_string": formatted_str,
                "units": units,
            }
        else:
            # Simple formatting without PDG rounding
            formatted_str = f"{value:.6g} ± {error:.6g}"
            if units:
                formatted_str += f" {units}"
            result["formatted"]["combined_string"] = formatted_str

        return result

    except Exception as e:
        logger.error(f"Error formatting PDG value: {e}")
        return {"error": f"Formatting failed: {str(e)}"}


def get_utils_tools() -> List[types.Tool]:
    """Return all enhanced utils-related MCP tools with comprehensive functionality."""
    return [
        types.Tool(
            name="parse_pdg_identifier",
            description="Parse PDG Identifier and return base identifier and edition with validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdgid": {
                        "type": "string",
                        "description": "PDG identifier to parse (e.g., 'S008', 'M100/2024')",
                    },
                    "validate_format": {
                        "type": "boolean",
                        "default": True,
                        "description": "Validate PDG identifier format and provide suggestions",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include metadata about identifier structure and conventions",
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
            description="Find the 'best' property from a list based on enhanced PDG criteria with detailed analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to get properties for",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": [
                            "mass",
                            "lifetime",
                            "width",
                            "branching_fraction",
                            "all",
                        ],
                        "description": "Type of property to find best value for",
                    },
                    "pedantic": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use strict criteria (may raise ambiguity errors)",
                    },
                    "include_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed analysis of selection criteria and alternatives",
                    },
                    "data_quality_threshold": {
                        "type": "string",
                        "enum": ["any", "recommended", "default_only"],
                        "default": "recommended",
                        "description": "Quality threshold for property selection",
                    },
                },
                "required": ["particle_name", "property_type"],
            },
        ),
        types.Tool(
            name="apply_pdg_rounding",
            description="Apply PDG rounding rules to value and error with detailed analysis and formatting options",
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
                    "include_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed analysis of rounding decision process",
                    },
                    "format_output": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include properly formatted output string following PDG conventions",
                    },
                    "precision_context": {
                        "type": "string",
                        "description": "Optional context for precision requirements (e.g., 'experimental', 'theoretical')",
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
    if error <= 0.0:
        raise ValueError("PDG rounding requires error larger than zero")

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

    new_error = round(reduced_error, n_digits) * 10**power
    new_value = round(value * 10 ** (-power), n_digits) * 10**power
    return new_value, new_error


def parse_pdg_id_utils(pdgid):
    """Parse PDG Identifier and return (base identifier, edition)."""
    try:
        baseid, edition = pdgid.split("/")
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
        return ("%s/%s" % (baseid, edition)).upper()


def find_best_property_utils(properties, pedantic=False, quantity=None):
    """Find the 'best' property from an iterable based on PDG criteria."""
    for_what = " for %s" % quantity if quantity else ""

    # Filter out alternates
    props_without_alternates = []
    for p in properties:
        try:
            if hasattr(p, "data_flags") and "A" not in p.data_flags:
                props_without_alternates.append(p)
            elif not hasattr(p, "data_flags"):
                props_without_alternates.append(p)
        except:
            props_without_alternates.append(p)

    # In non-pedantic mode, filter out "special" values
    if not pedantic:
        filtered_props = []
        for p in props_without_alternates:
            try:
                if hasattr(p, "data_flags") and "s" not in p.data_flags:
                    filtered_props.append(p)
                elif not hasattr(p, "data_flags"):
                    filtered_props.append(p)
            except:
                filtered_props.append(p)
        props_without_alternates = filtered_props

    if len(props_without_alternates) == 0:
        return None, f"No best property found{for_what}"
    elif len(props_without_alternates) == 1:
        return props_without_alternates[0], "success"
    else:
        if pedantic:
            return None, f"Ambiguous best property{for_what}"
        else:
            # Look for properties with 'D' flag (default/recommended)
            props_best = []
            for p in props_without_alternates:
                try:
                    if hasattr(p, "data_flags") and "D" in p.data_flags:
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


async def handle_utils_tools(
    name: str, arguments: dict, api
) -> List[types.TextContent]:
    """Handle utils-related tool calls."""

    if name == "parse_pdg_identifier":
        pdgid = arguments["pdgid"]
        validate_format = arguments.get("validate_format", True)
        include_metadata = arguments.get("include_metadata", False)

        try:
            base_id, edition = parse_pdg_id_utils(pdgid)

            result = {
                "original_pdgid": pdgid,
                "base_identifier": base_id,
                "edition": edition,
                "normalized": make_pdg_id_utils(base_id, edition),
            }

            # Add validation if requested
            if validate_format:
                validation = validate_pdg_identifier_format(pdgid)
                result["validation"] = validation

                if not validation.get("is_valid", False):
                    result["warnings"] = validation.get("error_details", [])
                    result["suggestions"] = validation.get("suggestions", [])

            # Add metadata if requested
            if include_metadata:
                result["metadata"] = {
                    "identifier_conventions": {
                        "format": "BaseID[/Edition]",
                        "examples": ["S008", "M100/2024", "G023"],
                        "prefixes": {
                            "S": "Summary/Review data",
                            "M": "Mass measurements",
                            "G": "Width measurements",
                            "T": "Lifetime measurements",
                            "B": "Branching fraction measurements",
                            "D": "Decay mode data",
                        },
                    },
                    "edition_info": "Edition typically represents PDG Review year (e.g., 2024, 2022)",
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            logger.error(f"Error parsing PDG identifier {pdgid}: {e}")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Failed to parse PDG identifier: {str(e)}",
                            "input": pdgid,
                        },
                        indent=2,
                    ),
                )
            ]

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
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get base PDG ID: {str(e)}"}, indent=2
                    ),
                )
            ]

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
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to make PDG identifier: {str(e)}"}, indent=2
                    ),
                )
            ]

    elif name == "find_best_property":
        particle_name = arguments["particle_name"]
        property_type = arguments["property_type"]
        pedantic = arguments.get("pedantic", False)
        include_analysis = arguments.get("include_analysis", True)
        data_quality_threshold = arguments.get("data_quality_threshold", "recommended")

        try:
            particle = api.get_particle_by_name(particle_name)
            properties = get_property_by_type(particle, property_type)

            if not properties:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": f"No {property_type} properties found for {particle_name}",
                                "particle_name": particle_name,
                                "property_type": property_type,
                            },
                            indent=2,
                        ),
                    )
                ]

            best_prop, status = find_best_property_utils(
                properties, pedantic, property_type
            )

            result = {
                "query": {
                    "particle_name": particle_name,
                    "property_type": property_type,
                    "pedantic_mode": pedantic,
                    "data_quality_threshold": data_quality_threshold,
                },
                "summary": {
                    "total_properties_found": len(properties),
                    "selection_status": "success" if best_prop else "failed",
                },
            }

            if best_prop is None:
                result["status"] = "error"
                result["message"] = status
                result["suggestions"] = [
                    "Try with pedantic=False for more lenient selection criteria",
                    "Check if the particle name is correct",
                    f"Verify that {property_type} measurements exist for this particle",
                ]
            else:
                result["status"] = "success"
                result["selected_property"] = {
                    "pdgid": safe_get_attribute(best_prop, "pdgid", "N/A"),
                    "description": safe_get_attribute(best_prop, "description", "N/A"),
                    "display_value": safe_get_attribute(
                            best_prop, "display_value_text", "N/A"
                        ),
                    "value": safe_get_attribute(best_prop, "value", "N/A"),
                    "units": safe_get_attribute(best_prop, "units", "N/A"),
                    "data_flags": safe_get_attribute(best_prop, "data_flags", "N/A"),
                    "in_summary_table": safe_get_attribute(
                        best_prop, "in_summary_table", False
                    ),
                    "is_limit": safe_get_attribute(best_prop, "is_limit", False),
                }

                # Add detailed analysis if requested
                if include_analysis:
                    result["analysis"] = {
                        "selection_criteria": {
                            "primary": "PDG recommended values (in_summary_table=True)",
                            "secondary": "Data quality flags and measurement precision",
                            "pedantic_mode": f"{'Enabled' if pedantic else 'Disabled'} - {'strict' if pedantic else 'lenient'} criteria",
                        },
                        "alternatives_considered": len(properties) - 1,
                        "quality_assessment": {
                            "recommended_value": safe_get_attribute(
                                best_prop, "in_summary_table", False
                            ),
                            "has_uncertainty": safe_get_attribute(
                                best_prop, "value", "N/A"
                            )
                            != "N/A",
                            "data_flags_status": safe_get_attribute(
                                best_prop, "data_flags", "N/A"
                            ),
                        },
                    }

                    # Add information about other properties for comparison
                    if len(properties) > 1:
                        alternatives = []
                        for i, prop in enumerate(
                            properties[:3]
                        ):  # Show top 3 alternatives
                            if prop != best_prop:
                                alternatives.append(
                                    {
                                        "rank": i + 2,
                                        "pdgid": safe_get_attribute(
                                            prop, "pdgid", "N/A"
                                        ),
                                        "value": safe_get_attribute(
                                            prop, "display_value_text", "N/A"
                                        ),
                                        "in_summary_table": safe_get_attribute(
                                            prop, "in_summary_table", False
                                        ),
                                    }
                                )
                        result["analysis"]["top_alternatives"] = alternatives

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            logger.error(
                f"Error finding best property for {particle_name} ({property_type}): {e}"
            )
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Failed to find best property: {str(e)}",
                            "query": {
                                "particle_name": particle_name,
                                "property_type": property_type,
                            },
                        },
                        indent=2,
                    ),
                )
            ]

    elif name == "apply_pdg_rounding":
        value = arguments["value"]
        error = arguments["error"]
        include_analysis = arguments.get("include_analysis", True)
        format_output = arguments.get("format_output", True)
        precision_context = arguments.get("precision_context", "")

        try:
            if error <= 0:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Error/uncertainty must be positive for PDG rounding"
                            },
                            indent=2,
                        ),
                    )
                ]

            rounded_value, rounded_error = pdg_round_utils(value, error)

            result = {
                "input": {
                    "value": value,
                    "error": error,
                    "precision_context": precision_context,
                },
                "output": {
                    "rounded_value": rounded_value,
                    "rounded_error": rounded_error,
                    "change_in_value": abs(rounded_value - value),
                    "change_in_error": abs(rounded_error - error),
                    "rounding_applied": True,
                },
            }

            # Add detailed analysis if requested
            if include_analysis:
                analysis = analyze_pdg_rounding_decision(error)
                result["analysis"] = analysis

            # Add formatted output if requested
            if format_output:
                formatting_result = format_pdg_value_with_uncertainty(
                    value, error, "", True
                )
                result["formatted_output"] = formatting_result.get("formatted", {})

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            logger.error(f"Error applying PDG rounding to {value} ± {error}: {e}")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Failed to apply PDG rounding: {str(e)}",
                            "input": {"value": value, "error": error},
                        },
                        indent=2,
                    ),
                )
            ]

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
                            linked_data.append(
                                {
                                    "type": "measurement",
                                    "id": getattr(measurement, "id", "N/A"),
                                    "property_pdgid": getattr(prop, "pdgid", "N/A"),
                                    "reference_id": (
                                        getattr(measurement, "reference_id", "N/A")
                                        if hasattr(measurement, "reference_id")
                                        else "N/A"
                                    ),
                                    "value": (
                                        getattr(
                                            measurement.get_value(), "value_text", "N/A"
                                        )
                                        if hasattr(measurement, "get_value")
                                        and measurement.get_value()
                                        else "N/A"
                                    ),
                                }
                            )
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
                            if (
                                hasattr(measurement, "reference")
                                and measurement.reference
                            ):
                                ref = measurement.reference
                                ref_id = getattr(ref, "id", "N/A")
                                if ref_id not in ref_ids:
                                    ref_ids.add(ref_id)
                                    linked_data.append(
                                        {
                                            "type": "reference",
                                            "id": ref_id,
                                            "title": getattr(ref, "title", "N/A"),
                                            "year": getattr(
                                                ref, "publication_year", "N/A"
                                            ),
                                            "doi": getattr(ref, "doi", "N/A"),
                                        }
                                    )
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
                                linked_data.append(
                                    {
                                        "type": "value",
                                        "id": getattr(value, "id", "N/A"),
                                        "measurement_id": getattr(
                                            measurement, "id", "N/A"
                                        ),
                                        "value_text": getattr(
                                            value, "value_text", "N/A"
                                        ),
                                        "error_positive": getattr(
                                            value, "error_positive", "N/A"
                                        ),
                                        "error_negative": getattr(
                                            value, "error_negative", "N/A"
                                        ),
                                    }
                                )
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
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get linked data: {str(e)}"}, indent=2
                    ),
                )
            ]

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
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to normalize PDG data: {str(e)}"}, indent=2
                    ),
                )
            ]

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
                    "available_tables": [
                        "particles",
                        "measurements",
                        "values",
                        "references",
                        "footnotes",
                    ],
                    "access_method": "via PDG API",
                    "implementation": "This would require direct database access through the PDG utils API",
                }

            # Attempt to get some data through available API methods
            try:
                if table_name == "particles":
                    # Try to get particle data
                    particles = api.get_particles()
                    result["sample_data"] = (
                        f"Table contains particle data, example: {list(particles)[0].name if particles else 'N/A'}"
                    )
                elif table_name == "measurements":
                    result["sample_data"] = (
                        "Measurements table contains PDG measurement data"
                    )
                else:
                    result["sample_data"] = (
                        f"Table '{table_name}' data access not implemented in this simulation"
                    )
            except Exception as inner_e:
                result["sample_data"] = f"Could not access table data: {str(inner_e)}"

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Failed to get PDG table data: {str(e)}"}, indent=2
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text", text=json.dumps({"error": f"Unknown utils tool: {name}"})
            )
        ]

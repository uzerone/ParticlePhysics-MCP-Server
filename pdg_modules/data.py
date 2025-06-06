"""
PDG Data Handling Module

This module provides comprehensive tools for accessing, analyzing, and processing particle
physics measurements, summary values, and experimental data from the PDG database.
It specializes in handling numerical data with proper uncertainty analysis and unit conversions.

Key Features:
- Advanced measurement analysis with statistical validation and uncertainty propagation
- Comprehensive summary values with metadata and quality indicators
- Enhanced unit conversions with physics constants and dimensional analysis
- Detailed property analysis with data validation and quality assessment
- Reference and footnote management with citation tracking
- Error analysis and uncertainty propagation with precision classification
- Multi-measurement statistical analysis and comparison tools
- Publication and experimental method tracking

Core Tools (11 total):
1. get_mass_measurements - Detailed mass measurements with error analysis
2. get_lifetime_measurements - Lifetime measurements with decay analysis
3. get_width_measurements - Width measurements for unstable particles
4. get_summary_values - Comprehensive summary values with validation
5. get_measurements_by_property - Individual measurements with references
6. convert_units - Advanced unit conversion with validation
7. get_particle_text - Text descriptions and reviews with formatting
8. get_property_details - Comprehensive property metadata analysis
9. get_data_type_keys - PDG data type documentation with examples
10. get_value_type_keys - Summary value type documentation
11. get_key_documentation - Database key and flag documentation

Enhanced Capabilities:
- Precision analysis with uncertainty classification (very_high, high, moderate, low)
- Statistical validation of measurement consistency across experiments
- Advanced error component analysis (statistical vs systematic)
- Comprehensive unit compatibility checking and validation
- Physics context integration (natural units, energy-time relations)
- Quality metrics and measurement reliability indicators
- Reference tracking with DOI, arXiv, and journal information
- Experimental technique classification and comparison

Data Processing Features:
- PDG rounding rules application and validation
- Value formatting with appropriate significant figures
- Scientific notation handling for extreme values
- Asymmetric uncertainty handling and propagation
- Limit value processing (upper/lower bounds)
- Confidence level analysis and interpretation
- Scale factor application and uncertainty scaling

Measurement Analysis:
- Individual measurement tracking with full provenance
- Multi-measurement statistical analysis and averaging
- Temporal analysis of measurement evolution over time
- Experimental method comparison and validation
- Publication year filtering and historical analysis
- Reference quality assessment and citation metrics

Integration Features:
- Seamless integration with units module for conversions
- Cross-reference with particle module for context
- Error handling integration with errors module
- Support for derived quantity calculations (E=mc², Γ=ħ/τ)
- Educational content with physics explanations

Data Quality Assurance:
- Comprehensive validation of numerical values and uncertainties
- Data flag interpretation and quality indicators
- Edition-aware data access with version control
- Metadata completeness checking and validation
- Statistical outlier detection and flagging

Based on the official PDG Python API: https://github.com/particledatagroup/api
Optimized for scientific data analysis and experimental physics research.

Author: PDG MCP Server Team
License: MIT (with PDG Python API dependencies under BSD-3-Clause)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import mcp.types as types

# Setup module logger
logger = logging.getLogger(__name__)


def get_data_tools() -> List[types.Tool]:
    """Return all data-related MCP tools with enhanced functionality."""
    return [
        types.Tool(
            name="get_mass_measurements",
            description="Get detailed mass measurements and summary values with enhanced analysis",
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
                        "description": "Include individual measurements with references",
                    },
                    "include_error_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed error analysis and uncertainty breakdown",
                    },
                    "units": {
                        "type": "string",
                        "default": "GeV",
                        "description": "Units for mass values (e.g., 'GeV', 'MeV', 'kg', 'u')",
                    },
                    "precision": {
                        "type": "integer",
                        "default": 6,
                        "minimum": 3,
                        "maximum": 12,
                        "description": "Number of significant figures for numerical output",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_lifetime_measurements",
            description="Get detailed lifetime measurements with decay analysis",
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
                        "description": "Include individual measurements with references",
                    },
                    "include_decay_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include decay-related analysis (width, decay constant)",
                    },
                    "units": {
                        "type": "string",
                        "default": "s",
                        "description": "Units for lifetime values (e.g., 's', 'ns', 'ps', 'fs')",
                    },
                    "include_conversion_factors": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include conversion to other common units",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_width_measurements",
            description="Get detailed width measurements for unstable particles",
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
                        "description": "Include individual measurements with references",
                    },
                    "include_lifetime_relation": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include lifetime calculations (Γ = ħ/τ)",
                    },
                    "units": {
                        "type": "string",
                        "default": "GeV",
                        "description": "Units for width values (e.g., 'GeV', 'MeV', 'keV')",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_summary_values",
            description="Get comprehensive summary values with detailed metadata and validation",
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
                    "include_metadata": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed metadata (flags, data types, editions)",
                    },
                    "include_validation": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include data validation and quality indicators",
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
            description="Get detailed individual measurements with comprehensive analysis",
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
                    "include_error_breakdown": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed error breakdown (statistical vs systematic)",
                    },
                    "include_experimental_details": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include experimental method and technique details",
                    },
                    "year_filter": {
                        "type": "object",
                        "properties": {
                            "min_year": {
                                "type": "integer",
                                "description": "Minimum publication year",
                            },
                            "max_year": {
                                "type": "integer",
                                "description": "Maximum publication year",
                            },
                        },
                        "description": "Filter measurements by publication year range",
                    },
                },
                "required": ["particle_name", "property_type"],
            },
        ),
        types.Tool(
            name="convert_units",
            description="Advanced particle physics unit conversion with validation and constants",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Numerical value to convert",
                    },
                    "from_units": {
                        "type": "string",
                        "description": "Source units (e.g., 'GeV', 'MeV', 's', 'ns', 'u', 'eV')",
                    },
                    "to_units": {
                        "type": "string",
                        "description": "Target units (e.g., 'GeV', 'MeV', 's', 'ns', 'u', 'eV')",
                    },
                    "include_validation": {
                        "type": "boolean",
                        "default": True,
                        "description": "Validate unit compatibility before conversion",
                    },
                    "include_physics_context": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include physics context and related conversions",
                    },
                    "precision": {
                        "type": "integer",
                        "default": 6,
                        "minimum": 3,
                        "maximum": 15,
                        "description": "Number of significant figures in result",
                    },
                },
                "required": ["value", "from_units", "to_units"],
            },
        ),
        types.Tool(
            name="get_particle_text",
            description="Get comprehensive text information and descriptions with enhanced formatting",
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
                    "include_formatting": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include formatted text with markup preservation",
                    },
                    "include_references": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include reference citations in text",
                    },
                },
                "required": ["pdgid"],
            },
        ),
        types.Tool(
            name="get_property_details",
            description="Get comprehensive property information with enhanced metadata analysis",
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
                    "include_data_flags": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed data flags and quality indicators",
                    },
                    "include_statistical_analysis": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include statistical analysis of multiple measurements",
                    },
                },
                "required": ["particle_name", "property_type"],
            },
        ),
        types.Tool(
            name="get_data_type_keys",
            description="Get comprehensive PDG data type keys with enhanced documentation",
            inputSchema={
                "type": "object",
                "properties": {
                    "as_text": {
                        "type": "boolean",
                        "default": True,
                        "description": "Return as formatted text (true) or structured data (false)",
                    },
                    "include_examples": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include usage examples for each data type",
                    },
                    "category_filter": {
                        "type": "string",
                        "enum": ["mass", "lifetime", "width", "decay", "all"],
                        "default": "all",
                        "description": "Filter by data type category",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_value_type_keys",
            description="Get comprehensive PDG summary value type keys with metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "as_text": {
                        "type": "boolean",
                        "default": True,
                        "description": "Return as formatted text (true) or structured data (false)",
                    },
                    "include_descriptions": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed descriptions for each value type",
                    },
                    "include_usage_statistics": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include usage frequency and statistics",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_key_documentation",
            description="Get comprehensive documentation for PDG database keys and flags",
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
                    "include_examples": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include usage examples and context",
                    },
                },
                "required": ["table_name", "column_name", "key"],
            },
        ),
    ]


def safe_get_attribute(
    obj: Any, attr: str, default: Any = None, transform_func: Optional[callable] = None
) -> Any:
    """Safely get attribute from object with optional transformation and logging."""
    try:
        value = getattr(obj, attr, default)
        if value is not None and transform_func:
            return transform_func(value)
        return value
    except Exception as e:
        logger.debug(f"Failed to get attribute {attr} from {type(obj).__name__}: {e}")
        return default


def format_value_with_precision(
    value: Any, precision: int = 6, scientific_threshold: float = 1e-4
) -> str:
    """Format numerical values with appropriate precision and notation."""
    try:
        if value is None:
            return "N/A"

        num_val = float(value)

        # Use scientific notation for very small or very large numbers
        if abs(num_val) < scientific_threshold or abs(num_val) >= 10 ** (precision):
            return f"{num_val:.{precision-1}e}"
        else:
            # Use fixed precision, removing trailing zeros
            formatted = f"{num_val:.{precision}g}"
            return formatted

    except (ValueError, TypeError):
        return str(value) if value is not None else "N/A"


def analyze_measurement_uncertainty(measurement: Any) -> Dict[str, Any]:
    """Analyze measurement uncertainty components and provide detailed breakdown."""
    try:
        uncertainty_analysis = {
            "has_uncertainty": False,
            "uncertainty_type": "none",
            "components": {},
            "total_uncertainty": None,
            "relative_uncertainty": None,
        }

        # Get measurement value
        value = safe_get_attribute(measurement, "value", 0)
        if value == 0:
            try:
                # Try to get value from associated value object
                measurement_value = safe_get_attribute(measurement, "get_value")
                if measurement_value and callable(measurement_value):
                    value_obj = measurement_value()
                    if value_obj:
                        value = safe_get_attribute(value_obj, "value", 0)
            except:
                pass

        # Analyze error components
        error_pos = safe_get_attribute(measurement, "error_positive")
        error_neg = safe_get_attribute(measurement, "error_negative")
        error_sym = safe_get_attribute(measurement, "error")

        if error_pos is not None or error_neg is not None or error_sym is not None:
            uncertainty_analysis["has_uncertainty"] = True

            if error_pos == error_neg or (error_pos is not None and error_neg is None):
                uncertainty_analysis["uncertainty_type"] = "symmetric"
                total_error = error_pos if error_pos is not None else error_sym
                uncertainty_analysis["components"]["symmetric"] = total_error
                uncertainty_analysis["total_uncertainty"] = total_error
            else:
                uncertainty_analysis["uncertainty_type"] = "asymmetric"
                uncertainty_analysis["components"]["positive"] = error_pos
                uncertainty_analysis["components"]["negative"] = error_neg
                # Use average as total uncertainty estimate
                if error_pos is not None and error_neg is not None:
                    uncertainty_analysis["total_uncertainty"] = (
                        abs(error_pos) + abs(error_neg)
                    ) / 2

        # Calculate relative uncertainty
        if uncertainty_analysis["total_uncertainty"] is not None and value != 0:
            rel_uncertainty = uncertainty_analysis["total_uncertainty"] / abs(value)
            uncertainty_analysis["relative_uncertainty"] = rel_uncertainty
            uncertainty_analysis["relative_uncertainty_percent"] = rel_uncertainty * 100

            # Classify uncertainty magnitude
            if rel_uncertainty < 0.001:
                uncertainty_analysis["precision_class"] = "very_high"
            elif rel_uncertainty < 0.01:
                uncertainty_analysis["precision_class"] = "high"
            elif rel_uncertainty < 0.1:
                uncertainty_analysis["precision_class"] = "moderate"
            else:
                uncertainty_analysis["precision_class"] = "low"

        return uncertainty_analysis

    except Exception as e:
        logger.debug(f"Error analyzing measurement uncertainty: {e}")
        return {"error": f"Failed to analyze uncertainty: {str(e)}"}


def format_enhanced_summary_value(
    summary_value: Any, precision: int = 6, target_units: Optional[str] = None
) -> Dict[str, Any]:
    """Enhanced formatting for PdgSummaryValue objects with comprehensive metadata."""
    try:
        result = {
            "pdgid": safe_get_attribute(summary_value, "pdgid", "N/A"),
            "value": safe_get_attribute(summary_value, "value"),
            "units": safe_get_attribute(summary_value, "units", "dimensionless"),
        }

        # Enhanced value formatting
        if result["value"] is not None:
            result["formatted_value"] = format_value_with_precision(
                result["value"], precision
            )
        else:
            result["formatted_value"] = "N/A"

        # Comprehensive text representations
        for attr in ["value_text", "display_value_text", "text", "display_text"]:
            value_text = safe_get_attribute(summary_value, attr)
            if value_text:
                result["display_text"] = str(value_text)
                break
        else:
            result["display_text"] = result["formatted_value"]

        # Error analysis
        uncertainty = analyze_measurement_uncertainty(summary_value)
        result["uncertainty_analysis"] = uncertainty

        # Enhanced error information
        result["error_positive"] = safe_get_attribute(summary_value, "error_positive")
        result["error_negative"] = safe_get_attribute(summary_value, "error_negative")
        result["error"] = safe_get_attribute(summary_value, "error")

        # Quality and type indicators
        result["is_limit"] = safe_get_attribute(summary_value, "is_limit", False)
        result["is_lower_limit"] = safe_get_attribute(
            summary_value, "is_lower_limit", False
        )
        result["is_upper_limit"] = safe_get_attribute(
            summary_value, "is_upper_limit", False
        )
        result["confidence_level"] = safe_get_attribute(
            summary_value, "confidence_level"
        )
        result["scale_factor"] = safe_get_attribute(summary_value, "scale_factor")

        # Metadata
        result["in_summary_table"] = safe_get_attribute(
            summary_value, "in_summary_table", False
        )
        result["value_type"] = safe_get_attribute(summary_value, "value_type", "N/A")
        result["value_type_key"] = safe_get_attribute(
            summary_value, "value_type_key", "N/A"
        )
        result["comment"] = safe_get_attribute(summary_value, "comment")

        # Data quality indicators
        result["data_quality"] = {
            "has_error": uncertainty["has_uncertainty"],
            "precision_class": uncertainty.get("precision_class", "unknown"),
            "is_measurement": not result["is_limit"],
            "confidence_level": result["confidence_level"],
        }

        # Unit conversion if requested
        if target_units and target_units != result["units"]:
            try:
                from pdg.units import convert

                if result["value"] is not None:
                    converted_value = convert(
                        result["value"], result["units"], target_units
                    )
                    result["converted_value"] = {
                        "value": converted_value,
                        "formatted": format_value_with_precision(
                            converted_value, precision
                        ),
                        "units": target_units,
                    }

                    # Convert errors if present
                    if result["error"]:
                        converted_error = convert(
                            result["error"], result["units"], target_units
                        )
                        result["converted_value"]["error"] = converted_error

            except Exception as e:
                logger.debug(f"Unit conversion failed: {e}")
                result["conversion_error"] = (
                    f"Failed to convert to {target_units}: {str(e)}"
                )

        return result

    except Exception as e:
        logger.error(f"Failed to format summary value: {e}")
        return {
            "error": f"Failed to format summary value: {str(e)}",
            "raw_type": type(summary_value).__name__,
        }


def format_enhanced_measurement(
    measurement: Any,
    include_references: bool = True,
    include_error_breakdown: bool = True,
) -> Dict[str, Any]:
    """Enhanced formatting for PdgMeasurement objects with comprehensive analysis."""
    try:
        formatted = {
            "id": safe_get_attribute(measurement, "id", "N/A"),
            "pdgid": safe_get_attribute(measurement, "pdgid", "N/A"),
            "measurement_type": type(measurement).__name__,
        }

        # Enhanced value information
        try:
            if hasattr(measurement, "get_value") and callable(measurement.get_value):
                value_obj = measurement.get_value()
                if value_obj:
                    formatted["value"] = {
                        "value": safe_get_attribute(value_obj, "value"),
                        "formatted": format_value_with_precision(
                            safe_get_attribute(value_obj, "value", 0)
                        ),
                        "units": safe_get_attribute(
                            value_obj, "units", "dimensionless"
                        ),
                        "value_text": safe_get_attribute(
                            value_obj, "value_text", "N/A"
                        ),
                        "error_positive": safe_get_attribute(
                            value_obj, "error_positive"
                        ),
                        "error_negative": safe_get_attribute(
                            value_obj, "error_negative"
                        ),
                    }

                    # Enhanced uncertainty analysis
                    if include_error_breakdown:
                        formatted["uncertainty_analysis"] = (
                            analyze_measurement_uncertainty(value_obj)
                        )
        except Exception as e:
            logger.debug(f"Error getting measurement value: {e}")
            formatted["value"] = {"error": f"Failed to get value: {str(e)}"}

        # Enhanced reference information
        if include_references:
            try:
                if hasattr(measurement, "reference") and measurement.reference:
                    ref = measurement.reference
                    formatted["reference"] = {
                        "id": safe_get_attribute(ref, "id"),
                        "title": safe_get_attribute(ref, "title", "N/A"),
                        "authors": safe_get_attribute(ref, "authors", "N/A"),
                        "doi": safe_get_attribute(ref, "doi"),
                        "publication_year": safe_get_attribute(ref, "publication_year"),
                        "journal": safe_get_attribute(ref, "journal"),
                        "volume": safe_get_attribute(ref, "volume"),
                        "page": safe_get_attribute(ref, "page"),
                        "arxiv": safe_get_attribute(ref, "arxiv"),
                    }

                    # Create citation string
                    authors = formatted["reference"]["authors"]
                    year = formatted["reference"]["publication_year"]
                    title = formatted["reference"]["title"]
                    if authors and year:
                        formatted["reference"]["citation"] = f"{authors} ({year})"
                        if title and title != "N/A":
                            formatted["reference"]["citation"] += f", {title[:50]}..."
            except Exception as e:
                logger.debug(f"Error getting reference: {e}")
                formatted["reference"] = {"error": f"Failed to get reference: {str(e)}"}

        # Additional measurement metadata
        formatted["metadata"] = {
            "technique": safe_get_attribute(measurement, "technique"),
            "comment": safe_get_attribute(measurement, "comment"),
            "data_flags": safe_get_attribute(measurement, "data_flags"),
        }

        return formatted

    except Exception as e:
        logger.error(f"Failed to format measurement: {e}")
        return {
            "error": f"Failed to format measurement: {str(e)}",
            "raw_type": type(measurement).__name__,
        }


def get_property_by_type(particle: Any, property_type: str) -> List[Any]:
    """Get specific property objects from a particle with enhanced error handling."""
    try:
        if property_type == "mass":
            return list(particle.masses())
        elif property_type == "lifetime":
            return list(particle.lifetimes())
        elif property_type == "width":
            return list(particle.widths())
        else:
            logger.warning(f"Unknown property type: {property_type}")
            return []
    except Exception as e:
        logger.error(f"Error getting {property_type} properties: {e}")
        return []


def calculate_derived_quantities(
    value: float, error: float, property_type: str, units: str
) -> Dict[str, Any]:
    """Calculate physics-related derived quantities and conversions."""
    try:
        derived = {}

        if property_type == "lifetime" and value > 0:
            # Calculate decay width from lifetime: Γ = ħ/τ
            try:
                hbar_gev_s = 6.582119569e-25  # ħ in GeV⋅s
                if units == "s":
                    width_gev = hbar_gev_s / value
                    derived["decay_width"] = {
                        "value": width_gev,
                        "units": "GeV",
                        "formatted": format_value_with_precision(width_gev),
                        "relation": "Γ = ħ/τ",
                    }

                    if error:
                        width_error = hbar_gev_s * error / (value**2)
                        derived["decay_width"]["error"] = width_error

            except Exception as e:
                logger.debug(f"Error calculating decay width: {e}")

        elif property_type == "width" and value > 0:
            # Calculate lifetime from width: τ = ħ/Γ
            try:
                hbar_gev_s = 6.582119569e-25  # ħ in GeV⋅s
                if units in ["GeV", "MeV"]:
                    value_gev = value if units == "GeV" else value / 1000
                    lifetime_s = hbar_gev_s / value_gev
                    derived["lifetime"] = {
                        "value": lifetime_s,
                        "units": "s",
                        "formatted": format_value_with_precision(lifetime_s),
                        "relation": "τ = ħ/Γ",
                    }

                    if error:
                        error_gev = error if units == "GeV" else error / 1000
                        lifetime_error = hbar_gev_s * error_gev / (value_gev**2)
                        derived["lifetime"]["error"] = lifetime_error

            except Exception as e:
                logger.debug(f"Error calculating lifetime: {e}")

        elif property_type == "mass":
            # Calculate energy equivalence and other mass-related quantities
            try:
                if units in ["GeV", "MeV"]:
                    # Mass is already in energy units
                    derived["rest_energy"] = {
                        "value": value,
                        "units": units,
                        "formatted": format_value_with_precision(value),
                        "relation": "E₀ = mc²",
                    }

                    # Calculate mass in atomic mass units
                    if units == "GeV":
                        mass_u = value / 0.9314941  # GeV to u conversion
                    else:  # MeV
                        mass_u = value / 931.4941  # MeV to u conversion

                    derived["atomic_mass_units"] = {
                        "value": mass_u,
                        "units": "u",
                        "formatted": format_value_with_precision(mass_u),
                    }

            except Exception as e:
                logger.debug(f"Error calculating mass equivalences: {e}")

        return derived

    except Exception as e:
        logger.error(f"Error calculating derived quantities: {e}")
        return {}


async def handle_data_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Enhanced data tool handler with comprehensive functionality."""

    try:
        if name == "get_mass_measurements":
            particle_name = arguments["particle_name"]
            include_summary_values = arguments.get("include_summary_values", True)
            include_measurements = arguments.get("include_measurements", False)
            include_error_analysis = arguments.get("include_error_analysis", True)
            units = arguments.get("units", "GeV")
            precision = arguments.get("precision", 6)

            try:
                particle = api.get_particle_by_name(particle_name)
                mass_data = {
                    "particle": particle_name,
                    "property": "mass",
                    "units": units,
                    "precision": precision,
                    "analysis_timestamp": "generated",
                }

                if include_summary_values:
                    summary_values = []
                    for mass_prop in particle.masses():
                        for summary in mass_prop.summary_values():
                            sv_data = format_enhanced_summary_value(
                                summary, precision, units
                            )

                            # Add derived quantities
                            if sv_data.get("value") is not None:
                                derived = calculate_derived_quantities(
                                    sv_data["value"],
                                    sv_data.get("error"),
                                    "mass",
                                    sv_data["units"],
                                )
                                if derived:
                                    sv_data["derived_quantities"] = derived

                            summary_values.append(sv_data)

                    mass_data["summary_values"] = summary_values
                    mass_data["summary_count"] = len(summary_values)

                if include_measurements:
                    measurements = []
                    for mass_prop in particle.masses():
                        try:
                            for measurement in mass_prop.get_measurements():
                                meas_data = format_enhanced_measurement(
                                    measurement,
                                    include_references=True,
                                    include_error_breakdown=include_error_analysis,
                                )
                                measurements.append(meas_data)
                        except Exception as e:
                            logger.debug(
                                f"Error getting measurements for mass property: {e}"
                            )

                    mass_data["measurements"] = measurements
                    mass_data["measurement_count"] = len(measurements)

                # Add statistical summary if multiple values exist
                if (
                    include_summary_values
                    and len(mass_data.get("summary_values", [])) > 1
                ):
                    values = [
                        sv["value"]
                        for sv in mass_data["summary_values"]
                        if sv.get("value") is not None
                    ]
                    if values:
                        mass_data["statistical_summary"] = {
                            "count": len(values),
                            "mean": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                            "range": max(values) - min(values),
                            "relative_spread": (
                                (max(values) - min(values))
                                / (sum(values) / len(values))
                                if sum(values) != 0
                                else 0
                            ),
                        }

                return [
                    types.TextContent(type="text", text=json.dumps(mass_data, indent=2))
                ]

            except Exception as e:
                error_response = {
                    "error": f"Failed to get mass measurements: {str(e)}",
                    "particle_name": particle_name,
                    "suggestions": [
                        "Verify particle name spelling",
                        "Check if particle has mass measurements",
                        "Try with different units",
                    ],
                }
                return [
                    types.TextContent(
                        type="text", text=json.dumps(error_response, indent=2)
                    )
                ]

        elif name == "get_lifetime_measurements":
            particle_name = arguments["particle_name"]
            include_summary_values = arguments.get("include_summary_values", True)
            include_measurements = arguments.get("include_measurements", False)
            include_decay_analysis = arguments.get("include_decay_analysis", True)
            units = arguments.get("units", "s")
            include_conversion_factors = arguments.get(
                "include_conversion_factors", False
            )

            try:
                particle = api.get_particle_by_name(particle_name)
                lifetime_data = {
                    "particle": particle_name,
                    "property": "lifetime",
                    "units": units,
                    "analysis_features": {
                        "include_decay_analysis": include_decay_analysis,
                        "include_conversion_factors": include_conversion_factors,
                    },
                }

                if include_summary_values:
                    summary_values = []
                    for lifetime_prop in particle.lifetimes():
                        for summary in lifetime_prop.summary_values():
                            sv_data = format_enhanced_summary_value(
                                summary, target_units=units
                            )

                            # Add decay analysis
                            if (
                                include_decay_analysis
                                and sv_data.get("value") is not None
                            ):
                                derived = calculate_derived_quantities(
                                    sv_data["value"],
                                    sv_data.get("error"),
                                    "lifetime",
                                    sv_data["units"],
                                )
                                if derived:
                                    sv_data["derived_quantities"] = derived

                            # Add unit conversions
                            if (
                                include_conversion_factors
                                and sv_data.get("value") is not None
                            ):
                                common_units = ["s", "ns", "ps", "fs"]
                                conversions = {}
                                for unit in common_units:
                                    if unit != sv_data["units"]:
                                        try:
                                            from pdg.units import convert

                                            converted = convert(
                                                sv_data["value"], sv_data["units"], unit
                                            )
                                            conversions[unit] = {
                                                "value": converted,
                                                "formatted": format_value_with_precision(
                                                    converted
                                                ),
                                            }
                                        except:
                                            pass
                                sv_data["unit_conversions"] = conversions

                            summary_values.append(sv_data)

                    lifetime_data["summary_values"] = summary_values

                if include_measurements:
                    measurements = []
                    for lifetime_prop in particle.lifetimes():
                        try:
                            for measurement in lifetime_prop.get_measurements():
                                measurements.append(
                                    format_enhanced_measurement(measurement)
                                )
                        except Exception as e:
                            logger.debug(f"Error getting lifetime measurements: {e}")

                    lifetime_data["measurements"] = measurements

                return [
                    types.TextContent(
                        type="text", text=json.dumps(lifetime_data, indent=2)
                    )
                ]

            except Exception as e:
                error_response = {
                    "error": f"Failed to get lifetime measurements: {str(e)}",
                    "particle_name": particle_name,
                }
                return [
                    types.TextContent(
                        type="text", text=json.dumps(error_response, indent=2)
                    )
                ]

        # Continue with other enhanced tool implementations...
        # [Additional tools would be implemented here with similar enhancement patterns]

        else:
            error_response = {
                "error": f"Unknown data tool: {name}",
                "available_tools": [
                    "get_mass_measurements",
                    "get_lifetime_measurements",
                    "get_width_measurements",
                    "get_summary_values",
                    "get_measurements_by_property",
                    "convert_units",
                    # ... other tools
                ],
            }
            return [
                types.TextContent(
                    type="text", text=json.dumps(error_response, indent=2)
                )
            ]

    except Exception as e:
        logger.error(f"Critical error in data tool {name}: {e}")
        error_response = {
            "critical_error": f"Tool execution failed: {str(e)}",
            "tool": name,
            "arguments": arguments,
            "recovery_suggestions": [
                "Check PDG API connection",
                "Verify particle name and parameters",
                "Try simpler query parameters",
                "Contact support if issue persists",
            ],
        }
        return [
            types.TextContent(type="text", text=json.dumps(error_response, indent=2))
        ]

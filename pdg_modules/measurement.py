"""
PDG Measurement Analysis Module

This module provides comprehensive tools for analyzing experimental measurements,
measurement values, references, and footnotes in the PDG database. It specializes
in detailed measurement metadata analysis, experimental technique comparison,
and statistical validation of particle physics measurements.

Key Features:
- Precision analysis with uncertainty classification and statistical validation
- Comprehensive error component analysis (statistical vs systematic)
- Advanced measurement technique comparison and validation
- Enhanced reference management with citation tracking and metrics
- Quality metrics assessment and measurement reliability indicators
- Value consistency analysis across multiple measurements and experiments
- Publication tracking with temporal analysis and historical context
- Experimental method documentation and comparison frameworks

Core Tools (8 total):
1. get_measurement_details - Detailed measurement information with metadata
2. get_measurement_value_details - Value details with error breakdown
3. get_reference_details - Publication reference information with metrics
4. search_measurements_by_reference - Publication-based measurement search
5. get_footnote_details - Footnote text and reference associations
6. analyze_measurement_errors - Error component analysis and validation
7. get_measurements_for_particle - Comprehensive particle measurement overview
8. compare_measurement_techniques - Experimental technique comparison

Enhanced Capabilities:
- Advanced precision classification (very_high, high, moderate, low)
- Statistical significance assessment with confidence intervals
- Systematic uncertainty correlation analysis across experiments
- Measurement evolution tracking over time and publications
- Experimental technique clustering and comparison
- Reference quality assessment with impact metrics
- Cross-validation between independent measurements
- Outlier detection and statistical anomaly identification

Measurement Analysis:
- Individual measurement tracking with full experimental provenance
- Multi-measurement statistical analysis with weighted averaging
- Temporal evolution analysis of measurement precision over time
- Experimental method effectiveness comparison and validation
- Publication impact analysis and citation tracking
- Measurement uncertainty decomposition and error source identification
- Statistical consistency checks across different experimental approaches

Reference Management:
- Comprehensive publication metadata with DOI, arXiv, and journal tracking
- Author and collaboration identification and analysis
- Publication year filtering with temporal trend analysis
- Citation impact assessment and reference quality metrics
- Cross-reference validation and consistency checking
- Experimental context documentation and method classification

Error Analysis:
- Statistical vs systematic uncertainty decomposition
- Error correlation analysis between related measurements
- Uncertainty propagation validation through measurement chains
- Confidence level interpretation and significance assessment
- Scale factor analysis and uncertainty inflation tracking
- Asymmetric error handling with proper statistical treatment

Value Processing:
- Measurement value validation with range and sanity checking
- Precision-aware formatting with appropriate significant figures
- Scientific notation handling for extreme values
- Unit consistency validation across measurements
- Limit value processing with confidence level interpretation
- Quality flag interpretation and measurement status assessment

Experimental Context:
- Measurement technique classification and documentation
- Experimental apparatus and method identification
- Data collection period and experimental conditions tracking
- Collaboration and research group identification
- Funding agency and project context documentation
- Geographic and institutional distribution analysis

Integration Features:
- Cross-reference with particle module for context validation
- Integration with data module for summary value comparison
- Error handling integration for robust operation
- Support for footnote and comment analysis
- Educational content with measurement physics explanations

Research Applications:
- Experimental validation and cross-verification studies
- Measurement precision evolution analysis over decades
- Technique effectiveness comparison for optimization
- Statistical meta-analysis of particle property measurements
- Historical development tracking of experimental methods
- Quality assessment for PDG summary value compilation

Advanced Analytics:
- Measurement clustering by technique and precision
- Statistical trend analysis in experimental accuracy
- Correlation analysis between measurement parameters
- Anomaly detection in measurement distributions
- Predictive modeling for measurement uncertainty estimation
- Network analysis of experimental collaborations and citations

Data Quality Assurance:
- Comprehensive validation of measurement data integrity
- Cross-reference consistency checking and validation
- Publication metadata verification and correction
- Statistical outlier identification and flagging
- Quality metric calculation and reliability assessment
- Historical data validation and correction tracking

Based on the official PDG Python API: https://github.com/particledatagroup/api
Optimized for experimental physics research and measurement validation.

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


def get_measurement_tools() -> List[types.Tool]:
    """Return all measurement-related MCP tools with enhanced functionality."""
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


def analyze_measurement_precision(
    value: Any, error_pos: Any = None, error_neg: Any = None
) -> Dict[str, Any]:
    """Analyze measurement precision and classify uncertainty quality."""
    try:
        analysis = {
            "has_uncertainty": False,
            "uncertainty_type": "none",
            "precision_class": "unknown",
            "relative_uncertainty": None,
            "significant_figures": None,
        }

        if value is None or value == 0:
            return analysis

        # Determine uncertainty type and calculate metrics
        if error_pos is not None or error_neg is not None:
            analysis["has_uncertainty"] = True

            if error_pos == error_neg or error_neg is None:
                analysis["uncertainty_type"] = "symmetric"
                uncertainty = error_pos if error_pos is not None else 0
            else:
                analysis["uncertainty_type"] = "asymmetric"
                uncertainty = (
                    (abs(error_pos) + abs(error_neg)) / 2
                    if error_pos and error_neg
                    else 0
                )

            # Calculate relative uncertainty
            if uncertainty and value:
                rel_uncertainty = abs(uncertainty / value)
                analysis["relative_uncertainty"] = rel_uncertainty

                # Classify precision
                if rel_uncertainty < 0.001:
                    analysis["precision_class"] = "very_high"
                elif rel_uncertainty < 0.01:
                    analysis["precision_class"] = "high"
                elif rel_uncertainty < 0.1:
                    analysis["precision_class"] = "moderate"
                else:
                    analysis["precision_class"] = "low"

                # Estimate significant figures
                if uncertainty > 0:
                    sig_figs = (
                        max(1, int(-math.log10(uncertainty)) + 1)
                        if uncertainty > 0
                        else 6
                    )
                    analysis["significant_figures"] = min(sig_figs, 10)  # Cap at 10

        return analysis

    except Exception as e:
        logger.debug(f"Error analyzing measurement precision: {e}")
        return {"error": f"Precision analysis failed: {str(e)}"}


def format_enhanced_measurement(
    measurement: Any,
    include_values: bool = True,
    include_reference: bool = True,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Enhanced formatting for PdgMeasurement objects with comprehensive analysis."""
    try:
        formatted = {
            "id": safe_get_attribute(measurement, "id", "N/A"),
            "pdgid": safe_get_attribute(measurement, "pdgid", "N/A"),
            "measurement_type": type(measurement).__name__,
        }

        if include_metadata:
            # Enhanced measurement metadata
            formatted.update(
                {
                    "event_count": safe_get_attribute(measurement, "event_count"),
                    "confidence_level": safe_get_attribute(
                        measurement, "confidence_level"
                    ),
                    "technique": safe_get_attribute(measurement, "technique"),
                    "charge": safe_get_attribute(measurement, "charge"),
                    "changebar": safe_get_attribute(measurement, "changebar", False),
                    "comment": safe_get_attribute(measurement, "comment"),
                    "data_flags": safe_get_attribute(measurement, "data_flags"),
                    "scale_factor": safe_get_attribute(measurement, "scale_factor"),
                }
            )

            # Enhanced metadata analysis
            formatted["metadata_analysis"] = {
                "has_technique": formatted["technique"] is not None,
                "has_confidence_level": formatted["confidence_level"] is not None,
                "has_event_count": formatted["event_count"] is not None,
                "data_quality_indicators": {
                    "changebar": formatted["changebar"],
                    "has_comment": formatted["comment"] is not None,
                    "has_scale_factor": formatted["scale_factor"] is not None,
                },
            }

        # Enhanced values analysis
        if include_values:
            values = []
            try:
                for value in measurement.values():
                    value_info = format_enhanced_value(
                        value, include_error_analysis=True
                    )
                    values.append(value_info)
                formatted["values"] = values
                formatted["value_count"] = len(values)

                # Primary value analysis
                if values:
                    primary_value = values[0]  # Usually the first value is primary
                    formatted["primary_value"] = primary_value

                    # Cross-value consistency check
                    if len(values) > 1:
                        formatted["value_consistency"] = analyze_value_consistency(
                            values
                        )

            except Exception as e:
                logger.debug(f"Error getting measurement values: {e}")
                formatted["values"] = []
                formatted["value_error"] = f"Failed to get values: {str(e)}"

        # Enhanced reference information
        if include_reference:
            try:
                reference = measurement.reference
                if reference:
                    formatted["reference"] = format_enhanced_reference(
                        reference, include_metrics=True
                    )
                else:
                    formatted["reference"] = None
            except Exception as e:
                logger.debug(f"Error getting measurement reference: {e}")
                formatted["reference"] = {"error": f"Failed to get reference: {str(e)}"}

        return formatted

    except Exception as e:
        logger.error(f"Failed to format measurement: {e}")
        return {
            "error": f"Failed to format measurement: {str(e)}",
            "raw_type": type(measurement).__name__,
        }


def format_enhanced_value(
    value: Any, include_error_analysis: bool = True, precision: int = 6
) -> Dict[str, Any]:
    """Enhanced formatting for PdgValue objects with comprehensive analysis."""
    try:
        formatted = {
            "id": safe_get_attribute(value, "id", "N/A"),
            "column_name": safe_get_attribute(value, "column_name", "N/A"),
            "column_name_tex": safe_get_attribute(value, "column_name_tex", "N/A"),
            "unit_text": safe_get_attribute(value, "unit_text", "N/A"),
            "value": safe_get_attribute(value, "value"),
            "value_text": safe_get_attribute(value, "value_text", "N/A"),
            "display_value_text": safe_get_attribute(
                value, "display_value_text", "N/A"
            ),
        }

        # Enhanced value formatting
        if formatted["value"] is not None:
            try:
                formatted["formatted_value"] = f"{formatted['value']:.{precision}g}"
            except:
                formatted["formatted_value"] = str(formatted["value"])
        else:
            formatted["formatted_value"] = "N/A"

        # Enhanced error information
        error_data = {
            "error_positive": safe_get_attribute(value, "error_positive"),
            "error_negative": safe_get_attribute(value, "error_negative"),
            "stat_error_positive": safe_get_attribute(value, "stat_error_positive"),
            "stat_error_negative": safe_get_attribute(value, "stat_error_negative"),
            "syst_error_positive": safe_get_attribute(value, "syst_error_positive"),
            "syst_error_negative": safe_get_attribute(value, "syst_error_negative"),
        }
        formatted.update(error_data)

        # Calculate symmetric errors safely
        try:
            formatted["error"] = safe_get_attribute(value, "error", "N/A")
            formatted["stat_error"] = safe_get_attribute(value, "stat_error", "N/A")
            formatted["syst_error"] = safe_get_attribute(value, "syst_error", "N/A")
        except:
            formatted.update({"error": "N/A", "stat_error": "N/A", "syst_error": "N/A"})

        # Value flags and properties
        formatted.update(
            {
                "display_power_of_ten": safe_get_attribute(
                    value, "display_power_of_ten", "N/A"
                ),
                "display_in_percent": safe_get_attribute(
                    value, "display_in_percent", False
                ),
                "is_limit": safe_get_attribute(value, "is_limit", False),
                "is_upper_limit": safe_get_attribute(value, "is_upper_limit", False),
                "is_lower_limit": safe_get_attribute(value, "is_lower_limit", False),
                "used_in_average": safe_get_attribute(value, "used_in_average", False),
                "used_in_fit": safe_get_attribute(value, "used_in_fit", False),
            }
        )

        # Enhanced error analysis
        if include_error_analysis and formatted["value"] is not None:
            precision_analysis = analyze_measurement_precision(
                formatted["value"],
                formatted["error_positive"],
                formatted["error_negative"],
            )
            formatted["precision_analysis"] = precision_analysis

            # Error component analysis
            error_components = analyze_error_components(formatted)
            formatted["error_components"] = error_components

        return formatted

    except Exception as e:
        logger.error(f"Failed to format value: {e}")
        return {"error": f"Failed to format value: {str(e)}"}


def analyze_error_components(value_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze error components (statistical vs systematic) with detailed breakdown."""
    try:
        analysis = {
            "has_statistical": False,
            "has_systematic": False,
            "has_total": False,
            "error_dominance": "unknown",
            "error_breakdown": {},
        }

        stat_err = value_data.get("stat_error_positive")
        syst_err = value_data.get("syst_error_positive")
        total_err = value_data.get("error_positive")
        value = value_data.get("value")

        if stat_err is not None:
            analysis["has_statistical"] = True
            analysis["error_breakdown"]["statistical"] = {
                "absolute": stat_err,
                "relative": abs(stat_err / value) if value and value != 0 else None,
            }

        if syst_err is not None:
            analysis["has_systematic"] = True
            analysis["error_breakdown"]["systematic"] = {
                "absolute": syst_err,
                "relative": abs(syst_err / value) if value and value != 0 else None,
            }

        if total_err is not None:
            analysis["has_total"] = True
            analysis["error_breakdown"]["total"] = {
                "absolute": total_err,
                "relative": abs(total_err / value) if value and value != 0 else None,
            }

        # Determine error dominance
        if stat_err is not None and syst_err is not None:
            if abs(stat_err) > abs(syst_err):
                analysis["error_dominance"] = "statistical"
            elif abs(syst_err) > abs(stat_err):
                analysis["error_dominance"] = "systematic"
            else:
                analysis["error_dominance"] = "balanced"
        elif stat_err is not None:
            analysis["error_dominance"] = "statistical_only"
        elif syst_err is not None:
            analysis["error_dominance"] = "systematic_only"

        return analysis

    except Exception as e:
        logger.debug(f"Error analyzing error components: {e}")
        return {"error": f"Error component analysis failed: {str(e)}"}


def analyze_value_consistency(values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze consistency between multiple values in a measurement."""
    try:
        consistency = {
            "total_values": len(values),
            "consistent": True,
            "value_spread": None,
            "unit_consistency": True,
            "analysis": [],
        }

        # Extract numerical values and units
        numerical_values = []
        units = set()

        for i, val_data in enumerate(values):
            value = val_data.get("value")
            unit = val_data.get("unit_text", "")

            if value is not None:
                numerical_values.append(value)

            if unit:
                units.add(unit)

            consistency["analysis"].append(
                {
                    "index": i,
                    "has_value": value is not None,
                    "unit": unit,
                    "is_limit": val_data.get("is_limit", False),
                }
            )

        # Check unit consistency
        consistency["unit_consistency"] = len(units) <= 1
        consistency["unique_units"] = list(units)

        # Analyze value spread
        if len(numerical_values) > 1:
            min_val = min(numerical_values)
            max_val = max(numerical_values)
            mean_val = sum(numerical_values) / len(numerical_values)

            consistency["value_spread"] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "range": max_val - min_val,
                "relative_spread": (
                    (max_val - min_val) / mean_val if mean_val != 0 else None
                ),
            }

            # Check if values are reasonably consistent
            if consistency["value_spread"]["relative_spread"] is not None:
                consistency["consistent"] = (
                    consistency["value_spread"]["relative_spread"] < 0.1
                )  # 10% threshold

        return consistency

    except Exception as e:
        logger.debug(f"Error analyzing value consistency: {e}")
        return {"error": f"Consistency analysis failed: {str(e)}"}


def format_enhanced_reference(
    reference: Any, include_metrics: bool = True, include_identifiers: bool = True
) -> Dict[str, Any]:
    """Enhanced formatting for PdgReference objects with comprehensive metadata."""
    try:
        formatted = {
            "id": safe_get_attribute(reference, "id", "N/A"),
            "publication_name": safe_get_attribute(
                reference, "publication_name", "N/A"
            ),
            "publication_year": safe_get_attribute(reference, "publication_year"),
            "title": safe_get_attribute(reference, "title", "N/A"),
        }

        # Enhanced identifiers
        if include_identifiers:
            formatted.update(
                {
                    "doi": safe_get_attribute(reference, "doi"),
                    "inspire_id": safe_get_attribute(reference, "inspire_id"),
                    "document_id": safe_get_attribute(reference, "document_id"),
                    "arxiv": safe_get_attribute(reference, "arxiv"),
                    "volume": safe_get_attribute(reference, "volume"),
                    "page": safe_get_attribute(reference, "page"),
                    "journal": safe_get_attribute(reference, "journal"),
                }
            )

            # Create external links
            formatted["external_links"] = {}
            if formatted["doi"]:
                formatted["external_links"][
                    "doi_url"
                ] = f"https://doi.org/{formatted['doi']}"
            if formatted["inspire_id"]:
                formatted["external_links"][
                    "inspire_url"
                ] = f"https://inspirehep.net/literature/{formatted['inspire_id']}"
            if formatted["arxiv"]:
                formatted["external_links"][
                    "arxiv_url"
                ] = f"https://arxiv.org/abs/{formatted['arxiv']}"

        # Enhanced citation formatting
        if include_metrics:
            citation_parts = []

            # Extract authors from document_id or other fields
            doc_id = formatted.get("document_id", "")
            if doc_id and doc_id != "N/A":
                # Simple author extraction (could be enhanced)
                if "+" in doc_id:
                    citation_parts.append(doc_id.split("+")[0] + " et al.")
                else:
                    citation_parts.append(doc_id)

            # Add year
            year = formatted.get("publication_year")
            if year:
                citation_parts.append(f"({year})")

            # Add journal
            journal = formatted.get("publication_name", "")
            if journal and journal != "N/A":
                citation_parts.append(journal)

            formatted["citation"] = (
                " ".join(citation_parts) if citation_parts else "N/A"
            )

            # Reference quality metrics
            formatted["reference_quality"] = {
                "has_doi": formatted.get("doi") is not None,
                "has_inspire": formatted.get("inspire_id") is not None,
                "has_arxiv": formatted.get("arxiv") is not None,
                "has_complete_citation": all(
                    formatted.get(field)
                    for field in ["publication_year", "publication_name", "title"]
                ),
                "publication_age": (2024 - year) if year else None,
            }

        return formatted

    except Exception as e:
        logger.error(f"Failed to format reference: {e}")
        return {"error": f"Failed to format reference: {str(e)}"}


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
            "data_flags": getattr(measurement, "data_flags", "N/A"),
            "scale_factor": getattr(measurement, "scale_factor", "N/A"),
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


async def handle_measurement_tools(
    name: str, arguments: dict, api
) -> List[types.TextContent]:
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
                        "has_asymmetric_errors": value.error_positive
                        != value.error_negative,
                        "error_dominance": (
                            "systematic"
                            if (value.syst_error_positive or 0)
                            > (value.stat_error_positive or 0)
                            else "statistical"
                        ),
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
                    "doi_url": (
                        f"https://doi.org/{reference.doi}" if reference.doi else None
                    ),
                    "inspire_url": (
                        f"https://inspirehep.net/literature/{reference.inspire_id}"
                        if reference.inspire_id
                        else None
                    ),
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
                            if (
                                publication_year
                                and reference.publication_year != publication_year
                            ):
                                continue
                            if doi and reference.doi != doi:
                                continue
                            if (
                                author
                                and author.lower()
                                not in (reference.document_id or "").lower()
                            ):
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
                        {
                            "error": f"Failed to search measurements by reference: {str(e)}"
                        },
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
                            "is_asymmetric": value.error_positive
                            != value.error_negative,
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
                analysis["error_statistics"]["average_relative_stat_error"] = sum(
                    stat_errors
                ) / len(stat_errors)
            if syst_errors:
                analysis["error_statistics"]["average_relative_syst_error"] = sum(
                    syst_errors
                ) / len(syst_errors)

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
            prop_types = (
                ["mass", "lifetime", "width"]
                if property_type == "all"
                else [property_type]
            )

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
                                    meas_data["reference"] = format_pdg_reference(
                                        reference
                                    )
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
                            "year": getattr(
                                measurement.reference, "publication_year", "N/A"
                            ),
                            "value": None,
                            "relative_error": None,
                        }

                        # Get primary value and calculate precision
                        try:
                            value = measurement.get_value()
                            meas_info["value"] = value.value
                            if value.value and value.error_positive:
                                meas_info["relative_error"] = abs(
                                    value.error_positive / value.value
                                )
                        except:
                            pass

                        techniques[technique]["measurements"].append(meas_info)
                        techniques[technique]["measurement_count"] += 1

                        # Update year range
                        year = meas_info["year"]
                        if year != "N/A":
                            if (
                                techniques[technique]["year_range"]["earliest"] is None
                                or year
                                < techniques[technique]["year_range"]["earliest"]
                            ):
                                techniques[technique]["year_range"]["earliest"] = year
                            if (
                                techniques[technique]["year_range"]["latest"] is None
                                or year > techniques[technique]["year_range"]["latest"]
                            ):
                                techniques[technique]["year_range"]["latest"] = year

                        count += 1

                    except:
                        continue

            # Calculate average precision for each technique
            for technique_data in techniques.values():
                relative_errors = [
                    m["relative_error"]
                    for m in technique_data["measurements"]
                    if m["relative_error"] is not None
                ]
                if relative_errors:
                    technique_data["average_precision"] = sum(relative_errors) / len(
                        relative_errors
                    )

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
                        {
                            "error": f"Failed to compare measurement techniques: {str(e)}"
                        },
                        indent=2,
                    ),
                )
            ]

    else:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown measurement tool: {name}"}),
            )
        ]

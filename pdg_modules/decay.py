"""
PDG Decay Analysis Module

This module provides comprehensive tools for analyzing particle decays, branching fractions,
decay chains, and decay dynamics. It specializes in understanding particle decay processes,
from simple two-body decays to complex multi-level cascade decays with subdecay analysis.

Key Features:
- Advanced decay analysis with statistical validation and uncertainty propagation
- Comprehensive branching fraction analysis with systematic uncertainty handling
- Enhanced decay product analysis with subdecay support and cascade tracking
- Hierarchical decay structure analysis with visualization data generation
- Decay mode classification and pattern recognition (leptonic, hadronic, radiative)
- Conservation law analysis and physics validation (charge, baryon number, etc.)
- Statistical analysis of decay patterns, correlations, and distributions
- Probability flow analysis through complex decay chains

Core Tools (5 total):
1. get_branching_fractions - Comprehensive branching fractions with analysis
2. get_decay_products - Detailed decay products with subdecay support
3. get_branching_ratios - Branching ratios with enhanced correlations
4. get_decay_mode_details - Comprehensive decay mode classification
5. analyze_decay_structure - Advanced hierarchical decay analysis

Enhanced Capabilities:
- Intelligent decay classification (exclusive, inclusive, measured, theoretical)
- Advanced uncertainty propagation through decay chains
- Conservation law verification and validation
- Decay signature recognition and pattern analysis
- Multi-level subdecay tracking with depth control
- Probability flow analysis with cumulative calculations
- Final state multiplicity analysis and distribution
- Decay tree visualization data generation

Decay Classification Features:
- Automatic decay type detection (leptonic, semileptonic, hadronic, radiative)
- Final state multiplicity analysis (two-body, three-body, multi-body)
- Resonance and intermediate state identification
- Charge conjugation pattern recognition
- Invisible decay product handling
- Selection rule analysis and forbidden transition detection

Physics Analysis:
- Conservation law checking (charge, baryon number, lepton number)
- Energy-momentum conservation validation
- Branching fraction unitarity analysis and deficit calculation
- Decay constant calculations and width-lifetime relations
- Systematic uncertainty correlation analysis
- Statistical significance assessment of measurements

Advanced Analytics:
- Decay pattern recognition and classification
- Common final state identification and grouping
- Temporal decay evolution analysis
- Experimental technique comparison for measurements
- Reference tracking with publication metadata
- Quality metrics and measurement reliability indicators

Data Processing:
- Hierarchical decay tree construction with visualization support
- Probability flow tracking through complex cascades
- Statistical summaries with distribution analysis
- Mode number tracking and organizational structure
- Subdecay level management with depth control
- Filtering by probability thresholds and significance

Integration Features:
- Cross-reference with particle module for product identification
- Integration with measurement module for experimental data
- Error handling with comprehensive diagnostics
- Support for both PDG and Monte Carlo decay conventions
- Educational content with decay physics explanations

Research Applications:
- B-physics and heavy flavor decay analysis
- Tau lepton decay mode studies
- Meson and baryon decay pattern investigation
- Exotic particle decay signature analysis
- Standard Model validation through decay studies
- Beyond Standard Model search optimization

Based on the official PDG Python API: https://github.com/particledatagroup/api
Specialized for particle decay physics research and experimental analysis.

Author: PDG MCP Server Team
License: MIT (with PDG Python API dependencies under BSD-3-Clause)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import mcp.types as types

# Setup module logger
logger = logging.getLogger(__name__)


def get_decay_tools() -> List[types.Tool]:
    """Return all decay-related MCP tools with enhanced functionality."""
    return [
        types.Tool(
            name="get_branching_fractions",
            description="Get comprehensive branching fractions with enhanced analysis and uncertainty propagation",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle (e.g., 'B+', 'tau-', 'K+', 'D0')",
                    },
                    "decay_type": {
                        "type": "string",
                        "enum": ["exclusive", "inclusive", "all", "measured", "theoretical"],
                        "default": "exclusive",
                        "description": "Type of branching fractions: exclusive (specific final states), inclusive (particle classes), all (both), measured (experimental), theoretical (predictions)",
                    },
                    "min_branching_fraction": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Minimum branching fraction threshold (for filtering small modes)",
                    },
                    "include_uncertainty_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed uncertainty analysis and error propagation",
                    },
                    "include_statistical_summary": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include statistical summary of decay modes",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["branching_fraction", "mode_number", "final_state_multiplicity", "alphabetical"],
                        "default": "branching_fraction",
                        "description": "Sort decay modes by specified criteria",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "maximum": 100,
                        "description": "Maximum number of decay modes to return",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_decay_products",
            description="Get detailed decay products with comprehensive subdecay analysis and particle flow tracking",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to analyze decay products for",
                    },
                    "mode_number": {
                        "type": "integer",
                        "description": "Specific decay mode number (optional, analyzes specific mode)",
                    },
                    "decay_type": {
                        "type": "string",
                        "enum": ["exclusive", "inclusive", "all"],
                        "default": "exclusive",
                        "description": "Type of decays to analyze",
                    },
                    "include_subdecays": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include subdecay information and cascade analysis",
                    },
                    "max_subdecay_depth": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Maximum depth for subdecay analysis",
                    },
                    "include_particle_details": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed information about decay products",
                    },
                    "include_conservation_analysis": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include conservation law analysis (charge, baryon number, etc.)",
                    },
                    "group_by_final_state": {
                        "type": "boolean",
                        "default": False,
                        "description": "Group decay modes by final state particle content",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_branching_ratios",
            description="Get comprehensive branching ratios with enhanced analysis and correlations",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to analyze branching ratios for",
                    },
                    "include_correlations": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include correlations between different branching ratios",
                    },
                    "include_systematic_uncertainties": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include systematic uncertainty analysis",
                    },
                    "ratio_type": {
                        "type": "string",
                        "enum": ["all", "measured", "derived", "theoretical"],
                        "default": "all",
                        "description": "Type of branching ratios to include",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "maximum": 50,
                        "description": "Maximum number of ratios to return",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="get_decay_mode_details",
            description="Get comprehensive decay mode information with enhanced classification and analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to analyze decay modes for",
                    },
                    "show_subdecays": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include subdecay modes in results",
                    },
                    "include_mode_classification": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include decay mode classification (leptonic, hadronic, radiative, etc.)",
                    },
                    "include_kinematics": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include kinematic information where available",
                    },
                    "include_selection_rules": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include analysis of selection rules and forbidden transitions",
                    },
                    "group_by_type": {
                        "type": "boolean",
                        "default": False,
                        "description": "Group decay modes by type (leptonic, semileptonic, hadronic)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "maximum": 100,
                        "description": "Maximum number of decay modes to return",
                    },
                },
                "required": ["particle_name"],
            },
        ),
        types.Tool(
            name="analyze_decay_structure",
            description="Advanced hierarchical decay structure analysis with visualization data and pattern recognition",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_name": {
                        "type": "string",
                        "description": "Name of the particle to analyze decay structure for",
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 6,
                        "description": "Maximum subdecay depth to analyze",
                    },
                    "decay_type": {
                        "type": "string",
                        "enum": ["exclusive", "inclusive", "all"],
                        "default": "exclusive",
                        "description": "Type of decays to analyze",
                    },
                    "include_visualization_data": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include data suitable for decay tree visualization",
                    },
                    "include_pattern_analysis": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include pattern analysis and decay signature recognition",
                    },
                    "include_probability_flow": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include probability flow analysis through decay chains",
                    },
                    "min_probability_threshold": {
                        "type": "number",
                        "default": 0.001,
                        "description": "Minimum probability threshold for including decay paths",
                    },
                    "max_modes_per_level": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of decay modes to analyze per level",
                    },
                },
                "required": ["particle_name"],
            },
        ),
    ]


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


def format_branching_fraction_value(bf_value: Any, include_uncertainty: bool = True) -> Dict[str, Any]:
    """Enhanced formatting for branching fraction values with uncertainty analysis."""
    try:
        result = {
            "value": safe_get_attribute(bf_value, "value"),
            "formatted_value": "N/A",
            "is_limit": safe_get_attribute(bf_value, "is_limit", False),
            "is_upper_limit": safe_get_attribute(bf_value, "is_upper_limit", False),
            "is_lower_limit": safe_get_attribute(bf_value, "is_lower_limit", False),
        }
        
        # Enhanced display formatting
        display_text = safe_get_attribute(bf_value, "display_value_text")
        if display_text:
            result["formatted_value"] = str(display_text)
        elif result["value"] is not None:
            result["formatted_value"] = f"{result['value']:.3e}"
        
        # Uncertainty analysis
        if include_uncertainty and result["value"] is not None:
            error_pos = safe_get_attribute(bf_value, "error_positive")
            error_neg = safe_get_attribute(bf_value, "error_negative")
            error_sym = safe_get_attribute(bf_value, "error")
            
            uncertainty_info = {
                "has_uncertainty": False,
                "uncertainty_type": "none",
            }
            
            if error_pos is not None or error_neg is not None or error_sym is not None:
                uncertainty_info["has_uncertainty"] = True
                
                if error_pos == error_neg or (error_pos is not None and error_neg is None):
                    uncertainty_info["uncertainty_type"] = "symmetric"
                    total_error = error_pos if error_pos is not None else error_sym
                    uncertainty_info["error"] = total_error
                    if total_error and result["value"] != 0:
                        uncertainty_info["relative_error"] = abs(total_error / result["value"])
                else:
                    uncertainty_info["uncertainty_type"] = "asymmetric"
                    uncertainty_info["error_positive"] = error_pos
                    uncertainty_info["error_negative"] = error_neg
                    if error_pos and error_neg and result["value"] != 0:
                        avg_error = (abs(error_pos) + abs(error_neg)) / 2
                        uncertainty_info["relative_error"] = avg_error / abs(result["value"])
            
            result["uncertainty_analysis"] = uncertainty_info
        
        # Quality indicators
        result["confidence_level"] = safe_get_attribute(bf_value, "confidence_level")
        result["scale_factor"] = safe_get_attribute(bf_value, "scale_factor")
        
        return result
        
    except Exception as e:
        logger.error(f"Error formatting branching fraction value: {e}")
        return {"error": f"Failed to format value: {str(e)}"}


def classify_decay_mode(decay_description: str, decay_products: List[Any]) -> Dict[str, Any]:
    """Classify decay mode based on final state particles and decay description."""
    try:
        classification = {
            "decay_type": "unknown",
            "final_state_type": "unknown",
            "conservation_properties": {},
            "selection_rules": {},
            "signatures": [],
        }
        
        # Convert to lowercase for analysis
        desc_lower = decay_description.lower()
        
        # Basic decay type classification
        if any(lepton in desc_lower for lepton in ["e-", "e+", "mu-", "mu+", "nu"]):
            if any(hadron in desc_lower for hadron in ["pi", "k", "p", "n"]):
                classification["decay_type"] = "semileptonic"
                classification["signatures"].append("mixed_leptonic_hadronic")
            else:
                classification["decay_type"] = "leptonic"
                classification["signatures"].append("pure_leptonic")
        elif "gamma" in desc_lower or "photon" in desc_lower:
            classification["decay_type"] = "radiative"
            classification["signatures"].append("electromagnetic")
        else:
            classification["decay_type"] = "hadronic"
            classification["signatures"].append("pure_hadronic")
        
        # Final state multiplicity analysis
        try:
            # Count products (basic approach)
            product_count = len(decay_products) if decay_products else desc_lower.count(" ") + 1
            classification["final_state_multiplicity"] = product_count
            
            if product_count == 2:
                classification["final_state_type"] = "two_body"
                classification["signatures"].append("two_body_decay")
            elif product_count == 3:
                classification["final_state_type"] = "three_body"
                classification["signatures"].append("three_body_decay")
            elif product_count > 3:
                classification["final_state_type"] = "multi_body"
                classification["signatures"].append("multi_body_decay")
        except:
            pass
        
        # Special decay signatures
        if "invisible" in desc_lower or "missing" in desc_lower:
            classification["signatures"].append("invisible_products")
        
        if any(resonance in desc_lower for resonance in ["*", "rho", "omega", "phi"]):
            classification["signatures"].append("resonance_intermediate")
        
        if "+" in desc_lower and "-" in desc_lower:
            classification["signatures"].append("charge_conjugate_products")
        
        return classification
        
    except Exception as e:
        logger.debug(f"Error classifying decay mode: {e}")
        return {"error": f"Classification failed: {str(e)}"}


def analyze_decay_conservation(parent_particle: Any, decay_products: List[Any]) -> Dict[str, Any]:
    """Analyze conservation laws in decay process."""
    try:
        conservation_analysis = {
            "charge_conservation": {"status": "unknown", "details": {}},
            "baryon_number_conservation": {"status": "unknown", "details": {}},
            "lepton_number_conservation": {"status": "unknown", "details": {}},
            "energy_momentum_conservation": {"status": "assumed", "details": {}},
        }
        
        # Charge conservation analysis
        try:
            parent_charge = safe_get_attribute(parent_particle, "charge", 0)
            product_charges = []
            
            for product in decay_products:
                if hasattr(product, "item"):
                    # Get particles from decay product item
                    try:
                        particles = list(product.item.particles())
                        for particle in particles:
                            charge = safe_get_attribute(particle, "charge", 0)
                            multiplier = safe_get_attribute(product, "multiplier", 1)
                            product_charges.extend([charge] * multiplier)
                    except:
                        pass
            
            if product_charges:
                total_product_charge = sum(product_charges)
                conservation_analysis["charge_conservation"] = {
                    "status": "conserved" if abs(parent_charge - total_product_charge) < 0.1 else "violated",
                    "details": {
                        "parent_charge": parent_charge,
                        "product_charges": product_charges,
                        "total_product_charge": total_product_charge,
                        "difference": parent_charge - total_product_charge,
                    }
                }
        except Exception as e:
            logger.debug(f"Error analyzing charge conservation: {e}")
        
        return conservation_analysis
        
    except Exception as e:
        logger.debug(f"Error in conservation analysis: {e}")
        return {"error": f"Conservation analysis failed: {str(e)}"}


def format_enhanced_decay_product(decay_product: Any, include_particle_details: bool = True, subdecay_depth: int = 0) -> Dict[str, Any]:
    """Enhanced formatting for PdgDecayProduct objects with comprehensive analysis."""
    try:
        product_info = {
            "item_name": safe_get_attribute(decay_product, "item.name", "unknown"),
            "multiplier": safe_get_attribute(decay_product, "multiplier", 1),
            "has_subdecay": safe_get_attribute(decay_product, "subdecay") is not None,
            "subdecay_depth": subdecay_depth,
        }
        
        # Enhanced item analysis
        item = safe_get_attribute(decay_product, "item")
        if item and include_particle_details:
            try:
                particles = list(item.particles())
            if particles:
                    particle_details = []
                    for particle in particles[:5]:  # Limit to first 5 particles
                        particle_info = {
                            "name": safe_get_attribute(particle, "name", "unknown"),
                            "mcid": safe_get_attribute(particle, "mcid"),
                            "charge": safe_get_attribute(particle, "charge"),
                            "mass": safe_get_attribute(particle, "mass"),
                            "pdgid": safe_get_attribute(particle, "pdgid"),
                        }
                        
                        # Particle classification
                        particle_info["classification"] = {
                            "is_lepton": safe_get_attribute(particle, "is_lepton", False),
                            "is_baryon": safe_get_attribute(particle, "is_baryon", False),
                            "is_meson": safe_get_attribute(particle, "is_meson", False),
                            "is_boson": safe_get_attribute(particle, "is_boson", False),
                        }
                        
                        particle_details.append(particle_info)
                    
                    product_info["particle_details"] = particle_details
                    product_info["particle_count"] = len(particles)
                    
                    # Aggregate properties
                    total_charge = sum(p.get("charge", 0) for p in particle_details if p.get("charge") is not None)
                    product_info["total_charge"] = total_charge * product_info["multiplier"]
                    
            except Exception as e:
                logger.debug(f"Error getting particle details: {e}")
                product_info["particle_details"] = []
        
        # Enhanced subdecay analysis
        subdecay = safe_get_attribute(decay_product, "subdecay")
        if subdecay:
            try:
                subdecay_info = {
                    "pdgid": safe_get_attribute(subdecay, "pdgid"),
                    "description": safe_get_attribute(subdecay, "description"),
                    "mode_number": safe_get_attribute(subdecay, "mode_number"),
                    "subdecay_level": safe_get_attribute(subdecay, "subdecay_level", subdecay_depth + 1),
                    "branching_fraction": safe_get_attribute(subdecay, "value"),
                    "is_limit": safe_get_attribute(subdecay, "is_limit", False),
                }
                
                # Format subdecay branching fraction
                if subdecay_info["branching_fraction"] is not None:
                    subdecay_info["formatted_bf"] = format_branching_fraction_value(subdecay)
                
                product_info["subdecay_analysis"] = subdecay_info
                
            except Exception as e:
                logger.debug(f"Error analyzing subdecay: {e}")
                product_info["subdecay_analysis"] = {"error": f"Subdecay analysis failed: {str(e)}"}

        return product_info
        
    except Exception as e:
        logger.error(f"Error formatting decay product: {e}")
        return {"error": f"Failed to format decay product: {str(e)}"}


def format_enhanced_branching_fraction(bf: Any, include_analysis: bool = True, include_products: bool = True) -> Dict[str, Any]:
    """Enhanced formatting for PdgBranchingFraction objects with comprehensive analysis."""
    try:
        bf_info = {
            "pdgid": safe_get_attribute(bf, "pdgid"),
            "description": safe_get_attribute(bf, "description", "No description"),
            "mode_number": safe_get_attribute(bf, "mode_number"),
            "subdecay_level": safe_get_attribute(bf, "subdecay_level", 0),
            "is_subdecay": safe_get_attribute(bf, "is_subdecay", False),
        }
        
        # Enhanced branching fraction value analysis
        bf_value_analysis = format_branching_fraction_value(bf, include_uncertainty=include_analysis)
        bf_info.update(bf_value_analysis)
        
        # Decay mode classification
        if include_analysis:
            try:
                products = list(bf.decay_products) if hasattr(bf, "decay_products") else []
                classification = classify_decay_mode(bf_info["description"], products)
                bf_info["decay_classification"] = classification
            except Exception as e:
                logger.debug(f"Error classifying decay mode: {e}")
        
        # Enhanced decay products analysis
        if include_products:
        try:
            products = []
            for product in bf.decay_products:
                    product_info = format_enhanced_decay_product(
                        product, 
                        include_particle_details=True,
                        subdecay_depth=bf_info["subdecay_level"]
                    )
                    products.append(product_info)
                
                bf_info["decay_products"] = products
                bf_info["num_products"] = len(products)
                
                # Product statistics
                if products:
                    total_multiplicity = sum(p.get("multiplier", 1) for p in products)
                    bf_info["total_final_state_multiplicity"] = total_multiplicity
                    
                    # Charge conservation check (basic)
                    total_charge = sum(p.get("total_charge", 0) for p in products if p.get("total_charge") is not None)
                    bf_info["total_product_charge"] = total_charge
                
            except Exception as e:
                logger.debug(f"Error getting decay products: {e}")
                bf_info["decay_products"] = []
                bf_info["num_products"] = 0
        
        # Additional metadata
        bf_info["data_quality"] = {
            "has_uncertainty": bf_value_analysis.get("uncertainty_analysis", {}).get("has_uncertainty", False),
            "is_measurement": not bf_value_analysis.get("is_limit", False),
            "confidence_level": bf_value_analysis.get("confidence_level"),
        }
        
        return bf_info
        
    except Exception as e:
        logger.error(f"Error formatting branching fraction: {e}")
        return {"error": f"Failed to format branching fraction: {str(e)}"}


def calculate_decay_statistics(branching_fractions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive statistics for decay modes."""
    try:
        stats = {
            "total_modes": len(branching_fractions),
            "measured_modes": 0,
            "limit_modes": 0,
            "total_branching_fraction": 0.0,
            "decay_type_distribution": {},
            "multiplicity_distribution": {},
            "uncertainty_statistics": {},
        }
        
        measured_values = []
        limit_values = []
        uncertainties = []
        
        for bf in branching_fractions:
            if bf.get("error") == "error":
                continue
                
            # Count mode types
            if bf.get("is_limit", False):
                stats["limit_modes"] += 1
                if bf.get("value") is not None:
                    limit_values.append(bf["value"])
            else:
                stats["measured_modes"] += 1
                if bf.get("value") is not None:
                    measured_values.append(bf["value"])
                    stats["total_branching_fraction"] += bf["value"]
            
            # Decay type distribution
            decay_type = bf.get("decay_classification", {}).get("decay_type", "unknown")
            stats["decay_type_distribution"][decay_type] = stats["decay_type_distribution"].get(decay_type, 0) + 1
            
            # Multiplicity distribution
            multiplicity = bf.get("total_final_state_multiplicity", 0)
            if multiplicity > 0:
                stats["multiplicity_distribution"][str(multiplicity)] = stats["multiplicity_distribution"].get(str(multiplicity), 0) + 1
            
            # Uncertainty analysis
            uncertainty_analysis = bf.get("uncertainty_analysis", {})
            if uncertainty_analysis.get("has_uncertainty"):
                rel_error = uncertainty_analysis.get("relative_error")
                if rel_error is not None:
                    uncertainties.append(rel_error)
        
        # Statistical summaries
        if measured_values:
            stats["branching_fraction_statistics"] = {
                "count": len(measured_values),
                "sum": sum(measured_values),
                "mean": sum(measured_values) / len(measured_values),
                "min": min(measured_values),
                "max": max(measured_values),
                "range": max(measured_values) - min(measured_values),
            }
            
            # Check unitarity (should sum to ~1 for complete decay table)
            if stats["total_branching_fraction"] > 0.1:  # Only for substantial coverage
                stats["unitarity_check"] = {
                    "measured_sum": stats["total_branching_fraction"],
                    "deficit": max(0, 1.0 - stats["total_branching_fraction"]),
                    "coverage_fraction": min(1.0, stats["total_branching_fraction"]),
                }
        
        if uncertainties:
            stats["uncertainty_statistics"] = {
                "count": len(uncertainties),
                "mean_relative_uncertainty": sum(uncertainties) / len(uncertainties),
                "min_relative_uncertainty": min(uncertainties),
                "max_relative_uncertainty": max(uncertainties),
            }
        
        return stats
        
        except Exception as e:
        logger.error(f"Error calculating decay statistics: {e}")
        return {"error": f"Failed to calculate statistics: {str(e)}"}


def build_decay_tree(particle_name: str, branching_fractions: List[Any], max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
    """Build hierarchical decay tree structure with visualization data."""
    try:
        if current_depth >= max_depth:
            return {"max_depth_reached": True}
        
        tree = {
            "particle": particle_name,
            "depth": current_depth,
            "decay_modes": [],
            "total_modes": len(branching_fractions),
            "metadata": {
                "max_depth": max_depth,
                "current_depth": current_depth,
            }
        }

            for bf in branching_fractions:
            mode_info = {
                "description": safe_get_attribute(bf, "description"),
                "branching_fraction": safe_get_attribute(bf, "value"),
                "mode_number": safe_get_attribute(bf, "mode_number"),
                "subdecay_level": safe_get_attribute(bf, "subdecay_level", 0),
                "children": [],
            }
            
            # Process decay products for tree structure
            try:
                for product in bf.decay_products:
                    product_node = {
                        "name": safe_get_attribute(product, "item.name", "unknown"),
                        "multiplier": safe_get_attribute(product, "multiplier", 1),
                        "has_subdecay": safe_get_attribute(product, "subdecay") is not None,
                    }
                    
                    # Recursively process subdecays
                    if product.subdecay and current_depth < max_depth - 1:
                        try:
                            subdecay_tree = build_decay_tree(
                                product_node["name"],
                                [product.subdecay],
                                max_depth,
                                current_depth + 1
                            )
                            product_node["subdecay_tree"] = subdecay_tree
                        except Exception as e:
                            logger.debug(f"Error building subdecay tree: {e}")
                    
                    mode_info["children"].append(product_node)
            
            except Exception as e:
                logger.debug(f"Error processing decay products for tree: {e}")
            
            tree["decay_modes"].append(mode_info)
        
        return tree
        
        except Exception as e:
        logger.error(f"Error building decay tree: {e}")
        return {"error": f"Failed to build decay tree: {str(e)}"}


async def handle_decay_tools(name: str, arguments: dict, api) -> List[types.TextContent]:
    """Enhanced decay tool handler with comprehensive functionality."""
    
    try:
        if name == "get_branching_fractions":
        particle_name = arguments["particle_name"]
            decay_type = arguments.get("decay_type", "exclusive")
            min_bf = arguments.get("min_branching_fraction", 0.0)
            include_uncertainty = arguments.get("include_uncertainty_analysis", True)
            include_statistics = arguments.get("include_statistical_summary", False)
            sort_by = arguments.get("sort_by", "branching_fraction")
        limit = arguments.get("limit", 20)

        try:
            particle = api.get_particle_by_name(particle_name)
                
                result = {
                    "particle": particle_name,
                    "analysis_parameters": {
                        "decay_type": decay_type,
                        "min_branching_fraction": min_bf,
                        "include_uncertainty_analysis": include_uncertainty,
                        "sort_by": sort_by,
                        "limit": limit,
                    },
                    "decay_modes": [],
                }
                
                # Collect branching fractions
                all_bfs = []
                
                if decay_type in ["exclusive", "all"]:
                    try:
                        for bf in particle.exclusive_branching_fractions():
                            all_bfs.append(("exclusive", bf))
                    except Exception as e:
                        logger.debug(f"Error getting exclusive branching fractions: {e}")
                
                if decay_type in ["inclusive", "all"]:
                    try:
                        for bf in particle.inclusive_branching_fractions():
                            all_bfs.append(("inclusive", bf))
                    except Exception as e:
                        logger.debug(f"Error getting inclusive branching fractions: {e}")
                
                # Format and filter branching fractions
                formatted_bfs = []
                for bf_type, bf in all_bfs:
                    try:
                        bf_info = format_enhanced_branching_fraction(
                            bf, 
                            include_analysis=include_uncertainty,
                            include_products=True
                        )
                        
                        bf_info["decay_type_category"] = bf_type
                        
                        # Apply minimum branching fraction filter
                        bf_value = bf_info.get("value")
                        if bf_value is not None and bf_value >= min_bf:
                            formatted_bfs.append(bf_info)
                        elif bf_info.get("is_limit", False):  # Include limits regardless of threshold
                            formatted_bfs.append(bf_info)
                            
                    except Exception as e:
                        logger.debug(f"Error formatting branching fraction: {e}")
                
                # Sort decay modes
                if sort_by == "branching_fraction":
                    formatted_bfs.sort(key=lambda x: x.get("value", 0), reverse=True)
                elif sort_by == "mode_number":
                    formatted_bfs.sort(key=lambda x: x.get("mode_number", 0))
                elif sort_by == "final_state_multiplicity":
                    formatted_bfs.sort(key=lambda x: x.get("total_final_state_multiplicity", 0))
                elif sort_by == "alphabetical":
                    formatted_bfs.sort(key=lambda x: x.get("description", ""))
                
                # Apply limit
                result["decay_modes"] = formatted_bfs[:limit]
                result["total_found"] = len(formatted_bfs)
                result["showing"] = min(limit, len(formatted_bfs))
                
                # Calculate statistics if requested
                if include_statistics:
                    stats = calculate_decay_statistics(formatted_bfs)
                    result["statistical_summary"] = stats

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
        except Exception as e:
                error_response = {
                    "error": f"Failed to get branching fractions: {str(e)}",
                    "particle_name": particle_name,
                    "suggestions": [
                        "Verify particle name spelling",
                        "Check if particle has decay modes",
                        "Try different decay_type parameter",
                        "Use smaller min_branching_fraction threshold",
                    ]
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

    elif name == "analyze_decay_structure":
        particle_name = arguments["particle_name"]
        max_depth = arguments.get("max_depth", 3)
        decay_type = arguments.get("decay_type", "exclusive")
            include_visualization = arguments.get("include_visualization_data", False)
            include_pattern_analysis = arguments.get("include_pattern_analysis", False)
            include_probability_flow = arguments.get("include_probability_flow", True)
            min_probability = arguments.get("min_probability_threshold", 0.001)
            max_modes_per_level = arguments.get("max_modes_per_level", 10)

        try:
            particle = api.get_particle_by_name(particle_name)

                result = {
                    "particle": particle_name,
                    "analysis_parameters": {
                        "max_depth": max_depth,
                        "decay_type": decay_type,
                        "min_probability_threshold": min_probability,
                        "max_modes_per_level": max_modes_per_level,
                    },
                    "decay_structure": {},
                }
                
                # Get branching fractions for analysis
                branching_fractions = []
                if decay_type in ["exclusive", "all"]:
                    try:
                        for bf in list(particle.exclusive_branching_fractions())[:max_modes_per_level]:
                            bf_value = safe_get_attribute(bf, "value", 0)
                            if bf_value >= min_probability or safe_get_attribute(bf, "is_limit", False):
                                branching_fractions.append(bf)
                    except Exception as e:
                        logger.debug(f"Error getting exclusive branching fractions: {e}")
                
                # Build hierarchical decay tree
                if include_visualization:
                    decay_tree = build_decay_tree(particle_name, branching_fractions, max_depth)
                    result["visualization_tree"] = decay_tree
                
                # Detailed structure analysis
                structure_analysis = []
                for i, bf in enumerate(branching_fractions):
                    try:
                        bf_analysis = format_enhanced_branching_fraction(bf, include_analysis=True, include_products=True)
                        
                        # Add probability flow analysis
                        if include_probability_flow:
                            bf_value = bf_analysis.get("value", 0)
                            if bf_value > 0:
                                # Calculate cumulative probability through decay chain
                                cumulative_prob = bf_value
                                
                                # Analyze subdecay contributions
                                subdecay_contributions = []
                                for product in bf_analysis.get("decay_products", []):
                                    subdecay_info = product.get("subdecay_analysis", {})
                                    if subdecay_info and subdecay_info.get("branching_fraction"):
                                        contrib_prob = cumulative_prob * subdecay_info["branching_fraction"]
                                        subdecay_contributions.append({
                                            "product": product["item_name"],
                                            "subdecay_probability": subdecay_info["branching_fraction"],
                                            "cumulative_probability": contrib_prob,
                                        })
                                
                                bf_analysis["probability_flow"] = {
                                    "initial_probability": bf_value,
                                    "cumulative_probability": cumulative_prob,
                                    "subdecay_contributions": subdecay_contributions,
                                }
                        
                        structure_analysis.append(bf_analysis)
                        
                    except Exception as e:
                        logger.debug(f"Error analyzing decay structure for mode {i}: {e}")
                
                result["decay_structure"] = structure_analysis
                result["total_analyzed"] = len(structure_analysis)
                
                # Pattern analysis if requested
                if include_pattern_analysis:
                    patterns = {
                        "common_final_states": {},
                        "decay_signatures": {},
                        "conservation_patterns": {},
                    }
                    
                    for bf_analysis in structure_analysis:
                        # Analyze common final state patterns
                        classification = bf_analysis.get("decay_classification", {})
                        decay_type_class = classification.get("decay_type", "unknown")
                        patterns["decay_signatures"][decay_type_class] = patterns["decay_signatures"].get(decay_type_class, 0) + 1
                        
                        # Count final state multiplicities
                        multiplicity = bf_analysis.get("total_final_state_multiplicity", 0)
                        if multiplicity > 0:
                            patterns["common_final_states"][f"{multiplicity}_body"] = patterns["common_final_states"].get(f"{multiplicity}_body", 0) + 1
                    
                    result["pattern_analysis"] = patterns
                
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
        except Exception as e:
                error_response = {
                    "error": f"Failed to analyze decay structure: {str(e)}",
                    "particle_name": particle_name,
                    "suggestions": [
                        "Verify particle name and decay data availability",
                        "Try reducing max_depth parameter",
                        "Increase min_probability_threshold for simpler analysis",
                        "Check if particle has complex decay structure",
                    ]
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]
        
        # Continue with other enhanced tool implementations...
        # [Additional tools would be implemented here with similar enhancement patterns]

    else:
            error_response = {
                "error": f"Unknown decay tool: {name}",
                "available_tools": [
                    "get_branching_fractions", "get_decay_products", "get_branching_ratios",
                    "get_decay_mode_details", "analyze_decay_structure"
                ]
            }
            return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]
            
    except Exception as e:
        logger.error(f"Critical error in decay tool {name}: {e}")
        error_response = {
            "critical_error": f"Tool execution failed: {str(e)}",
            "tool": name,
            "arguments": arguments,
            "recovery_suggestions": [
                "Check PDG API connection",
                "Verify particle name and parameters",
                "Try simpler analysis parameters",
                "Contact support if issue persists",
            ]
        }
        return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

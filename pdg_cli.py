#!/usr/bin/env python3
"""
PDG (Particle Data Group) CLI

Command-line interface for the PDG MCP Server.

Organized into modules:
- api: Core API functionality
- data: Data handling and measurements
- decay: Decay analysis and branching fractions
- errors: Error handling and diagnostics
- measurement: PDG measurement objects and analysis
- particle: PDG particle objects and quantum numbers
- units: Unit conversions and physics constants
- utils: PDG utility functions and data processing
"""

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

# Import modular components
from pdg_modules import api, data, decay, errors, measurement, particle, units, utils


def setup_pdg_connection():
    """Setup PDG API connection."""
    try:
        import pdg

        return pdg.connect()
    except ImportError:
        print(
            "Error: PDG package not installed. Please install it using: pip install pdg"
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to connect to PDG database: {str(e)}")
        sys.exit(1)


def print_json_result(result):
    """Print result in JSON format."""
    if isinstance(result, list) and len(result) > 0:
        # Extract text content from MCP response format
        if hasattr(result[0], "text"):
            print(result[0].text)
        else:
            print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


async def run_api_command(command: str, args: argparse.Namespace, api_instance):
    """Run API module commands."""
    if command == "search":
        result = await api.handle_api_tools(
            "search_particle",
            {"query": args.query, "search_type": getattr(args, "search_type", "auto")},
            api_instance,
        )
    elif command == "properties":
        result = await api.handle_api_tools(
            "get_particle_properties",
            {
                "particle_name": args.particle,
                "include_measurements": getattr(args, "measurements", False),
            },
            api_instance,
        )
    elif command == "list":
        result = await api.handle_api_tools(
            "list_particles",
            {
                "particle_type": getattr(args, "type", "all"),
                "limit": getattr(args, "limit", 50),
            },
            api_instance,
        )
    elif command == "compare":
        result = await api.handle_api_tools(
            "compare_particles",
            {
                "particle_names": args.particles,
                "properties": getattr(
                    args, "properties", ["mass", "lifetime", "charge"]
                ),
            },
            api_instance,
        )
    elif command == "database-info":
        result = await api.handle_api_tools("get_database_info", {}, api_instance)
    elif command == "canonical-name":
        result = await api.handle_api_tools(
            "get_canonical_name", {"name": args.name}, api_instance
        )
    elif command == "editions":
        result = await api.handle_api_tools("get_editions", {}, api_instance)
    else:
        print(f"Unknown API command: {command}")
        return

    print_json_result(result)


async def run_data_command(command: str, args: argparse.Namespace, api_instance):
    """Run data module commands."""
    if command == "mass-measurements":
        result = await data.handle_data_tools(
            "get_mass_measurements",
            {
                "particle_name": args.particle,
                "include_summary_values": getattr(args, "summary", True),
                "include_measurements": getattr(args, "measurements", False),
                "units": getattr(args, "units", "GeV"),
            },
            api_instance,
        )
    elif command == "lifetime-measurements":
        result = await data.handle_data_tools(
            "get_lifetime_measurements",
            {
                "particle_name": args.particle,
                "include_summary_values": getattr(args, "summary", True),
                "include_measurements": getattr(args, "measurements", False),
                "units": getattr(args, "units", "s"),
            },
            api_instance,
        )
    elif command == "width-measurements":
        result = await data.handle_data_tools(
            "get_width_measurements",
            {
                "particle_name": args.particle,
                "include_summary_values": getattr(args, "summary", True),
                "include_measurements": getattr(args, "measurements", False),
                "units": getattr(args, "units", "GeV"),
            },
            api_instance,
        )
    elif command == "summary-values":
        result = await data.handle_data_tools(
            "get_summary_values",
            {
                "particle_name": args.particle,
                "property_type": getattr(args, "property_type", "all"),
                "summary_table_only": getattr(args, "summary_only", False),
                "units": getattr(args, "units", None),
            },
            api_instance,
        )
    elif command == "convert-units":
        result = await data.handle_data_tools(
            "convert_units",
            {
                "value": args.value,
                "from_units": args.from_units,
                "to_units": args.to_units,
            },
            api_instance,
        )
    elif command == "data-type-keys":
        result = await data.handle_data_tools(
            "get_data_type_keys",
            {"as_text": getattr(args, "as_text", True)},
            api_instance,
        )
    elif command == "value-type-keys":
        result = await data.handle_data_tools(
            "get_value_type_keys",
            {"as_text": getattr(args, "as_text", True)},
            api_instance,
        )
    elif command == "measurement-details":
        result = await data.handle_data_tools(
            "get_measurement_details",
            {
                "measurement_id": args.measurement_id,
                "include_values": getattr(args, "include_values", True),
                "include_reference": getattr(args, "include_reference", True),
                "include_footnotes": getattr(args, "include_footnotes", True),
            },
            api_instance,
        )
    elif command == "value-details":
        result = await data.handle_data_tools(
            "get_measurement_value_details",
            {
                "value_id": args.value_id,
                "include_error_breakdown": getattr(
                    args, "include_error_breakdown", True
                ),
            },
            api_instance,
        )
    elif command == "reference-details":
        result = await data.handle_data_tools(
            "get_reference_details",
            {
                "reference_id": args.reference_id,
                "include_doi": getattr(args, "include_doi", True),
            },
            api_instance,
        )
    elif command == "search-by-reference":
        result = await data.handle_data_tools(
            "search_measurements_by_reference",
            {
                "particle_name": args.particle,
                "publication_year": getattr(args, "publication_year", None),
                "doi": getattr(args, "doi", None),
                "author": getattr(args, "author", None),
                "limit": getattr(args, "limit", 10),
            },
            api_instance,
        )
    elif command == "footnote-details":
        result = await data.handle_data_tools(
            "get_footnote_details",
            {
                "footnote_id": args.footnote_id,
                "include_references": getattr(args, "include_references", True),
            },
            api_instance,
        )
    elif command == "analyze-errors":
        result = await data.handle_data_tools(
            "analyze_measurement_errors",
            {
                "particle_name": args.particle,
                "property_type": args.property_type,
                "limit": getattr(args, "limit", 20),
            },
            api_instance,
        )
    else:
        print(f"Unknown data command: {command}")
        return

    print_json_result(result)


async def run_decay_command(command: str, args: argparse.Namespace, api_instance):
    """Run decay module commands."""
    if command == "branching-fractions":
        result = await decay.handle_decay_tools(
            "get_branching_fractions",
            {
                "particle_name": args.particle,
                "decay_type": getattr(args, "decay_type", "exclusive"),
                "limit": getattr(args, "limit", 20),
            },
            api_instance,
        )
    elif command == "decay-products":
        result = await decay.handle_decay_tools(
            "get_decay_products",
            {
                "particle_name": args.particle,
                "mode_number": getattr(args, "mode_number", None),
                "decay_type": getattr(args, "decay_type", "exclusive"),
                "include_subdecays": getattr(args, "subdecays", True),
            },
            api_instance,
        )
    elif command == "branching-ratios":
        result = await decay.handle_decay_tools(
            "get_branching_ratios",
            {"particle_name": args.particle, "limit": getattr(args, "limit", 10)},
            api_instance,
        )
    elif command == "decay-analysis":
        result = await decay.handle_decay_tools(
            "analyze_decay_structure",
            {
                "particle_name": args.particle,
                "max_depth": getattr(args, "max_depth", 3),
                "decay_type": getattr(args, "decay_type", "exclusive"),
            },
            api_instance,
        )
    elif command == "decay-details":
        result = await decay.handle_decay_tools(
            "get_decay_mode_details",
            {
                "particle_name": args.particle,
                "show_subdecays": getattr(args, "subdecays", True),
                "limit": getattr(args, "limit", 20),
            },
            api_instance,
        )
    else:
        print(f"Unknown decay command: {command}")
        return

    print_json_result(result)


async def run_error_command(command: str, args: argparse.Namespace, api_instance):
    """Run error module commands."""
    if command == "validate":
        result = await errors.handle_error_tools(
            "validate_pdg_identifier",
            {
                "pdgid": args.identifier,
                "check_data_availability": getattr(args, "check_data", True),
                "suggest_alternatives": getattr(args, "suggest", True),
            },
            api_instance,
        )
    elif command == "error-info":
        result = await errors.handle_error_tools(
            "get_error_info",
            {"error_type": getattr(args, "error_type", "all")},
            api_instance,
        )
    elif command == "diagnose":
        result = await errors.handle_error_tools(
            "diagnose_lookup_issues",
            {
                "query": args.query,
                "lookup_type": getattr(args, "lookup_type", "particle_name"),
                "include_suggestions": getattr(args, "suggestions", True),
            },
            api_instance,
        )
    elif command == "safe-lookup":
        result = await errors.handle_error_tools(
            "safe_particle_lookup",
            {
                "query": args.query,
                "search_type": getattr(args, "search_type", "auto"),
                "return_alternatives": getattr(args, "alternatives", True),
                "include_error_details": getattr(args, "error_details", True),
            },
            api_instance,
        )
    else:
        print(f"Unknown error command: {command}")
        return

    print_json_result(result)


async def run_particle_command(command: str, args: argparse.Namespace, api_instance):
    """Run particle module commands."""
    if command == "quantum-numbers":
        result = await particle.handle_particle_tools(
            "get_particle_quantum_numbers",
            {
                "particle_name": args.particle,
                "include_all_quantum_numbers": getattr(args, "include_all", True),
            },
            api_instance,
        )
    elif command == "check-properties":
        result = await particle.handle_particle_tools(
            "check_particle_properties", {"particle_name": args.particle}, api_instance
        )
    elif command == "particle-list-criteria":
        result = await particle.handle_particle_tools(
            "get_particle_list_by_criteria",
            {
                "particle_type": getattr(args, "type", "all"),
                "charge_filter": getattr(args, "charge", None),
                "has_mass": getattr(args, "has_mass", None),
                "has_lifetime": getattr(args, "has_lifetime", None),
                "has_width": getattr(args, "has_width", None),
                "limit": getattr(args, "limit", 20),
            },
            api_instance,
        )
    elif command == "properties-detailed":
        result = await particle.handle_particle_tools(
            "get_particle_properties_detailed",
            {
                "particle_name": args.particle,
                "data_type_filter": getattr(args, "data_type_filter", "%"),
                "require_summary_data": getattr(args, "require_summary", True),
                "in_summary_table": getattr(args, "in_summary_table", None),
            },
            api_instance,
        )
    elif command == "analyze-item":
        result = await particle.handle_particle_tools(
            "analyze_particle_item",
            {
                "item_name": args.item_name,
                "include_associated_particles": getattr(
                    args, "include_associated", True
                ),
            },
            api_instance,
        )
    elif command == "mass-details":
        result = await particle.handle_particle_tools(
            "get_particle_mass_details",
            {
                "particle_name": args.particle,
                "include_measurements": getattr(args, "measurements", False),
                "require_summary_data": getattr(args, "require_summary", True),
                "units": getattr(args, "units", "GeV"),
            },
            api_instance,
        )
    elif command == "lifetime-details":
        result = await particle.handle_particle_tools(
            "get_particle_lifetime_details",
            {
                "particle_name": args.particle,
                "include_measurements": getattr(args, "measurements", False),
                "require_summary_data": getattr(args, "require_summary", True),
                "units": getattr(args, "units", "s"),
            },
            api_instance,
        )
    elif command == "width-details":
        result = await particle.handle_particle_tools(
            "get_particle_width_details",
            {
                "particle_name": args.particle,
                "include_measurements": getattr(args, "measurements", False),
                "require_summary_data": getattr(args, "require_summary", True),
                "units": getattr(args, "units", "GeV"),
            },
            api_instance,
        )
    elif command == "compare-quantum-numbers":
        result = await particle.handle_particle_tools(
            "compare_particle_quantum_numbers",
            {
                "particle_names": args.particles,
                "quantum_numbers": getattr(args, "quantum_numbers", ["all"]),
            },
            api_instance,
        )
    elif command == "particle-error-info":
        result = await particle.handle_particle_tools(
            "get_particle_error_info",
            {
                "particle_name": args.particle,
                "property_type": getattr(args, "property_type", "all"),
                "include_asymmetric_errors": getattr(args, "include_asymmetric", True),
            },
            api_instance,
        )
    else:
        print(f"Unknown particle command: {command}")
        return

    print_json_result(result)


async def run_units_command(command: str, args: argparse.Namespace, api_instance):
    """Run units module commands."""
    if command == "convert-advanced":
        result = await units.handle_units_tools(
            "convert_units_advanced",
            {
                "value": args.value,
                "from_units": args.from_units,
                "to_units": args.to_units,
                "validate_compatibility": getattr(args, "validate_compatibility", True),
            },
            api_instance,
        )
    elif command == "unit-factors":
        result = await units.handle_units_tools(
            "get_unit_conversion_factors",
            {
                "unit_type": getattr(args, "unit_type", "all"),
                "include_factors": getattr(args, "include_factors", True),
            },
            api_instance,
        )
    elif command == "physics-constants":
        result = await units.handle_units_tools(
            "get_physics_constants",
            {
                "constant_name": getattr(args, "constant_name", "all"),
                "include_description": getattr(args, "include_description", True),
            },
            api_instance,
        )
    elif command == "validate-compatibility":
        result = await units.handle_units_tools(
            "validate_unit_compatibility",
            {
                "unit1": args.unit1,
                "unit2": args.unit2,
                "explain_incompatibility": getattr(
                    args, "explain_incompatibility", True
                ),
            },
            api_instance,
        )
    elif command == "unit-info":
        result = await units.handle_units_tools(
            "get_unit_info",
            {
                "unit": args.unit,
                "include_examples": getattr(args, "include_examples", True),
            },
            api_instance,
        )
    elif command == "natural-units":
        result = await units.handle_units_tools(
            "convert_between_natural_units",
            {
                "value": args.value,
                "conversion_type": args.conversion_type,
                "input_units": args.input_units,
                "output_units": args.output_units,
            },
            api_instance,
        )
    elif command == "common-conversions":
        result = await units.handle_units_tools(
            "get_common_conversions",
            {
                "category": getattr(args, "category", "all"),
                "include_examples": getattr(args, "include_examples", True),
            },
            api_instance,
        )
    else:
        print(f"Unknown units command: {command}")
        return

    print_json_result(result)


async def run_utils_command(command: str, args: argparse.Namespace, api_instance):
    """Run utils module commands."""
    if command == "parse-pdg-id":
        result = await utils.handle_utils_tools(
            "parse_pdg_identifier", {"pdgid": args.pdgid}, api_instance
        )
    elif command == "base-pdg-id":
        result = await utils.handle_utils_tools(
            "get_base_pdg_id", {"pdgid": args.pdgid}, api_instance
        )
    elif command == "make-pdg-id":
        result = await utils.handle_utils_tools(
            "make_pdg_identifier",
            {"baseid": args.baseid, "edition": getattr(args, "edition", None)},
            api_instance,
        )
    elif command == "find-best-property":
        result = await utils.handle_utils_tools(
            "find_best_property",
            {
                "particle_name": args.particle,
                "property_type": args.property_type,
                "pedantic": getattr(args, "pedantic", False),
            },
            api_instance,
        )
    elif command == "pdg-rounding":
        result = await utils.handle_utils_tools(
            "apply_pdg_rounding",
            {"value": args.value, "error": args.error},
            api_instance,
        )
    elif command == "linked-data":
        result = await utils.handle_utils_tools(
            "get_linked_data",
            {
                "particle_name": args.particle,
                "link_type": args.link_type,
                "limit": getattr(args, "limit", 10),
            },
            api_instance,
        )
    elif command == "normalize-data":
        result = await utils.handle_utils_tools(
            "normalize_pdg_data",
            {
                "data_input": args.data_input,
                "data_type": args.data_type,
                "strict": getattr(args, "strict", False),
            },
            api_instance,
        )
    elif command == "table-data":
        result = await utils.handle_utils_tools(
            "get_pdg_table_data",
            {
                "table_name": args.table_name,
                "row_id": args.row_id,
                "include_metadata": getattr(args, "include_metadata", True),
            },
            api_instance,
        )
    else:
        print(f"Unknown utils command: {command}")
        return

    print_json_result(result)


async def run_measurement_command(command: str, args: argparse.Namespace, api_instance):
    """Run measurement module commands."""
    if command == "measurement-details":
        result = await measurement.handle_measurement_tools(
            "get_measurement_details",
            {
                "measurement_id": args.measurement_id,
                "include_values": getattr(args, "include_values", True),
                "include_reference": getattr(args, "include_reference", True),
                "include_footnotes": getattr(args, "include_footnotes", True),
            },
            api_instance,
        )
    elif command == "value-details":
        result = await measurement.handle_measurement_tools(
            "get_measurement_value_details",
            {
                "value_id": args.value_id,
                "include_error_breakdown": getattr(
                    args, "include_error_breakdown", True
                ),
            },
            api_instance,
        )
    elif command == "reference-details":
        result = await measurement.handle_measurement_tools(
            "get_reference_details",
            {
                "reference_id": args.reference_id,
                "include_doi": getattr(args, "include_doi", True),
            },
            api_instance,
        )
    elif command == "search-by-reference":
        result = await measurement.handle_measurement_tools(
            "search_measurements_by_reference",
            {
                "particle_name": args.particle,
                "publication_year": getattr(args, "publication_year", None),
                "doi": getattr(args, "doi", None),
                "author": getattr(args, "author", None),
                "limit": getattr(args, "limit", 10),
            },
            api_instance,
        )
    elif command == "footnote-details":
        result = await measurement.handle_measurement_tools(
            "get_footnote_details",
            {
                "footnote_id": args.footnote_id,
                "include_references": getattr(args, "include_references", True),
            },
            api_instance,
        )
    elif command == "analyze-errors":
        result = await measurement.handle_measurement_tools(
            "analyze_measurement_errors",
            {
                "particle_name": args.particle,
                "property_type": args.property_type,
                "limit": getattr(args, "limit", 20),
            },
            api_instance,
        )
    elif command == "measurements-for-particle":
        result = await measurement.handle_measurement_tools(
            "get_measurements_for_particle",
            {
                "particle_name": args.particle,
                "property_type": getattr(args, "property_type", "all"),
                "include_references": getattr(args, "include_references", True),
                "limit": getattr(args, "limit", 50),
            },
            api_instance,
        )
    elif command == "compare-techniques":
        result = await measurement.handle_measurement_tools(
            "compare_measurement_techniques",
            {
                "particle_name": args.particle,
                "property_type": args.property_type,
                "limit": getattr(args, "limit", 20),
            },
            api_instance,
        )
    else:
        print(f"Unknown measurement command: {command}")
        return

    print_json_result(result)


def create_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="PDG MCP Server CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modules and commands:

API Module:
  search               Search for particles
  properties          Get particle properties  
  list                List particles by type
  compare             Compare multiple particles
  database-info       Get database information
  canonical-name      Get canonical particle name
  editions            List PDG editions

Data Module:
  mass-measurements   Get mass measurements
  lifetime-measurements Get lifetime measurements
  width-measurements  Get width measurements
  summary-values      Get summary values
  convert-units       Convert between units
  data-type-keys      Get data type keys
  value-type-keys     Get value type keys

Measurement Module:
  measurement-details Get detailed measurement information
  value-details       Get measurement value details with errors
  reference-details   Get publication reference details
  search-by-reference Search measurements by reference criteria
  footnote-details    Get footnote text and references
  analyze-errors      Analyze measurement error components
  measurements-for-particle Get all measurements for a particle
  compare-techniques  Compare measurement techniques

Particle Module:
  quantum-numbers     Get particle quantum numbers (J, P, C, G, I)
  check-properties    Check particle classification and entries
  particle-list-criteria List particles by specific criteria
  properties-detailed Get comprehensive particle properties
  analyze-item        Analyze PDG items from decay descriptions
  mass-details        Get detailed mass information with all entries
  lifetime-details    Get detailed lifetime information with all entries
  width-details       Get detailed width information with all entries
  compare-quantum-numbers Compare quantum numbers across particles
  particle-error-info Get error information for particle properties

Units Module:
  convert-advanced    Advanced unit conversion with validation
  unit-factors        Get available unit conversion factors
  physics-constants   Get physics constants (ħ, c, etc.)
  validate-compatibility Check unit compatibility for conversion
  unit-info           Get detailed information about specific units
  natural-units       Convert between natural units (E↔λ, E↔τ, etc.)
  common-conversions  Get common unit conversions in particle physics

Utils Module:
  parse-pdg-id        Parse PDG identifier into base ID and edition
  base-pdg-id         Get base part of PDG identifier
  make-pdg-id         Create normalized PDG identifier with edition
  find-best-property  Find best property using PDG criteria
  pdg-rounding        Apply PDG rounding rules to values and errors
  linked-data         Get linked data from PDG database tables
  normalize-data      Normalize and validate PDG data structures
  table-data          Get raw data from PDG database tables

Decay Module:
  branching-fractions Get branching fractions
  decay-products      Get decay products
  branching-ratios    Get branching ratios  
  decay-analysis      Analyze decay structure
  decay-details       Get decay mode details

Error Module:
  validate            Validate PDG identifier
  error-info          Get error information
  diagnose            Diagnose lookup issues
  safe-lookup         Safe particle lookup

Examples:
  python pdg_cli.py search pi+
  python pdg_cli.py properties --particle e-
  python pdg_cli.py mass-measurements --particle proton
  python pdg_cli.py branching-fractions --particle tau-
  python pdg_cli.py validate --identifier S008
  python pdg_cli.py measurement-details --measurement-id 12345
  python pdg_cli.py search-by-reference --particle electron --publication-year 2020
  python pdg_cli.py analyze-errors --particle muon --property-type mass
  python pdg_cli.py quantum-numbers --particle proton
  python pdg_cli.py compare-quantum-numbers --particles e- mu- tau-
  python pdg_cli.py particle-list-criteria --type baryon --charge 1
  python pdg_cli.py convert-advanced --value 1 --from-units MeV --to-units GeV
  python pdg_cli.py physics-constants --constant-name hbar
  python pdg_cli.py validate-compatibility --unit1 GeV --unit2 MeV
  python pdg_cli.py parse-pdg-id --pdgid "S008/2024"
  python pdg_cli.py find-best-property --particle proton --property-type mass
  python pdg_cli.py pdg-rounding --value 1.23456 --error 0.00123
        """,
    )

    parser.add_argument("command", help="Command to execute")

    # Common arguments
    parser.add_argument("--particle", help="Particle name")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--limit", type=int, help="Limit number of results")

    # API module arguments
    parser.add_argument(
        "--search-type",
        choices=["name", "mcid", "pdgid", "auto"],
        default="auto",
        help="Type of search (default: auto)",
    )
    parser.add_argument(
        "--measurements", action="store_true", help="Include measurements"
    )
    parser.add_argument(
        "--type",
        choices=["all", "baryon", "meson", "lepton", "boson", "quark"],
        help="Particle type filter",
    )
    parser.add_argument("--particles", nargs="+", help="List of particles to compare")
    parser.add_argument(
        "--properties",
        nargs="+",
        choices=["mass", "lifetime", "charge", "spin", "quantum_numbers"],
        help="Properties to compare",
    )
    parser.add_argument("--name", help="Particle name for canonical lookup")

    # Data module arguments
    parser.add_argument(
        "--summary", action="store_true", default=True, help="Include summary values"
    )
    parser.add_argument("--units", help="Units for measurements")
    parser.add_argument(
        "--property-type",
        choices=["mass", "lifetime", "width", "all"],
        help="Type of property",
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="Only summary table values"
    )
    parser.add_argument("--value", type=float, help="Value to convert")
    parser.add_argument("--from-units", help="Source units")
    parser.add_argument("--to-units", help="Target units")
    parser.add_argument(
        "--as-text", action="store_true", default=True, help="Return as text format"
    )

    # Decay module arguments
    parser.add_argument(
        "--decay-type", choices=["exclusive", "inclusive", "all"], help="Type of decay"
    )
    parser.add_argument("--mode-number", type=int, help="Specific decay mode number")
    parser.add_argument(
        "--subdecays", action="store_true", default=True, help="Include subdecays"
    )
    parser.add_argument("--max-depth", type=int, help="Maximum decay depth")

    # Error module arguments
    parser.add_argument("--identifier", help="PDG identifier to validate")
    parser.add_argument(
        "--check-data",
        action="store_true",
        default=True,
        help="Check data availability",
    )
    parser.add_argument(
        "--suggest", action="store_true", default=True, help="Suggest alternatives"
    )
    parser.add_argument(
        "--error-type",
        choices=[
            "all",
            "PdgApiError",
            "PdgInvalidPdgIdError",
            "PdgNoDataError",
            "PdgAmbiguousValueError",
            "PdgRoundingError",
        ],
        help="Specific error type",
    )
    parser.add_argument(
        "--lookup-type",
        choices=["particle_name", "pdg_id", "mcid", "property"],
        help="Type of lookup",
    )
    parser.add_argument(
        "--suggestions", action="store_true", default=True, help="Include suggestions"
    )
    parser.add_argument(
        "--alternatives", action="store_true", default=True, help="Return alternatives"
    )
    parser.add_argument(
        "--error-details",
        action="store_true",
        default=True,
        help="Include error details",
    )

    # New measurement module arguments
    parser.add_argument("--measurement-id", type=int, help="PDG measurement ID")
    parser.add_argument("--value-id", type=int, help="PDG value ID")
    parser.add_argument("--reference-id", type=int, help="PDG reference ID")
    parser.add_argument("--footnote-id", type=int, help="PDG footnote ID")
    parser.add_argument(
        "--include-values",
        action="store_true",
        default=True,
        help="Include measurement values",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        default=True,
        help="Include reference information",
    )
    parser.add_argument(
        "--include-footnotes",
        action="store_true",
        default=True,
        help="Include footnotes",
    )
    parser.add_argument(
        "--include-error-breakdown",
        action="store_true",
        default=True,
        help="Include detailed error breakdown",
    )
    parser.add_argument(
        "--include-doi",
        action="store_true",
        default=True,
        help="Include DOI and external links",
    )
    parser.add_argument("--publication-year", type=int, help="Publication year filter")
    parser.add_argument("--doi", help="DOI to search for")
    parser.add_argument("--author", help="Author name to search for")
    parser.add_argument(
        "--include-references",
        action="store_true",
        default=True,
        help="Include references",
    )

    # New particle module arguments
    parser.add_argument(
        "--include-all",
        action="store_true",
        default=True,
        help="Include all quantum numbers and additional info",
    )
    parser.add_argument("--charge", type=float, help="Filter by charge")
    parser.add_argument(
        "--has-mass", action="store_true", help="Filter particles with mass"
    )
    parser.add_argument(
        "--has-lifetime", action="store_true", help="Filter particles with lifetime"
    )
    parser.add_argument(
        "--has-width", action="store_true", help="Filter particles with width"
    )
    parser.add_argument("--data-type-filter", help="Data type filter (%% for all)")
    parser.add_argument(
        "--require-summary",
        action="store_true",
        default=True,
        help="Require summary data",
    )
    parser.add_argument(
        "--in-summary-table",
        action="store_true",
        help="Filter properties in Summary Table",
    )
    parser.add_argument("--item-name", help="PDG item name to analyze")
    parser.add_argument(
        "--include-associated",
        action="store_true",
        default=True,
        help="Include associated particles",
    )
    parser.add_argument(
        "--quantum-numbers",
        nargs="+",
        choices=["J", "P", "C", "G", "I", "all"],
        help="Quantum numbers to compare",
    )
    parser.add_argument(
        "--include-asymmetric",
        action="store_true",
        default=True,
        help="Include asymmetric error information",
    )

    # Units module arguments
    parser.add_argument(
        "--validate-compatibility",
        action="store_true",
        default=True,
        help="Validate unit compatibility",
    )
    parser.add_argument(
        "--unit-type", choices=["all", "energy", "time"], help="Filter by unit type"
    )
    parser.add_argument(
        "--include-factors",
        action="store_true",
        default=True,
        help="Include conversion factors",
    )
    parser.add_argument(
        "--constant-name",
        choices=["all", "hbar", "hbar_gev_s"],
        help="Specific physics constant",
    )
    parser.add_argument(
        "--include-description",
        action="store_true",
        default=True,
        help="Include descriptions",
    )
    parser.add_argument("--unit1", help="First unit for compatibility check")
    parser.add_argument("--unit2", help="Second unit for compatibility check")
    parser.add_argument(
        "--explain-incompatibility",
        action="store_true",
        default=True,
        help="Explain incompatibility",
    )
    parser.add_argument("--unit", help="Unit to get information about")
    parser.add_argument(
        "--include-examples",
        action="store_true",
        default=True,
        help="Include conversion examples",
    )
    parser.add_argument(
        "--conversion-type",
        choices=[
            "energy_to_length",
            "energy_to_time",
            "mass_to_energy",
            "length_to_energy",
            "time_to_energy",
            "energy_to_mass",
        ],
        help="Type of natural unit conversion",
    )
    parser.add_argument("--input-units", help="Input units for natural conversion")
    parser.add_argument("--output-units", help="Output units for natural conversion")
    parser.add_argument(
        "--category",
        choices=["all", "energy", "time", "mass", "length"],
        help="Category of common conversions",
    )

    # Utils module arguments
    parser.add_argument("--pdgid", help="PDG identifier to parse or manipulate")
    parser.add_argument("--baseid", help="Base PDG identifier")
    parser.add_argument("--edition", help="PDG edition")
    parser.add_argument(
        "--pedantic",
        action="store_true",
        default=False,
        help="Use strict/pedantic criteria",
    )
    parser.add_argument("--error", type=float, help="Error value for PDG rounding")
    parser.add_argument(
        "--link-type",
        choices=["measurements", "references", "footnotes", "values"],
        help="Type of linked data",
    )
    parser.add_argument("--data-input", help="Data to normalize")
    parser.add_argument(
        "--data-type",
        choices=["particle_name", "pdg_id", "value", "measurement"],
        help="Type of data to normalize",
    )
    parser.add_argument(
        "--strict", action="store_true", default=False, help="Use strict normalization"
    )
    parser.add_argument("--table-name", help="PDG database table name")
    parser.add_argument("--row-id", type=int, help="Database row ID")
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        default=True,
        help="Include table metadata",
    )

    return parser


async def async_main():
    """Main CLI entry point (async version)."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup PDG connection
    api_instance = setup_pdg_connection()

    # Determine which module the command belongs to
    api_commands = [
        "search",
        "properties",
        "list",
        "compare",
        "database-info",
        "canonical-name",
        "editions",
    ]
    data_commands = [
        "mass-measurements",
        "lifetime-measurements",
        "width-measurements",
        "summary-values",
        "convert-units",
        "data-type-keys",
        "value-type-keys",
    ]
    decay_commands = [
        "branching-fractions",
        "decay-products",
        "branching-ratios",
        "decay-analysis",
        "decay-details",
    ]
    error_commands = ["validate", "error-info", "diagnose", "safe-lookup"]
    measurement_commands = [
        "measurement-details",
        "value-details",
        "reference-details",
        "search-by-reference",
        "footnote-details",
        "analyze-errors",
        "measurements-for-particle",
        "compare-techniques",
    ]
    particle_commands = [
        "quantum-numbers",
        "check-properties",
        "particle-list-criteria",
        "properties-detailed",
        "analyze-item",
        "mass-details",
        "lifetime-details",
        "width-details",
        "compare-quantum-numbers",
        "particle-error-info",
    ]
    units_commands = [
        "convert-advanced",
        "unit-factors",
        "physics-constants",
        "validate-compatibility",
        "unit-info",
        "natural-units",
        "common-conversions",
    ]
    utils_commands = [
        "parse-pdg-id",
        "base-pdg-id",
        "make-pdg-id",
        "find-best-property",
        "pdg-rounding",
        "linked-data",
        "normalize-data",
        "table-data",
    ]

    command = args.command

    try:
        if command in api_commands:
            await run_api_command(command, args, api_instance)
        elif command in data_commands:
            await run_data_command(command, args, api_instance)
        elif command in decay_commands:
            await run_decay_command(command, args, api_instance)
        elif command in error_commands:
            await run_error_command(command, args, api_instance)
        elif command in measurement_commands:
            await run_measurement_command(command, args, api_instance)
        elif command in particle_commands:
            await run_particle_command(command, args, api_instance)
        elif command in units_commands:
            await run_units_command(command, args, api_instance)
        elif command in utils_commands:
            await run_utils_command(command, args, api_instance)
        else:
            print(f"Unknown command: {command}")
            print("Use --help to see available commands")
            sys.exit(1)

    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point (sync wrapper)."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

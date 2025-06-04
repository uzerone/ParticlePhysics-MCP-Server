#!/usr/bin/env python3
"""
PDG MCP Server Usage Examples

This file demonstrates various ways to use the PDG MCP server
for particle physics research tasks.
"""

import asyncio
import json
from typing import Any, Dict

# Import the MCP server module
import pdg_mcp_server as pdg_server


async def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to call a tool and return parsed JSON result."""
    try:
        result = await pdg_server.handle_call_tool(tool_name, arguments)
        if result and len(result) > 0:
            return json.loads(result[0].text)
        return {"error": "No result returned"}
    except Exception as e:
        return {"error": f"Tool call failed: {str(e)}"}


def print_example_header(title: str):
    """Print a formatted header for examples."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {title}")
    print("=" * 60)


async def example_basic_particle_search():
    """Example 1: Basic particle search and information retrieval."""
    print_example_header("Basic Particle Search")

    # Search for different particles using different methods
    particles_to_search = [
        {"query": "electron", "method": "name"},
        {"query": "211", "method": "Monte Carlo ID"},
        {"query": "S008", "method": "PDG ID"},
    ]

    for particle in particles_to_search:
        print(
            f"\nSearching for particle using {particle['method']}: {particle['query']}"
        )
        result = await call_tool("search_particle", {"query": particle["query"]})

        if "error" not in result:
            for particle_info in result:
                if "name" in particle_info:
                    print(f"  Found: {particle_info['name']}")
                    print(f"  MCID: {particle_info.get('mcid', 'N/A')}")
                    print(f"  Charge: {particle_info.get('charge', 'N/A')}")
                    print(f"  Mass: {particle_info.get('mass', 'N/A')}")
        else:
            print(f"  Error: {result['error']}")


async def example_lepton_family_study():
    """Example 2: Study the lepton family properties."""
    print_example_header("Lepton Family Study")

    # Get detailed properties of all charged leptons
    leptons = ["electron", "muon", "tau-"]

    print("Comparing charged lepton properties:\n")

    # Compare leptons
    comparison_result = await call_tool(
        "compare_particles",
        {
            "particle_names": leptons,
            "properties": ["mass", "lifetime", "charge", "spin"],
        },
    )

    if "error" not in comparison_result:
        for particle in comparison_result["particles"]:
            print(f"{particle['name']}:")
            for prop, value in particle.items():
                if prop != "name":
                    print(f"  {prop}: {value}")
            print()

    # Get tau decay modes (tau is the only lepton that decays)
    print("\nTau decay modes:")
    tau_decays = await call_tool(
        "get_branching_fractions",
        {"particle_name": "tau-", "decay_type": "exclusive", "limit": 10},
    )

    if "error" not in tau_decays:
        for i, decay in enumerate(tau_decays["decay_modes"][:5], 1):
            print(f"  {i}. {decay['description']}")
            print(f"     Branching fraction: {decay['display_value']}")


async def example_meson_decay_analysis():
    """Example 3: Analyze meson decay patterns."""
    print_example_header("Meson Decay Analysis")

    # Study kaon decay modes
    kaon_types = ["K+", "K-", "K0"]

    for kaon in kaon_types:
        print(f"\n{kaon} decay modes:")

        decays = await call_tool(
            "get_branching_fractions",
            {"particle_name": kaon, "decay_type": "exclusive", "limit": 5},
        )

        if "error" not in decays:
            for i, decay in enumerate(decays["decay_modes"], 1):
                print(f"  {i}. {decay['description']}")
                print(f"     BR: {decay['display_value']}")
                if decay["is_limit"]:
                    print("     (This is a limit, not a measurement)")
        else:
            print(f"  Error getting decays: {decays['error']}")


async def example_heavy_quark_physics():
    """Example 4: Heavy quark physics - B meson studies."""
    print_example_header("Heavy Quark Physics - B Mesons")

    # Compare B meson masses
    b_mesons = ["B+", "B0", "B_s0", "B_c+"]

    print("B meson mass comparison:")
    comparison = await call_tool(
        "compare_particles",
        {"particle_names": b_mesons, "properties": ["mass", "lifetime"]},
    )

    if "error" not in comparison:
        for particle in comparison["particles"]:
            if "error" not in particle:
                print(
                    f"  {particle['name']:<6}: Mass = {particle.get('mass', 'N/A'):<15} Lifetime = {particle.get('lifetime', 'N/A')}"
                )

    # Look at B+ decay modes involving J/psi
    print(f"\nB+ decays involving J/psi:")
    b_decays = await call_tool(
        "get_branching_fractions",
        {"particle_name": "B+", "decay_type": "exclusive", "limit": 50},
    )

    if "error" not in b_decays:
        jpsi_decays = [
            decay
            for decay in b_decays["decay_modes"]
            if "J/psi" in decay["description"]
        ]

        for i, decay in enumerate(jpsi_decays[:5], 1):
            print(f"  {i}. {decay['description']}")
            print(f"     BR: {decay['display_value']}")


async def example_particle_discovery_timeline():
    """Example 5: Look at fundamental particles and their properties."""
    print_example_header("Fundamental Particles Overview")

    # Get information about fundamental particles
    fundamental_particles = {
        "Quarks": ["u", "d", "c", "s", "t", "b"],
        "Leptons": ["electron", "muon", "tau-", "nu_e", "nu_mu", "nu_tau"],
        "Gauge Bosons": ["photon", "W+", "W-", "Z0", "g"],
    }

    for category, particles in fundamental_particles.items():
        print(f"\n{category}:")

        for particle_name in particles:
            properties = await call_tool(
                "get_particle_properties", {"particle_name": particle_name}
            )

            if "error" not in properties:
                mass = properties.get("mass", "N/A")
                charge = properties.get("charge", "N/A")
                print(f"  {particle_name:<12}: Mass = {mass:<15} Charge = {charge}")
            else:
                print(f"  {particle_name:<12}: Could not retrieve data")


async def example_monte_carlo_id_reference():
    """Example 6: Monte Carlo ID reference for common particles."""
    print_example_header("Monte Carlo ID Reference")

    # Common Monte Carlo IDs
    common_mcids = [
        (11, "electron"),
        (13, "muon"),
        (15, "tau"),
        (22, "photon"),
        (211, "pi+"),
        (321, "K+"),
        (2212, "proton"),
        (2112, "neutron"),
    ]

    print("Common particle Monte Carlo IDs:")
    print(f"{'MCID':<8} {'Name':<12} {'Mass (GeV)':<15} {'Charge'}")
    print("-" * 50)

    for mcid, expected_name in common_mcids:
        particle = await call_tool("get_particle_by_mcid", {"mcid": mcid})

        if "error" not in particle:
            name = particle.get("name", "N/A")
            mass = particle.get("mass", "N/A")
            charge = particle.get("charge", "N/A")
            print(f"{mcid:<8} {name:<12} {mass:<15} {charge}")
        else:
            print(f"{mcid:<8} {expected_name:<12} {'Error':<15} {'N/A'}")


async def example_research_workflow():
    """Example 7: A complete research workflow example."""
    print_example_header("Complete Research Workflow")

    print("Research Question: What are the main decay modes of D mesons?")
    print("\nStep 1: List available D mesons")

    # First, let's find D mesons
    d_mesons = ["D+", "D0", "D_s+"]

    for d_meson in d_mesons:
        print(f"\nAnalyzing {d_meson}:")

        # Get basic properties
        properties = await call_tool(
            "get_particle_properties", {"particle_name": d_meson}
        )

        if "error" not in properties:
            print(f"  Mass: {properties.get('mass', 'N/A')}")
            print(f"  Lifetime: {properties.get('lifetime', 'N/A')}")

        # Get main decay modes
        decays = await call_tool(
            "get_branching_fractions",
            {"particle_name": d_meson, "decay_type": "exclusive", "limit": 3},
        )

        if "error" not in decays and decays["decay_modes"]:
            print("  Top decay modes:")
            for i, decay in enumerate(decays["decay_modes"], 1):
                print(f"    {i}. {decay['description']} ({decay['display_value']})")
        else:
            print("  No decay data available")


async def example_database_exploration():
    """Example 8: Explore the database contents and capabilities."""
    print_example_header("Database Exploration")

    # Get database info
    db_info = await call_tool("get_database_info", {})

    if "error" not in db_info:
        print("PDG Database Information:")
        for key, value in db_info.items():
            if key != "info_keys":
                print(f"  {key}: {value}")

    # List particles by type
    particle_types = ["baryon", "meson", "lepton", "boson"]

    print(f"\nParticle counts by type:")
    for ptype in particle_types:
        particles = await call_tool(
            "list_particles", {"particle_type": ptype, "limit": 100}
        )

        if "error" not in particles:
            count = particles.get("count", 0)
            print(f"  {ptype.capitalize()}s: {count} particles")


async def example_advanced_measurements():
    """Example 9: Advanced PDG measurements and summary values."""
    print_example_header("Advanced PDG Measurements")

    # Get detailed mass measurements for electron
    print("Electron mass measurements:")
    mass_data = await call_tool(
        "get_mass_measurements",
        {"particle_name": "electron", "include_summary_values": True, "units": "MeV"},
    )

    if "error" not in mass_data:
        print(f"  Particle: {mass_data['particle']}")
        print(f"  Units: {mass_data['units']}")
        if "summary_values" in mass_data:
            for i, sv in enumerate(mass_data["summary_values"][:3], 1):
                if "error" not in sv:
                    print(f"  {i}. {sv['value_text']}")
                    if sv.get("converted_value"):
                        print(
                            f"     Converted: {sv['converted_value']} {sv['converted_units']}"
                        )
                    print(f"     In Summary Table: {sv['in_summary_table']}")

    # Get lifetime measurements for muon
    print(f"\nMuon lifetime measurements:")
    lifetime_data = await call_tool(
        "get_lifetime_measurements",
        {"particle_name": "muon", "include_summary_values": True, "units": "ns"},
    )

    if "error" not in lifetime_data:
        if "summary_values" in lifetime_data:
            for i, sv in enumerate(lifetime_data["summary_values"][:2], 1):
                if "error" not in sv:
                    print(f"  {i}. {sv['value_text']}")
                    if sv.get("converted_value"):
                        print(
                            f"     Converted: {sv['converted_value']} {sv['converted_units']}"
                        )

    # Unit conversion examples
    print(f"\nUnit conversion examples:")
    conversions = [
        {"value": 0.511, "from_units": "MeV", "to_units": "GeV"},
        {"value": 2.2e-6, "from_units": "s", "to_units": "ns"},
        {"value": 938.272, "from_units": "MeV", "to_units": "kg"},
    ]

    for conv in conversions:
        result = await call_tool("convert_units", conv)
        if "error" not in result:
            print(
                f"  {result['original_value']} {result['original_units']} = {result['converted_value']} {result['converted_units']}"
            )


async def example_summary_values_analysis():
    """Example 10: Detailed summary values analysis."""
    print_example_header("Summary Values Analysis")

    # Get all summary values for proton
    print("Proton summary values (all properties):")
    summary_data = await call_tool(
        "get_summary_values",
        {"particle_name": "proton", "property_type": "all", "summary_table_only": True},
    )

    if "error" not in summary_data:
        for prop_type, values in summary_data["summary_values"].items():
            if values:
                print(f"\n  {prop_type.upper()}:")
                for i, sv in enumerate(values[:2], 1):  # Limit to first 2
                    if "error" not in sv:
                        print(f"    {i}. {sv['value_text']}")
                        if sv.get("comment"):
                            print(f"       Comment: {sv['comment']}")
                        print(f"       Value type: {sv.get('value_type', 'N/A')}")

    # Get property details for W boson mass
    print(f"\nW boson mass property details:")
    prop_details = await call_tool(
        "get_property_details", {"particle_name": "W+", "property_type": "mass"}
    )

    if "error" not in prop_details:
        for i, prop in enumerate(prop_details["properties"][:1], 1):  # Just first one
            if "error" not in prop:
                print(f"  Property {i}:")
                print(f"    PDG ID: {prop['pdgid']}")
                print(f"    Description: {prop['description']}")
                print(f"    Data type: {prop['data_type']}")
                if "best_summary" in prop:
                    best = prop["best_summary"]
                    print(f"    Best value: {best['value_text']}")
                    print(
                        f"    Error: +{best.get('error_positive', 'N/A')} -{best.get('error_negative', 'N/A')}"
                    )


async def example_pdg_metadata_exploration():
    """Example 11: Explore PDG database metadata and structure."""
    print_example_header("PDG Database Metadata Exploration")

    # Get available PDG editions
    print("Available PDG Review editions:")
    editions_result = await call_tool("get_editions", {})
    if "error" not in editions_result:
        print(f"  Default edition: {editions_result['default_edition']}")
        print(f"  Total editions: {editions_result['total_editions']}")
        print(f"  Available: {editions_result['available_editions']}")

    # Get data type keys
    print(f"\nPDG Data Type Keys (sample):")
    data_types_result = await call_tool("get_data_type_keys", {"as_text": True})
    if "error" not in data_types_result:
        keys_text = data_types_result["data_type_keys"]
        # Show first few lines only
        lines = keys_text.split("\n")[:6]
        for line in lines:
            print(f"  {line}")
        print("  ...")

    # Get canonical names for common particles
    print(f"\nCanonical particle names:")
    test_particles = ["e-", "mu-", "pi+", "K0"]
    for particle in test_particles:
        canonical_result = await call_tool("get_canonical_name", {"name": particle})
        if "error" not in canonical_result:
            canonical_name = canonical_result["canonical_name"]
            print(f"  {particle} -> {canonical_name}")
        else:
            print(f"  {particle} -> Error: {canonical_result['error']}")

    # Get all identifiers for mass measurements (limited sample)
    print(f"\nMass-type PDG identifiers (sample):")
    identifiers_result = await call_tool(
        "get_all_pdg_identifiers", {"data_type_key": "M", "limit": 5}
    )
    if "error" not in identifiers_result:
        print(f"  Found {identifiers_result['count']} identifiers:")
        for identifier in identifiers_result["identifiers"]:
            print(f"    {identifier['pdgid']}: {identifier['description']}")
            print(f"      Type: {identifier['data_type']}")

    # Search for particles by name pattern
    print(f"\nParticles with 'pi' in their names:")
    particles_result = await call_tool(
        "get_particles_by_name", {"name": "pi", "case_sensitive": False, "limit": 3}
    )
    if "error" not in particles_result:
        print(f"  Found {particles_result['count']} particles:")
        for particle in particles_result["particles"]:
            if "error" not in particle:
                print(f"    {particle['name']}: Mass = {particle.get('mass', 'N/A')}")

    # Get specific PDG object by identifier
    print(f"\nStandard Model PDG entry (S008):")
    pdg_obj_result = await call_tool("get_pdg_by_identifier", {"pdgid": "S008"})
    if "error" not in pdg_obj_result:
        print(f"  Type: {pdg_obj_result['object_type']}")
        print(f"  Description: {pdg_obj_result.get('description', 'N/A')}")
        if "content" in pdg_obj_result:
            content = pdg_obj_result["content"]
            # Show first 100 characters
            print(
                f"  Content: {content[:100]}..."
                if len(content) > 100
                else f"  Content: {content}"
            )


async def main():
    """Run all examples."""
    print("PDG MCP Server Usage Examples")
    print("These examples demonstrate various research scenarios")
    print("using the PDG particle physics database.\n")


async def example_advanced_decay_analysis():
    """Example 12: Advanced decay analysis using PDG decay module."""
    print_example_header("Advanced Decay Analysis")

    # Get detailed decay products for tau
    print("Tau decay products with detailed information:")
    decay_products = await call_tool(
        "get_decay_products",
        {"particle_name": "tau-", "decay_type": "exclusive", "include_subdecays": True},
    )

    if "error" not in decay_products:
        print(f"Found {decay_products['total_modes']} decay modes:")
        for i, mode in enumerate(decay_products["decay_modes"][:3], 1):  # Show first 3
            print(f"\n  {i}. {mode['description']}")
            print(f"     Mode #{mode['mode_number']}, BR: {mode['display_value']}")
            print(f"     Subdecay level: {mode['subdecay_level']}")
            print(f"     Products ({mode['num_products']}):")
            for product in mode["decay_products"]:
                if "error" not in product:
                    mult = (
                        f"{product['multiplier']}×" if product["multiplier"] > 1 else ""
                    )
                    print(f"       - {mult}{product['item_name']}")

    # Analyze decay structure for B meson
    print(f"\nB+ meson decay structure analysis:")
    structure_analysis = await call_tool(
        "analyze_decay_structure",
        {"particle_name": "B+", "max_depth": 2, "decay_type": "exclusive"},
    )

    if "error" not in structure_analysis:
        print(f"Analyzed {structure_analysis['total_analyzed']} decay chains:")
        for i, decay in enumerate(
            structure_analysis["decay_structure"][:2], 1
        ):  # Show first 2
            print(f"\n  {i}. Depth {decay['depth']}: {decay['description']}")
            print(f"     BR: {decay['branching_fraction']}")
            print(
                f"     Mode: {decay['mode_number']}, Subdecay: {decay['is_subdecay']}"
            )
            if decay.get("products"):
                for product in decay["products"][:3]:  # First 3 products
                    mult = (
                        f"{product['multiplier']}×" if product["multiplier"] > 1 else ""
                    )
                    print(f"     → {mult}{product['item_name']}")

    # Get decay mode details for kaon
    print(f"\nK+ decay mode details:")
    mode_details = await call_tool(
        "get_decay_mode_details",
        {"particle_name": "K+", "show_subdecays": True, "limit": 4},
    )

    if "error" not in mode_details:
        print(f"Found {mode_details['total_modes']} decay modes:")
        for i, mode in enumerate(mode_details["decay_modes"], 1):
            print(f"  {i}. Mode {mode['mode_number']}: {mode['description']}")
            print(f"     BR: {mode['branching_fraction']}")
            print(
                f"     Subdecay: {mode['is_subdecay']} (level {mode['subdecay_level']})"
            )
            print(f"     Products: {mode['num_products']}")

    # Get branching ratios for D meson
    print(f"\nD+ branching ratios:")
    ratios = await call_tool(
        "get_branching_ratios", {"particle_name": "D+", "limit": 3}
    )

    if "error" not in ratios:
        print(f"Found {ratios['total_found']} branching ratios:")
        for i, ratio in enumerate(ratios["branching_ratios"], 1):
            print(f"  {i}. {ratio['description']}")
            print(f"     Value: {ratio['display_value']}")
            print(f"     Associated with: {ratio['associated_bf_description']}")

    # List of example functions
    examples = [
        example_basic_particle_search,
        example_lepton_family_study,
        example_meson_decay_analysis,
        example_heavy_quark_physics,
        example_particle_discovery_timeline,
        example_monte_carlo_id_reference,
        example_research_workflow,
        example_database_exploration,
        example_advanced_measurements,
        example_summary_values_analysis,
        example_pdg_metadata_exploration,
        example_advanced_decay_analysis,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            await example_func()

            # Pause between examples for readability
            if i < len(examples):
                input(f"\nPress Enter to continue to example {i+1}...")

        except Exception as e:
            print(f"\nError in example {i}: {e}")
            continue

    print(f"\n{'='*60}")
    print("EXAMPLES COMPLETED")
    print("=" * 60)
    print("\nThese examples show just a fraction of what's possible with")
    print("the PDG MCP server. You can adapt these patterns for your")
    print("own research needs!")
    print("\nFor more information:")
    print("- Check the README.md file")
    print("- Run the test suite: python test_modular.py")
    print("- Explore the PDG documentation: https://pdgapi.lbl.gov/doc/")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()

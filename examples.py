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


async def main():
    """Run all examples."""
    print("PDG MCP Server Usage Examples")
    print("These examples demonstrate various research scenarios")
    print("using the PDG particle physics database.\n")

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
    print("- Run the test suite: python test_pdg_server.py")
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

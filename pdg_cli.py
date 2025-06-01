#!/usr/bin/env python3
"""
PDG MCP Server CLI Wrapper
Provides terminal-style commands for easy testing and usage
"""

import argparse
import asyncio
import json
import sys
from typing import List, Optional

import pdg_mcp_server as server

async def search_particle(query: str, search_type: str = "auto") -> None:
    """Search for a particle"""
    result = await server.handle_call_tool('search_particle', {
        'query': query,
        'search_type': search_type
    })
    data = json.loads(result[0].text)
    
    if isinstance(data, list):
        for particle in data:
            if 'error' in particle:
                print(f"Error: {particle['error']}")
            else:
                print(f"Name: {particle.get('name', 'N/A')}")
                print(f"MCID: {particle.get('mcid', 'N/A')}")
                print(f"Mass: {particle.get('mass', 'N/A')}")
                print(f"Charge: {particle.get('charge', 'N/A')}")
                print()
    else:
        print(json.dumps(data, indent=2))

async def get_particle_properties(particle_name: str, include_measurements: bool = False) -> None:
    """Get particle properties"""
    result = await server.handle_call_tool('get_particle_properties', {
        'particle_name': particle_name,
        'include_measurements': include_measurements
    })
    data = json.loads(result[0].text)
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        print(f"Particle: {data.get('name', 'N/A')}")
        print(f"Mass: {data.get('mass', 'N/A')}")
        print(f"Lifetime: {data.get('lifetime', 'N/A')}")
        print(f"Charge: {data.get('charge', 'N/A')}")
        print(f"Width: {data.get('width', 'N/A')}")
        
        if 'quantum_numbers' in data:
            print("Quantum numbers:")
            for qn, value in data['quantum_numbers'].items():
                print(f"  {qn}: {value}")

async def get_branching_fractions(particle_name: str, decay_type: str = "exclusive", limit: int = 20) -> None:
    """Get branching fractions"""
    result = await server.handle_call_tool('get_branching_fractions', {
        'particle_name': particle_name,
        'decay_type': decay_type,
        'limit': limit
    })
    data = json.loads(result[0].text)
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        print(f"Decay modes for {data['particle']} (found {data['total_found']}):")
        for i, decay in enumerate(data['decay_modes'], 1):
            print(f"{i:2d}. {decay['description']}")
            print(f"    BR: {decay['display_value']}")
            if decay.get('is_limit'):
                print("    (This is a limit)")
            print()

async def list_particles(particle_type: str = "all", limit: int = 50) -> None:
    """List particles"""
    result = await server.handle_call_tool('list_particles', {
        'particle_type': particle_type,
        'limit': limit
    })
    data = json.loads(result[0].text)
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        print(f"Particles (type: {data['filter']}, showing {data['count']}):")
        for particle in data['particles']:
            print(f"  {particle['name']:<15} MCID: {str(particle.get('mcid', 'N/A')):<10} Mass: {particle.get('mass', 'N/A')}")

async def get_particle_by_mcid(mcid: int) -> None:
    """Get particle by Monte Carlo ID"""
    result = await server.handle_call_tool('get_particle_by_mcid', {
        'mcid': mcid
    })
    data = json.loads(result[0].text)
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        print(f"MCID {mcid}:")
        print(f"  Name: {data.get('name', 'N/A')}")
        print(f"  Mass: {data.get('mass', 'N/A')}")
        print(f"  Charge: {data.get('charge', 'N/A')}")

async def compare_particles(particle_names: List[str], properties: List[str] = None) -> None:
    """Compare particles"""
    if properties is None:
        properties = ["mass", "lifetime", "charge"]
    
    result = await server.handle_call_tool('compare_particles', {
        'particle_names': particle_names,
        'properties': properties
    })
    data = json.loads(result[0].text)
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        print("Particle comparison:")
        print(f"{'Name':<15} " + " ".join(f"{prop:<15}" for prop in properties))
        print("-" * (15 + sum(16 for _ in properties)))
        
        for particle in data['particles']:
            if 'error' in particle:
                print(f"{particle['name']:<15} Error: {particle['error']}")
            else:
                row = f"{particle['name']:<15} "
                for prop in properties:
                    value = particle.get(prop, 'N/A')
                    if isinstance(value, dict):
                        value = str(value)
                    row += f"{str(value):<15} "
                print(row)

async def get_database_info() -> None:
    """Get database information"""
    result = await server.handle_call_tool('get_database_info', {})
    data = json.loads(result[0].text)
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        print("PDG Database Information:")
        for key, value in data.items():
            if key != 'info_keys':
                print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description='PDG MCP Server CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # search_particle
    search_parser = subparsers.add_parser('search', help='Search for a particle')
    search_parser.add_argument('--query', required=True, help='Particle name, MCID, or PDG ID')
    search_parser.add_argument('--type', choices=['name', 'mcid', 'pdgid', 'auto'], default='auto', help='Search type')

    # get_particle_properties
    props_parser = subparsers.add_parser('properties', help='Get particle properties')
    props_parser.add_argument('--particle', required=True, help='Particle name')
    props_parser.add_argument('--measurements', action='store_true', help='Include measurements')

    # get_branching_fractions
    br_parser = subparsers.add_parser('decays', help='Get branching fractions')
    br_parser.add_argument('--particle', required=True, help='Particle name')
    br_parser.add_argument('--type', choices=['exclusive', 'inclusive', 'all'], default='exclusive', help='Decay type')
    br_parser.add_argument('--limit', type=int, default=20, help='Maximum results')

    # list_particles
    list_parser = subparsers.add_parser('list', help='List particles')
    list_parser.add_argument('--type', choices=['all', 'baryon', 'meson', 'lepton', 'boson', 'quark'], default='all', help='Particle type')
    list_parser.add_argument('--limit', type=int, default=50, help='Maximum results')

    # get_particle_by_mcid
    mcid_parser = subparsers.add_parser('mcid', help='Get particle by Monte Carlo ID')
    mcid_parser.add_argument('--id', type=int, required=True, help='Monte Carlo ID')

    # compare_particles
    compare_parser = subparsers.add_parser('compare', help='Compare particles')
    compare_parser.add_argument('--particles', nargs='+', required=True, help='Particle names')
    compare_parser.add_argument('--properties', nargs='+', default=['mass', 'lifetime', 'charge'], help='Properties to compare')

    # get_database_info
    subparsers.add_parser('info', help='Get database information')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'search':
            asyncio.run(search_particle(args.query, args.type))
        elif args.command == 'properties':
            asyncio.run(get_particle_properties(args.particle, args.measurements))
        elif args.command == 'decays':
            asyncio.run(get_branching_fractions(args.particle, args.type, args.limit))
        elif args.command == 'list':
            asyncio.run(list_particles(args.type, args.limit))
        elif args.command == 'mcid':
            asyncio.run(get_particle_by_mcid(args.id))
        elif args.command == 'compare':
            asyncio.run(compare_particles(args.particles, args.properties))
        elif args.command == 'info':
            asyncio.run(get_database_info())
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
ParticlePhysics MCP Server Help System

This script generates comprehensive documentation about all available tools
to help LLMs understand the capabilities of the MCP server.
"""

import json
import sys
from typing import Dict, List, Any

# Import all modules to get their tools
from modules import api, data, decay, errors, measurement, particle, units, utils


def format_tool_info(tool) -> Dict[str, Any]:
    """Format tool information in a structured way."""
    tool_info = {
        "name": tool.name,
        "description": tool.description,
        "parameters": {}
    }
    
    # Extract parameters from inputSchema
    if hasattr(tool, 'inputSchema') and 'properties' in tool.inputSchema:
        for param_name, param_info in tool.inputSchema['properties'].items():
            tool_info["parameters"][param_name] = {
                "type": param_info.get("type", "unknown"),
                "description": param_info.get("description", "No description"),
                "required": param_name in tool.inputSchema.get("required", []),
                "default": param_info.get("default", None),
                "enum": param_info.get("enum", None),
                "min": param_info.get("minimum", None),
                "max": param_info.get("maximum", None)
            }
    
    return tool_info


def generate_markdown_help() -> str:
    """Generate markdown-formatted help documentation."""
    md_lines = []
    
    # Header
    md_lines.append("# ParticlePhysics MCP Server - Available Tools\n")
    md_lines.append("This document lists all available tools in the ParticlePhysics MCP Server.")
    md_lines.append("Each tool includes its description and parameter details.\n")
    
    # Table of Contents
    md_lines.append("## Table of Contents\n")
    modules_info = [
        ("API Tools", api.get_api_tools()),
        ("Data Tools", data.get_data_tools()),
        ("Decay Tools", decay.get_decay_tools()),
        ("Error Tools", errors.get_error_tools()),
        ("Measurement Tools", measurement.get_measurement_tools()),
        ("Particle Tools", particle.get_particle_tools()),
        ("Units Tools", units.get_units_tools()),
        ("Utils Tools", utils.get_utils_tools())
    ]
    
    for module_name, _ in modules_info:
        anchor = module_name.lower().replace(" ", "-")
        md_lines.append(f"- [{module_name}](#{anchor})")
    
    md_lines.append("\n---\n")
    
    # Generate documentation for each module
    for module_name, tools in modules_info:
        anchor = module_name.lower().replace(" ", "-")
        md_lines.append(f"## {module_name}\n")
        
        for tool in tools:
            tool_info = format_tool_info(tool)
            
            # Tool name and description
            md_lines.append(f"### `{tool_info['name']}`\n")
            md_lines.append(f"**Description:** {tool_info['description']}\n")
            
            # Parameters
            if tool_info['parameters']:
                md_lines.append("**Parameters:**\n")
                for param_name, param_info in tool_info['parameters'].items():
                    required_mark = "**[Required]**" if param_info['required'] else "[Optional]"
                    md_lines.append(f"- `{param_name}` {required_mark}")
                    md_lines.append(f"  - Type: `{param_info['type']}`")
                    md_lines.append(f"  - Description: {param_info['description']}")
                    
                    if param_info['default'] is not None:
                        md_lines.append(f"  - Default: `{param_info['default']}`")
                    
                    if param_info['enum']:
                        md_lines.append(f"  - Allowed values: {', '.join([f'`{v}`' for v in param_info['enum']])}")
                    
                    if param_info['min'] is not None:
                        md_lines.append(f"  - Minimum: `{param_info['min']}`")
                    
                    if param_info['max'] is not None:
                        md_lines.append(f"  - Maximum: `{param_info['max']}`")
                    
                    md_lines.append("")
            
            md_lines.append("---\n")
    
    return "\n".join(md_lines)


def generate_json_help() -> Dict[str, Any]:
    """Generate JSON-formatted help documentation."""
    help_data = {
        "server": "ParticlePhysics MCP Server",
        "version": "1.0.0",
        "description": "Model Context Protocol server for Particle Data Group (PDG) physics data",
        "total_tools": 0,
        "modules": {}
    }
    
    modules_info = [
        ("api", api.get_api_tools()),
        ("data", data.get_data_tools()),
        ("decay", decay.get_decay_tools()),
        ("errors", errors.get_error_tools()),
        ("measurement", measurement.get_measurement_tools()),
        ("particle", particle.get_particle_tools()),
        ("units", units.get_units_tools()),
        ("utils", utils.get_utils_tools())
    ]
    
    total_tools = 0
    
    for module_name, tools in modules_info:
        module_tools = []
        for tool in tools:
            tool_info = format_tool_info(tool)
            module_tools.append(tool_info)
            total_tools += 1
        
        help_data["modules"][module_name] = {
            "description": f"Tools for {module_name} operations",
            "tool_count": len(module_tools),
            "tools": module_tools
        }
    
    help_data["total_tools"] = total_tools
    
    return help_data


def generate_simple_list() -> str:
    """Generate a simple list of all tools with brief descriptions."""
    lines = []
    lines.append("PARTICLEPHYSICS MCP SERVER - TOOL LIST")
    lines.append("=" * 60)
    lines.append("")
    
    modules_info = [
        ("API", api.get_api_tools()),
        ("Data", data.get_data_tools()),
        ("Decay", decay.get_decay_tools()),
        ("Error", errors.get_error_tools()),
        ("Measurement", measurement.get_measurement_tools()),
        ("Particle", particle.get_particle_tools()),
        ("Units", units.get_units_tools()),
        ("Utils", utils.get_utils_tools())
    ]
    
    total_tools = 0
    
    for module_name, tools in modules_info:
        lines.append(f"{module_name} Module ({len(tools)} tools):")
        lines.append("-" * 40)
        
        for tool in tools:
            lines.append(f"  • {tool.name}")
            lines.append(f"    {tool.description}")
            lines.append("")
            total_tools += 1
        
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"TOTAL TOOLS: {total_tools}")
    
    return "\n".join(lines)


def generate_llm_prompt_helper() -> str:
    """Generate a helper text specifically formatted for LLM consumption."""
    lines = []
    lines.append("# ParticlePhysics MCP Server Tool Reference for LLMs\n")
    lines.append("## Overview")
    lines.append("The ParticlePhysics MCP Server provides access to the Particle Data Group (PDG) database.")
    lines.append("Below are all available tools grouped by category.\n")
    
    modules_info = [
        ("API Tools - General particle data queries", api.get_api_tools()),
        ("Data Tools - Detailed measurements and values", data.get_data_tools()),
        ("Decay Tools - Particle decay information", decay.get_decay_tools()),
        ("Error Tools - Error handling and validation", errors.get_error_tools()),
        ("Measurement Tools - Experimental measurements", measurement.get_measurement_tools()),
        ("Particle Tools - Particle properties and quantum numbers", particle.get_particle_tools()),
        ("Units Tools - Unit conversions for physics", units.get_units_tools()),
        ("Utils Tools - Utility functions and helpers", utils.get_utils_tools())
    ]
    
    lines.append("## Quick Reference\n")
    
    # Common use cases
    lines.append("### Common Use Cases:")
    lines.append("- **Find particle info:** Use `search_particle` with particle name/ID")
    lines.append("- **Get mass/lifetime:** Use `get_particle_properties` or specific measurement tools")
    lines.append("- **List decay modes:** Use `get_branching_fractions` or `get_decay_products`")
    lines.append("- **Convert units:** Use `convert_units_advanced`")
    lines.append("- **Compare particles:** Use `compare_particles`\n")
    
    lines.append("## Tool Categories\n")
    
    for module_desc, tools in modules_info:
        module_name = module_desc.split(" - ")[0]
        lines.append(f"### {module_desc}\n")
        
        for tool in tools:
            tool_info = format_tool_info(tool)
            
            # Simple format for LLM consumption
            lines.append(f"**{tool_info['name']}**")
            lines.append(f"- Purpose: {tool_info['description']}")
            
            # Show required parameters
            required_params = [p for p, info in tool_info['parameters'].items() if info['required']]
            optional_params = [p for p, info in tool_info['parameters'].items() if not info['required']]
            
            if required_params:
                lines.append(f"- Required: {', '.join(required_params)}")
            
            if optional_params:
                lines.append(f"- Optional: {', '.join(optional_params)}")
            
            lines.append("")
    
    # Add examples section
    lines.append("## Example Tool Calls\n")
    lines.append("```json")
    lines.append('// Search for a particle')
    lines.append('{')
    lines.append('  "tool": "search_particle",')
    lines.append('  "arguments": {')
    lines.append('    "query": "electron"')
    lines.append('  }')
    lines.append('}')
    lines.append("")
    lines.append('// Get particle properties')
    lines.append('{')
    lines.append('  "tool": "get_particle_properties",')
    lines.append('  "arguments": {')
    lines.append('    "particle_name": "muon",')
    lines.append('    "include_measurements": true')
    lines.append('  }')
    lines.append('}')
    lines.append("")
    lines.append('// Convert units')
    lines.append('{')
    lines.append('  "tool": "convert_units_advanced",')
    lines.append('  "arguments": {')
    lines.append('    "value": 0.511,')
    lines.append('    "from_units": "MeV",')
    lines.append('    "to_units": "GeV"')
    lines.append('  }')
    lines.append('}')
    lines.append("```")
    
    return "\n".join(lines)


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate help documentation for ParticlePhysics MCP Server tools"
    )
    parser.add_argument(
        "--format", 
        choices=["markdown", "json", "simple", "llm"],
        default="simple",
        help="Output format (default: simple)"
    )
    parser.add_argument(
        "--output",
        help="Output file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    # Generate help in requested format
    if args.format == "markdown":
        output = generate_markdown_help()
    elif args.format == "json":
        output = json.dumps(generate_json_help(), indent=2)
    elif args.format == "llm":
        output = generate_llm_prompt_helper()
    else:  # simple
        output = generate_simple_list()
    
    # Output to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Help documentation written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main() 
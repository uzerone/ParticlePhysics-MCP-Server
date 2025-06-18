# ParticlePhysics MCP Server

A **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). This production-ready server enables AI assistants and applications to query comprehensive particle physics information through 64 specialized tools across 8 modules with enterprise-grade security, caching, and performance features.

## Installation (No Local Setup Required)

### For Claude Desktop/IDE Users:
Simply add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "uv",
      "args": [
        "tool",
        "run",
        "--from",
        "git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git",
        "pp-mcp-server"
      ]
    }
  }
}
```

**That's it!** No local installation needed. The server will be automatically downloaded and run when needed.

## Available Tools (64 Total)

### API Module (11 tools)
1. **search_particle** - Advanced particle search by name, Monte Carlo ID, or PDG ID with fuzzy matching
2. **get_particle_properties** - Get comprehensive particle properties with enhanced metadata and validation
3. **list_particles** - List particles with advanced filtering, sorting, and pagination
4. **get_particle_by_mcid** - Get particle information using Monte Carlo particle ID with validation
5. **compare_particles** - Advanced particle comparison with statistical analysis and visualization data
6. **get_database_info** - Get comprehensive PDG database information and metadata
7. **get_canonical_name** - Get canonical PDG name with alternative name suggestions
8. **get_particles_by_name** - Advanced name-based particle search with fuzzy matching and filtering
9. **get_editions** - Get comprehensive PDG Review editions information with metadata
10. **get_pdg_by_identifier** - Enhanced PDG data object retrieval with validation and metadata
11. **get_all_pdg_identifiers** - Get filtered and paginated PDG identifiers with enhanced metadata

### Data Module (11 tools)
12. **get_mass_measurements** - Get detailed mass measurements and summary values with enhanced analysis
13. **get_lifetime_measurements** - Get detailed lifetime measurements with decay analysis
14. **get_width_measurements** - Get detailed width measurements for unstable particles
15. **get_summary_values** - Get comprehensive summary values with detailed metadata and validation
16. **get_measurements_by_property** - Get detailed individual measurements with comprehensive analysis
17. **convert_units** - Advanced particle physics unit conversion with validation and constants
18. **get_particle_text** - Get comprehensive text information and descriptions with enhanced formatting
19. **get_property_details** - Get comprehensive property information with enhanced metadata analysis
20. **get_data_type_keys** - Get comprehensive PDG data type keys with enhanced documentation
21. **get_value_type_keys** - Get comprehensive PDG summary value type keys with metadata
22. **get_key_documentation** - Get comprehensive documentation for PDG database keys and flags

### Decay Module (5 tools)
23. **get_branching_fractions** - Get comprehensive branching fractions with enhanced analysis and uncertainty propagation
24. **get_decay_products** - Get detailed decay products with comprehensive subdecay analysis and particle flow tracking
25. **get_branching_ratios** - Get comprehensive branching ratios with enhanced analysis and correlations
26. **get_decay_mode_details** - Get comprehensive decay mode information with enhanced classification and analysis
27. **analyze_decay_structure** - Advanced hierarchical decay structure analysis with visualization data and pattern recognition

### Error Module (4 tools)
28. **validate_pdg_identifier** - Comprehensive PDG identifier validation with enhanced error analysis and suggestions
29. **get_error_info** - Get information about PDG API error types and their meanings
30. **diagnose_lookup_issues** - Diagnose common issues with particle or data lookups
31. **safe_particle_lookup** - Safely lookup particle with comprehensive error handling and alternatives

### Measurement Module (8 tools)
32. **get_measurement_details** - Get detailed information about a specific PDG measurement including values and references
33. **get_measurement_value_details** - Get detailed information about a specific measurement value including errors and units
34. **get_reference_details** - Get detailed publication information for a PDG reference
35. **search_measurements_by_reference** - Search for measurements by reference properties (year, DOI, etc.)
36. **get_footnote_details** - Get footnote text and associated references
37. **analyze_measurement_errors** - Analyze error components (statistical vs systematic) across multiple measurements
38. **get_measurements_for_particle** - Get all measurements for a specific particle with detailed breakdown
39. **compare_measurement_techniques** - Compare different measurement techniques for a particle property

### Particle Module (10 tools)
40. **get_particle_quantum_numbers** - Get quantum numbers (spin, parity, isospin, etc.) for a particle
41. **check_particle_properties** - Check particle classification (is_baryon, is_meson, is_lepton, etc.)
42. **get_particle_list_by_criteria** - Get list of particles matching specific criteria (type, charge, etc.)
43. **get_particle_properties_detailed** - Get comprehensive particle properties including all available data
44. **analyze_particle_item** - Analyze PDG items from decay descriptions and product lists
45. **get_particle_mass_details** - Get detailed mass information including all mass entries for a particle
46. **get_particle_lifetime_details** - Get detailed lifetime information including all lifetime entries for a particle
47. **get_particle_width_details** - Get detailed width information including all width entries for a particle
48. **compare_particle_quantum_numbers** - Compare quantum numbers across multiple particles
49. **get_particle_error_info** - Get error information for particle mass, lifetime, and width

### Units Module (7 tools)
50. **convert_units_advanced** - Convert particle physics values between different units with validation
51. **get_unit_conversion_factors** - Get available unit conversion factors and their base units
52. **get_physics_constants** - Get physics constants used in particle physics calculations
53. **validate_unit_compatibility** - Check if two units are compatible for conversion
54. **get_unit_info** - Get detailed information about a specific unit
55. **convert_between_natural_units** - Convert between natural units common in particle physics
56. **get_common_conversions** - Get common unit conversions used in particle physics

### Utils Module (8 tools)
57. **parse_pdg_identifier** - Parse PDG Identifier and return base identifier and edition with validation
58. **get_base_pdg_id** - Get the normalized base part of a PDG identifier
59. **make_pdg_identifier** - Create a normalized full PDG identifier with optional edition
60. **find_best_property** - Find the 'best' property from a list based on enhanced PDG criteria with detailed analysis
61. **apply_pdg_rounding** - Apply PDG rounding rules to value and error with detailed analysis and formatting options
62. **get_linked_data** - Get linked data from PDG database tables
63. **normalize_pdg_data** - Normalize and validate PDG data structures
64. **get_pdg_table_data** - Get raw data from PDG database tables

## Maintainers

This project is maintained by:
- [@uzerone](https://github.com/uzerone)
- [@bee4come](https://github.com/bee4come)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

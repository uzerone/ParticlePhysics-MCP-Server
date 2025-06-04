# PDG MCP Server

A comprehensive **Model Context Protocol (MCP) server** providing seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). Perfect for researchers, students, and AI assistants working with particle physics.

## 🚀 Features

- **64 MCP Tools** across **8 specialized modules**
- **Complete PDG API Coverage** with modular organization
- **Multiple Installation Methods**: uvx, pip, or direct
- **CLI Interface** for terminal usage
- **Production Ready** with comprehensive testing
- **Python 3.10+** required (MCP dependency)

## 📦 Quick Installation

### Option 1: uvx (Recommended)
```bash
uvx --from git+https://github.com/uzerone/PDG-MCP-Server.git pdg-mcp-server
```

### Option 2: Direct Python/pip
```bash
git clone https://github.com/uzerone/PDG-MCP-Server.git
cd PDG-MCP-Server
pip install -r requirements.txt
python pdg_mcp_server.py
```

### Option 3: Development Setup
```bash
git clone https://github.com/uzerone/PDG-MCP-Server.git
cd PDG-MCP-Server
pip install -r requirements.txt
python test_modular.py  # Run tests
```

## 🛠️ MCP Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "pdg": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/uzerone/PDG-MCP-Server.git", "pdg-mcp-server"]
    }
  }
}
```

## 💬 Example Queries for AI Assistants

### Basic Research
- *"What is the mass of the proton?"*
- *"Find information about the electron"*
- *"Compare the masses of all quarks"*

### Advanced Analysis
- *"What are the main decay modes of the tau lepton?"*
- *"Get electron mass measurements in MeV with error bars"*
- *"Analyze the hierarchical structure of B+ meson decay chains"*
- *"Compare quantum numbers of leptons"*

### PDG Data & Utilities
- *"Convert 0.511 MeV to GeV using PDG units"*
- *"Parse PDG identifier S008/2024"*
- *"Validate PDG identifier and suggest alternatives"*
- *"Apply PDG rounding rules to experimental data"*

## 🖥️ Terminal CLI Usage

### Quick Commands
```bash
# Basic particle operations
./pdg search --query "e-"
./pdg properties --particle "proton"
./pdg mass-measurements --particle "electron" --units "MeV"

# Advanced analysis
./pdg decay-products --particle "tau-" --type exclusive
./pdg quantum-numbers --particle "proton"
./pdg convert-advanced --value 0.511 --from-units MeV --to-units GeV

# Utilities
./pdg parse-pdg-id --pdgid "S008/2024"
./pdg find-best-property --particle proton --property-type mass
./pdg validate --identifier S008 --check-data
```

## 🔧 Key Tool Categories

### Core Physics Tools
- **search_particle** - Search by name, MCID, or PDG ID
- **get_particle_properties** - Detailed particle properties
- **get_mass_measurements** - Mass with error bars and units
- **get_branching_fractions** - Decay modes and branching ratios
- **get_particle_quantum_numbers** - Quantum numbers (J, P, C, G, I)

### Data Analysis Tools
- **convert_units_advanced** - Physics unit conversions with validation
- **get_summary_values** - PDG summary table values
- **analyze_decay_structure** - Hierarchical decay analysis
- **compare_particles** - Side-by-side particle comparison
- **get_measurements_by_property** - Individual measurements with references

### Utility Tools
- **parse_pdg_identifier** - Parse PDG IDs into components
- **validate_pdg_identifier** - Validate and diagnose PDG IDs
- **apply_pdg_rounding** - Apply PDG rounding rules
- **safe_particle_lookup** - Safe lookup with error handling

## 🐍 Python API Example

```python
import asyncio
import json
import pdg_mcp_server as pdg_server

async def research_example():
    # Compare lepton masses
    result = await pdg_server.handle_call_tool('compare_particles', {
        'particle_names': ['e-', 'mu-', 'tau-'],
        'properties': ['mass', 'lifetime', 'charge']
    })
    data = json.loads(result[0].text)
    
    for particle in data['particles']:
        print(f"{particle['name']}: {particle['mass']}")
    
    # Get tau decay modes
    result = await pdg_server.handle_call_tool('get_branching_fractions', {
        'particle_name': 'tau-',
        'decay_type': 'exclusive',
        'limit': 5
    })
    decays = json.loads(result[0].text)
    
    for decay in decays['decay_modes']:
        print(f"{decay['description']}: {decay['display_value']}")

asyncio.run(research_example())
```

## 🧪 Testing & Development

```bash
# Run all tests
python test_modular.py

# Test specific functionality
python pdg_cli.py search pi+
python pdg_cli.py quantum-numbers --particle proton

# Run examples
python examples.py
```

## 📊 Complete Tool Reference

### API Module (11 tools)
- `search_particle` - Search for particles by name, MCID, or PDG ID
- `get_particle_properties` - Get detailed particle properties
- `list_particles` - List particles by type with filtering
- `get_particle_by_mcid` - Get particle by Monte Carlo ID
- `compare_particles` - Compare multiple particles side-by-side
- `get_database_info` - Get PDG database information
- `get_canonical_name` - Get canonical particle names
- `get_particles_by_name` - Search particles by name patterns
- `get_editions` - List available PDG editions
- `get_pdg_by_identifier` - Get particle by PDG identifier
- `get_all_pdg_identifiers` - List all PDG identifiers

### Data Module (11 tools)
- `get_mass_measurements` - Mass measurements with errors and units
- `get_lifetime_measurements` - Lifetime measurements with units
- `get_width_measurements` - Width measurements for unstable particles
- `get_summary_values` - PDG summary table values
- `get_measurements_by_property` - Individual measurements with references
- `convert_units` - Convert between physics units
- `get_particle_text` - Get text descriptions and reviews
- `get_property_details` - Detailed property information
- `get_data_type_keys` - Available data type keys
- `get_value_type_keys` - Available value type keys
- `get_key_documentation` - Documentation for specific keys

### Measurement Module (8 tools)
- `get_measurement_details` - Detailed measurement information
- `get_measurement_value_details` - Value details with error breakdown
- `get_reference_details` - Publication reference information
- `search_measurements_by_reference` - Search by publication criteria
- `get_footnote_details` - Footnote text and references
- `analyze_measurement_errors` - Error component analysis
- `get_measurements_for_particle` - All measurements for a particle
- `compare_measurement_techniques` - Compare measurement methods

### Particle Module (10 tools)
- `get_particle_quantum_numbers` - Quantum numbers (J, P, C, G, I)
- `check_particle_properties` - Particle classification checks
- `get_particle_list_by_criteria` - Filter particles by criteria
- `get_particle_properties_detailed` - Comprehensive properties
- `analyze_particle_item` - Analyze PDG items from descriptions
- `get_particle_mass_details` - Detailed mass information
- `get_particle_lifetime_details` - Detailed lifetime information
- `get_particle_width_details` - Detailed width information
- `compare_particle_quantum_numbers` - Compare quantum numbers
- `get_particle_error_info` - Error information for properties

### Units Module (7 tools)
- `convert_units_advanced` - Advanced unit conversion with validation
- `get_unit_conversion_factors` - Available conversion factors
- `get_physics_constants` - Physics constants (ħ, c, etc.)
- `validate_unit_compatibility` - Check unit compatibility
- `get_unit_info` - Detailed unit information
- `convert_between_natural_units` - Natural unit conversions
- `get_common_conversions` - Common physics unit conversions

### Utils Module (8 tools)
- `parse_pdg_identifier` - Parse PDG IDs into components
- `get_base_pdg_id` - Get base PDG identifier
- `make_pdg_identifier` - Create normalized PDG identifiers
- `find_best_property` - Find best property using PDG criteria
- `apply_pdg_rounding` - Apply PDG rounding rules
- `get_linked_data` - Get linked database data
- `normalize_pdg_data` - Normalize and validate data
- `get_pdg_table_data` - Access raw database tables

### Decay Module (5 tools)
- `get_branching_fractions` - Branching fractions (exclusive/inclusive)
- `get_decay_products` - Decay products with subdecay support
- `get_branching_ratios` - Branching ratios between modes
- `get_decay_mode_details` - Detailed decay mode information
- `analyze_decay_structure` - Hierarchical decay analysis

### Error Module (4 tools)
- `validate_pdg_identifier` - Validate PDG identifiers
- `get_error_info` - Error type information
- `diagnose_lookup_issues` - Diagnose lookup problems
- `safe_particle_lookup` - Safe lookup with error handling

## 📊 Common Particle Reference

| Particle | Name | MCID | Mass (GeV) | Charge |
|----------|------|------|------------|--------|
| Electron | `e-` | 11 | 0.000511 | -1 |
| Muon | `mu-` | 13 | 0.105658 | -1 |
| Tau | `tau-` | 15 | 1.77686 | -1 |
| Proton | `p` | 2212 | 0.938272 | +1 |
| Neutron | `n` | 2112 | 0.939565 | 0 |
| Pion+ | `pi+` | 211 | 0.139570 | +1 |

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Note**: PDG data is subject to [PDG's license terms](https://pdg.lbl.gov/about/terms). Starting with 2024 edition, published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## 🔗 Links

- **PDG Website**: https://pdg.lbl.gov/
- **PDG Python API**: https://github.com/particledatagroup/api  
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Repository**: https://github.com/uzerone/PDG-MCP-Server

## 👥 Authors

- [@uzerone](https://github.com/uzerone) - Project maintainer
- [@bee4come](https://github.com/bee4come) - Project maintainer

---

**🔬 Happy particle physics research!** ⚛️
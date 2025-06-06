# PDG MCP Server

A comprehensive **Model Context Protocol (MCP) server** providing seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). Enhanced with advanced analytics, intelligent error handling, and comprehensive research tools perfect for researchers, students, and AI assistants working with particle physics.

> **🆕 Latest Update**: All 8 modules have been significantly enhanced with advanced analytics, intelligent error handling, fuzzy search capabilities, statistical validation, and comprehensive physics context. The server now provides 73+ tools with enhanced functionality based on the official PDG Python API patterns.

## 🚀 Enhanced Features

- **64+ Enhanced MCP Tools** across **8 specialized modules** with comprehensive analytics
- **Complete PDG API Coverage** with intelligent enhancements and advanced error handling
- **Advanced Data Analysis** with uncertainty propagation and statistical validation
- **Intelligent Search & Validation** with fuzzy matching and smart suggestions
- **Multiple Installation Methods**: uvx, pip, or direct development setup
- **Enhanced CLI Interface** with comprehensive commands and analysis tools
- **Production Ready** with comprehensive testing and robust error handling
- **Research Optimized** with educational content and physics context
- **Python 3.10+** required (MCP dependency)

## ✨ New Enhanced Capabilities

### 🧠 Intelligent Features
- **Smart Particle Search** with fuzzy matching and auto-detection
- **Comprehensive Error Handling** with intelligent recovery and suggestions
- **Advanced Uncertainty Analysis** with precision classification
- **Statistical Validation** of measurements and experimental data
- **Educational Context** with physics explanations and background

### 📊 Advanced Analytics
- **Decay Structure Analysis** with hierarchical visualization data
- **Conservation Law Verification** for physics validation
- **Measurement Technique Comparison** across experiments
- **Temporal Analysis** of measurement evolution
- **Quality Metrics** and reliability indicators

### 🔧 Enhanced Tools
- **Pattern Recognition** for decay signatures and particle classification
- **Unit Conversion Validation** with dimensional analysis
- **PDG Identifier Intelligence** with format analysis and suggestions
- **Safe Operations** with comprehensive fallback mechanisms
- **Reference Tracking** with citation metrics and DOI integration

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
python test_modular.py  # Run comprehensive tests
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

## 💬 Enhanced Example Queries for AI Assistants

### 🔬 Basic Research (Enhanced)
- *"What is the mass of the proton with uncertainty analysis?"*
- *"Find comprehensive information about the electron including quantum numbers"*
- *"Compare the masses of all quarks with statistical analysis and uncertainty propagation"*
- *"Search for 'muon' with fuzzy matching and alternative suggestions"*

### 🧮 Advanced Physics Analysis
- *"Analyze the complete decay structure of B+ mesons with subdecay visualization"*
- *"Get electron mass measurements from different experiments with technique comparison"*
- *"Perform conservation law analysis for tau lepton decay modes"*
- *"Compare quantum numbers of all leptons with detailed explanations"*
- *"Analyze branching fraction uncertainties for D meson decays"*

### 🔍 Intelligent Data Queries
- *"Validate PDG identifier 'S008/2024' with format analysis and suggestions"*
- *"Find best mass property for the proton using PDG criteria"*
- *"Diagnose lookup issues for particle name 'elektron' with smart suggestions"*
- *"Get measurement error component analysis for muon lifetime"*

### 🧰 Enhanced Utilities & Conversions
- *"Convert 0.511 MeV to GeV with dimensional validation and physics context"*
- *"Apply PDG rounding rules to experimental data with decision analysis"*
- *"Parse compound PDG identifier 'M100/2024' with comprehensive validation"*
- *"Get unit conversion factors between natural units with physics constants"*

### 📈 Research & Analysis Workflows
- *"Perform temporal analysis of Z boson mass measurements over time"*
- *"Analyze decay pattern signatures for exotic particle searches"*
- *"Compare measurement techniques across different experiments for W boson width"*
- *"Generate comprehensive particle classification report for all baryons"*

## 🖥️ Enhanced Terminal CLI Usage

### Quick Commands with Enhanced Features
```bash
# Enhanced particle operations with smart search
./pdg search --query "elektron" --fuzzy-match --include-suggestions
./pdg properties --particle "proton" --include-quantum-numbers --include-decays
./pdg mass-measurements --particle "electron" --units "MeV" --include-error-analysis

# Advanced decay analysis with visualization data
./pdg decay-structure --particle "B+" --max-depth 3 --include-visualization
./pdg branching-fractions --particle "tau-" --include-uncertainty-analysis --sort-by branching_fraction
./pdg conservation-analysis --particle "D0" --include-verification

# Enhanced quantum and particle analysis
./pdg quantum-numbers --particle "proton" --include-jpc-notation --with-explanations
./pdg particle-classification --criteria "is_baryon=True" --include-metadata
./pdg compare-particles --particles "e-,mu-,tau-" --include-ratios --include-uncertainties

# Intelligent utilities and validation
./pdg validate-identifier --pdgid "S008" --check-data-availability --suggest-alternatives
./pdg diagnose-lookup --query "elektron" --include-suggestions --analyze-patterns
./pdg convert-advanced --value 0.511 --from-units MeV --to-units GeV --include-validation

# Advanced measurement analysis
./pdg measurement-techniques --particle "muon" --property "lifetime" --compare-methods
./pdg error-analysis --particle "Z0" --property "mass" --include-systematic-errors
./pdg reference-search --particle "higgs" --year-filter "2012-2024" --include-doi
```

## 🔧 Enhanced Tool Categories

### 🔬 Core Physics Tools (Enhanced)
- **search_particle** - Intelligent search with fuzzy matching and auto-detection
- **get_particle_properties** - Comprehensive properties with enhanced metadata
- **get_mass_measurements** - Mass with detailed error analysis and uncertainty propagation
- **get_branching_fractions** - Decay modes with advanced uncertainty analysis
- **get_particle_quantum_numbers** - Quantum numbers with educational context and notation

### 📊 Advanced Data Analysis Tools
- **convert_units_advanced** - Enhanced conversions with dimensional validation
- **get_summary_values** - PDG values with comprehensive metadata and quality indicators
- **analyze_decay_structure** - Hierarchical analysis with visualization data
- **compare_particles** - Advanced comparison with statistical analysis and correlations
- **get_measurements_by_property** - Measurements with technique comparison and temporal analysis

### 🛡️ Intelligent Utility Tools
- **validate_pdg_identifier** - Comprehensive validation with format analysis
- **diagnose_lookup_issues** - Smart diagnosis with pattern recognition
- **safe_particle_lookup** - Enhanced lookup with intelligent fallbacks
- **apply_pdg_rounding** - PDG rules with decision analysis and formatting

### 🔬 Research & Analytics Tools
- **analyze_measurement_errors** - Advanced error component analysis
- **compare_measurement_techniques** - Experimental method comparison
- **get_decay_mode_details** - Enhanced classification with physics analysis
- **get_physics_constants** - Constants with unit relationships and conversions

## 🐍 Enhanced Python API Example

```python
import asyncio
import json
import pdg_mcp_server as pdg_server

async def advanced_research_example():
    # Enhanced particle comparison with uncertainties
    result = await pdg_server.handle_call_tool('compare_particles', {
        'particle_names': ['e-', 'mu-', 'tau-'],
        'properties': ['mass', 'lifetime', 'charge', 'quantum_numbers'],
        'include_ratios': True,
        'include_uncertainties': True
    })
    data = json.loads(result[0].text)
    
    print("=== Enhanced Lepton Comparison ===")
    for particle in data['particles']:
        mass_info = particle['mass']
        print(f"{particle['name']}: {mass_info['formatted']} (precision: {mass_info.get('precision_class', 'N/A')})")
    
    # Property ratios with statistical analysis
    if 'property_ratios' in data:
        mass_ratios = data['property_ratios'].get('mass', {})
        print(f"\nMass ratios: {mass_ratios.get('ratios', {})}")
    
    # Advanced decay structure analysis
    result = await pdg_server.handle_call_tool('analyze_decay_structure', {
        'particle_name': 'B+',
        'max_depth': 3,
        'include_visualization_data': True,
        'include_probability_flow': True,
        'min_probability_threshold': 0.001
    })
    decay_data = json.loads(result[0].text)
    
    print(f"\n=== B+ Meson Decay Structure ===")
    print(f"Total analyzed modes: {decay_data.get('total_analyzed', 0)}")
    
    # Pattern analysis if available
    if 'pattern_analysis' in decay_data:
        patterns = decay_data['pattern_analysis']
        print(f"Decay signatures: {patterns.get('decay_signatures', {})}")
    
    # Enhanced measurement analysis with error components
    result = await pdg_server.handle_call_tool('analyze_measurement_errors', {
        'particle_name': 'muon',
        'property_type': 'lifetime',
        'limit': 5
    })
    error_data = json.loads(result[0].text)
    
    print(f"\n=== Muon Lifetime Error Analysis ===")
    if 'error_summary' in error_data:
        summary = error_data['error_summary']
        print(f"Statistical vs Systematic: {summary.get('error_breakdown', {})}")

asyncio.run(advanced_research_example())
```

## 🧪 Enhanced Testing & Development

```bash
# Run comprehensive test suite
python test_modular.py

# Test enhanced functionality with validation
python pdg_cli.py search "elektron" --fuzzy-match
python pdg_cli.py validate-identifier "s008" --suggest-alternatives
python pdg_cli.py decay-structure "tau-" --include-visualization

# Run enhanced examples with new capabilities
python examples.py  # Now includes advanced analysis examples

# Development testing with enhanced error handling
python -c "import pdg_modules; print('All enhanced modules imported successfully')"
```

## 📊 Complete Enhanced Tool Reference

### 🔬 API Module (11 Enhanced Tools)
- `search_particle` - **Enhanced**: Intelligent search with fuzzy matching, auto-detection, and confidence scoring
- `get_particle_properties` - **Enhanced**: Comprehensive properties with quantum number formatting and educational content
- `list_particles` - **Enhanced**: Advanced filtering with mass ranges, charge filters, and statistical analysis
- `get_particle_by_mcid` - **Enhanced**: Validation with related particle discovery and comprehensive metadata
- `compare_particles` - **Enhanced**: Multi-particle analysis with statistical ratios and uncertainty propagation
- `get_database_info` - **Enhanced**: Comprehensive metadata with statistics and capability documentation
- `get_canonical_name` - **Enhanced**: Intelligent name resolution with alternative suggestions
- `get_particles_by_name` - **Enhanced**: Fuzzy matching with confidence scores and pattern recognition
- `get_editions` - **Enhanced**: Edition metadata with publication details and change tracking
- `get_pdg_by_identifier` - **Enhanced**: Validation with format analysis and related object discovery
- `get_all_pdg_identifiers` - **Enhanced**: Filtered listings with metadata and category organization

### 📊 Data Module (11 Enhanced Tools)
- `get_mass_measurements` - **Enhanced**: Detailed error analysis with precision classification and derived quantities
- `get_lifetime_measurements` - **Enhanced**: Decay analysis with width relations and unit conversion factors
- `get_width_measurements` - **Enhanced**: Lifetime calculations with uncertainty propagation and physics context
- `get_summary_values` - **Enhanced**: Comprehensive metadata with validation and quality indicators
- `get_measurements_by_property` - **Enhanced**: Experimental details with technique comparison and temporal analysis
- `convert_units` - **Enhanced**: Validation with physics context and dimensional analysis
- `get_particle_text` - **Enhanced**: Formatted text with markup preservation and reference integration
- `get_property_details` - **Enhanced**: Comprehensive metadata with statistical analysis and data flags
- `get_data_type_keys` - **Enhanced**: Documentation with examples and category filtering
- `get_value_type_keys` - **Enhanced**: Descriptions with usage statistics and frequency analysis
- `get_key_documentation` - **Enhanced**: Context with examples and comprehensive usage guidance

### 🔬 Measurement Module (8 Enhanced Tools)
- `get_measurement_details` - **Enhanced**: Comprehensive information with value breakdown and metadata analysis
- `get_measurement_value_details` - **Enhanced**: Error component analysis with statistical vs systematic breakdown
- `get_reference_details` - **Enhanced**: Publication information with DOI integration and citation metrics
- `search_measurements_by_reference` - **Enhanced**: Advanced filtering with author search and temporal analysis
- `get_footnote_details` - **Enhanced**: Context with reference tracking and detailed explanations
- `analyze_measurement_errors` - **Enhanced**: Statistical analysis with precision classification and correlation study
- `get_measurements_for_particle` - **Enhanced**: Comprehensive breakdown with quality metrics and technique comparison
- `compare_measurement_techniques` - **Enhanced**: Experimental method analysis with statistical validation

### ⚛️ Particle Module (10 Enhanced Tools)
- `get_particle_quantum_numbers` - **Enhanced**: Detailed explanations with JPC notation and physics context
- `check_particle_properties` - **Enhanced**: Classification with comprehensive metadata and physics background
- `get_particle_list_by_criteria` - **Enhanced**: Advanced filtering with intelligent suggestions and statistical summaries
- `get_particle_properties_detailed` - **Enhanced**: Comprehensive analysis with uncertainty quantification and validation
- `analyze_particle_item` - **Enhanced**: Item analysis with associated particle information and context
- `get_particle_mass_details` - **Enhanced**: Detailed mass information with measurement tracking and error analysis
- `get_particle_lifetime_details` - **Enhanced**: Lifetime analysis with decay constant calculations and physics relations
- `get_particle_width_details` - **Enhanced**: Width information with lifetime calculations and uncertainty analysis
- `compare_particle_quantum_numbers` - **Enhanced**: Quantum number comparison with detailed explanations and patterns
- `get_particle_error_info` - **Enhanced**: Error analysis with asymmetric uncertainty handling and precision classification

### 🔧 Units Module (7 Enhanced Tools)
- `convert_units_advanced` - **Enhanced**: Dimensional validation with physics context and uncertainty propagation
- `get_unit_conversion_factors` - **Enhanced**: Comprehensive factors with base units and physics constants
- `get_physics_constants` - **Enhanced**: Constants with descriptions and unit relationships
- `validate_unit_compatibility` - **Enhanced**: Compatibility checking with detailed explanations and suggestions
- `get_unit_info` - **Enhanced**: Detailed information with conversion examples and physics context
- `convert_between_natural_units` - **Enhanced**: Natural unit conversions with dimensional analysis
- `get_common_conversions` - **Enhanced**: Common physics conversions with examples and validation

### 🛠️ Utils Module (8 Enhanced Tools)
- `parse_pdg_identifier` - **Enhanced**: Comprehensive parsing with format validation and detailed analysis
- `get_base_pdg_id` - **Enhanced**: Base identifier extraction with validation and normalization
- `make_pdg_identifier` - **Enhanced**: Identifier creation with format validation and edition handling
- `find_best_property` - **Enhanced**: Intelligent selection using official PDG criteria with confidence scoring
- `apply_pdg_rounding` - **Enhanced**: Rounding rules with decision analysis and formatting options
- `get_linked_data` - **Enhanced**: Database linking with metadata and comprehensive reference tracking
- `normalize_pdg_data` - **Enhanced**: Data validation with quality metrics and integrity checking
- `get_pdg_table_data` - **Enhanced**: Raw data access with metadata and comprehensive table information

### 🔬 Decay Module (5 Enhanced Tools)
- `get_branching_fractions` - **Enhanced**: Advanced uncertainty analysis with statistical validation and classification
- `get_decay_products` - **Enhanced**: Comprehensive analysis with conservation law verification and particle flow tracking
- `get_branching_ratios` - **Enhanced**: Enhanced correlations with systematic uncertainty analysis
- `get_decay_mode_details` - **Enhanced**: Classification with physics analysis and selection rule validation
- `analyze_decay_structure` - **Enhanced**: Hierarchical analysis with visualization data and pattern recognition

### 🛡️ Error Module (4 Enhanced Tools)
- `validate_pdg_identifier` - **Enhanced**: Comprehensive validation with format analysis and intelligent suggestions
- `get_error_info` - **Enhanced**: Detailed error documentation with recovery guidance and examples
- `diagnose_lookup_issues` - **Enhanced**: Pattern recognition with smart suggestions and confidence scoring
- `safe_particle_lookup` - **Enhanced**: Intelligent fallbacks with alternative search strategies and error context

## 📊 Common Particle Reference (Enhanced)

| Particle | Name | MCID | Mass (GeV) | Charge | Lifetime (s) | Key Properties |
|----------|------|------|------------|--------|--------------|----------------|
| Electron | `e-` | 11 | 0.0005109989 ± 0.0000000010 | -1 | stable | Fundamental lepton |
| Muon | `mu-` | 13 | 0.1056583745 ± 0.0000000024 | -1 | 2.1969811 × 10⁻⁶ | Heavy electron analog |
| Tau | `tau-` | 15 | 1.77686 ± 0.00012 | -1 | 2.903 × 10⁻¹³ | Heaviest lepton |
| Proton | `p` | 2212 | 0.9382720813 ± 0.0000000058 | +1 | stable | Light quark baryon |
| Neutron | `n` | 2112 | 0.9395654133 ± 0.0000000058 | 0 | 879.4 ± 0.6 | Free neutron decay |
| Pion+ | `pi+` | 211 | 0.13957039 ± 0.00000018 | +1 | 2.6033 × 10⁻⁸ | Light meson |

## 📦 Dependencies & Licensing

### Core Dependencies

This project relies on several key dependencies, each with their own licensing terms:

#### 🧬 PDG Python API (Primary Dependency)
- **Package**: `pdg` - Python package for machine-readable access to PDG data
- **Source**: Lawrence Berkeley National Laboratory (LBNL)
- **Copyright**: © 2023, The Regents of the University of California, through Lawrence Berkeley National Laboratory
- **License**: BSD-3-Clause-style License (LBNL)
- **Repository**: https://github.com/particledatagroup/api
- **Purpose**: Provides direct access to Particle Data Group database

#### 🔧 MCP Framework
- **Package**: `mcp` - Model Context Protocol implementation
- **License**: MIT License
- **Purpose**: Enables AI assistant integration via MCP protocol

#### 🐍 Python Standard Dependencies
- **asyncio**: Built-in Python async framework
- **json**: Built-in JSON handling
- **argparse**: Built-in command-line interface
- **logging**: Built-in logging functionality

### Development Dependencies
- **typing**: Type hints support
- **pathlib**: Modern path handling

### License Compliance Summary

| Component | License | Copyright Holder |
|-----------|---------|------------------|
| **PDG MCP Server** (this project) | MIT License | PDG MCP Server Contributors |
| **PDG Python API** | BSD-3-Clause-style | UC Regents / LBNL |
| **PDG Data** | CC BY 4.0 (2024+) | Particle Data Group |
| **MCP Framework** | MIT License | Model Context Protocol Contributors |

### 📋 Attribution Requirements

1. **PDG Python API**: This software depends on the PDG Python API developed by Lawrence Berkeley National Laboratory. Full license terms are included in our [LICENSE](LICENSE) file.

2. **PDG Data**: The particle physics data accessed through this server is provided by the Particle Data Group (PDG) and is subject to their licensing terms.

3. **Installation**: The PDG Python API is automatically installed as a dependency when you install this package.

### 🔒 License Compatibility

All dependencies use permissive licenses that are compatible with each other:
- MIT License (this project, MCP)
- BSD-3-Clause-style (PDG Python API)  
- CC BY 4.0 (PDG data)

This ensures the entire software stack can be used together without licensing conflicts.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for complete details including PDG Python API attribution.

**Important**: This project includes multiple license components:
- **PDG MCP Server code**: MIT License
- **PDG Python API**: BSD-3-Clause-style License (LBNL)
- **PDG Data**: CC BY 4.0 License (starting 2024 edition)

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
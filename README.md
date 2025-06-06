# PDG MCP Server

A **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). This server enables AI assistants and applications to query comprehensive particle physics information through 64 specialized tools.

## Features

- **64 MCP Tools** across 8 specialized modules
- **Complete PDG API Coverage** with intelligent enhancements
- **Advanced Search & Analysis** with fuzzy matching and validation
- **Multiple Installation Methods** - uvx, pip, or development setup
- **CLI Interface** with comprehensive commands
- **Python 3.10+ Compatible** (required for MCP)

## Quick Installation

### Option 1: uvx (Recommended)
```bash
uvx --from git+https://github.com/uzerone/PDG-MCP-Server.git pdg-mcp-server
```

### Option 2: Direct Installation
```bash
git clone https://github.com/uzerone/PDG-MCP-Server.git
cd PDG-MCP-Server
pip install -r requirements.txt
python pdg_mcp_server.py
```

## MCP Configuration

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

## Usage Examples

### Basic Particle Queries
- *"What is the mass of the proton?"*
- *"Find information about the electron"*
- *"Compare the masses of all quarks"*
- *"Search for muon decay modes"*

### Advanced Analysis
- *"Analyze the decay structure of B+ mesons"*
- *"Get electron mass measurements from different experiments"*
- *"Compare quantum numbers of all leptons"*
- *"Convert 0.511 MeV to GeV"*

## CLI Usage

```bash
# Search for particles
python pdg_cli.py search "electron"

# Get particle properties
python pdg_cli.py properties --particle "proton"

# Mass measurements
python pdg_cli.py mass-measurements --particle "muon"

# Decay analysis
python pdg_cli.py branching-fractions --particle "tau-"

# Unit conversion
python pdg_cli.py convert-units --value 1 --from-units MeV --to-units GeV
```

## Available Modules

| Module | Tools | Description |
|--------|-------|-------------|
| **API** | 11 | Core particle search and properties |
| **Data** | 11 | Mass, lifetime, and measurement data |
| **Particle** | 10 | Quantum numbers and classifications |
| **Measurement** | 8 | Experimental measurements and references |
| **Units** | 7 | Unit conversions and physics constants |
| **Utils** | 8 | PDG utilities and data processing |
| **Decay** | 5 | Branching fractions and decay analysis |
| **Errors** | 4 | Error handling and diagnostics |

## Common Particles Reference

| Particle | Symbol | Mass (GeV) | Charge | Type |
|----------|--------|------------|--------|------|
| Electron | `e-` | 0.000511 | -1 | Lepton |
| Muon | `mu-` | 0.106 | -1 | Lepton |
| Proton | `p` | 0.938 | +1 | Baryon |
| Neutron | `n` | 0.940 | 0 | Baryon |
| Pion+ | `pi+` | 0.140 | +1 | Meson |

## Testing

```bash
# Run test suite
python test_modular.py

# Test CLI
python pdg_cli.py --help

# Verify installation
python -c "import pdg; print('✓ PDG API ready')"
```

## Dependencies

- **Python 3.10+** (required for MCP)
- **PDG Python API** (automatically installed)
- **MCP Framework** (for AI assistant integration)

## Maintainers

This project is maintained by:
- [@uzerone](https://github.com/uzerone)
- [@bee4come](https://github.com/bee4come)

## License

MIT License - see [LICENSE](LICENSE) file for details.

**Note**: This project includes dependencies with different licenses:
- PDG MCP Server: MIT License
- PDG Python API: BSD-3-Clause License (Lawrence Berkeley National Laboratory)
- PDG Data: CC BY 4.0 License (Particle Data Group)

## Links

- **PDG Website**: https://pdg.lbl.gov/
- **PDG Python API**: https://github.com/particledatagroup/api  
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Repository**: https://github.com/uzerone/PDG-MCP-Server

---

**🔬 Empowering AI assistants with particle physics knowledge** ⚛️
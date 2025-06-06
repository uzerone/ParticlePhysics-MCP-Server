# ParticlePhysics MCP Server

A **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). This production-ready server enables AI assistants and applications to query comprehensive particle physics information through 60 specialized tools across 8 modules with enterprise-grade security, caching, and performance features.

## Features

- **60 MCP Tools** across 8 specialized modules
- **Enterprise Security** - Input validation, rate limiting, XSS protection
- **High Performance** - LRU caching, connection pooling, optimized algorithms
- **Production Ready** - Comprehensive configuration, logging, health monitoring
- **Complete PDG API Coverage** with intelligent enhancements
- **Advanced CLI Interface** with `pp` wrapper script for convenience
- **Python 3.10+ Compatible** (required for MCP)

## Quick Installation

### Option 1: uvx (Recommended)
```bash
uvx --from git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git pdg-mcp-server
```

### Option 2: Direct Installation
```bash
git clone https://github.com/uzerone/ParticlePhysics-MCP-Server.git
cd ParticlePhysics-MCP-Server
pip install -r requirements.txt
python pdg_mcp_server.py
```

### Option 3: Development Setup
```bash
git clone https://github.com/uzerone/ParticlePhysics-MCP-Server.git
cd ParticlePhysics-MCP-Server
pip install -e .
```

## MCP Configuration

### For Claude Desktop

Add to your `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "python",
      "args": ["/path/to/ParticlePhysics-MCP-Server/pdg_mcp_server.py"],
      "env": {
        "PDG_ENVIRONMENT": "production",
        "PDG_CACHE_ENABLED": "true",
        "PDG_RATE_LIMIT_ENABLED": "true"
      }
    }
  }
}
```

### For uvx Installation

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git", "pdg-mcp-server"]
    }
  }
}
```

### Configuration Options

The server supports environment variables for configuration:

- `PDG_ENVIRONMENT`: Set to `production` or `development` (default: `development`)
- `PDG_CACHE_ENABLED`: Enable/disable caching (default: `true`)
- `PDG_CACHE_SIZE`: Cache size limit (default: `1000`)
- `PDG_CACHE_TTL`: Cache TTL in seconds (default: `3600`)
- `PDG_RATE_LIMIT_ENABLED`: Enable/disable rate limiting (default: `true`)
- `PDG_DEBUG`: Enable debug logging (default: `false`)

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

### Using the `pp` wrapper script (recommended)

```bash
# Search for particles
./pp search --query "electron"

# Get particle properties
./pp properties --particle "proton"

# Mass measurements
./pp mass-measurements --particle "muon"

# Unit conversion (advanced)
./pp convert-advanced --value 0.511 --from-units MeV --to-units GeV

# Quantum numbers
./pp quantum-numbers --particle "proton"
```

### Using Python directly

```bash
# Search for particles
python pdg_cli.py search --query "electron"

# Get particle properties
python pdg_cli.py properties --particle "proton"

# Mass measurements
python pdg_cli.py mass-measurements --particle "muon"

# Decay analysis
python pdg_cli.py branching-fractions --particle "tau-"

# Unit conversion
python pdg_cli.py convert-advanced --value 1 --from-units MeV --to-units GeV
```

## Available Modules

| Module | Tools | Description |
|--------|-------|-------------|
| **API** | 11 | Core particle search and properties |
| **Data** | 8 | Mass, lifetime, and measurement data |
| **Particle** | 10 | Quantum numbers and classifications |
| **Measurement** | 8 | Experimental measurements and references |
| **Units** | 7 | Unit conversions and physics constants |
| **Utils** | 8 | PDG utilities and data processing |
| **Decay** | 5 | Branching fractions and decay analysis |
| **Errors** | 4 | Error handling and diagnostics |

### Enterprise Features

- **🔒 Security**: Input validation, rate limiting, XSS protection, security logging
- **⚡ Performance**: LRU cache with TTL, connection pooling, optimized algorithms  
- **🔧 Configuration**: Environment-based settings, feature flags, structured config
- **🧪 Testing**: 100% test coverage, comprehensive test suite, CI/CD integration
- **📊 Monitoring**: Health checks, performance metrics, cache statistics


## Testing

```bash
# Run comprehensive test suite (100% success rate)
python test_modular.py

# Test CLI wrapper
./pp --help

# Test CLI directly  
python pdg_cli.py --help

# Verify installation
python -c "import pdg; print('✓ PDG API ready')"

# Test MCP server
python pdg_mcp_server.py &
```

### Test Results

The project includes a comprehensive test suite with:
- ✅ **100% Test Success Rate**
- ✅ **9 Test Suites** covering all components
- ✅ **Security Testing** with XSS detection
- ✅ **Performance Testing** with benchmarking
- ✅ **Integration Testing** with live PDG API
- ✅ **Configuration Testing** with validation


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

---

**🔬 Empowering AI assistants with particle physics knowledge** ⚛️
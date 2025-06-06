# ParticlePhysics MCP Server

A **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). This production-ready server enables AI assistants and applications to query comprehensive particle physics information through 60 specialized tools across 8 modules with enterprise-grade security, caching, and performance features.

## One-Click Installation (No Local Setup Required!)

### For Claude Desktop/IDE Users:
Simply add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git",
        "pp-mcp-server"
      ]
    }
  }
}
```

**That's it!** No local installation needed. The server will be automatically downloaded and run when needed.

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

## Features

### 60 Specialized Tools Across 8 Modules:
- **Particle Search**: Find particles by name, ID, or properties
- **Mass & Lifetime Data**: Get precise measurements with uncertainties
- **Decay Analysis**: Explore decay modes and branching fractions
- **Unit Conversion**: Convert between particle physics units
- **Quantum Numbers**: Access spin, parity, and other quantum properties
- **Error Handling**: Smart validation and helpful suggestions
- **Database Access**: Direct PDG database integration
- **Measurement Analysis**: Statistical analysis of experimental data

### Enterprise Features:
- **Caching**: Intelligent caching for fast response times
- **Rate Limiting**: Built-in protection against API abuse
- **Input Validation**: Comprehensive security and data validation
- **Error Recovery**: Robust error handling with helpful suggestions
- **Async Support**: High-performance asynchronous operations

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

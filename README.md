# ParticlePhysics MCP Server

A **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). This production-ready server enables AI assistants and applications to query comprehensive particle physics information through 60 specialized tools across 8 modules with enterprise-grade security, caching, and performance features.

## Quick Installation

### Option 1: uvx (Recommended)
```bash
uvx --from git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git pp-mcp-server
```

### Option 2: Direct Installation
```bash
git clone https://github.com/uzerone/ParticlePhysics-MCP-Server.git
cd ParticlePhysics-MCP-Server
pip install -r requirements.txt
python pp_mcp_server.py
```

### Option 3: Development Setup
```bash
git clone https://github.com/uzerone/ParticlePhysics-MCP-Server.git
cd ParticlePhysics-MCP-Server
pip install -e .
```

## MCP Configuration

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git", "pp-mcp-server"]
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

**Empowering AI assistants with particle physics knowledge**
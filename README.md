# ParticlePhysics MCP Server

A **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). This production-ready server enables AI assistants and applications to query comprehensive particle physics information through 60 specialized tools across 8 modules with enterprise-grade security, caching, and performance features.

## Quick Installation

### Option 1: Direct Python Script (Recommended for Local Development)
```bash
git clone https://github.com/uzerone/ParticlePhysics-MCP-Server.git
cd ParticlePhysics-MCP-Server
pip install -r requirements.txt
```

### Option 2: Development Installation
```bash
git clone https://github.com/uzerone/ParticlePhysics-MCP-Server.git
cd ParticlePhysics-MCP-Server
pip install -e .
```

## MCP Configuration for Claude Desktop

Choose the configuration method that works best for your setup:

### Method 1: Absolute Path (Most Reliable)
Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/TO/ParticlePhysics-MCP-Server/pp_mcp_server.py"]
    }
  }
}
```

**Note**: Replace `/ABSOLUTE/PATH/TO/` with your actual path. For example:
- macOS: `/Users/username/Documents/ParticlePhysics-MCP-Server/pp_mcp_server.py`
- Windows: `C:\\Users\\username\\Documents\\ParticlePhysics-MCP-Server\\pp_mcp_server.py`
- Linux: `/home/username/ParticlePhysics-MCP-Server/pp_mcp_server.py`

### Method 2: Python Module (After Installation)
If you've installed the package with `pip install -e .`:

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "python",
      "args": ["-m", "pp_mcp_server"]
    }
  }
}
```

### Method 3: Working Directory Method
```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "python",
      "args": ["pp_mcp_server.py"],
      "cwd": "/ABSOLUTE/PATH/TO/ParticlePhysics-MCP-Server"
    }
  }
}
```

## Configuration File Locations

The Claude Desktop configuration file should be placed at:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

After updating the configuration:
1. Save the file
2. Restart Claude Desktop
3. Look for the tools icon in Claude to verify the server is connected

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
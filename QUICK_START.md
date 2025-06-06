# 🚀 ParticlePhysics MCP Server - Quick Start Guide

Get particle physics data in your AI assistant in **less than 2 minutes**!

## 📋 Prerequisites

You only need ONE of these:
- `uvx` (recommended): `pip install uvx`
- `pipx`: `pip install pipx`
- `npm` (for npx method)

## 🎯 Fastest Setup (30 seconds)

### For Claude Desktop:

1. **Find your config file:**
   - Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. **Add this configuration:**
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

3. **Restart Claude Desktop**

4. **Start using!** Try these queries:
   - "What is the mass of the proton?"
   - "Compare the properties of all quarks"
   - "Find decay modes of the B+ meson"

## 🔧 Alternative Installation Methods

### Method 1: Using pipx
```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "pipx",
      "args": [
        "run",
        "--spec",
        "git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git",
        "pp-mcp-server"
      ]
    }
  }
}
```

### Method 2: Direct from GitHub Release
```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "python",
      "args": [
        "-m",
        "pip",
        "install",
        "--user",
        "git+https://github.com/uzerone/ParticlePhysics-MCP-Server.git",
        "&&",
        "pp-mcp-server"
      ]
    }
  }
}
```

## 🧪 Test Your Installation

In Claude, try:
1. "List available particle physics tools"
2. "What particles are available in the PDG database?"
3. "Get electron mass with uncertainties"

## 🛠️ Troubleshooting

### "Command not found" error
- Make sure `uvx` is installed: `pip install uvx`
- Or try the `pipx` method instead

### "PDG package not installed" error
- The server will automatically install dependencies on first run
- If it persists, manually install: `pip install pdg`

### Claude doesn't show the tools
1. Make sure you saved the config file in the correct location
2. Completely quit and restart Claude Desktop
3. Check the MCP icon in Claude's interface

## 📚 What Can You Do?

The server provides **60 specialized tools** across 8 modules:

- **Particle Search**: Find particles by name, ID, or properties
- **Mass & Lifetime Data**: Get precise measurements with uncertainties
- **Decay Analysis**: Explore decay modes and branching fractions
- **Unit Conversion**: Convert between particle physics units
- **Quantum Numbers**: Access spin, parity, and other quantum properties
- **Error Handling**: Smart validation and helpful suggestions

## 🎉 That's It!

You now have access to the entire Particle Data Group database directly in Claude!

### Example Queries to Try:
```
- "Search for Higgs boson properties"
- "Compare masses of electron, muon, and tau"
- "What are the main decay modes of the K+ meson?"
- "Convert 125 GeV to MeV"
- "Get quantum numbers for the proton"
```

## 📖 Learn More

- [Full Documentation](README.md)
- [60 Available Tools](README.md#usage-examples)
- [PDG Website](https://pdg.lbl.gov/)

---

**Need help?** Open an issue on [GitHub](https://github.com/uzerone/ParticlePhysics-MCP-Server/issues) 
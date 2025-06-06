# ParticlePhysics MCP Server

A **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). This production-ready server enables AI assistants and applications to query comprehensive particle physics information through 60 specialized tools across 8 modules with enterprise-grade security, caching, and performance features.

## Installation (No Local Setup Required)

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

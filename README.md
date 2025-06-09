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

## Maintainers

This project is maintained by:
- [@uzerone](https://github.com/uzerone)
- [@bee4come](https://github.com/bee4come)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

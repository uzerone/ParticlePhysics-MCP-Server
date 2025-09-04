# ParticlePhysics MCP Server

A Model Context Protocol (MCP) server that provides access to the particle data. This server enables Claude Desktop and other MCP clients to search and retrieve particle physics data in real-time.

## Status and Progress

- 🛠️ Work on displaying particle values with associated errors is ongoing.

## Configuration


```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "uvx",
      "args": ["--from", ".", "python", "-m", "particlephysics_mcp_server"]
    }
  }
}
```

## Tools

### `search_particle`
Returns key properties inline.
- **query** (required): Name or symbol of the particle.

### `list_decays`
Returns all decay modes for a specified particle.
- **query** (required): Name or symbol of the particle.

## Local Testing

- To use the inspector, run the restart script:
```
./restart_mcp_inspector.sh
```

- To run end-to-end tests with Playwright:
```
uvx pytest tests/test_e2e_playwright.py
```

## Maintainers

This project is developed and maintained by:
- [@uzerone](https://github.com/uzerone)
- [@bee4come](https://github.com/bee4come)

## License

MIT License - see [LICENSE](LICENSE).

## References
- [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro)
- [Particle Data Group](https://pdg.lbl.gov/)


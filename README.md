# ParticlePhysics MCP Server

A Model Context Protocol (MCP) server that provides access to the particle data. This server enables Claude Desktop and other MCP clients to search and retrieve particle physics data in real-time.

## Status and Progress

- Work on displaying particle values with associated errors is ongoing.
- Next: deploy the server so it can run 24/7 remotely and be used by LLMs or IDEs.

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
- **particle_id** (required): Name, symbol, or PDG ID of the particle.

## VS Code Skill

This repo includes a VS Code Copilot skill that documents the MCP workflow:
- [ .github/skills/particlephysics-skill/SKILL.md ](.github/skills/particlephysics-skill/SKILL.md)

Trigger it with phrases like "mcp inspector", "particle physics mcp", or "pp".

## Maintainers

This project is developed and maintained by:
- [@uzerone](https://github.com/uzerone)

## License

MIT License - see [LICENSE](LICENSE).

## References
- [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro)
- [Particle Data Group](https://pdg.lbl.gov/)


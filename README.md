# ParticlePhysics MCP Server

Minimal MCP server for particle physics data.

It lets MCP clients (Claude Desktop, IDEs, etc.) query particles and decay modes quickly.

## Quick Config

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

- Skill file: `.github/skills/particlephysics-skill/SKILL.md`
- Trigger phrase: `particle physics mcp` or `pp`

## Maintainer

- [@uzerone](https://github.com/uzerone)

## License

MIT - see `LICENSE`.


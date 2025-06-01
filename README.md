# PDG MCP Server

Model Context Protocol server for particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/).

**Features:** Search particles, get properties, decay modes, compare particles, list by type.

## Quick Command Reference

### Terminal Commands
```bash
# Search particles
./pdg search --query "e-"                    # By name
./pdg search --query "211" --type mcid       # By Monte Carlo ID
./pdg mcid --id 2212                         # Direct MCID lookup

# Get properties
./pdg properties --particle "e-"             # Basic info
./pdg properties --particle "tau-" --measurements  # With references

# Decay modes
./pdg decays --particle "tau-" --limit 10    # Tau decays
./pdg decays --particle "B+" --type all      # All B+ decays

# List particles
./pdg list --type lepton --limit 20          # Leptons only
./pdg list --type meson                      # Mesons only

# Compare particles  
./pdg compare --particles "e-" "mu-" "tau-"  # Basic comparison
./pdg compare --particles "pi+" "pi-" --properties mass charge

# Database info
./pdg info                                   # PDG database details
```

### MCP Client Usage
Ask natural language questions in Claude Desktop:
- "What is the mass of the proton?"
- "Show me tau decay modes"
- "Compare electron and muon properties"

## Installation

**uvx (Recommended):**
```bash
# From local directory
uvx --from . pdg-mcp-server

# From GitHub
uvx --from git+https://github.com/uzerone/pdg-mcp-server.git pdg-mcp-server
```

**Docker:**
```bash
git clone https://github.com/uzerone/pdg-mcp-server
cd pdg-mcp-server
./docker-run.sh build
./docker-run.sh test
```

**Direct:**
```bash
pip install -r requirements.txt
python test_pdg_server.py
```

## Usage Examples

**Terminal:**
```bash
./pdg search --query "proton"               # Find proton
./pdg decays --particle "tau-" --limit 5    # Tau decay modes
./pdg compare --particles "e-" "mu-"        # Compare leptons
```

**MCP Integration:**
```json
{
  "mcpServers": {
    "pdg": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/uzerone/pdg-mcp-server.git", "pdg-mcp-server"]
    }
  }
}
```

**Python:**
```python
import pdg_mcp_server as server
import asyncio

async def example():
    result = await server.handle_call_tool('search_particle', {'query': 'e-'})
    print(result[0].text)

asyncio.run(example())
```

## Available Tools

| Tool | Description | Example |
|------|-------------|---------|
| `search_particle` | Find particles by name/ID | `search --query "e-"` |
| `get_particle_properties` | Get mass, lifetime, etc. | `properties --particle "proton"` |
| `get_branching_fractions` | Decay modes | `decays --particle "tau-"` |
| `list_particles` | Filter by type | `list --type lepton` |
| `get_particle_by_mcid` | Lookup by Monte Carlo ID | `mcid --id 211` |
| `compare_particles` | Side-by-side comparison | `compare --particles "e-" "mu-"` |
| `get_database_info` | Database metadata | `info` |

## Docker Commands

```bash
./docker-run.sh build     # Build image
./docker-run.sh test      # Run tests  
./docker-run.sh examples  # Interactive examples
./docker-run.sh run       # Start server
./docker-run.sh logs      # View logs
./docker-run.sh cleanup   # Remove everything
```

## Testing

```bash
# Test everything
./docker-run.sh test

# Try some commands
./pdg search --query "e-"
./pdg decays --particle "mu-" --limit 3
./pdg compare --particles "pi+" "pi-"

# Run examples
docker run --rm -v "$(pwd)/examples.py:/app/examples.py" pdg-mcp-server:latest python examples.py
```

## Common Particle Names

| Particle | Name | MCID |
|----------|------|------|
| Electron | `e-` | 11 |
| Muon | `mu-` | 13 |
| Tau | `tau-` | 15 |
| Proton | `p` | 2212 |
| Neutron | `n` | 2112 |
| Pion+ | `pi+` | 211 |
| Kaon+ | `K+` | 321 |

## Authors

- [@uzerone](https://github.com/uzerone) 
- [@bee4come](https://github.com/bee4come) 

## License

MIT License. PDG data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Links

- [PDG Website](https://pdg.lbl.gov/)
- [PDG Python API](https://github.com/particledatagroup/api)
- [MCP Protocol](https://modelcontextprotocol.io/)

---

**🔬 Happy particle physics research!** 
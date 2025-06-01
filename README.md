# PDG MCP Server

A powerful **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). Perfect for researchers, students, and AI assistants working with particle physics.

## 🚀 Features

- **7 MCP Tools** for comprehensive particle physics research
- **Multiple Installation Methods**: uvx, Docker, pip, or direct
- **CLI Interface** for terminal usage  
- **Comprehensive Testing** suite included
- **Production Ready** with Docker support

---

## 📦 Installation

### Option 1: uvx (Recommended)
```bash
# From GitHub (latest)
uvx --from git+https://github.com/uzerone/PDG-MCP-Server.git pdg-mcp-server

# From local clone  
git clone https://github.com/uzerone/PDG-MCP-Server.git
cd PDG-MCP-Server
uvx --from . pdg-mcp-server
```

### Option 2: Docker
```bash
git clone https://github.com/uzerone/PDG-MCP-Server.git
cd PDG-MCP-Server
./docker-run.sh build
./docker-run.sh run
```

### Option 3: Python/pip
```bash
git clone https://github.com/uzerone/PDG-MCP-Server.git
cd PDG-MCP-Server
pip install -r requirements.txt
python pdg_mcp_server.py
```

### Option 4: Direct Terminal CLI
```bash
git clone https://github.com/uzerone/PDG-MCP-Server.git
cd PDG-MCP-Server
./pdg search --query "electron"  # Requires Docker
```

---

## 🛠️ MCP Configuration

Add to your MCP client configuration:

### uvx Method
```json
{
  "mcpServers": {
    "pdg": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/uzerone/PDG-MCP-Server.git", "pdg-mcp-server"]
    }
  }
}
```

### Docker Method  
```json
{
  "mcpServers": {
    "pdg": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "pdg-mcp-server:latest"]
    }
  }
}
```

---

## 🔧 Available MCP Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| **search_particle** | Find particles by name, MCID, or PDG ID | `"Find information about the muon"` |
| **get_particle_properties** | Get detailed particle properties | `"What is the mass of the proton?"` |
| **get_branching_fractions** | Get decay modes and branching ratios | `"Show me tau decay modes"` |
| **list_particles** | List particles by type | `"List all leptons"` |
| **get_particle_by_mcid** | Lookup by Monte Carlo ID | `"Get particle with MCID 211"` |
| **compare_particles** | Side-by-side particle comparison | `"Compare electron and muon"` |
| **get_database_info** | Database metadata and info | `"What PDG database version?"` |

---

## 💬 Example Queries for AI Assistants

Ask your AI assistant (Claude, ChatGPT, etc.) with the MCP server:

### Basic Queries
- *"What is the mass of the proton?"*
- *"Find information about the electron"*
- *"Show me the properties of pi+"*

### Research Queries  
- *"What are the main decay modes of the tau lepton?"*
- *"Compare the masses of all quarks"*
- *"List all particles with charge +1"*

### Analysis Queries
- *"Which B meson decays involve J/psi?"*
- *"Compare the lifetimes of charged and neutral kaons"*
- *"What are the quantum numbers of the W boson?"*

---

## 🖥️ Terminal CLI Usage

### Quick Commands
```bash
# Particle search
./pdg search --query "e-"                    # Electron by name
./pdg search --query "211" --type mcid       # Pi+ by Monte Carlo ID

# Properties and mass
./pdg properties --particle "proton"         # Basic properties  
./pdg properties --particle "tau-" --measurements  # With references

# Decay modes  
./pdg decays --particle "tau-" --limit 10    # Tau decays
./pdg decays --particle "B+" --type all      # All B+ decays

# List and compare
./pdg list --type lepton --limit 20          # List leptons
./pdg compare --particles "e-" "mu-" "tau-"  # Compare leptons

# Database info
./pdg info                                   # PDG database details
```

### Using uvx
```bash
# Test the server
uvx --from . python test_pdg_server.py

# Run CLI commands
uvx --from . python pdg_cli.py search --query "proton"
uvx --from . python pdg_cli.py decays --particle "tau-"
```

---

## 🐍 Python API Usage

```python
import asyncio
import pdg_mcp_server as server

async def example():
    # Search for a particle
    result = await server.handle_call_tool('search_particle', {'query': 'electron'})
    print(result[0].text)
    
    # Get particle properties
    result = await server.handle_call_tool('get_particle_properties', 
                                         {'particle_name': 'proton'})
    print(result[0].text)
    
    # Compare particles
    result = await server.handle_call_tool('compare_particles', {
        'particle_names': ['electron', 'muon'], 
        'properties': ['mass', 'lifetime']
    })
    print(result[0].text)

# Run example
asyncio.run(example())
```

---

## 🧪 Testing

### Run All Tests
```bash
# Docker method
./docker-run.sh test

# uvx method  
uvx --from . python test_pdg_server.py

# Direct method
python test_pdg_server.py
```

### Interactive Examples
```bash
# Docker
./docker-run.sh examples

# uvx
uvx --from . python examples.py

# Direct
python examples.py
```

---

## 📊 Common Particle Reference

| Particle | Name | MCID | Mass (GeV) | Charge |
|----------|------|------|------------|--------|
| Electron | `e-` | 11 | 0.000511 | -1 |
| Muon | `mu-` | 13 | 0.105658 | -1 |
| Tau | `tau-` | 15 | 1.77686 | -1 |
| Proton | `p` | 2212 | 0.938272 | +1 |
| Neutron | `n` | 2112 | 0.939565 | 0 |
| Pion+ | `pi+` | 211 | 0.139570 | +1 |
| Kaon+ | `K+` | 321 | 0.493677 | +1 |

---

## 🚀 Docker Commands

```bash
./docker-run.sh build     # Build image
./docker-run.sh test      # Run tests  
./docker-run.sh examples  # Interactive examples
./docker-run.sh run       # Start server
./docker-run.sh logs      # View logs
./docker-run.sh cleanup   # Remove everything
./docker-run.sh shell     # Interactive shell
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `./docker-run.sh test`  
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Note**: PDG data is subject to [PDG's own license terms](https://pdg.lbl.gov/about/terms). Starting with 2024 edition, published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 🔗 Links

- **PDG Website**: https://pdg.lbl.gov/
- **PDG Python API**: https://github.com/particledatagroup/api  
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Repository**: https://github.com/uzerone/PDG-MCP-Server

---

## 👥 Authors

- [@uzerone](https://github.com/uzerone) - Project maintainer
- [@bee4come](https://github.com/bee4come) - Project maintainer

---

**🔬 Happy particle physics research!** ⚛️ 
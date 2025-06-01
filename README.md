# PDG MCP Server

A powerful **Model Context Protocol (MCP) server** that provides seamless access to particle physics data from the [Particle Data Group (PDG)](https://pdg.lbl.gov/). Perfect for researchers, students, and AI assistants working with particle physics.

## 🚀 Features

- **7 MCP Tools** for comprehensive particle physics research
- **Multiple Installation Methods**: uvx, Docker, pip, or direct
- **CLI Interface** for terminal usage  
- **Comprehensive Testing** suite included
- **Production Ready** with Docker support
- **Python 3.10+** required (MCP dependency)

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

### Basic Example
```python
import asyncio
import json
import pdg_mcp_server as pdgmcp

async def basic_example():
    # Search for a particle
    result = await pdgmcp.handle_call_tool('search_particle', {'query': 'electron'})
    data = json.loads(result[0].text)
    print(f"Found: {data[0]['name']} (mass: {data[0]['mass']})")
    
    # Get detailed properties
    result = await pdgmcp.handle_call_tool('get_particle_properties', 
                                         {'particle_name': 'proton'})
    data = json.loads(result[0].text)
    print(f"Proton mass: {data['mass']}, lifetime: {data['lifetime']}")

asyncio.run(basic_example())
```

### Advanced Research Example
```python
import asyncio
import json
import pdg_mcp_server as pdgmcp

async def research_workflow():
    """Complete particle physics research workflow."""
    
    # 1. Compare lepton family
    print("=== Lepton Family Comparison ===")
    result = await pdgmcp.handle_call_tool('compare_particles', {
        'particle_names': ['e-', 'mu-', 'tau-'],
        'properties': ['mass', 'lifetime', 'charge']
    })
    leptons = json.loads(result[0].text)
    
    for particle in leptons['particles']:
        print(f"{particle['name']}: {particle['mass']}, τ={particle['lifetime']}")
    
    # 2. Study tau decays
    print("\n=== Tau Decay Modes ===")
    result = await pdgmcp.handle_call_tool('get_branching_fractions', {
        'particle_name': 'tau-',
        'decay_type': 'exclusive',
        'limit': 5
    })
    decays = json.loads(result[0].text)
    
    for i, decay in enumerate(decays['decay_modes'], 1):
        print(f"{i}. {decay['description']} ({decay['display_value']})")
    
    # 3. Monte Carlo ID lookup
    print("\n=== Monte Carlo ID Lookup ===")
    mcids = [11, 13, 15, 211, 2212]  # e-, mu-, tau-, pi+, proton
    
    for mcid in mcids:
        result = await pdgmcp.handle_call_tool('get_particle_by_mcid', {'mcid': mcid})
        particle = json.loads(result[0].text)
        print(f"MCID {mcid}: {particle['name']} ({particle['mass']})")

asyncio.run(research_workflow())
```

### Error Handling Example
```python
import asyncio
import json
import pdg_mcp_server as pdgmcp

async def safe_particle_lookup(particle_name):
    """Safely look up particle with error handling."""
    try:
        result = await pdgmcp.handle_call_tool('get_particle_properties', 
                                             {'particle_name': particle_name})
        data = json.loads(result[0].text)
        
        if 'error' in data:
            print(f"Error: {data['error']}")
            return None
        
        return {
            'name': data['name'],
            'mass': data.get('mass', 'Unknown'),
            'charge': data.get('charge', 'Unknown')
        }
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

async def error_example():
    # Valid particle
    proton = await safe_particle_lookup('proton')
    if proton:
        print(f"✓ {proton['name']}: {proton['mass']}")
    
    # Invalid particle
    invalid = await safe_particle_lookup('invalid_particle')
    if not invalid:
        print("✗ Particle not found")

asyncio.run(error_example())
```

### Batch Processing Example
```python
import asyncio
import json
import pdg_mcp_server as pdgmcp

async def batch_particle_analysis():
    """Process multiple particles efficiently."""
    
    particles_of_interest = [
        'electron', 'muon', 'tau-',
        'pi+', 'pi-', 'K+', 'K-',
        'proton', 'neutron'
    ]
    
    results = []
    
    for particle in particles_of_interest:
        try:
            result = await pdgmcp.handle_call_tool('get_particle_properties', 
                                                 {'particle_name': particle})
            data = json.loads(result[0].text)
            
            if 'error' not in data:
                results.append({
                    'name': data['name'],
                    'mass': data.get('mass', 'N/A'),
                    'charge': data.get('charge', 'N/A'),
                    'mcid': data.get('mcid', 'N/A')
                })
        except:
            continue
    
    # Sort by mass
    results.sort(key=lambda x: float(x['mass'].split()[0]) if 'GeV' in x['mass'] else 0)
    
    print("Particles sorted by mass:")
    for particle in results:
        print(f"{particle['name']:<10} {particle['mass']:<15} {particle['charge']:<5} MCID:{particle['mcid']}")

asyncio.run(batch_particle_analysis())
```

### Custom Helper Functions
```python
import asyncio
import json
import pdg_mcp_server as pdgmcp

class PDGHelper:
    """Helper class for common PDG operations."""
    
    @staticmethod
    async def get_mass(particle_name):
        """Get particle mass in GeV."""
        result = await pdgmcp.handle_call_tool('get_particle_properties', 
                                             {'particle_name': particle_name})
        data = json.loads(result[0].text)
        return data.get('mass', 'Unknown') if 'error' not in data else None
    
    @staticmethod
    async def find_by_charge(charge, particle_type='all', limit=10):
        """Find particles with specific charge."""
        result = await pdgmcp.handle_call_tool('list_particles', {
            'particle_type': particle_type,
            'limit': 100
        })
        data = json.loads(result[0].text)
        
        if 'error' in data:
            return []
        
        filtered = [p for p in data['particles'] if p.get('charge') == charge]
        return filtered[:limit]
    
    @staticmethod
    async def mass_comparison(particles):
        """Compare masses of multiple particles."""
        result = await pdgmcp.handle_call_tool('compare_particles', {
            'particle_names': particles,
            'properties': ['mass']
        })
        data = json.loads(result[0].text)
        return data['particles'] if 'error' not in data else []

async def helper_example():
    # Use helper functions
    print("Electron mass:", await PDGHelper.get_mass('electron'))
    
    charged_particles = await PDGHelper.find_by_charge(1.0, 'meson', 5)
    print("Charged mesons:", [p['name'] for p in charged_particles])
    
    mass_data = await PDGHelper.mass_comparison(['proton', 'neutron'])
    for p in mass_data:
        print(f"{p['name']}: {p.get('mass', 'N/A')}")

asyncio.run(helper_example())
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
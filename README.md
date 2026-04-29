# ParticlePhysics MCP Server

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](CHANGELOG.md)

A Model Context Protocol server that lets Claude Desktop, IDEs, and other MCP clients
look up particle properties and decay modes.

Supports natural-language queries (`muon plus`, `pion zero`, `antiproton`, `anti up quark`),
case-insensitive lookup, MC IDs (`-13`), and returns both human-readable text and a
structured JSON payload.

## Table of Contents

- [Features](#features)
- [Install](#install)
- [Configure your MCP client](#configure-your-mcp-client)
- [Tools](#tools)
  - [`search_particle`](#search_particle)
  - [`list_decays`](#list_decays)
- [Claude Skill](#claude-skill)
- [Changelog](#changelog)
- [Maintainer](#maintainer)
- [License](#license)

## Features

- `search_particle` — mass (MeV + GeV), charge, spin, color, parity / C / I / G,
  lifetime or width, MCID, review ID
- `list_decays` — exclusive / inclusive branching fractions with the source method named
- Natural-language input — `muon plus`, `positive tau`, `pion zero`, `kaon minus`
- Anti-particle support — `antimuon`, `anti up quark`, `ubar`, `u bar`, `u_bar`, `u~`,
  `antineutron`; resolved via MCID negation, no name guessing
- MC ID lookup — query directly with `11`, `-2212`, etc.
- Self-conjugate aware — `anti photon` resolves to `gamma`, `anti pi0` to `pi0`
- Structured output — every response includes a fenced ` ```json ` block alongside
  the human-readable text

## Install

```bash
git clone https://github.com/uzerone/particlephysics-mcp-server.git
cd particlephysics-mcp-server
pip install -e .
```

## Configure your MCP client

Add one of the following to your client's MCP config (e.g. `claude_desktop_config.json`).

**Using `uvx` from the cloned repo** (no global install needed):

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "uvx",
      "args": ["--from", "/absolute/path/to/particlephysics-mcp-server",
               "python", "-m", "particlephysics_mcp_server"]
    }
  }
}
```

**Using a local virtualenv** (after `pip install -e .`):

```json
{
  "mcpServers": {
    "particlephysics": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "particlephysics_mcp_server"]
    }
  }
}
```

## Tools

### `search_particle`

| Input | Example |
|---|---|
| Canonical name | `mu+`, `pi0`, `K-`, `Sigma+`, `gamma`, `Lambda`, `H` |
| English alias | `muon`, `pion`, `higgs`, `electron`, `top quark` |
| Anti-particle | `antimuon`, `anti up quark`, `ubar`, `u bar`, `u_bar`, `u~`, `antiproton` |
| Natural-language charge | `muon plus`, `positive tau`, `pion zero`, `kaon minus` |
| MC ID | `11` (electron), `-13` (mu+), `2212` (proton) |

Sample call: `search_particle({"query": "muon plus"})`

````text
Found 1 particle(s) matching 'muon plus':

1. mu
   Name: mu+
   PDG ID: -13
   PDG Review ID: S004/2025
   Mass: 105.6583755 MeV (0.1056583755 GeV)
   Spin (J): 1/2
   Charge: 1
   Color: singlet
   Quantum numbers: J=1/2
   Lifetime: 2.196981148893498e-06

```json
{
  "query": "muon plus",
  "count": 1,
  "particles": [{
    "name": "mu+",
    "mcid": -13,
    "pdg_review_id": "S004/2025",
    "mass": {"mev": 105.6583755, "gev": 0.1056583755},
    "charge": {"value": 1.0, "fraction": "1"},
    "spin": "1/2",
    "color": {"multiplicity": 1, "label": "singlet"},
    "quantum_numbers": {"J": "1/2"},
    "lifetime": {"seconds": 2.197e-06, "stable": false, "text": "..."}
  }]
}
```
````

### `list_decays`

Same identifier formats as `search_particle`. Tries `exclusive_branching_fractions` →
`branching_fractions` → `inclusive_branching_fractions` and reports which source it used.
Returns an empty decay list with `stable=true` for stable particles.

Sample call: `list_decays({"particle_id": "tau"})`

````text
Decay modes for particle 'tau':

1. tau- --> mu- nubar_mu nu_tau (BR: 17.39 ± 0.04 %)
2. tau- --> e- nubar_e nu_tau  (BR: 17.82 ± 0.04 %)
…

```json
{
  "particle": {"name": "tau-", "mcid": 15, ...},
  "source": "exclusive_branching_fractions",
  "count": 137,
  "decays": [{
    "description": "tau- --> mu- nubar_mu nu_tau",
    "branching_ratio_text": "17.39 ± 0.04",
    "value_text": "17.39E-2",
    "is_limit": false
  }, ...]
}
```
````

## Claude Skill

A Claude Code skill spec lives at [`.github/skills/particlephysics-skill/SKILL.md`](.github/skills/particlephysics-skill/SKILL.md). It runs the inspector, validates both tools, and exercises the natural-language / anti-particle / MC ID query surface.

Trigger phrase: `particle physics mcp` or `pp`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Maintainer

[@uzerone](https://github.com/uzerone)

## License

MIT — see [LICENSE.txt](LICENSE.txt).

---
name: particlephysics-skill
description: "Use when: particle physics MCP, mcp inspector, particle/decay queries, pp server, or running the ParticlePhysics MCP workflow."
---

# ParticlePhysics MCP Skill

## Overview
Use this skill to launch the MCP inspector, validate the two tools, and sanity-check common queries against the local server.

## Checklist
1. Install (once): `pip install -e .`
2. Start / restart the MCP inspector: `./restart_mcp_inspector.sh`
3. Verify the server responds to `search_particle` and `list_decays`.
4. Run tests: `python -m pytest tests/`

## Quick Queries

### `search_particle`
Try these to cover the full input surface:

- Canonical names: `electron`, `proton`, `Higgs`, `up quark`, `mu+`, `pi0`, `Sigma+`
- Anti-particles: `antimuon`, `anti up quark`, `ubar`, `u_bar`, `u~`, `antiproton`, `antineutron`
- Natural-language charge: `muon plus`, `positive tau`, `pion zero`, `kaon minus`
- MC IDs: `11` (electron), `-13` (mu+), `2212` (proton)
- Self-conjugate: `anti photon` (should resolve to `gamma`)

Each response includes a fenced ` ```json ` block alongside the human-readable text — verify both exist.

### `list_decays`
- `muon`, `tau`, `pion`, `B0`, `K+`
- Anti-particle decays: `antimuon` (mcid=-13, same decay structure as muon)

Verify the JSON payload includes a `source` field (`exclusive_branching_fractions` / `branching_fractions` / `inclusive_branching_fractions`).

## Notes
- If `uvx` is not available, the inspector script falls back to `python -m`.
- If `pdg` import fails, re-run `pip install -e .` (or `pip install -e ".[dev]"` if you also want pytest).

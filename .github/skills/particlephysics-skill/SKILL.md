---
name: particlephysics-skill
description: "Use when: particle physics MCP, mcp inspector, PDG queries, pp server, or running the ParticlePhysics MCP workflow."
---

# ParticlePhysics MCP Skill

## Overview
Use this skill to run the MCP inspector, validate tools, and sanity-check common queries.

## Checklist
1. Start/restart the MCP inspector: `./restart_mcp_inspector.sh`.
2. Verify the server responds to `search_particle` and `list_decays`.
3. Run tests: `python -m pytest tests/test_e2e_playwright.py`.

## Quick Queries
- `search_particle`:
  - `electron`, `proton`, `Higgs`, `up quark`, `anti up quark`
- `list_decays`:
  - `muon`, `tau`, `pion`

## Notes
- If `uvx` is not available, use `python -m pytest` for tests.
- If `pdg` import fails, ensure `python -m pip install -r requirements.txt` has been run.

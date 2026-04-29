# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-04-29

Quality-of-life release: natural-language queries, principled antiparticle resolution
via MCID negation, structured JSON output, async-correct execution, and a corrected
mass-unit bug (PDG returns GeV, not MeV).

### Added
- **Natural-language queries** for `search_particle` and `list_decays`:
  - Charge words: `muon plus`, `muon minus`, `pion zero`, `positive tau`, `negative pion`, `kaon minus`, etc.
  - Anti-particle markers: `antimuon`, `anti muon`, `anti-muon`, `antiproton`, `antineutron`, `anti up quark`, `ubar`, `u bar`, `u_bar`, `u~`.
  - Self-conjugate aware: `anti photon` resolves to `gamma`, `anti pi0` to `pi0`, etc.
- **MCID-based antiparticle resolution.** Instead of guessing PDG names (`pbar` vs `p_bar` vs `p~`), the resolver fetches the antiparticle by negating the MCID (`api.get_particle_by_mcid(-mcid)`). Works uniformly across leptons, baryons, mesons and quarks.
- **Structured JSON output.** Both tools append a fenced ```` ```json ```` block to the human-readable text containing a typed payload (`{query, count, particles:[...]}` for `search_particle`; `{particle, source, count, decays:[...]}` for `list_decays`).
- **Async-correct execution.** Blocking PDG calls now run on a worker thread via `asyncio.to_thread`, so the MCP event loop is no longer pinned during slow lookups.
- **`PdgParticle` and `PdgDecay` Protocol types** for typed access to the PDG client without depending on its concrete classes.
- **Tool descriptions** expanded with input examples (PDG names, English aliases, anti-particle markers, NL charge phrases, MC IDs) and response-schema documentation.
- **Test coverage** grew from 13 → 27 cases covering NL queries, anti-particles via prefix/suffix, mass-unit consistency, MCID lookup, generic-quark ambiguity guard, not-found paths, JSON-shape validation, and antiparticle decays.
- **`aliases.json`** as a standalone data file, replacing a 200-line hardcoded dict in `server.py`.

### Changed
- Particle resolver split into `_resolve_simple` (name/MCID/alias) and `_resolve_particle` (anti-particle orchestration).
- Search is now case-insensitive at the user-input layer while preserving the original case for PDG (which is case-sensitive — `Sigma+` resolves, `sigma+` previously did not).
- PDG API connection (`pdg.connect()`) cached as a process-wide singleton instead of being re-established per call.
- Generic-quark ambiguity guard extended to cover `anti quark`, `antiquarks`, `quark bar`, etc.

### Fixed
- **Mass units (pre-existing bug).** PDG stores masses in **GeV**, not MeV; the previous formatter divided by 1000 in the wrong direction, displaying e.g. proton mass as `0.938 MeV` instead of `938 MeV`. Both human and JSON outputs now report correct magnitudes (proton: 938.27 MeV / 0.93827 GeV, Higgs: 125 199 MeV / 125.2 GeV).
- **`Sigma+` resolved to `Z0`.** The substring fallback iterated `api.get_all()`, which yields `PdgProperty` identifiers whose descriptions can mention other particles; replaced with an exact, case-insensitive name scan over `api.get_particles()`.
- **Double charge negation.** Anti-particle queries that resolved to the actual antiparticle (correct charge from PDG) were being negated again at the display layer, flipping `ubar`'s `-2/3` to `2/3`. The post-hoc `anti_view` flip is gone.
- **`_negate_numeric_like(0)` returned `"-0"`** — now short-circuits to `"0"`.
- **Dead code:** unreachable `return` after `except` in the original `list_decays`.
- **Bare `except:` clauses** (12 of them) changed to `except Exception:` so they no longer swallow `KeyboardInterrupt` / `SystemExit`.

### Removed
- `subprocess`-based `setup_module_paths()` hack that called `uvx pip show mcp` and `uvx pip show pdg` to inject `sys.path`. Redundant when launched from a properly-configured venv (and was the source of slow start-up under uvx).
- Unused runtime dependencies: `fastapi`, `uvicorn`, `httpx`. None were imported anywhere in the codebase.
- `pytest` moved from runtime `dependencies` to `optional-dependencies.dev` (`pip install -e ".[dev]"`).
- `_load_name_mappings()` / references to `generated/name_mappings.json` (the file never existed in the repository).
- `anti_view` heuristic in the search-result formatter (the resolver now returns the actual antiparticle entry, so post-hoc sign-flipping is no longer needed and was producing wrong values).

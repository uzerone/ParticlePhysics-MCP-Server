---
name: README scope and tone preferences
description: For this project's README, omit Development sections, omit "Particle Data Group" / "PDG group" organizational references, and avoid emojis.
type: feedback
---

When editing the README (or analogous user-facing docs) for this project, omit the following:

1. **Any "Development" / "Contributing" / dev-tooling section.** Don't include `pip install -e ".[dev]"`, `pytest`, `restart_mcp_inspector.sh`, or other developer-workflow content. The README is a user-facing reference, not a contributor guide.
2. **References to "Particle Data Group" or "PDG group" as an organization/source.** Don't link to pdg.lbl.gov, don't say "queries the Particle Data Group database", don't credit PDG as a data source in prose. Field-name occurrences inside actual server output (e.g. `PDG ID:` printed by the tool) are OK because they're part of the technical interface, not organizational branding.
3. **Emojis.** See `feedback_no_emoji.md`.

**Why:** User explicitly issued each of these as corrections during README work — "dont write anything about development", "dont say anything about PDG group", "no emoji". Pattern is consistent: keep the README minimal, user-focused, and free of branding/dev-noise.

**How to apply:** Default to product-feature documentation only — what the tools do, how to install, how to configure the MCP client, sample input/output, license. If a section feels developer-oriented or like data-source attribution, drop it before asking.

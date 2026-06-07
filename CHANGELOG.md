# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-04-29

Easier to ask, more trustworthy to read.

### Added
- **Ask in plain English** — phrases like `muon plus` or `pion zero` now work, no special notation needed.
- **Antiparticles just work** — `antimuon`, `antiproton`, and shorthand all find the right particle.
- **Faster responses** — lookups no longer freeze the server.

### Fixed
- **Correct masses** — fixed numbers that were off by a large factor (proton now ~938 MeV, not 0.938).
- **More reliable searches** — fixed cases that returned the wrong particle or a flipped charge.
- **Capitalization no longer matters** — searches now ignore case.

### Removed
- **Lighter, faster setup** — dropped a slow start-up workaround and unused components.

#!/usr/bin/env python3
"""
ParticlePhysics MCP Server

A Model Context Protocol server that provides access to the Particle Data Group Database.
"""

import asyncio
import json
import logging
import math
import os
import re
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Tool
import mcp.types as types


@runtime_checkable
class PdgParticle(Protocol):
    """Subset of the PDG ``PdgParticle`` API the server consumes.

    Defined as a Protocol so the server has a typed contract without depending on
    pdg's concrete classes (which differ between pdg releases).
    """

    name: str | None
    description: str | None
    mcid: int | None
    pdgid: str | None
    mass: float | None
    charge: float | None
    lifetime: float | None
    width: float | None


@runtime_checkable
class PdgDecay(Protocol):
    """Subset of a decay-mode entry returned by ``branching_fractions()`` and friends."""

    description: str | None
    display_value_text: str | None

# Configure logging: send to stderr so stdout remains clean for MCP JSON-RPC
_log_level = os.getenv("MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("particlephysics-mcp-server")

server = Server("particlephysics-mcp-server")


# ---------------------------------------------------------------------------
# Aliases & PDG API singleton
# ---------------------------------------------------------------------------

_ALIASES_CACHE: dict[str, str] | None = None


def _load_aliases() -> dict[str, str]:
    """Load aliases.json once and cache it (lowercase keys -> canonical PDG name)."""
    global _ALIASES_CACHE
    if _ALIASES_CACHE is not None:
        return _ALIASES_CACHE
    path = os.path.join(os.path.dirname(__file__), "aliases.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _ALIASES_CACHE = {
            k.strip().lower(): v
            for k, v in raw.items()
            if isinstance(k, str) and isinstance(v, str) and not k.startswith("_")
        }
    except Exception as e:
        logger.warning(f"Failed to load aliases.json: {e}")
        _ALIASES_CACHE = {}
    return _ALIASES_CACHE


_PDG_API: Any = None


def _get_pdg_api() -> Any:
    """Return a cached pdg.connect() handle. Raises ImportError if pdg is missing."""
    global _PDG_API
    if _PDG_API is not None:
        return _PDG_API
    import pdg  # noqa: F401  -- ImportError surfaces to caller
    _PDG_API = pdg.connect()
    return _PDG_API


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_charge(value: Any) -> str:
    """Format electric charge as a fraction (0.3333 -> 1/3, -0.6666 -> -2/3)."""
    try:
        if isinstance(value, str):
            value = float(value)
        if value is None:
            return "unknown"
        if float(value) == 0.0:
            return "0"
        frac = Fraction(float(value)).limit_denominator(6)
        if frac.denominator == 1:
            return f"{frac.numerator}"
        return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return str(value)


def _format_lifetime(value: Any) -> str:
    """Format lifetime, showing 'stable (infinite)' for infinities."""
    try:
        if value is None:
            return "unknown"
        s = str(value).strip().lower()
        if s in {"inf", "+inf", "infinity"}:
            return "stable (infinite)"
        try:
            if float(value) == float("inf"):
                return "stable (infinite)"
        except Exception:
            pass
        return str(value)
    except Exception:
        return str(value)


def _format_width(value: Any) -> str:
    """Format decay width, showing '0 (stable)' for zero."""
    try:
        if value is None:
            return "unknown"
        if float(value) == 0.0:
            return "0 (stable)"
        return str(value)
    except Exception:
        return str(value)


def _to_float(value: Any) -> float | None:
    """Best-effort convert to float; tolerates trailing units like 'MeV'."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    try:
        s = str(value).strip()
        match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)
        if match:
            return float(match.group(0))
    except Exception:
        pass
    return None


def _format_mass_mev_gev(value: Any) -> tuple[str, str]:
    """Return (MeV_str, GeV_str) for a mass value. PDG stores mass in GeV."""
    gv = _to_float(value)
    if gv is None:
        return (str(value) if value is not None else "unknown", "unknown")
    return (f"{gv * 1000.0}", f"{gv}")


# ---------------------------------------------------------------------------
# Anti-particle helpers
# ---------------------------------------------------------------------------

def _is_anti_query(text: str) -> bool:
    """Heuristic: detect if a query refers to an anti-particle."""
    try:
        s = (text or "").strip().lower()
        if not s:
            return False
        if s.startswith("anti-") or s.startswith("anti "):
            return True
        if s.endswith("~") or s.endswith("_bar"):
            return True
        if any(s.endswith(x) or x in s for x in ["ubar", "dbar", "sbar", "cbar", "bbar", "tbar"]):
            return True
        if "antineutrino" in s:
            return True
    except Exception:
        pass
    return False


def _negate_numeric_like(value: Any) -> str:
    """Negate a numeric-like value (number, '2/3', etc.). Zero stays '0'."""
    try:
        f = float(value)
        if f == 0.0:
            return "0"
        nf = -f
        if nf.is_integer():
            return str(int(nf))
        return str(nf)
    except Exception:
        pass
    s = str(value).strip()
    if s in {"0", "+0", "-0"}:
        return "0"
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            num = num.strip()
            den = den.strip()
            if num.startswith("-"):
                return f"{num[1:]}/{den}"
            if num.startswith("+"):
                return f"-{num[1:]}/{den}"
            return f"-{num}/{den}"
        if s.startswith("-"):
            return s[1:]
        if s.startswith("+"):
            return f"-{s[1:]}"
        return f"-{s}"
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
# Quantum-number / classification helpers
# ---------------------------------------------------------------------------

def _infer_color_multiplicity(particle: Any) -> str:
    """Infer QCD color multiplicity from PDG flags: Q->3, G->8, else 1."""
    try:
        flags = getattr(particle, "data_flags", None)
        if flags is None and hasattr(particle, "particle_list"):
            flags = getattr(particle.particle_list, "data_flags", None)
        if isinstance(flags, str):
            if "G" in flags:
                return "8"
            if "Q" in flags:
                return "3"
        name = (getattr(particle, "description", None) or "").lower()
        if name in {"g", "gluon"}:
            return "8"
        if name in {"u", "d", "s", "c", "b", "t", "up", "down", "strange", "charm", "bottom", "top"}:
            return "3"
    except Exception:
        pass
    return "1"


def _format_color_label(color_code: str) -> str:
    return {"1": "singlet", "3": "triplet", "8": "octet"}.get(str(color_code), str(color_code))


def _is_quark(particle: Any) -> bool:
    try:
        flags = getattr(particle, "data_flags", None)
        if isinstance(flags, str) and "Q" in flags:
            return True
        name = (getattr(particle, "description", None) or "").lower().strip()
        return name in {"u", "d", "s", "c", "b", "t", "up", "down", "strange", "charm", "bottom", "top"}
    except Exception:
        return False


def _format_quark_mass_from_measurements(particle: Any) -> str | None:
    """Fall back to PDG measurement entries for quarks where .mass is unset."""
    try:
        for method_name in ["masses", "mass_measurements"]:
            method = getattr(particle, method_name, None)
            if not callable(method):
                continue
            try:
                entries = list(method())
            except Exception:
                entries = []
            if not entries:
                continue
            for entry in entries:
                vt = getattr(entry, "value_text", None)
                if vt:
                    return str(vt)
            first = entries[0]
            val = getattr(first, "value", None)
            units = getattr(first, "units", "") or ""
            if val is not None:
                return f"{val} {units}".strip()
    except Exception:
        pass
    return None


def _get_first_attr(particle: Any, candidate_names: list[str]) -> Any:
    """Return the first non-None attribute among candidate_names, also checking nested particle_list."""
    sources = [particle]
    if hasattr(particle, "particle_list"):
        sources.append(getattr(particle, "particle_list"))
    for src in sources:
        for name in candidate_names:
            try:
                if hasattr(src, name):
                    value = getattr(src, name)
                    if value is not None:
                        return value
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Particle resolver (unified)
# ---------------------------------------------------------------------------

def _strip_anti_markers(s: str) -> tuple[str, bool]:
    """Strip anti-particle markers and return (base, was_stripped).

    Recognised forms (all case-insensitive):
        '_bar' / 'bar' / ' bar' suffix      -> antiparticle
        '~' suffix                           -> antiparticle
        'anti-' / 'anti ' / 'anti' prefix    -> antiparticle
    """
    stripped = False
    # Suffix forms
    if s.endswith("_bar"):
        s = s[:-4]
        stripped = True
    elif s.endswith(" bar"):
        s = s[:-4]
        stripped = True
    elif s.endswith("bar") and len(s) > 3:
        s = s[:-3]
        stripped = True
    if s.endswith("~"):
        s = s[:-1]
        stripped = True
    # Prefix forms
    if s.startswith("anti-") or s.startswith("anti "):
        s = s[5:]
        stripped = True
    elif s.startswith("anti") and len(s) > 4:
        # 'antimuon' -> 'muon', 'antineutron' -> 'neutron'
        s = s[4:]
        stripped = True
    s = s.strip()
    s = (
        s.replace("nu(e)", "nu_e")
        .replace("nu(mu)", "nu_mu")
        .replace("nu(tau)", "nu_tau")
        .replace("k(s)", "K0S")
        .replace("k(l)", "K0L")
    )
    return s, stripped


_CHARGE_SUFFIX_WORDS: dict[str, str] = {
    "plus": "+",
    "minus": "-",
    "zero": "0",
    "neutral": "0",
    "naught": "0",
}
_CHARGE_PREFIX_WORDS: dict[str, str] = {
    "positive": "+",
    "negative": "-",
}


def _extract_charge_modifier(s: str) -> tuple[str, str] | None:
    """Pull a natural-language charge modifier off a multi-word query.

    'muon plus' -> ('muon', '+')
    'positive tau' -> ('tau', '+')
    'pion zero' -> ('pion', '0')
    """
    parts = s.lower().strip().split()
    if len(parts) < 2:
        return None
    if parts[-1] in _CHARGE_SUFFIX_WORDS:
        return (" ".join(parts[:-1]).strip(), _CHARGE_SUFFIX_WORDS[parts[-1]])
    if parts[0] in _CHARGE_PREFIX_WORDS:
        return (" ".join(parts[1:]).strip(), _CHARGE_PREFIX_WORDS[parts[0]])
    return None


def _retarget_charge(canonical: str, sign: str) -> str:
    """Replace any trailing charge marker on a canonical name with `sign`."""
    base = canonical
    if base.endswith("_bar"):
        base = base[:-4]
    elif base.endswith("~"):
        base = base[:-1]
    elif base and base[-1] in {"+", "-"}:
        base = base[:-1]
    return f"{base}{sign}"


def _get_antiparticle(api: Any, particle: Any) -> Any:
    """Fetch the antiparticle by negating the MCID. Returns None if self-conjugate or not found."""
    if particle is None:
        return None
    try:
        mcid = particle.mcid
    except Exception:
        return None
    if not mcid:
        return None
    try:
        anti = api.get_particle_by_mcid(-mcid)
    except Exception:
        return None
    anti = _to_individual(anti)
    if anti is None:
        return None
    # Self-conjugate (e.g., photon, pi0) — PDG returns the same entry.
    if getattr(anti, "mcid", None) == mcid:
        return None
    return anti


def _to_individual(obj: Any) -> Any:
    """Flatten a PdgParticleList to its first concrete PdgParticle when possible."""
    if obj is None:
        return None
    try:
        if hasattr(obj, "get_particles"):
            parts = obj.get_particles()
            if parts:
                return parts[0]
    except Exception:
        pass
    return obj


def _case_variants(s: str) -> list[str]:
    """Generate common case variants of a name for PDG lookup (which is case-sensitive)."""
    s = s.strip()
    if not s:
        return []
    out = [s]
    lo = s.lower()
    up = s.upper()
    if lo != s:
        out.append(lo)
    if up != s:
        out.append(up)
    # Capitalised first letter ('sigma+' -> 'Sigma+', 'lambda' -> 'Lambda').
    if s[0].islower():
        out.append(s[0].upper() + s[1:])
    # Dedup, preserve order.
    seen: set[str] = set()
    deduped: list[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _resolve_simple(api: Any, aliases: dict[str, str], query: str) -> Any:
    """Resolve a particle by name/MCID/alias. No anti-particle handling."""
    if not query:
        return None

    candidates: list[str] = []

    def _add(t: str) -> None:
        t = (t or "").strip()
        if t and t not in candidates:
            candidates.append(t)

    # Direct alias hit takes priority (alias keys are lowercased).
    direct = aliases.get(query.strip().lower())
    if direct:
        for v in _case_variants(direct):
            _add(v)

    # Original query and case variants.
    for v in _case_variants(query):
        _add(v)

    # Catch-all: alias-map any current candidate.
    for term in list(candidates):
        m = aliases.get(term.lower())
        if not m:
            continue
        for v in _case_variants(m):
            _add(v)

    lookups = [
        lambda t: api.get(t),
        lambda t: api.get_particle_by_name(t),
        lambda t: api.get_particle_by_mcid(t),
    ]
    for term in candidates:
        for lookup in lookups:
            try:
                result = lookup(term)
            except Exception:
                continue
            individual = _to_individual(result)
            if individual is not None:
                return individual

    # Last resort: list-style lookup.
    for term in candidates:
        try:
            results = api.get_particles_by_name(term)
        except Exception:
            results = None
        if results:
            for r in results:
                individual = _to_individual(r)
                if individual is not None:
                    return individual

    return None


def _resolve_particle(api: Any, particle_id: str) -> Any:
    """Resolve a particle from name / alias / MCID, including natural-language anti markers."""
    if not particle_id:
        return None

    aliases = _load_aliases()
    lowered = particle_id.lower().strip()

    # Detect anti-particle intent (anti-/anti /antiX prefix or _bar/bar/~ suffix).
    base_query, wants_anti = _strip_anti_markers(lowered)
    if not base_query:
        base_query = lowered

    # Natural-language charge ('muon plus', 'positive tau', 'pion zero') on the base form.
    charge_mod = _extract_charge_modifier(base_query)
    base_particle: Any = None
    if charge_mod is not None:
        nl_base, nl_sign = charge_mod
        nl_alias = aliases.get(nl_base)
        # Try the alias-retargeted symbolic form first (e.g., 'muon plus' -> 'mu+').
        if nl_alias:
            base_particle = _resolve_simple(api, aliases, _retarget_charge(nl_alias, nl_sign))
        # Then try the direct symbolic form (e.g., 'muon+' if PDG knows it).
        if base_particle is None and nl_base:
            base_particle = _resolve_simple(api, aliases, f"{nl_base}{nl_sign}")

    if base_particle is None:
        base_particle = _resolve_simple(api, aliases, base_query)

    # Fall back to original query if anti-stripping yielded nothing useful.
    if base_particle is None and base_query != lowered:
        base_particle = _resolve_simple(api, aliases, lowered)

    if wants_anti and base_particle is not None:
        anti = _get_antiparticle(api, base_particle)
        if anti is not None:
            return anti
        # Self-conjugate or PDG missing the entry — return base as best-effort.

    if base_particle is not None:
        return base_particle

    # Case-insensitive name scan across actual particles (slow path).
    try:
        for particle in api.get_particles():
            n = getattr(particle, "name", None)
            if n and n.lower() == lowered:
                return _to_individual(particle)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

_SEARCH_PARTICLE_DESCRIPTION = (
    "Look up a particle in the Particle Data Group (PDG) database and return its key "
    "properties: mass (MeV and GeV), charge, spin, color, parity/C/I/G quantum numbers, "
    "lifetime or decay width, MCID, and PDG review identifier.\n\n"
    "Accepts a wide range of inputs:\n"
    "  - PDG canonical names: 'mu+', 'pi0', 'K-', 'Sigma+', 'gamma', 'Lambda', 'H'\n"
    "  - Common English names: 'muon', 'pion', 'higgs', 'photon', 'electron', 'top quark'\n"
    "  - Anti-particle markers: 'antimuon', 'anti-up quark', 'ubar', 'u bar', 'u_bar', 'p~'\n"
    "  - Natural-language charge: 'muon plus', 'positive tau', 'pion zero', 'kaon minus'\n"
    "  - Numeric MC ID strings (e.g. '11' for electron, '-13' for mu+)\n\n"
    "Returns Markdown-formatted text followed by a fenced ```json``` block with a structured "
    "payload {query, count, particles:[{name, description, mcid, pdg_review_id, mass:{mev,gev}, "
    "charge:{value,fraction}, spin, color, quantum_numbers, lifetime|width}]} for programmatic use."
)

_LIST_DECAYS_DESCRIPTION = (
    "List decay modes (with branching ratios where available) for a given particle from the "
    "PDG database.\n\n"
    "Accepts the same wide range of identifiers as search_particle (PDG names, English aliases, "
    "anti-particle markers, natural-language charge, MC IDs).\n\n"
    "Tries exclusive_branching_fractions first, then branching_fractions, then "
    "inclusive_branching_fractions; reports which source was used. For stable particles, "
    "returns an empty decay list with stable=true.\n\n"
    "Returns Markdown-formatted text followed by a fenced ```json``` block with a structured "
    "payload {particle, source, count, decays:[{description, branching_ratio_text, value, ...}]}."
)


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_particle",
            description=_SEARCH_PARTICLE_DESCRIPTION,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Particle name, symbol, alias, or numeric MC ID. "
                            "Examples: 'electron', 'mu+', 'muon plus', 'antimuon', "
                            "'anti up quark', 'higgs', 'pion zero', '-13'."
                        ),
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_decays",
            description=_LIST_DECAYS_DESCRIPTION,
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_id": {
                        "type": "string",
                        "description": (
                            "Particle name, alias, anti-particle marker, or numeric MC ID. "
                            "Examples: 'tau', 'mu-', 'positive tau', 'antimuon', 'B0', '321'."
                        ),
                    }
                },
                "required": ["particle_id"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls."""
    if arguments is None:
        arguments = {}
    try:
        if name == "search_particle":
            return await search_particle(arguments)
        if name == "list_decays":
            return await list_decays(arguments)
        raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """Coerce a PDG value to something JSON can represent (no NaN/Inf)."""
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return str(value)


def _particle_to_dict(particle: Any) -> dict[str, Any]:
    """Serialize a PdgParticle to a JSON-friendly dict."""
    out: dict[str, Any] = {
        "name": getattr(particle, "name", None),
        "description": getattr(particle, "description", None),
        "mcid": _json_safe(getattr(particle, "mcid", None)),
        "pdg_review_id": _json_safe(getattr(particle, "pdgid", None)),
    }

    mass_raw = None
    try:
        mass_raw = getattr(particle, "mass", None)
    except Exception:
        mass_raw = None
    mass_gev = _to_float(mass_raw)
    if mass_gev is not None:
        out["mass"] = {"mev": mass_gev * 1000.0, "gev": mass_gev}
    elif _is_quark(particle):
        qm = _format_quark_mass_from_measurements(particle)
        if qm:
            out["mass"] = {"text": qm}

    try:
        charge_raw = getattr(particle, "charge", None)
    except Exception:
        charge_raw = None
    if charge_raw is not None:
        cv = _to_float(charge_raw)
        out["charge"] = {
            "value": cv,
            "fraction": _format_charge(charge_raw),
        }

    spin = _get_first_attr(particle, ["quantum_J", "J", "spin"])
    if spin is not None:
        out["spin"] = _json_safe(spin)

    color_code = _infer_color_multiplicity(particle)
    out["color"] = {"multiplicity": int(color_code), "label": _format_color_label(color_code)}

    quantum_numbers: dict[str, Any] = {}
    for symbol, attrs in [
        ("J", ["quantum_J", "J"]),
        ("P", ["quantum_P", "P"]),
        ("C", ["quantum_C", "C"]),
        ("I", ["quantum_I", "I"]),
        ("G", ["quantum_G", "G"]),
    ]:
        v = _get_first_attr(particle, attrs)
        if v is not None:
            quantum_numbers[symbol] = _json_safe(v)
    t3 = _get_first_attr(particle, ["weak_isospin", "t3", "T3"])
    if t3 is not None:
        quantum_numbers["T3"] = _json_safe(t3)
    y = _get_first_attr(particle, ["weak_hypercharge", "y", "Y"])
    if y is not None:
        quantum_numbers["Y"] = _json_safe(y)
    if quantum_numbers:
        out["quantum_numbers"] = quantum_numbers

    try:
        lifetime = getattr(particle, "lifetime", None)
    except Exception:
        lifetime = None
    if lifetime is not None:
        lf = _to_float(lifetime)
        out["lifetime"] = {
            "seconds": lf,
            "stable": (lf is None and str(lifetime).strip().lower() in {"inf", "+inf", "infinity"})
                      or (lf is not None and math.isinf(lf)),
            "text": _format_lifetime(lifetime),
        }
    else:
        try:
            width = getattr(particle, "width", None)
        except Exception:
            width = None
        if width is not None:
            wf = _to_float(width)
            out["width"] = {
                "value": wf,
                "stable": wf == 0.0,
                "text": _format_width(width),
            }

    flags = _get_first_attr(particle, ["data_flags"])
    if isinstance(flags, str) and flags:
        out["data_flags"] = flags

    return out


def _decay_to_dict(decay: Any) -> dict[str, Any]:
    """Serialize a decay-mode entry to a JSON-friendly dict."""
    out: dict[str, Any] = {
        "description": getattr(decay, "description", None),
        "branching_ratio_text": getattr(decay, "display_value_text", None),
    }
    for attr in ("value", "value_text", "is_limit"):
        v = getattr(decay, attr, None)
        if v is not None:
            out[attr] = _json_safe(v)
    if not out["description"] and not out["branching_ratio_text"]:
        out["raw"] = str(decay)
    return out


def _json_block(payload: dict[str, Any]) -> str:
    """Render a payload as a fenced JSON code block."""
    return "```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_PDG_NOT_INSTALLED_MSG = "Error: pdg package not installed. Please install with: pip install pdg"


_GENERIC_QUARK_TERMS = {
    "quark", "a quark", "the quark", "any quark", "quarks",
    "anti quark", "anti-quark", "antiquark", "anti quarks", "anti-quarks", "antiquarks",
    "quark bar", "quarkbar", "quark_bar",
}


def _search_particle_sync(arguments: dict) -> list[types.TextContent]:
    """Synchronous core for ``search_particle``. Wrapped in to_thread by the async entrypoint."""
    query = arguments.get("query", "")
    if isinstance(query, str):
        query = query.strip()
    if not query:
        return [types.TextContent(type="text", text="Error: query parameter is required")]

    if query.lower().strip() in _GENERIC_QUARK_TERMS:
        return [types.TextContent(
            type="text",
            text=(
                "The term 'quark' is ambiguous. Please specify a quark type: "
                "up (u), down (d), strange (s), charm (c), bottom (b), or top (t). "
                "For antiquarks, prefix 'anti' or suffix 'bar' (e.g., 'anti up quark', 'ubar')."
            ),
        )]

    try:
        api = _get_pdg_api()
    except ImportError:
        return [types.TextContent(type="text", text=_PDG_NOT_INSTALLED_MSG)]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error connecting to PDG: {e}")]

    found_particles: list[Any] = []

    def _append_if_new(p: Any) -> None:
        pid = getattr(p, "pdgid", None)
        if pid is None:
            return
        if not any(getattr(x, "pdgid", None) == pid for x in found_particles):
            found_particles.append(p)

    try:
        exact = _resolve_particle(api, query)
        if exact is not None:
            _append_if_new(exact)

        try:
            for p in api.get_particles_by_name(query) or []:
                _append_if_new(p)
        except Exception as e:
            logger.debug(f"get_particles_by_name failed for '{query}': {e}")

        try:
            by_mcid = api.get_particle_by_mcid(query)
        except Exception:
            by_mcid = None
        if by_mcid is not None:
            try:
                if hasattr(by_mcid, "get_particles"):
                    extras = by_mcid.get_particles() or []
                elif isinstance(by_mcid, (list, tuple)):
                    extras = list(by_mcid)
                else:
                    extras = [by_mcid]
                for p in extras:
                    _append_if_new(p)
            except Exception as e:
                logger.debug(f"MCID expansion failed for '{query}': {e}")
    except Exception as e:
        if not found_particles:
            logger.warning(f"Search error: {e}")

    if not found_particles:
        return [types.TextContent(type="text", text=f"No particles found matching '{query}'")]

    lines: list[str] = [f"Found {len(found_particles)} particle(s) matching '{query}':", ""]
    particles_payload: list[dict[str, Any]] = []

    for i, particle in enumerate(found_particles, start=1):
        try:
            data = _particle_to_dict(particle)
            particles_payload.append(data)

            lines.append(f"{i}. {data.get('description') or data.get('name') or 'Unknown particle'}")
            if data.get("name"):
                lines.append(f"   Name: {data['name']}")
            if data.get("mcid") is not None:
                lines.append(f"   PDG ID: {data['mcid']}")
            if data.get("pdg_review_id") is not None:
                lines.append(f"   PDG Review ID: {data['pdg_review_id']}")

            mass = data.get("mass")
            if isinstance(mass, dict):
                if "mev" in mass:
                    lines.append(f"   Mass: {mass['mev']} MeV ({mass['gev']} GeV)")
                elif "text" in mass:
                    lines.append(f"   Mass: {mass['text']}")

            spin = data.get("spin")
            if spin is not None:
                lines.append(f"   Spin (J): {spin}")

            charge = data.get("charge")
            if isinstance(charge, dict) and charge.get("fraction") is not None:
                lines.append(f"   Charge: {charge['fraction']}")

            color = data.get("color")
            if isinstance(color, dict) and color.get("label"):
                lines.append(f"   Color: {color['label']}")

            qn = data.get("quantum_numbers") or {}
            if qn:
                lines.append("   Quantum numbers: " + ", ".join(f"{k}={v}" for k, v in qn.items()))

            lifetime = data.get("lifetime")
            width = data.get("width")
            if isinstance(lifetime, dict) and lifetime.get("text"):
                lines.append(f"   Lifetime: {lifetime['text']}")
            elif isinstance(width, dict) and width.get("text"):
                lines.append(f"   Width: {width['text']}")

            lines.append("")
        except Exception as e:
            lines.append(f"   Error retrieving particle info: {e}")
            lines.append("")

    payload = {"query": query, "count": len(particles_payload), "particles": particles_payload}
    text = "\n".join(lines).rstrip() + "\n\n" + _json_block(payload) + "\n"
    return [types.TextContent(type="text", text=text)]


def _list_decays_sync(arguments: dict) -> list[types.TextContent]:
    """Synchronous core for ``list_decays``."""
    particle_id = arguments.get("particle_id", "")
    if isinstance(particle_id, str):
        particle_id = particle_id.strip()
    if not particle_id:
        return [types.TextContent(type="text", text="Error: particle_id parameter is required")]

    try:
        api = _get_pdg_api()
    except ImportError:
        return [types.TextContent(type="text", text=_PDG_NOT_INSTALLED_MSG)]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error connecting to PDG: {e}")]

    particle = _resolve_particle(api, particle_id)
    if particle is None:
        return [types.TextContent(type="text", text=f"Particle '{particle_id}' not found")]

    try:
        decay_entries: list[Any] = []
        source: str | None = None
        for method_name in [
            "exclusive_branching_fractions",
            "branching_fractions",
            "inclusive_branching_fractions",
        ]:
            method = getattr(particle, method_name, None)
            if not callable(method):
                continue
            try:
                entries = method()
                decay_entries = list(entries) if entries is not None else []
            except Exception:
                decay_entries = []
            if decay_entries:
                source = method_name
                break

        if not decay_entries:
            for attr in ["decay_modes", "decays"]:
                value = getattr(particle, attr, None)
                if value:
                    decay_entries = list(value)
                    source = attr
                    break
            if not decay_entries:
                getter = getattr(particle, "get_decay_modes", None)
                if callable(getter):
                    try:
                        dm = getter()
                        decay_entries = list(dm) if dm else []
                        if decay_entries:
                            source = "get_decay_modes"
                    except Exception:
                        decay_entries = []

        particle_label = getattr(particle, "description", None) or particle_id
        if not decay_entries:
            payload = {
                "particle": _particle_to_dict(particle),
                "decays": [],
                "stable": True,
            }
            return [types.TextContent(
                type="text",
                text=(
                    f"No decay modes found for particle '{particle_label}'. "
                    "This particle may be stable or decay information may not be available.\n\n"
                    + _json_block(payload) + "\n"
                ),
            )]

        header_lines = [f"Decay modes for particle '{particle_label}':", ""]
        if _is_anti_query(particle_id):
            header_lines.append(
                "Note: If the particle decays, its antiparticle decays through the same processes "
                "but with charges flipped."
            )
            header_lines.append("")

        body_lines: list[str] = []
        decays_payload: list[dict[str, Any]] = []
        for idx, decay in enumerate(decay_entries, start=1):
            try:
                d = _decay_to_dict(decay)
                decays_payload.append(d)
                desc = d.get("description")
                br = d.get("branching_ratio_text")
                if desc and br:
                    body_lines.append(f"{idx}. {desc} (BR: {br})")
                elif desc:
                    body_lines.append(f"{idx}. {desc}")
                else:
                    body_lines.append(f"{idx}. {d.get('raw') or decay}")
            except Exception as e:
                body_lines.append(f"{idx}. [Decay mode info unavailable: {e}]")

        payload = {
            "particle": _particle_to_dict(particle),
            "source": source,
            "count": len(decays_payload),
            "decays": decays_payload,
        }
        text = (
            "\n".join(header_lines)
            + "\n".join(body_lines)
            + "\n\n"
            + _json_block(payload)
            + "\n"
        )
        return [types.TextContent(type="text", text=text)]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error retrieving decay modes: {e}")]


async def search_particle(arguments: dict) -> list[types.TextContent]:
    """Async wrapper that runs the blocking PDG calls off the event loop."""
    return await asyncio.to_thread(_search_particle_sync, arguments)


async def list_decays(arguments: dict) -> list[types.TextContent]:
    """Async wrapper that runs the blocking PDG calls off the event loop."""
    return await asyncio.to_thread(_list_decays_sync, arguments)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Main entry point for the server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="particlephysics-mcp-server",
                server_version="1.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(
                        tools_changed=False,
                        resources_changed=False,
                        prompts_changed=False,
                    ),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

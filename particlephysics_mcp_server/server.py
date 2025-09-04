#!/usr/bin/env python3
"""
ParticlePhysics MCP Server

A Model Context Protocol server that provides access to the Particle Data Group Database.
"""

import asyncio
import logging
import os
import sys
import subprocess
import json
from fractions import Fraction
from typing import Any, Sequence

# Function to find and add the correct module paths
def setup_module_paths():
    """Find and add the correct paths for MCP and PDG modules."""
    try:
        # Get the location of installed packages using uvx
        result = subprocess.run(['uvx', 'pip', 'show', 'mcp'], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                mcp_path = line.split(':', 1)[1].strip()
                if mcp_path not in sys.path:
                    sys.path.insert(0, mcp_path)
                break
        
        result = subprocess.run(['uvx', 'pip', 'show', 'pdg'], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                pdg_path = line.split(':', 1)[1].strip()
                if pdg_path not in sys.path:
                    sys.path.insert(0, pdg_path)
                break
    except Exception as e:
        logging.warning(f"Could not automatically find module paths: {e}")

# Setup module paths before importing
setup_module_paths()

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.types as types

# Configure logging: send to stderr so stdout remains clean for MCP JSON-RPC
_log_level = os.getenv("MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("particlephysics-mcp-server")

# Initialize the MCP server
server = Server("particlephysics-mcp-server")


_NAME_MAPPINGS_CACHE: dict[str, str] | None = None

def _load_name_mappings() -> dict[str, str]:
    """Load generated/name_mappings.json if available and cache it.

    Returns a lowercase-key mapping from alias/description to canonical PDG description.
    If file is missing or invalid, returns an empty dict.
    """
    global _NAME_MAPPINGS_CACHE
    if _NAME_MAPPINGS_CACHE is not None:
        return _NAME_MAPPINGS_CACHE
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        path = os.path.join(root, 'generated', 'name_mappings.json')
        if not os.path.exists(path):
            _NAME_MAPPINGS_CACHE = {}
            return _NAME_MAPPINGS_CACHE
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        # Normalize to lowercase keys
        mapping: dict[str, str] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    mapping[k.strip().lower()] = v
        _NAME_MAPPINGS_CACHE = mapping
        return _NAME_MAPPINGS_CACHE
    except Exception as e:
        logger.warning(f"Failed to load name mappings: {e}")
        _NAME_MAPPINGS_CACHE = {}
        return _NAME_MAPPINGS_CACHE

def _format_charge(value: Any) -> str:
    """Format electric charge nicely (e.g., 0.3333 -> 1/3, -0.6666 -> -2/3).

    Falls back to the original representation if conversion is not possible.
    """
    try:
        # Handle strings like '0.3333'
        if isinstance(value, str):
            value = float(value)
        # Handle None
        if value is None:
            return "unknown"
        # Exact zero
        if float(value) == 0.0:
            return "0"
        # Convert to a Fraction with bounded denominator to capture common charges
        frac = Fraction(float(value)).limit_denominator(6)
        # If denominator is 1, show integer
        if frac.denominator == 1:
            return f"{frac.numerator}"
        # Otherwise show numerator/denominator with sign handled naturally
        return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        # Fallback to string
        return str(value)


def _format_lifetime(value: Any) -> str:
    """Format lifetime, showing stable for infinities when appropriate."""
    try:
        if value is None:
            return "unknown"
        s = str(value).strip().lower()
        if s in {"inf", "+inf", "infinity"}:
            return "stable (infinite)"
        # numeric check against infinity
        try:
            if float(value) == float("inf"):
                return "stable (infinite)"
        except Exception:
            pass
        return str(value)
    except Exception:
        return str(value)


def _format_width(value: Any) -> str:
    """Format decay width, showing 0 (stable) for zero values."""
    try:
        if value is None:
            return "unknown"
        if float(value) == 0.0:
            return "0 (stable)"
        return str(value)
    except Exception:
        return str(value)


def _to_float(value: Any) -> float | None:
    """Best-effort convert to float; return None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        try:
            # Sometimes PDG may provide strings with units; strip non-numeric
            s = str(value).strip()
            # Keep digits, dot, minus, exponent, and slash (handled later)
            return float(s)
        except Exception:
            return None


def _format_mass_mev_gev(value: Any) -> tuple[str, str]:
    """Return (MeV_str, GeV_str) for a given mass value.
    If conversion fails, return (str(value), 'unknown').
    """
    mv = _to_float(value)
    if mv is None:
        return (str(value) if value is not None else "unknown", "unknown")
    # Assume PDG mass is in MeV (common for PDG tables); provide both
    mass_mev = mv
    mass_gev = mv / 1000.0
    return (f"{mass_mev}", f"{mass_gev}")


def _is_anti_query(text: str) -> bool:
    """Heuristic: determine if the user query refers to an anti-particle.

    Looks for markers like 'anti-', ' anti ', '~', or *_bar and common 'ubar', 'dbar', etc.
    """
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
    """Negate a numeric-like value represented as number or string (supports simple rationals like '2/3')."""
    # Try float
    try:
        f = float(value)
        nf = -f
        # Preserve integer-like presentation when possible
        if nf.is_integer():
            return str(int(nf))
        return str(nf)
    except Exception:
        pass
    # Try rational a/b
    try:
        s = str(value).strip()
        if "/" in s:
            num, den = s.split("/", 1)
            num = num.strip()
            den = den.strip()
            if num.startswith("-"):
                return f"{num[1:]}/{den}"
            if num.startswith("+"):
                return f"-{num[1:]}/{den}"
            return f"-{num}/{den}"
        # Fallback: add/remove leading '-'
        if s.startswith("-"):
            return s[1:]
        if s.startswith("+"):
            return f"-{s[1:]}"
        return f"-{s}"
    except Exception:
        return str(value)


def _infer_color_multiplicity(particle: Any) -> str:
    """Infer QCD color multiplicity from PDG flags when available.
    - 'Q' (quark) -> 3 (triplet)
    - 'G' (gluon) -> 8 (octet)
    Otherwise -> 1 (singlet)
    """
    try:
        flags = getattr(particle, 'data_flags', None)
        if flags is None and hasattr(particle, 'particle_list'):
            flags = getattr(particle.particle_list, 'data_flags', None)
        if isinstance(flags, str):
            if 'G' in flags:
                return '8'
            if 'Q' in flags:
                return '3'
        # For named cases if flags not present
        name = (getattr(particle, 'description', None) or '').lower()
        if name in {"g", "gluon"}:
            return '8'
        if name in {"u","d","s","c","b","t","up","down","strange","charm","bottom","top"}:
            return '3'
    except Exception:
        pass
    return '1'


def _format_color_label(color_code: str) -> str:
    """Map color multiplicity to human label."""
    mapping = {
        '1': 'singlet',
        '3': 'triplet',
        '8': 'octet',
    }
    return mapping.get(str(color_code), str(color_code))


def _is_quark(particle: Any) -> bool:
    """Best-effort check if particle is a quark using PDG flags or description."""
    try:
        flags = getattr(particle, 'data_flags', None)
        if isinstance(flags, str) and 'Q' in flags:
            return True
        name = (getattr(particle, 'description', None) or '').lower().strip()
        return name in {'u','d','s','c','b','t','up','down','strange','charm','bottom','top'}
    except Exception:
        return False


def _format_quark_mass_from_measurements(particle: Any) -> str | None:
    """Extract a human-readable quark mass from PDG measurement methods if available.

    Returns a concise text (prefer PDG's value_text if present)."""
    try:
        # PDG may expose masses() or mass_measurements() methods
        for method_name in ['masses', 'mass_measurements']:
            method = getattr(particle, method_name, None)
            if callable(method):
                try:
                    entries = list(method())
                except Exception:
                    entries = []
                if not entries:
                    continue
                # Prefer an entry with value_text
                for entry in entries:
                    vt = getattr(entry, 'value_text', None)
                    if vt:
                        return str(vt)
                # Fallback to numeric value + optional units
                first = entries[0]
                val = getattr(first, 'value', None)
                units = getattr(first, 'units', '') or ''
                if val is not None:
                    return f"{val} {units}".strip()
        return None
    except Exception:
        return None

def _get_first_attr(particle: Any, candidate_names: list[str]) -> Any:
    """Return the first non-None attribute among candidate_names if present on particle or its particle_list wrapper."""
    for name in candidate_names:
        try:
            if hasattr(particle, name):
                value = getattr(particle, name)
                if value is not None:
                    return value
        except Exception:
            pass
    # Try the nested particle_list if present
    if hasattr(particle, 'particle_list'):
        nested = getattr(particle, 'particle_list')
        for name in candidate_names:
            try:
                if hasattr(nested, name):
                    value = getattr(nested, name)
                    if value is not None:
                        return value
            except Exception:
                pass
    return None

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_particle",
            description="Search for particles by name or properties in the PDG database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (particle name, symbol, or property)"
                    }
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="list_decays",
            description="List decay modes for a specific particle",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_id": {
                        "type": "string",
                        "description": "Particle identifier (PDG ID or name)"
                    }
                },
                "required": ["particle_id"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls."""
    if arguments is None:
        arguments = {}
    
    try:
        if name == "search_particle":
            return await search_particle(arguments)
        
        elif name == "list_decays":
            return await list_decays(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


def _find_particle_by_alias(api, particle_id: str):
    """Helper function to find particle by alias or name."""
    # Start from generated mappings if available
    generated = _load_name_mappings()
    # Common aliases for well-known particles (using actual PDG names), used as fallback/augmentation
    common_aliases = {
        # Photons
        'photon': 'gamma',
        'gamma': 'gamma',
        'light particle': 'gamma',
        
        # Leptons
        'electron': 'e-',
        'e-': 'e-',
        'positron': 'e+',
        'e+': 'e+',
        'e': 'e-',
        # Anti-electron (positron) aliases
        'anti-electron': 'e+',
        'anti electron': 'e+',
        'antielectron': 'e+',
        'e_bar': 'e+',
        'e~': 'e+',
        'muon': 'mu-',
        'mu-': 'mu-',
        'mu': 'mu-',
        'antimu': 'mu+',
        'antimuon': 'mu+',
        'anti-muon': 'mu+',
        'anti muon': 'mu+',
        'mu+': 'mu+',
        # Anti-muon aliases
        'mu_bar': 'mu+',
        'mu~': 'mu+',
        'tau': 'tau-',
        'tauon': 'tau-',
        'tau-': 'tau-',
        'antitau': 'tau+',
        'antitauon': 'tau+',
        'anti-tau': 'tau+',
        'anti tau': 'tau+',
        'tau+': 'tau+',
        # Anti-tau aliases
        'tau_bar': 'tau+',
        'tau~': 'tau+',
        
        # Neutrinos
        'neutrino': 'nu_e',
        'electron neutrino': 'nu_e',
        'muon neutrino': 'nu_mu',
        'tau neutrino': 'nu_tau',
        'nu_e': 'nu_e',
        'nu_mu': 'nu_mu',
        'nu_tau': 'nu_tau',
        'nu(e)': 'nu_e',
        'nu(mu)': 'nu_mu',
        'nu(tau)': 'nu_tau',
        # Anti-neutrino (flavor-specific) aliases
        'electron antineutrino': 'nu_e_bar',
        'anti-electron neutrino': 'nu_e_bar',
        'anti electron neutrino': 'nu_e_bar',
        'nu_e_bar': 'nu_e_bar',
        'nu_e~': 'nu_e_bar',
        'muon antineutrino': 'nu_mu_bar',
        'anti-muon neutrino': 'nu_mu_bar',
        'anti muon neutrino': 'nu_mu_bar',
        'nu_mu_bar': 'nu_mu_bar',
        'nu_mu~': 'nu_mu_bar',
        'tau antineutrino': 'nu_tau_bar',
        'anti-tau neutrino': 'nu_tau_bar',
        'anti tau neutrino': 'nu_tau_bar',
        'nu_tau_bar': 'nu_tau_bar',
        'nu_tau~': 'nu_tau_bar',
        
        # Gauge bosons
        'gluon': 'g',
        'g': 'g',
        'w boson': 'W+',
        'w+': 'W+',
        'w-': 'W-',
        'z boson': 'Z0',
        'z': 'Z0',
        'higgs': 'H',
        'h': 'H',
        'god particle': 'H',
        # RPP-style kaon short/long
        'k(s)': 'K0S',
        'k(l)': 'K0L',
        
        # Mesons
        'pion': 'pi+',
        'pion+': 'pi+',
        'pion-': 'pi-',
        'pion0': 'pi0',
        'pi+': 'pi+',
        'pi-': 'pi-',
        'pi0': 'pi0',
        'eta': 'eta',
        'kaon': 'K+',
        'kaon+': 'K+',
        'kaon-': 'K-',
        'kaon0': 'K0',
        'K+': 'K+',
        'K-': 'K-',
        'K0': 'K0',
        
        # Baryons
        'proton': 'p',
        'p': 'p',
        'neutron': 'n',
        'n': 'n',
        'lambda': 'Lambda',
        'sigma': 'Sigma+',
        'xi': 'Xi0',
        'omega': 'Omega-',
        
        # Quarks
        'up quark': 'u',
        'up': 'u',
        'u': 'u',
        # Anti-up quark aliases
        'anti-up quark': 'u_bar',
        'anti up quark': 'u_bar',
        'antiup quark': 'u_bar',
        'anti-up': 'u_bar',
        'antiup': 'u_bar',
        'ubar': 'u_bar',
        'u_bar': 'u_bar',
        'u~': 'u_bar',
        'down quark': 'd',
        'down': 'd',
        'd': 'd',
        # Anti-down quark aliases
        'anti-down quark': 'd_bar',
        'anti down quark': 'd_bar',
        'antidown quark': 'd_bar',
        'anti-down': 'd_bar',
        'antidown': 'd_bar',
        'dbar': 'd_bar',
        'd_bar': 'd_bar',
        'd~': 'd_bar',
        'strange quark': 's',
        'strange': 's',
        's': 's',
        # Anti-strange quark aliases
        'anti-strange quark': 's_bar',
        'anti strange quark': 's_bar',
        'antistrange quark': 's_bar',
        'anti-strange': 's_bar',
        'antistrange': 's_bar',
        'sbar': 's_bar',
        's_bar': 's_bar',
        's~': 's_bar',
        'charm quark': 'c',
        'charm': 'c',
        'c': 'c',
        # Anti-charm quark aliases
        'anti-charm quark': 'c_bar',
        'anti charm quark': 'c_bar',
        'anticharm quark': 'c_bar',
        'anti-charm': 'c_bar',
        'anticharm': 'c_bar',
        'cbar': 'c_bar',
        'c_bar': 'c_bar',
        'c~': 'c_bar',
        'bottom quark': 'b',
        'bottom': 'b',
        'beauty quark': 'b',
        'beauty': 'b',
        'b': 'b',
        # Anti-bottom (beauty) quark aliases
        'anti-bottom quark': 'b_bar',
        'anti bottom quark': 'b_bar',
        'antibottom quark': 'b_bar',
        'anti-beauty quark': 'b_bar',
        'anti beauty quark': 'b_bar',
        'antibeauty quark': 'b_bar',
        'anti-bottom': 'b_bar',
        'antibottom': 'b_bar',
        'anti-beauty': 'b_bar',
        'antibeauty': 'b_bar',
        'bbar': 'b_bar',
        'b_bar': 'b_bar',
        'b~': 'b_bar',
        'top quark': 't',
        'top': 't',
        'truth quark': 't',
        'truth': 't',
        't': 't',
        # Anti-top (truth) quark aliases
        'anti-top quark': 't_bar',
        'anti top quark': 't_bar',
        'antitop quark': 't_bar',
        'anti-truth quark': 't_bar',
        'anti truth quark': 't_bar',
        'antitruth quark': 't_bar',
        'anti-top': 't_bar',
        'antitop': 't_bar',
        'anti-truth': 't_bar',
        'antitruth': 't_bar',
        'tbar': 't_bar',
        't_bar': 't_bar',
        't~': 't_bar'
    }
    
    def get_individual_particle(particle_list):
        """Return a concrete PDG particle if available; otherwise return input as-is."""
        try:
            if hasattr(particle_list, 'get_particles'):
                individuals = particle_list.get_particles()
                if individuals:
                    return individuals[0]
        except Exception:
            pass
        # Already a particle-like object; return as-is to preserve methods like masses()
        return particle_list
    
    # Normalize the input
    search_term = particle_id.lower().strip()
    # Build candidate terms to try (ensure anti-particles map to base particle too)
    candidate_terms = []
    def _append_unique(term: str):
        t = term.strip()
        if t and t not in candidate_terms:
            candidate_terms.append(t)

    _append_unique(particle_id)
    _append_unique(search_term)
    # Remove common anti markers to get base particle
    base_term = search_term
    if base_term.endswith('_bar'):
        base_term = base_term[:-4]
    elif base_term.endswith('bar') and len(base_term) > 3:
        # e.g., 'ubar' -> 'u'
        base_term = base_term[:-3]
    if base_term.endswith('~'):
        base_term = base_term[:-1]
    if base_term.startswith('anti-'):
        base_term = base_term[5:]
    if base_term.startswith('anti '):
        base_term = base_term[5:]
    # Normalize RPP-style neutrino and kaon names
    base_term = (
        base_term
        .replace('nu(e)', 'nu_e')
        .replace('nu(mu)', 'nu_mu')
        .replace('nu(tau)', 'nu_tau')
        .replace('k(s)', 'K0S')
        .replace('k(l)', 'K0L')
    )
    if base_term != search_term:
        _append_unique(base_term)
    
    # First, try the exact match
    for term in candidate_terms:
        try:
            result = api.get(term)
            if result:
                return get_individual_particle(result)
        except:
            pass
    
    # Try to get particle by name (handles ambiguity)
    for term in candidate_terms:
        try:
            result = api.get_particle_by_name(term)
            if result:
                return get_individual_particle(result)
        except Exception:
            # If there's ambiguity, try to get all particles with this name
            try:
                results = api.get_particles_by_name(term)
                if results:
                    return get_individual_particle(results[0])  # Return first match
            except:
                pass
    
    # Try generated mappings first
    for term in candidate_terms:
        if term in generated:
            mapped = generated[term]
            try:
                result = api.get_particle_by_name(mapped)
                if result:
                    return get_individual_particle(result)
            except:
                try:
                    result = api.get(mapped)
                    if result:
                        return get_individual_particle(result)
                except:
                    pass
            # If mapped is an anti-form, also try base form
            if isinstance(mapped, str):
                m = mapped
                if m.endswith('_bar'):
                    m = m[:-4]
                elif m.endswith('~'):
                    m = m[:-1]
                if m and m != mapped:
                    try:
                        result = api.get_particle_by_name(m)
                        if result:
                            return get_individual_particle(result)
                    except:
                        try:
                            result = api.get(m)
                            if result:
                                return get_individual_particle(result)
                        except:
                            pass

    # Try common aliases as fallback
    for term in candidate_terms:
        if term in common_aliases:
            mapped = common_aliases[term]
            try:
                # Try to get the particle by name first
                result = api.get_particle_by_name(mapped)
                if result:
                    return get_individual_particle(result)
            except:
                try:
                    # If that fails, try to get by PDG ID
                    result = api.get(mapped)
                    if result:
                        return get_individual_particle(result)
                except:
                    pass
            # If mapped is an anti-form, also try the base form
            if isinstance(mapped, str):
                m = mapped
                if m.endswith('_bar'):
                    m = m[:-4]
                elif m.endswith('~'):
                    m = m[:-1]
                if m and m != mapped:
                    try:
                        result = api.get_particle_by_name(m)
                        if result:
                            return get_individual_particle(result)
                    except:
                        try:
                            result = api.get(m)
                            if result:
                                return get_individual_particle(result)
                        except:
                            pass
    
    # Try searching in particle names/descriptions
    try:
        all_particles = api.get_all()
        for particle in all_particles:
            if hasattr(particle, 'description') and particle.description:
                if search_term in particle.description.lower():
                    return get_individual_particle(particle)
    except:
        pass
    
    # If nothing found, try searching with wildcards
    try:
        results = api.get_particles_by_name(particle_id)
        if results:
            # Return the first match
            for result in results:
                return get_individual_particle(result)
    except:
        pass
    
    return None


async def search_particle(arguments: dict) -> list[types.TextContent]:
    """Search for particles by name or properties."""
    try:
        # Try to import PDG module
        try:
            import pdg
        except ImportError:
            # Try to find and add PDG path
            try:
                result = subprocess.run(['uvx', 'pip', 'show', 'pdg'], 
                                      capture_output=True, text=True, check=True)
                for line in result.stdout.split('\n'):
                    if line.startswith('Location:'):
                        pdg_path = line.split(':', 1)[1].strip()
                        if pdg_path not in sys.path:
                            sys.path.insert(0, pdg_path)
                        import pdg
                        break
                else:
                    return [types.TextContent(type="text", text="Error: pdg package not installed. Please install with: pip install pdg")]
            except Exception:
                return [types.TextContent(type="text", text="Error: pdg package not installed. Please install with: pip install pdg")]
        
        query = arguments.get("query", "")
        if isinstance(query, str):
            query = query.strip()
        if not query:
            return [types.TextContent(type="text", text="Error: query parameter is required")]
        # Handle generic quark queries as ambiguous to prompt for specificity
        try:
            ql = query.lower()
            generic_quark = {"quark", "a quark", "the quark", "any quark", "quarks"}
            if ql in generic_quark:
                guidance = (
                    "The term 'quark' is ambiguous. Please specify a quark type: "
                    "up (u), down (d), strange (s), charm (c), bottom (b), or top (t)."
                )
                return [types.TextContent(type="text", text=guidance)]
        except Exception:
            pass
        
        # Initialize PDG API
        api = pdg.connect()
        
        results = []
        found_particles = []
        
        # Search for particles
        try:
            # Try exact match first
            exact_match = _find_particle_by_alias(api, query)
            if exact_match:
                found_particles.append(exact_match)
            
            # Search for additional matches (no limit)
            search_results = api.get_particles_by_name(query)
            for particle in search_results:
                # Check if this particle is already in our results
                if not any(getattr(p, 'pdgid', None) == getattr(particle, 'pdgid', None) for p in found_particles):
                    found_particles.append(particle)

            # Also search by MCID when possible
            try:
                by_mcid = api.get_particle_by_mcid(query)
            except Exception:
                by_mcid = None
            # Handle possible return types: single particle or list/iterable
            def _append_particle_if_new(p):
                try:
                    pid_existing = getattr(p, 'pdgid', None)
                    if pid_existing is None:
                        return
                    if not any(getattr(x, 'pdgid', None) == pid_existing for x in found_particles):
                        found_particles.append(p)
                except Exception:
                    pass
            if by_mcid is not None:
                try:
                    # If it has get_particles(), get the individual(s)
                    if hasattr(by_mcid, 'get_particles'):
                        parts = by_mcid.get_particles()
                        for p in parts or []:
                            _append_particle_if_new(p)
                    # If it's iterable (list/tuple/generator), iterate
                    elif isinstance(by_mcid, (list, tuple)):
                        for p in by_mcid:
                            _append_particle_if_new(p)
                    else:
                        _append_particle_if_new(by_mcid)
                except Exception:
                    pass
        except Exception as e:
            # Downgrade to debug if we already have results; warn only when nothing found
            if found_particles:
                logger.debug(f"Search partial error (continuing): {e}")
            else:
                logger.warning(f"Search error: {e}")
        
        if not found_particles:
            return [types.TextContent(type="text", text=f"No particles found matching '{query}'")]
        
        # Format results
        result_text = f"Found {len(found_particles)} particle(s) matching '{query}':\n\n"
        
        for i, particle in enumerate(found_particles):
            try:
                # Basic particle information
                result_text += f"{i+1}. {particle.description or 'Unknown particle'}\n"
                # If query suggests anti-particle, flip sign-displayed quantities
                anti_view = _is_anti_query(query)
                if anti_view:
                    result_text += (
                        "   Note: For antiparticles, mass and spin are identical to the particle; "
                        "charges and additive quantum numbers appear with opposite sign.\n"
                    )
                result_text += f"   PDG ID: {particle.pdgid}\n"
                
                # Mass (MeV and GeV) with quark fallback to measurement text
                mass_line_emitted = False
                if hasattr(particle, 'mass') and particle.mass is not None:
                    mev, gev = _format_mass_mev_gev(particle.mass)
                    result_text += f"   Mass: {mev} MeV ({gev} GeV)\n"
                    mass_line_emitted = True
                if not mass_line_emitted and _is_quark(particle):
                    qm = _format_quark_mass_from_measurements(particle)
                    if qm:
                        result_text += f"   Mass: {qm}\n"
                
                # Spin (J)
                j_val = _get_first_attr(particle, ['quantum_J', 'J', 'spin'])
                if j_val is not None:
                    result_text += f"   Spin (J): {j_val}\n"
                
                # Charge (format as rational when possible)
                if hasattr(particle, 'charge') and particle.charge is not None:
                    charge_text = _format_charge(particle.charge)
                    if anti_view:
                        charge_text = _negate_numeric_like(charge_text)
                    result_text += f"   Charge: {charge_text}\n"
                
                # Color multiplicity (inferred)
                color_code = _infer_color_multiplicity(particle)
                result_text += f"   Color: {_format_color_label(color_code)}\n"
                
                # Quantum numbers
                quantum_info = []
                j_val = _get_first_attr(particle, ['quantum_J', 'J'])
                if j_val is not None:
                    quantum_info.append(f"J={j_val}")
                p_val = _get_first_attr(particle, ['quantum_P', 'P'])
                if p_val is not None:
                    quantum_info.append(f"P={p_val}")
                c_val = _get_first_attr(particle, ['quantum_C', 'C'])
                if c_val is not None:
                    quantum_info.append(f"C={c_val}")
                i_val = _get_first_attr(particle, ['quantum_I', 'I'])
                if i_val is not None:
                    quantum_info.append(f"I={i_val}")
                g_val = _get_first_attr(particle, ['quantum_G', 'G'])
                if g_val is not None:
                    quantum_info.append(f"G={g_val}")

                # Weak isospin (T3) and weak hypercharge (Y) if available
                t3_val = _get_first_attr(particle, ['weak_isospin', 't3', 'T3'])
                if t3_val is not None:
                    t3_text = str(t3_val)
                    if anti_view:
                        t3_text = _negate_numeric_like(t3_text)
                    quantum_info.append(f"T3={t3_text}")
                y_val = _get_first_attr(particle, ['weak_hypercharge', 'y', 'Y'])
                if y_val is not None:
                    y_text = str(y_val)
                    if anti_view:
                        y_text = _negate_numeric_like(y_text)
                    quantum_info.append(f"Y={y_text}")
                
                if quantum_info:
                    result_text += f"   Quantum numbers: {', '.join(quantum_info)}\n"
                
                # Lifetime/Width (format stable/NA cases nicely)
                if hasattr(particle, 'lifetime') and particle.lifetime is not None:
                    lifetime_val = particle.lifetime
                    try:
                        if lifetime_val == float('inf') or str(lifetime_val).lower() in {"inf", "+inf", "infinity"}:
                            result_text += "   Lifetime: stable (infinite)\n"
                        else:
                            result_text += f"   Lifetime: {lifetime_val}\n"
                    except Exception:
                        result_text += f"   Lifetime: {lifetime_val}\n"
                elif hasattr(particle, 'width') and particle.width is not None:
                    width_val = particle.width
                    try:
                        if float(width_val) == 0.0:
                            result_text += "   Width: 0 (stable)\n"
                        else:
                            result_text += f"   Width: {width_val}\n"
                    except Exception:
                        result_text += f"   Width: {width_val}\n"
                
                result_text += "\n"
                
            except Exception as e:
                result_text += f"   Error retrieving particle info: {e}\n\n"
        
        return [types.TextContent(type="text", text=result_text)]
        
    except ImportError:
        return [types.TextContent(type="text", text="Error: pdg package not installed. Please install with: pip install pdg")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error searching particles: {str(e)}")]


# Removed get_property tool per new API: now properties are shown in search results


async def list_decays(arguments: dict) -> list[types.TextContent]:
    """List decay modes for a specific particle."""
    try:
        # Try to import PDG module
        try:
            import pdg
        except ImportError:
            # Try to find and add PDG path
            try:
                result = subprocess.run(['uvx', 'pip', 'show', 'pdg'], 
                                      capture_output=True, text=True, check=True)
                for line in result.stdout.split('\n'):
                    if line.startswith('Location:'):
                        pdg_path = line.split(':', 1)[1].strip()
                        if pdg_path not in sys.path:
                            sys.path.insert(0, pdg_path)
                        import pdg
                        break
                else:
                    return [types.TextContent(type="text", text="Error: pdg package not installed. Please install with: pip install pdg")]
            except Exception:
                return [types.TextContent(type="text", text="Error: pdg package not installed. Please install with: pip install pdg")]
        
        particle_id = arguments.get("particle_id", "")
        if not particle_id:
            return [types.TextContent(type="text", text="Error: particle_id parameter is required")]
        
        # Initialize PDG API
        api = pdg.connect()
        
        # Find the particle (ensure we return a raw PDG particle object, not a local wrapper)
        def _resolve_raw_particle(api, pid: str):
            candidates = []
            try:
                obj = api.get_particle_by_name(pid)
                if obj is not None:
                    candidates.append(obj)
            except Exception:
                pass
            try:
                obj = api.get(pid)
                if obj is not None:
                    candidates.append(obj)
            except Exception:
                pass
            # Also try MCID lookups
            try:
                by_mcid = api.get_particle_by_mcid(pid)
            except Exception:
                by_mcid = None
            if by_mcid is not None:
                try:
                    if hasattr(by_mcid, 'get_particles'):
                        parts = by_mcid.get_particles()
                        if parts:
                            candidates.extend(parts)
                    elif isinstance(by_mcid, (list, tuple)):
                        candidates.extend(list(by_mcid))
                    else:
                        candidates.append(by_mcid)
                except Exception:
                    pass
            try:
                lst = api.get_particles_by_name(pid)
                if lst:
                    candidates.extend(lst)
            except Exception:
                pass
            # Normalize to a concrete PdgParticle when possible
            for cand in candidates:
                try:
                    # Some returns have get_particles() to access individuals
                    if hasattr(cand, 'get_particles'):
                        parts = cand.get_particles()
                        if parts:
                            return parts[0]
                except Exception:
                    pass
                # Otherwise assume it's already a particle
                return cand
            return None

        # Prefer resolving via our alias helper first to disambiguate (e.g., 'muon' -> 'mu-')
        alias_particle = _find_particle_by_alias(api, particle_id)
        particle = None
        if alias_particle is not None:
            candidate_key = getattr(alias_particle, 'description', None) or getattr(alias_particle, 'pdgid', None)
            if candidate_key:
                particle = _resolve_raw_particle(api, candidate_key)
        if particle is None:
            particle = _resolve_raw_particle(api, particle_id)
        if particle is None and alias_particle is not None and hasattr(alias_particle, 'pdgid'):
            particle = _resolve_raw_particle(api, getattr(alias_particle, 'pdgid'))
        if particle is None:
            return [types.TextContent(type="text", text=f"Particle '{particle_id}' not found")]
        
        # Get decay modes
        try:
            decay_entries = []

            # Prefer exclusive branching fractions when available
            for method_name in ['exclusive_branching_fractions', 'branching_fractions', 'inclusive_branching_fractions']:
                method = getattr(particle, method_name, None)
                if callable(method):
                    try:
                        entries = method()
                        # Some implementations may return iterators/generators
                        decay_entries = list(entries) if entries is not None else []
                    except Exception:
                        decay_entries = []
                    if decay_entries:
                        break

            if not decay_entries:
                # Legacy fallbacks
                if hasattr(particle, 'decay_modes') and particle.decay_modes:
                    decay_entries = list(particle.decay_modes)
                elif hasattr(particle, 'decays') and particle.decays:
                    decay_entries = list(particle.decays)
                elif hasattr(particle, 'get_decay_modes'):
                    try:
                        dm = particle.get_decay_modes()
                        decay_entries = list(dm) if dm else []
                    except Exception:
                        pass

            if not decay_entries:
                return [types.TextContent(type="text", text=f"No decay modes found for particle '{particle.description or particle_id}'. This particle may be stable or decay information may not be available.")]

            # Format decay modes using PDG's original description and BR text without reconstruction
            result_text = f"Decay modes for particle '{particle.description or particle_id}':\n\n"
            # If user requested an antiparticle, add guidance note about charge-flipped decays
            try:
                if _is_anti_query(particle_id):
                    result_text += (
                        "Note: If the particle decays, its antiparticle decays through the same processes but with charges flipped.\n\n"
                    )
            except Exception:
                pass
            count = 0
            for decay in decay_entries:
                count += 1
                try:
                    desc = getattr(decay, 'description', None)
                    br_text = getattr(decay, 'display_value_text', None)
                    if desc and br_text:
                        result_text += f"{count}. {desc} (BR: {br_text})\n"
                    elif desc:
                        result_text += f"{count}. {desc}\n"
                    else:
                        result_text += f"{count}. {str(decay)}\n"
                except Exception as e:
                    result_text += f"{count}. [Decay mode info unavailable: {e}]\n"

            return [types.TextContent(type="text", text=result_text)]

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error retrieving decay modes: {str(e)}")]
        
        return [types.TextContent(type="text", text=result_text)]
        
    except ImportError:
        return [types.TextContent(type="text", text="Error: pdg package not installed. Please install with: pip install pdg")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error listing decays: {str(e)}")]


async def main():
    """Main entry point for the server."""
    # Import here to avoid issues if mcp is not installed
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="particlephysics-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(
                        tools_changed=False,
                        resources_changed=False,
                        prompts_changed=False
                    ),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

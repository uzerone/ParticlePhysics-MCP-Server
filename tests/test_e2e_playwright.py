import asyncio
import pytest

from particlephysics_mcp_server.server import search_particle, list_decays


def run(coro):
    return asyncio.run(coro)


def extract_text(contents):
    # contents is a list of mcp.types.TextContent; get combined text
    parts = []
    for c in contents:
        text = getattr(c, "text", "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def test_up_quark_mass():
    result = run(search_particle({"query": "up quark"}))
    text = extract_text(result)
    assert "Mass:" in text
    assert "u" in text or "up" in text


def test_tau_decays():
    result = run(list_decays({"particle_id": "tau"}))
    text = extract_text(result)
    assert "Decay modes for particle" in text
    assert "BR:" in text


# New test: anti up quark should resolve and show flipped charge
def test_anti_up_quark_alias_and_charge():
    result = run(search_particle({"query": "anti up quark"}))
    text = extract_text(result)
    assert "Mass:" in text
    assert "Charge: -2/3" in text


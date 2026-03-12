import asyncio

from particlephysics_mcp_server.server import search_particle, list_decays


def run(coro):
    return asyncio.run(coro)


def extract_text(contents):
    parts = []
    for c in contents:
        text = getattr(c, "text", "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def test_search_particle_electron():
    result = run(search_particle({"query": "electron"}))
    text = extract_text(result)
    assert "Found" in text
    assert "Mass:" in text
    assert "Charge:" in text


def test_search_particle_higgs():
    result = run(search_particle({"query": "Higgs"}))
    text = extract_text(result)
    assert "Found" in text
    assert "Spin" in text


def test_list_decays_muon():
    result = run(list_decays({"particle_id": "muon"}))
    text = extract_text(result)
    assert "Decay modes for particle" in text
    assert "BR:" in text


def test_list_decays_pion():
    result = run(list_decays({"particle_id": "pion"}))
    text = extract_text(result)
    assert "Decay modes for particle" in text
    assert "pi+" in text or "pi-" in text

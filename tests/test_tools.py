import asyncio
import json
import re

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


_JSON_BLOCK_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def extract_json(text: str) -> dict:
    m = _JSON_BLOCK_RE.search(text)
    assert m, f"no JSON block in:\n{text[:400]}"
    return json.loads(m.group(1))


# ---------- happy path ----------

def test_search_particle_electron():
    result = run(search_particle({"query": "electron"}))
    text = extract_text(result)
    assert "Found" in text
    assert "Mass:" in text
    assert "Charge:" in text
    assert "PDG Review ID:" in text


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


# ---------- natural-language queries ----------

def test_search_natural_language_charge():
    cases = [
        ("muon plus", "mu+", -13),
        ("muon minus", "mu-", 13),
        ("positive tau", "tau+", -15),
        ("pion zero", "pi0", 111),
        ("kaon minus", "K-", -321),
        ("positive electron", "e+", -11),
    ]
    for query, expected_name, expected_mcid in cases:
        payload = extract_json(extract_text(run(search_particle({"query": query}))))
        first = payload["particles"][0]
        assert first["name"] == expected_name, f"{query} -> {first['name']!r}"
        assert first["mcid"] == expected_mcid, f"{query} mcid={first['mcid']}"


def test_search_anti_particle_via_prefix():
    cases = [
        ("antimuon", -13),
        ("anti muon", -13),
        ("antiproton", -2212),
        ("antineutron", -2112),
        ("anti up quark", -2),
        ("anti charm quark", -4),
        ("antielectron", -11),
    ]
    for query, expected_mcid in cases:
        payload = extract_json(extract_text(run(search_particle({"query": query}))))
        assert payload["particles"][0]["mcid"] == expected_mcid, query


def test_search_anti_particle_via_suffix():
    cases = [
        ("ubar", -2),
        ("u bar", -2),
        ("u_bar", -2),
        ("u~", -2),
        ("p bar", -2212),
    ]
    for query, expected_mcid in cases:
        payload = extract_json(extract_text(run(search_particle({"query": query}))))
        assert payload["particles"][0]["mcid"] == expected_mcid, query


def test_search_anti_baryon_charge_is_negated():
    """ubar's charge IS -2/3 in PDG — verify we surface it (no double-flip)."""
    payload = extract_json(extract_text(run(search_particle({"query": "anti up quark"}))))
    first = payload["particles"][0]
    assert first["mcid"] == -2
    assert first["charge"]["fraction"] == "-2/3"


def test_search_self_conjugate_anti_returns_self():
    payload = extract_json(extract_text(run(search_particle({"query": "anti photon"}))))
    assert payload["particles"][0]["name"] == "gamma"


# ---------- error / edge paths ----------

def test_search_not_found():
    result = run(search_particle({"query": "definitely-not-a-particle-xyz123"}))
    text = extract_text(result)
    assert "No particles found" in text


def test_search_empty_query():
    result = run(search_particle({"query": ""}))
    text = extract_text(result)
    assert "Error" in text and "required" in text


def test_search_generic_quark_is_ambiguous():
    for q in ["quark", "anti quark", "antiquarks"]:
        text = extract_text(run(search_particle({"query": q})))
        assert "ambiguous" in text.lower() and "specify" in text.lower(), q


def test_list_decays_not_found():
    result = run(list_decays({"particle_id": "definitely-not-a-particle-xyz123"}))
    text = extract_text(result)
    assert "not found" in text.lower()


# ---------- MCID lookup ----------

def test_search_by_numeric_mcid_string():
    """search_particle accepts a numeric string MCID."""
    payload = extract_json(extract_text(run(search_particle({"query": "11"}))))
    names = {p["name"] for p in payload["particles"]}
    assert "e-" in names


# ---------- JSON output structure ----------

def test_search_returns_json_block_with_expected_keys():
    payload = extract_json(extract_text(run(search_particle({"query": "Higgs"}))))
    assert payload["query"] == "Higgs"
    assert payload["count"] >= 1
    p = payload["particles"][0]
    for key in ("name", "description", "mcid", "pdg_review_id", "spin", "color"):
        assert key in p, f"missing {key}"


def test_list_decays_returns_json_block_with_source():
    payload = extract_json(extract_text(run(list_decays({"particle_id": "tau"}))))
    assert "particle" in payload
    assert payload["count"] >= 1
    assert payload["source"] in {
        "exclusive_branching_fractions",
        "branching_fractions",
        "inclusive_branching_fractions",
        "decay_modes",
        "decays",
        "get_decay_modes",
    }
    first = payload["decays"][0]
    assert "description" in first


# ---------- anti-particle decays ----------

def test_list_decays_antimuon_returns_decays():
    """Anti-muon shares decay structure with muon (charge-flipped)."""
    text = extract_text(run(list_decays({"particle_id": "antimuon"})))
    assert "Decay modes" in text
    payload = extract_json(text)
    assert payload["particle"]["mcid"] == -13
    assert payload["count"] >= 1


# ---------- mass units sanity ----------

def test_mass_units_are_consistent():
    """PDG stores masses in GeV; the JSON payload exposes both with the right magnitudes."""
    cases = [
        ("electron", 0.511, 1.0),     # ~0.511 MeV
        ("muon", 105.7, 1.0),
        ("proton", 938.27, 1.0),
        ("Higgs", 125_000.0, 1000.0),  # ~125 GeV = 125000 MeV
    ]
    for query, expected_mev, tol in cases:
        payload = extract_json(extract_text(run(search_particle({"query": query}))))
        first = payload["particles"][0]
        m = first["mass"]
        assert "mev" in m and "gev" in m, query
        assert abs(m["gev"] * 1000 - m["mev"]) < 1e-6, f"{query}: gev*1000 != mev"
        assert abs(m["mev"] - expected_mev) < tol, f"{query}: expected ~{expected_mev} MeV, got {m['mev']}"

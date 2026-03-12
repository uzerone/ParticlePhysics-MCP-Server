import math

from particlephysics_mcp_server.server import (
    _format_charge,
    _format_lifetime,
    _format_width,
    _is_anti_query,
    _negate_numeric_like,
    _to_float,
)


def test_format_charge_rational():
    assert _format_charge(1 / 3) == "1/3"
    assert _format_charge(-2 / 3) == "-2/3"
    assert _format_charge(0.0) == "0"


def test_format_charge_string():
    assert _format_charge("0.5") == "1/2"


def test_is_anti_query():
    assert _is_anti_query("anti up quark") is True
    assert _is_anti_query("e~") is True
    assert _is_anti_query("proton") is False


def test_negate_numeric_like():
    assert _negate_numeric_like("2/3") == "-2/3"
    assert _negate_numeric_like("-2/3") == "2/3"
    assert _negate_numeric_like("+1") == "-1"


def test_format_lifetime_width():
    assert _format_lifetime(float("inf")) == "stable (infinite)"
    assert _format_width(0.0) == "0 (stable)"


def test_to_float_parses_units():
    assert _to_float("1.27 GeV") == 1.27
    assert _to_float("-3.5e-2 MeV") == -0.035
    assert _to_float("abc") is None

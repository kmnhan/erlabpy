import numpy as np
import pytest

from erlab.analysis import xps


def _sigma_at(curve, hv: float) -> float:
    indices = np.flatnonzero(np.isclose(curve.hv.values, hv))
    assert len(indices) == 1
    return float(curve.values[indices[0]])


def _sigma_values_at(curve, hv: float) -> list[float]:
    indices = np.flatnonzero(np.isclose(curve.hv.values, hv))
    return [float(curve.values[index]) for index in indices]


def test_get_cross_section_reuses_loaded_reference(monkeypatch) -> None:
    original_load = xps.np.load
    load_calls = 0

    def counted_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    xps._xsection_data.cache_clear()
    monkeypatch.setattr(xps.np, "load", counted_load)

    try:
        fe = xps.get_cross_section("Fe")
        li = xps.get_cross_section("Li")
        total = xps.get_total_cross_section("Li")
    finally:
        xps._xsection_data.cache_clear()

    assert load_calls == 1
    assert {"2p", "3d"} <= fe.keys()
    assert {"1s", "2s"} <= li.keys()
    assert _sigma_at(total, 10.2) == pytest.approx(1.235, abs=1e-4)


def test_get_cross_section_uses_elettra_reference_points() -> None:
    li = xps.get_cross_section("Li")
    fe = xps.get_cross_section("Fe")
    total_li = xps.get_total_cross_section("Li")

    assert _sigma_at(li["1s"], 60.0) == pytest.approx(3.248, abs=1e-4)
    assert _sigma_at(li["2s"], 10.2) == pytest.approx(1.235, abs=1e-4)
    assert _sigma_at(fe["2p"], 725.0) == pytest.approx(0.9731, abs=1e-4)
    assert _sigma_values_at(total_li, 60.0) == pytest.approx([0.06073, 3.309], abs=1e-4)


def test_get_cross_section_removes_n_electrons_metadata() -> None:
    li = xps.get_cross_section("Li")
    assert "n_electrons" not in li["1s"].attrs
    assert not any(key.endswith("__n") for key in xps._xsection_data())


def test_get_edge_preserves_legacy_mapping() -> None:
    levels = xps.get_edge("Li")

    assert isinstance(levels["1s"], float)
    assert isinstance(levels["2s"], float)
    assert levels["1s"] > levels["2s"] > 0.0


def test_get_edge_can_calculate_harmonic_kinetic_energies() -> None:
    base_levels = xps.get_edge("Li")

    levels = xps.get_edge(
        "Li",
        photon_energy=60.0,
        work_function=4.5,
        max_harmonic=3,
    )

    level = levels["2s"]
    assert isinstance(level, xps.CoreLevelEdge)
    assert level.edge == pytest.approx(base_levels["2s"])
    assert level.kinetic_energies == pytest.approx(
        {
            1: 60.0 - base_levels["2s"] - 4.5,
            2: 120.0 - base_levels["2s"] - 4.5,
            3: 180.0 - base_levels["2s"] - 4.5,
        }
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"photon_energy": None, "work_function": 1.0}, "work_function requires"),
        ({"photon_energy": None, "max_harmonic": 2}, "max_harmonic requires"),
        ({"photon_energy": 0.0}, "photon_energy must be positive"),
        ({"photon_energy": 10.0, "work_function": -1.0}, "work_function must be"),
        ({"photon_energy": 10.0, "max_harmonic": 0}, "max_harmonic must be"),
    ],
)
def test_get_edge_validates_harmonic_inputs(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        xps.get_edge("Li", **kwargs)


def test_get_cross_section_raises_for_unavailable_element() -> None:
    with pytest.raises(KeyError, match="Rf"):
        xps.get_cross_section("Rf")
    with pytest.raises(KeyError, match="Rf"):
        xps.get_total_cross_section("Rf")

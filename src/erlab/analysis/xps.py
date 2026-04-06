from __future__ import annotations

__all__ = [
    "CoreLevelEdge",
    "get_cross_section",
    "get_edge",
    "get_total_cross_section",
]

import functools
import importlib.resources
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
import xraydb

import erlab

if TYPE_CHECKING:
    from collections.abc import Mapping


IUPAC_TO_XPS: dict[str, str] = {
    "K": "1s",
    "L1": "2s",
    "L2": "2p1/2",
    "L3": "2p3/2",
    "M1": "3s",
    "M2": "3p1/2",
    "M3": "3p3/2",
    "M4": "3d3/2",
    "M5": "3d5/2",
    "N1": "4s",
    "N2": "4p1/2",
    "N3": "4p3/2",
    "N4": "4d3/2",
    "N5": "4d5/2",
    "N6": "4f5/2",
    "N7": "4f7/2",
    "O1": "5s",
    "O2": "5p1/2",
    "O3": "5p3/2",
    "O4": "5d3/2",
    "O5": "5d5/2",
    "O6": "5f5/2",
    "O7": "5f7/2",
    "O8": "5g7/2",
    "O9": "5g9/2",
    "P1": "6s",
    "P2": "6p1/2",
    "P3": "6p3/2",
    "P4": "6d3/2",
    "P5": "6d5/2",
    "P6": "6f5/2",
    "P7": "6f7/2",
    "P8": "6g7/2",
    "P9": "6g9/2",
    "P10": "6h9/2",
    "P11": "6h11/2",
    "Q1": "7s",
    "Q2": "7p1/2",
    "Q3": "7p3/2",
    "Q4": "7d3/2",
    "Q5": "7d5/2",
    "Q6": "7f5/2",
    "Q7": "7f7/2",
    "Q8": "7g7/2",
    "Q9": "7g9/2",
    "Q10": "7h9/2",
    "Q11": "7h11/2",
    "Q12": "7i11/2",
    "Q13": "7i13/2",
}


def _validate_kinetic_energy_inputs(
    photon_energy: float,
    *,
    work_function: float,
    max_harmonic: int,
) -> None:
    if photon_energy <= 0.0:
        raise ValueError("photon_energy must be positive")
    if work_function < 0.0:
        raise ValueError("work_function must be non-negative")
    if max_harmonic < 1:
        raise ValueError("max_harmonic must be at least 1")


@dataclass(frozen=True)
class CoreLevelEdge:
    """Absorption edge and harmonic kinetic energies for one core level.

    Attributes
    ----------
    edge
        Absorption edge in eV.
    kinetic_energies
        Mapping from harmonic order to photoelectron kinetic energy in eV.

    Examples
    --------
    >>> import erlab.analysis as era
    >>> au_4f = era.xps.CoreLevelEdge.from_edge(
    ...     84.0, photon_energy=1486.6, workfunction=4.5, max_harmonic=3
    ... )
    >>> au_4f.edge
    84.0
    >>> au_4f.kinetic_energies
    {1: 1398.1, 2: 2884.7}
    """

    edge: float
    kinetic_energies: dict[int, float]

    @classmethod
    def from_edge(
        cls,
        edge: float,
        *,
        photon_energy: float,
        workfunction: float = 0.0,
        max_harmonic: int = 1,
    ) -> CoreLevelEdge:
        """Build a :class:`CoreLevelEdge` from an absorption edge value.

        Parameters
        ----------
        edge
            Absorption edge in eV.
        photon_energy
            Fundamental photon energy in eV.
        workfunction
            Work function in eV subtracted from each harmonic kinetic energy.
        max_harmonic
            Highest harmonic order to include. The returned mapping contains orders
            ``1`` through ``max_harmonic``.

        Returns
        -------
        CoreLevelEdge
            A record containing the absorption edge and the per-harmonic kinetic
            energies.

        Examples
        --------
        >>> import erlab.analysis as era
        >>> edge = era.xps.CoreLevelEdge.from_edge(
        ...     706.8, photon_energy=1486.6, workfunction=4.5, max_harmonic=2
        ... )
        >>> edge.kinetic_energies[1]
        775.3
        """
        _validate_kinetic_energy_inputs(
            photon_energy,
            work_function=workfunction,
            max_harmonic=max_harmonic,
        )
        edge_value = float(edge)
        return cls(
            edge=edge_value,
            kinetic_energies={
                order: (float(photon_energy) * order) - edge_value - float(workfunction)
                for order in range(1, max_harmonic + 1)
            },
        )


@functools.cache
def _xsection_data() -> dict[str, np.ndarray]:
    with (
        importlib.resources.as_file(
            importlib.resources.files(erlab.analysis).joinpath(
                "_data/yeh_lindau_1985_pics.npz"
            )
        ) as f,
        np.load(f, allow_pickle=False) as archive,
    ):
        return {key: archive[key] for key in archive.files}


def _available_subshells(
    element: str | int,
    data: Mapping[str, np.ndarray] | None = None,
) -> tuple[str, ...]:
    symbol = xraydb.atomic_symbol(element)
    key = f"{symbol}__subshells"
    z = _xsection_data() if data is None else data
    try:
        arr = z[key]
    except KeyError as e:
        raise KeyError(f"No bundled cross-section data for {symbol}") from e
    return tuple(str(x) for x in arr.tolist())


def _cross_section_array(
    symbol: str, label: str, data: Mapping[str, np.ndarray]
) -> xr.DataArray:
    base = f"{symbol}__{label}"
    try:
        hv = data[f"{base}__hv"].astype(np.float64, copy=False)
        sigma = data[f"{base}__sigma"].astype(np.float64, copy=False)
    except KeyError as e:
        raise KeyError(f"No bundled cross-section data for {symbol} {label}") from e

    return xr.DataArray(
        data=sigma,
        coords={"hv": hv},
        dims=["hv"],
        name=f"{symbol}_{label}",
    )


def get_cross_section(element: str | int) -> dict[str, xr.DataArray]:
    """Get the photoionization cross section curves for a given element.

    Data are based on the `Elettra WebCrossSections
    <https://vuo.elettra.eu/services/elements/WebElements.html>`_ service, which is
    based on :cite:t:`yeh1985photoionization`.

    Parameters
    ----------
    element
        The element symbol, name, or atomic number. For example, ``"Fe"``, ``"Iron"``,
        or ``26``.

    Returns
    -------
    dict
        A dictionary mapping subshell orbital labels to 1D DataArrays with coordinate
        ``hv`` (photon energy in eV).

    Examples
    --------
    >>> import numpy as np
    >>> import erlab.analysis as era
    >>> curves = era.xps.get_cross_section("Cu")
    >>> list(curves.keys())
    ['2s', '2p', '3s', '3p', '3d', '4s']
    """
    symbol = xraydb.atomic_symbol(element)
    z = _xsection_data()

    sigma_curves: dict[str, xr.DataArray] = {}

    for subshell in _available_subshells(symbol, data=z):
        sigma_curves[subshell] = _cross_section_array(symbol, subshell, z)

    return sigma_curves


def get_total_cross_section(element: str | int) -> xr.DataArray:
    """Get the total photoionization cross section curve for a given element.

    Data are based on the `Elettra WebCrossSections
    <https://vuo.elettra.eu/services/elements/WebElements.html>`_ service, which is
    based on :cite:t:`yeh1985photoionization`.

    Parameters
    ----------
    element
        The element symbol, name, or atomic number. For example, ``"Fe"``, ``"Iron"``,
        or ``26``.

    Returns
    -------
    DataArray
        A 1D DataArray with coordinate ``hv`` (photon energy in eV).
    """
    symbol = xraydb.atomic_symbol(element)
    return _cross_section_array(symbol, "total", _xsection_data())


def _edge_map(element: str | int) -> dict[str, float]:
    xdb = xraydb.get_xraydb()
    elem = xdb.symbol(element)
    ltab = xdb.tables["xray_levels"]
    out: dict[str, float] = {}
    for r in xdb.query(ltab).filter(ltab.c.element == elem).all():
        out[IUPAC_TO_XPS[str(r.iupac_symbol)]] = r.absorption_edge
    return out


@overload
def get_edge(
    element: str | int,
    *,
    photon_energy: None = None,
    work_function: float = 0.0,
    max_harmonic: int = 1,
) -> dict[str, float]: ...


@overload
def get_edge(
    element: str | int,
    *,
    photon_energy: float,
    work_function: float = 0.0,
    max_harmonic: int = 1,
) -> dict[str, CoreLevelEdge]: ...


def get_edge(
    element: str | int,
    *,
    photon_energy: float | None = None,
    work_function: float = 0.0,
    max_harmonic: int = 1,
) -> dict[str, float] | dict[str, CoreLevelEdge]:
    """Get the x-ray absorption edges for a given element.

    The values are taken from :mod:`xraydb` :cite:p:`newville2023xraydb`, using the
    underlying X-ray level compilation described by :cite:t:`elam2002xraydb`. The values
    for core-level edges can be treated as the binding energies of the corresponding
    electrons. When ``photon_energy`` is provided, a conversion to kinetic energies is
    performed for each edge and its harmonics up to ``max_harmonic``, using the provided
    work function.

    Parameters
    ----------
    element
        The element symbol, name, or atomic number. For example, ``"Fe"``, ``"Iron"``,
        or ``26``.
    photon_energy
        Optional photon energy in eV. When provided, the returned mapping contains
        :class:`CoreLevelEdge` records with kinetic energies computed from each edge for
        the fundamental and higher harmonics up to ``max_harmonic``.
    work_function
        Work function in eV used for the kinetic-energy conversion. Only valid when
        ``photon_energy`` is provided.
    max_harmonic
        Highest harmonic order to include in the kinetic-energy calculation. Harmonics
        are computed as integer multiples of ``photon_energy`` from ``1`` through
        ``max_harmonic``. Only valid when ``photon_energy`` is provided.

    Returns
    -------
    dict
        A dictionary mapping subshell labels to absorption edges in eV when
        ``photon_energy`` is omitted. When ``photon_energy`` is provided, the mapping
        values are :class:`CoreLevelEdge` objects containing the absorption edge and
        per-harmonic kinetic energies.

    Notes
    -----
    The returned values come from `xraydb`. In many XPS contexts these numbers are close
    to, and interpreted similarly to, core-level binding energies.

    Examples
    --------
    >>> import erlab.analysis as era
    >>> edges = era.xps.get_edge("Fe")
    >>> edges["2p3/2"]
    706.8
    >>> edges["3p3/2"]
    52.7

    >>> edges_ke = era.xps.get_edge(
    ...     "Fe", photon_energy=1486.6, work_function=4.5, max_harmonic=2
    ... )
    >>> fe_2p = edges_ke["2p3/2"]
    >>> fe_2p.edge
    706.8
    >>> fe_2p.kinetic_energies
    {1: 775.3, 2: 2261.9}

    """
    out = _edge_map(element)
    if photon_energy is None:
        if work_function != 0.0:
            raise ValueError(
                "work_function requires photon_energy to calculate kinetic energies"
            )
        if max_harmonic != 1:
            raise ValueError(
                "max_harmonic requires photon_energy to calculate kinetic energies"
            )
        return out

    _validate_kinetic_energy_inputs(
        photon_energy,
        work_function=work_function,
        max_harmonic=max_harmonic,
    )
    return {
        label: CoreLevelEdge.from_edge(
            edge,
            photon_energy=photon_energy,
            workfunction=work_function,
            max_harmonic=max_harmonic,
        )
        for label, edge in out.items()
    }

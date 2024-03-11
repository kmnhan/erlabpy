"""
Physical constants and unit conversion.

.. currentmodule:: erlab.constants

"""

import numpy as np

e: float = 1.602176634e-19  #: Elementary charge :math:`e` (C)
c: float = 299792458.0  #: Speed of light :math:`c` (m/s)
m_e: float = 9.1093837015e-31  #: Electron mass :math:`m_e` (kg) (±0.000 000 0028e-31)

mcsq: float = (
    8.1871057769e-14  #: Electron rest energy :math:`m_e c^2` (J) (±0.000 000 0025e-14)
)
mcsq_eV: float = 510998.95000  #: Electron rest energy :math:`m_e c^2` (eV) (±0.000 15)

h: float = 6.62607015e-34  #: Planck constant :math:`h` (J·s)
hc: float = 1.9864458571489286e-25  #: :math:`hc`  (J·m)
hbar: float = 1.0545718176461565e-34  #: Dirac constant :math:`\hbar` (J·s)
hbarc: float = 3.1615267734966903e-26  #: :math:`\hbar c`  (J·m)
hbarsq: float = 1.1121217185735183e-68  #: :math:`\hbar^2` (J²·s²)

h_eV: float = 4.135667696923859e-15  #: Planck constant :math:`h` (eV·s)
hc_eV: float = 1.2398419843320028e-6  #: :math:`hc`  (eV·m)
hbar_eV: float = 6.582119569509067e-16  #: Dirac constant :math:`\hbar` (eV·s)
hbarsq_eV: float = 4.332429802731423e-31  #: :math:`\hbar^2` (eV²·s²)

rel_eV_nm: float = 1239.8419843320028  #: :math:`hc`  (eV·nm)
rel_kconv: float = (
    0.512316721967493  #: :math:`\frac{\sqrt{2 m_e}}{\hbar}`, Used in momentum conversion
)
rel_kzconv: float = (
    3.8099821161548606  #: :math:`\frac{\hbar^2}{2 m_e}`, Used in momentum conversion
)

kb: float = 1.380649e-23  #: Boltzmann constant :math:`k_B` (J/K)
kb_eV: float = 8.617333262145179e-5  #: Boltzmann constant :math:`k_B` (eV/K)


def conv_eV_nm(value: float) -> float:
    """Convert between energy and wavelength."""
    return np.divide(rel_eV_nm, np.float64(value))


def conv_watt_photons(value: float, wavelength: float) -> float:
    """Convert Watts to photons per second.

    Parameters
    ----------
    value
        Power in Watts.
    wavelength
        Wavelength in nanometers.

    Returns
    -------
    power : float
        Power in photons per second.

    """
    return value / (hc * 1e9 / wavelength)

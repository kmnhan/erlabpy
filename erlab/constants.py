"""Physical constants.

"""

E: float = 1.602176634e-19  #: Elementary charge :math:`e` (C)
C: float = 299792458.0  #: Speed of light :math:`c` (m/s)
ME: float = 9.1093837015e-31  #: Electron mass :math:`m_e` (kg) (±0.0000000028e-31)

H: float = 6.62607015e-34  #: Planck constant :math:`h` (J·s)
H_EV: float = 4.135667696923859e-15  #: Planck constant :math:`h` (eV·s)

HBAR: float = 1.0545718176461565e-34  #: Dirac constant :math:`\hbar` (J·s)
HBAR_EV: float = 6.582119569509067e-16  #: Dirac constant :math:`\hbar` (eV·s)

HC: float = 1.9864458571489286e-25  #: :math:`hc`  (J·m)
HBARSQ: float = 1.1121217185735183e-68  #: :math:`\hbar^2` (J²·s²)
HBARSQ_EV: float = 4.332429802731423e-31  #: :math:`\hbar^2` (eV²·s²)
HBARC: float = 3.1615267734966903e-26  #: :math:`\hbar c`  (J·m)

K_COEFF: float = 0.512316721967493  #: :math:`\frac{\sqrt{2 m_e}}{\hbar}`, Used in momentum conversion
KZ_COEFF: float = (
    3.8099821161548606  #: :math:`\frac{\hbar^2}{2 m_e}`, Used in momentum conversion
)

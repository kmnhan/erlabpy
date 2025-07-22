"""User customization options for interactive tools, mainly ImageTool.

When implementing new options or modifying existing ones, ensure that the attribute
`DEFAULT_OPTIONS` and the functions `make_parameter` and `parameter_to_dict` in
`defaults.py` are updated accordingly.
"""

import lazy_loader as _lazy
import pyqtgraph.parametertree

from erlab.interactive._options.parameters import (
    ColorListParameter,
    _CustomColorMapParameter,
)

pyqtgraph.parametertree.registerParameterType(
    "colorlist", ColorListParameter, override=True
)

pyqtgraph.parametertree.registerParameterType(
    "erlabpy_colormap", _CustomColorMapParameter, override=True
)

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

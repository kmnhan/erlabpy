"""User customization options for interactive tools, mainly ImageTool.

To implement new options or modify existing ones, modify the pydantic schema in
`schema.py`. The options UI is automatically generated from the schema. Metadata for the
UI like grouping, titles, descriptions, and some types that need special handling can be
defined in the schema using the `json_schema_extra` field attribute.
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

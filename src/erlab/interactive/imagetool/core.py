"""Compatibility shim for ImageTool internals.

The canonical implementation now lives in
:mod:`erlab.interactive.imagetool.viewer` and
:mod:`erlab.interactive.imagetool.plot_items`.

This module re-exports previous ``core`` symbols to preserve backwards
compatibility.

.. deprecated:: 3.21.0
   Import from :mod:`erlab.interactive.imagetool.viewer` and
   :mod:`erlab.interactive.imagetool.plot_items` instead.
"""

from __future__ import annotations

import warnings

from erlab.interactive.imagetool._viewer_dialogs import (
    _AssociatedCoordsDialog,
    _CursorColorCoordDialog,
)
from erlab.interactive.imagetool.plot_items import (
    ItoolColorBar,
    ItoolColorBarItem,
    ItoolCursorLine,
    ItoolCursorSpan,
    ItoolDisplayObject,
    ItoolGraphicsLayoutWidget,
    ItoolImageItem,
    ItoolPlotDataItem,
    ItoolPlotItem,
    ItoolPolyLineROI,
    _OptionKeyMenuFilter,
    _pad_1d_plot,
    _PolyROIEditDialog,
)
from erlab.interactive.imagetool.viewer import (
    ColorMapState,
    ImageSlicerArea,
    ImageSlicerState,
    PlotItemState,
    SlicerLinkProxy,
    _link_splitters,
    _make_cursor_colors,
    _parse_dataset,
    _parse_input,
    _processed_ndim,
    _supported_shape,
    _sync_splitters,
    link_slicer,
    record_history,
    suppress_history,
    suppressnanwarning,
)

warnings.warn(
    "`erlab.interactive.imagetool.core` is deprecated and will be removed in a "
    "future release. Import from `erlab.interactive.imagetool.viewer` (and "
    "`erlab.interactive.imagetool.plot_items` for plot classes) instead.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "ColorMapState",
    "ImageSlicerArea",
    "ImageSlicerState",
    "ItoolColorBar",
    "ItoolColorBarItem",
    "ItoolCursorLine",
    "ItoolCursorSpan",
    "ItoolDisplayObject",
    "ItoolGraphicsLayoutWidget",
    "ItoolImageItem",
    "ItoolPlotDataItem",
    "ItoolPlotItem",
    "ItoolPolyLineROI",
    "PlotItemState",
    "SlicerLinkProxy",
    "_AssociatedCoordsDialog",
    "_CursorColorCoordDialog",
    "_OptionKeyMenuFilter",
    "_PolyROIEditDialog",
    "_link_splitters",
    "_make_cursor_colors",
    "_pad_1d_plot",
    "_parse_dataset",
    "_parse_input",
    "_processed_ndim",
    "_supported_shape",
    "_sync_splitters",
    "link_slicer",
    "record_history",
    "suppress_history",
    "suppressnanwarning",
]

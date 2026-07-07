"""Data-source inspection widgets for Figure Composer."""

from __future__ import annotations

import logging
import typing

from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._figurecomposer._sources import (
    _public_source_data,
    _source_display_label,
)

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer._state import FigureSourceState


logger = logging.getLogger(__name__)


def _dims_text(dims: Sequence[typing.Hashable]) -> str:
    return ", ".join(str(dim) for dim in dims) or "scalar"


def _shape_text(shape: Sequence[int]) -> str:
    return " × ".join(str(size) for size in shape) or "scalar"


def _sizes_text(data: xr.DataArray) -> str:
    return ", ".join(f"{dim}: {size}" for dim, size in data.sizes.items()) or "scalar"


def source_metadata_tooltip(
    source: FigureSourceState | None,
    name: str,
    data: xr.DataArray | None,
) -> str:
    """Return a compact source tooltip with public DataArray metadata."""
    lines = [_source_display_label(source, name, disambiguate=False)]
    if data is None:
        lines.append("DataArray: unavailable")
        return "\n".join(lines)
    public = _public_source_data(data)
    lines.extend(
        (
            f"Dims: {_sizes_text(public)}",
            f"Shape: {_shape_text(public.shape)}",
            f"Dtype: {public.dtype}",
            f"Size: {erlab.utils.formatting.format_nbytes(public.nbytes)}",
        )
    )
    return "\n".join(lines)


def source_value_tooltip(
    data: xr.DataArray | None,
    value: tuple[str, str | None],
    *,
    axis: str,
) -> str:
    """Return a compact tooltip for a picked-value row."""
    axis_name = _source_value_axis_label(axis)
    kind, name = value
    if data is None:
        return f"{axis_name} values are unavailable because the source is missing."
    public = _public_source_data(data)
    if kind == "data":
        squeezed = public.squeeze(drop=True)
        return "\n".join(
            (
                f"Use DataArray values as {axis_name} values.",
                f"Dims: {_sizes_text(squeezed)}",
                f"Shape: {_shape_text(squeezed.shape)}",
                f"Dtype: {squeezed.dtype}",
            )
        )
    coord = _coord_by_name(public, name)
    if coord is None:
        return f"Coordinate {name!r} is not available."
    return "\n".join(
        (
            f"Use coordinate {name!r} as {axis_name} values.",
            f"Coord dims: {_dims_text(coord.dims)}",
            f"Shape: {_shape_text(coord.shape)}",
            f"Dtype: {coord.dtype}",
        )
    )


def _source_value_axis_label(axis: str) -> str:
    return {
        "x": "X",
        "y": "Y",
        "xerr": "X error",
        "yerr": "Y error",
    }.get(axis, axis.upper())


def _coord_by_name(data: xr.DataArray, name: str | None) -> xr.DataArray | None:
    if name is None:
        return None
    coord = data.coords.get(name)
    if coord is not None:
        return coord
    for coord_name, coord_data in data.coords.items():
        if str(coord_name) == name:
            return coord_data
    return None


def _source_data_with_loaded_coords(data: xr.DataArray) -> xr.DataArray:
    loaded_coords = {
        key: coord.copy(deep=False).load() for key, coord in data.coords.items()
    }
    return data.copy(deep=False).assign_coords(loaded_coords)


def _source_details_html(data: xr.DataArray) -> str:
    try:
        return erlab.utils.formatting.format_darr_html(
            _source_data_with_loaded_coords(data),
            show_size=True,
            show_summary=False,
        )
    except Exception:
        logger.debug(
            "Failed to load coordinates for Figure Composer source details",
            exc_info=True,
        )
        return erlab.utils.formatting.format_darr_html(
            data,
            show_size=True,
            show_summary=False,
            load_values=False,
        )


class SourceInspectorWidget(QtWidgets.QWidget):
    """Compact, source-tab-local inspector for Figure Composer source data."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureComposerSourceInspector")
        self._source_name: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.title_label = QtWidgets.QLabel("No source selected", self)
        self.title_label.setObjectName("figureComposerSourceInspectorTitle")
        self.title_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.title_label)

        self.subtitle_label = QtWidgets.QLabel("", self)
        self.subtitle_label.setObjectName("figureComposerSourceInspectorSubtitle")
        self.subtitle_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.subtitle_label)

        self.details_button = QtWidgets.QToolButton(self)
        self.details_button.setObjectName("figureComposerSourceInspectorDetailsButton")
        self.details_button.setText("Source details")
        self.details_button.setAccessibleName("Source Details")
        self.details_button.setCheckable(True)
        self.details_button.setChecked(False)
        self.details_button.setAutoRaise(True)
        self.details_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.details_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.details_button.setToolTip(
            "Show coordinates and attributes for the selected source."
        )
        self.details_button.toggled.connect(self._set_details_visible)
        layout.addWidget(self.details_button)

        self.details_scroll = QtWidgets.QScrollArea(self)
        self.details_scroll.setObjectName("figureComposerSourceInspectorDetailsScroll")
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setMaximumHeight(180)
        self.details_scroll.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.details_label = QtWidgets.QLabel(self.details_scroll)
        self.details_label.setObjectName("figureComposerSourceInspectorDetails")
        self.details_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.details_label.setMargin(6)
        self.details_scroll.setWidget(self.details_label)
        self.details_scroll.setVisible(False)
        layout.addWidget(self.details_scroll)

    @QtCore.Slot(bool)
    def _set_details_visible(self, visible: bool) -> None:
        self.details_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        )
        self.details_scroll.setVisible(visible)
        self.setProperty("figureComposerSourceDetailsExpanded", visible)

    def source_name(self) -> str | None:
        return self._source_name

    def details_html(self) -> str:
        return self.details_label.text()

    def set_context(
        self,
        *,
        source_name: str | None,
        source_state: FigureSourceState | None,
        data: xr.DataArray | None,
    ) -> None:
        self._source_name = source_name
        self.setProperty("figureComposerSourceAlias", source_name or "")
        if source_name is None:
            self.title_label.setText("No source selected")
            self.subtitle_label.setText("Select a source in this tab.")
            self.details_label.clear()
            self.details_button.setEnabled(False)
            self.setProperty("figureComposerSourceDims", ())
            return
        self.title_label.setText(_source_display_label(source_state, source_name))
        if data is None:
            self.subtitle_label.setText(f"Alias: {source_name}<br>Unavailable")
            self.details_label.clear()
            self.details_button.setEnabled(False)
            self.setProperty("figureComposerSourceDims", ())
            return
        public = _public_source_data(data)
        summary_html = erlab.utils.formatting.format_darr_shape_html(
            public.rename(None),
            show_size=True,
        )
        self.subtitle_label.setText(
            erlab.interactive.utils._apply_qt_accent_color(summary_html)
        )
        self.details_label.setText(
            erlab.interactive.utils._apply_qt_accent_color(
                _source_details_html(public.rename(None))
            )
        )
        self.details_button.setEnabled(True)
        self.setProperty(
            "figureComposerSourceDims",
            tuple(str(dim) for dim in public.dims),
        )
        self.setProperty("figureComposerSourceDtype", str(public.dtype))
        self.setProperty(
            "figureComposerSourceSize",
            erlab.utils.formatting.format_nbytes(public.nbytes),
        )

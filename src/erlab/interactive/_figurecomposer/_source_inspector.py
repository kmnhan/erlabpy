"""Data-source inspection widgets for Figure Composer."""

from __future__ import annotations

import html
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


def _original_source_name(data: xr.DataArray | None, alias: str) -> str | None:
    if data is None or data.name is None:
        return None
    original = str(data.name)
    return original if original and original != alias else None


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
    if original := _original_source_name(data, name):
        lines.append(f"Original name: {original}")
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
    """Source metadata summary with lazily rendered coordinate details."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureComposerSourceInspector")
        self._source_name: str | None = None
        self._data: xr.DataArray | None = None
        self._details_key: tuple[str, int] | None = None
        self._details_html: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

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

        self.details_label = QtWidgets.QLabel(self)
        self.details_label.setObjectName("figureComposerSourceInspectorDetails")
        self.details_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.details_label.setMargin(6)
        self.details_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.details_label.setVisible(False)
        layout.addWidget(self.details_label)

    @QtCore.Slot(bool)
    def _set_details_visible(self, visible: bool) -> None:
        if visible:
            self._ensure_details_html()
        self.details_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        )
        self.details_label.setVisible(visible)
        self.setProperty("figureComposerSourceDetailsExpanded", visible)

    def _ensure_details_html(self) -> None:
        if self._data is None or self._details_key is None:
            self.details_label.clear()
            return
        if self._details_html is None:
            self._details_html = erlab.interactive.utils._apply_qt_accent_color(
                _source_details_html(self._data)
            )
        self.details_label.setText(self._details_html)

    def invalidate_details(self) -> None:
        """Discard cached metadata after source data changes."""
        self._details_html = None
        self.details_label.clear()

    def source_name(self) -> str | None:
        return self._source_name

    def details_html(self) -> str:
        return self.details_label.text()

    def set_context(
        self,
        *,
        source_name: str | None,
        data: xr.DataArray | None,
        context_lines: Sequence[str] = (),
    ) -> None:
        self._source_name = source_name
        self.setProperty("figureComposerSourceAlias", source_name or "")
        if source_name is None:
            self._data = None
            self._details_key = None
            self._details_html = None
            self.subtitle_label.clear()
            self.details_label.clear()
            self.details_button.setEnabled(False)
            self.setProperty("figureComposerSourceDims", ())
            self.setProperty("figureComposerSourceDtype", "")
            self.setProperty("figureComposerSourceSize", "")
            return
        context_html = tuple(html.escape(line) for line in context_lines)
        if data is None:
            self._data = None
            self._details_key = None
            self._details_html = None
            self.subtitle_label.setText(
                "<br>".join(("Data unavailable", *context_html))
            )
            self.details_label.clear()
            self.details_button.setEnabled(False)
            self.setProperty("figureComposerSourceDims", ())
            self.setProperty("figureComposerSourceDtype", "")
            self.setProperty("figureComposerSourceSize", "")
            return
        public = _public_source_data(data)
        details_key = (source_name, id(data))
        if details_key != self._details_key:
            self._details_html = None
            self.details_label.clear()
        self._data = public.rename(None)
        self._details_key = details_key
        shape_html = erlab.utils.formatting.format_darr_shape_html(
            public.rename(None),
            show_size=True,
        )
        shape_html = (
            shape_html.removeprefix("<p>")
            .removesuffix("</p>")
            .replace("</p><p>", "<br>")
        )
        summary_lines: list[str] = []
        if original := _original_source_name(data, source_name):
            summary_lines.append(html.escape(original))
        summary_lines.extend((shape_html, *context_html))
        self.subtitle_label.setText(
            erlab.interactive.utils._apply_qt_accent_color("<br>".join(summary_lines))
        )
        self.details_button.setEnabled(True)
        if self.details_button.isChecked():
            self._ensure_details_html()
        self.setProperty(
            "figureComposerSourceDims",
            tuple(str(dim) for dim in public.dims),
        )
        self.setProperty("figureComposerSourceDtype", str(public.dtype))
        self.setProperty(
            "figureComposerSourceSize",
            erlab.utils.formatting.format_nbytes(public.nbytes),
        )

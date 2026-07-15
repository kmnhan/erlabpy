from __future__ import annotations

import pydantic
import pytest
from qtpy import QtCore, QtGui

import erlab.interactive._figurecomposer._model._sources as figurecomposer_sources
import erlab.interactive._figurecomposer._ui._widgets as figurecomposer_widgets
from erlab.interactive._figurecomposer import (
    FigureDataSelectionState,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError


def test_source_selection_state_helpers_cover_shared_paths() -> None:
    empty = FigureDataSelectionState(source="data")
    first = FigureDataSelectionState(source="a", qsel={"eV": 0.0})
    first_for_b = first.model_copy(update={"source": "b"})
    second = FigureDataSelectionState(source="b", isel={"eV": 1})

    assert not figurecomposer_sources.selection_has_effect(empty)
    assert figurecomposer_sources.selection_has_effect(first)
    assert figurecomposer_sources.selection_content_equal(first, first_for_b)
    assert figurecomposer_sources.shared_selection(()) == FigureDataSelectionState(
        source=""
    )
    assert figurecomposer_sources.shared_selection(
        (first, first_for_b)
    ) == first.model_copy(update={"source": ""})
    assert figurecomposer_sources.shared_selection((first, second)) is None


def test_source_selection_dimension_parsing_and_updates() -> None:
    selection = FigureDataSelectionState(
        source="data",
        isel={"kx": 1},
        qsel={"eV": 0.0, "eV_width": 0.2},
        mean_dims=("ky",),
    )

    assert figurecomposer_sources.selection_dim_mode(selection, "kx") == "isel"
    assert figurecomposer_sources.selection_dim_mode(selection, "eV") == "qsel"
    assert figurecomposer_sources.selection_dim_mode(selection, "eV_width") == "qsel"
    assert figurecomposer_sources.selection_dim_mode(selection, "ky") == "mean"
    assert figurecomposer_sources.selection_dim_mode(selection, "missing") == "keep"
    assert figurecomposer_sources.selection_dim_value_text(selection, "kx") == "1"
    assert figurecomposer_sources.selection_dim_value_text(selection, "eV") == "0.0"
    assert figurecomposer_sources.selection_dim_value_text(selection, "missing") == ""
    assert figurecomposer_sources.selection_width_key("eV") == "eV_width"
    assert figurecomposer_sources.selection_dim_width_text(selection, "eV") == "0.2"
    assert figurecomposer_sources.selection_dim_width_text(selection, "kx") == ""

    assert figurecomposer_sources.selection_value_from_text(" slice(0, 2) ") == slice(
        0, 2
    )
    assert figurecomposer_sources.selection_width_from_text("") is None
    assert figurecomposer_sources.selection_width_from_text("0.5") == 0.5
    with pytest.raises(FigureComposerInputError):
        figurecomposer_sources.selection_value_from_text("")
    with pytest.raises(FigureComposerInputError):
        figurecomposer_sources.selection_value_from_text("slice(")
    with pytest.raises(FigureComposerInputError):
        figurecomposer_sources.selection_width_from_text("slice(0, 2)")
    with pytest.raises(FigureComposerInputError):
        figurecomposer_sources.selection_width_from_text("bad(")

    assert figurecomposer_sources.selection_with_dimension(
        selection, "kx", "keep"
    ) == FigureDataSelectionState(
        source="data", qsel={"eV": 0.0, "eV_width": 0.2}, mean_dims=("ky",)
    )
    assert figurecomposer_sources.selection_with_dimension(
        selection, "eV", "isel", value=2
    ) == FigureDataSelectionState(
        source="data", isel={"kx": 1, "eV": 2}, mean_dims=("ky",)
    )
    assert figurecomposer_sources.selection_with_dimension(
        selection, "kx", "qsel", value=0.0, width=0.1
    ) == FigureDataSelectionState(
        source="data",
        qsel={"eV": 0.0, "eV_width": 0.2, "kx": 0.0, "kx_width": 0.1},
        mean_dims=("ky",),
    )
    assert figurecomposer_sources.selection_with_dimension(
        selection, "kx", "mean"
    ) == FigureDataSelectionState(
        source="data",
        qsel={"eV": 0.0, "eV_width": 0.2},
        mean_dims=("ky", "kx"),
    )
    with pytest.raises(FigureComposerInputError):
        figurecomposer_sources.selection_with_dimension(
            selection, "eV", "qsel", value=slice(0, 2), width=0.1
        )


def test_figure_composer_state_serializes_slice_values() -> None:
    source = FigureSourceState(
        name="data",
        isel={"kx": slice(1, 5, 2)},
        qsel={"eV": slice(-0.5, 0.5)},
    )
    restored_source = FigureSourceState.model_validate_json(source.model_dump_json())
    assert restored_source.isel["kx"] == slice(1, 5, 2)
    assert restored_source.qsel["eV"] == slice(-0.5, 0.5)

    selection = FigureDataSelectionState(
        source="data",
        isel={"kx": slice(None, 3)},
        qsel={"eV": slice(-1.0, None)},
    )
    restored_selection = FigureDataSelectionState.model_validate_json(
        selection.model_dump_json()
    )
    assert restored_selection.isel["kx"] == slice(None, 3)
    assert restored_selection.qsel["eV"] == slice(-1.0, None)

    operation = FigureOperationState.plot_slices(
        label="slice",
        sources=("data",),
    ).model_copy(
        update={
            "slice_kwargs": {"beta": slice(-0.5, 0.5)},
            "method_args": (slice(1, 3, 2), [slice(None, 1)]),
            "method_kwargs": {"region": slice(None, 2)},
        }
    )
    restored_operation = FigureOperationState.model_validate_json(
        operation.model_dump_json()
    )
    assert restored_operation.slice_kwargs["beta"] == slice(-0.5, 0.5)
    assert restored_operation.method_args == (slice(1, 3, 2), [slice(None, 1)])
    assert restored_operation.method_kwargs["region"] == slice(None, 2)

    panel_style = FigurePlotSlicesPanelStyleState(
        map_index=0,
        slice_index=0,
        norm_kwargs={"region": slice(1, 2)},
    )
    restored_panel_style = FigurePlotSlicesPanelStyleState.model_validate_json(
        panel_style.model_dump_json()
    )
    assert restored_panel_style.norm_kwargs["region"] == slice(1, 2)

    empty_source = FigureSourceState.model_validate({"name": "empty", "isel": None})
    assert empty_source.isel == {}
    with pytest.raises(pydantic.ValidationError):
        FigureSourceState.model_validate({"name": "invalid", "isel": "not-a-mapping"})
    malformed_slice_marker = FigureSourceState.model_validate(
        {
            "name": "malformed-marker",
            "isel": {"x": {"__erlab_figure_composer_slice__": [0, 1]}},
        }
    )
    assert malformed_slice_marker.isel == {
        "x": {"__erlab_figure_composer_slice__": [0, 1]}
    }

    ordinary_kwargs = FigureOperationState.method(
        family="axes",
        name="plot",
        kwargs={"kind": "slice", "start": 1},
    )
    restored_kwargs = FigureOperationState.model_validate(
        ordinary_kwargs.model_dump(mode="json")
    )
    assert restored_kwargs.method_kwargs == {"kind": "slice", "start": 1}


def test_figure_display_window_source_drop_event_branches(qtbot) -> None:
    window = figurecomposer_widgets._FigureComposerDisplayWindow(FigureSubplotsState())
    qtbot.addWidget(window)
    mime = QtCore.QMimeData()
    assert figurecomposer_widgets._false_mime_state(mime) is False
    assert window._handle_source_drag_event(None) is False
    assert (
        window._handle_source_drag_event(QtCore.QEvent(QtCore.QEvent.Type.DragEnter))
        is False
    )

    def drag_event() -> QtGui.QDragEnterEvent:
        return QtGui.QDragEnterEvent(
            QtCore.QPoint(0, 0),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    def drop_event() -> QtGui.QDropEvent:
        return QtGui.QDropEvent(
            QtCore.QPointF(0.0, 0.0),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    assert window._handle_source_drag_event(drag_event()) is False

    window.set_source_drop_callbacks(can_drop=lambda data: data is mime)
    accepted_drag = drag_event()
    assert window._handle_source_drag_event(accepted_drag)
    assert accepted_drag.isAccepted()

    window.set_source_drop_callbacks(
        can_drop=lambda _data: True,
        drop=lambda _data: False,
    )
    assert window._handle_source_drag_event(drop_event()) is False

    window.set_source_drop_callbacks(
        can_drop=lambda _data: True,
        drop=lambda _data: True,
    )
    accepted_drop = drop_event()
    assert window._handle_source_drag_event(accepted_drop)
    assert accepted_drop.isAccepted()

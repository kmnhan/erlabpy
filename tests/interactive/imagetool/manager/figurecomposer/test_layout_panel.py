import pytest

from erlab.interactive._figurecomposer import (
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._layout_panel import FigureLayoutPanel


def _grid_setup(*, child: FigureGridSpecGridState | None = None) -> FigureSubplotsState:
    return FigureSubplotsState(
        layout_mode="gridspec",
        figsize=(6.0, 4.0),
        dpi=150.0,
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root",
                nrows=2,
                ncols=2,
                axes=(
                    FigureGridSpecAxesState(
                        axes_id="left",
                        span=FigureGridSpecSpanState(
                            row_start=0,
                            row_stop=1,
                            col_start=0,
                            col_stop=1,
                        ),
                    ),
                ),
                child_grids=() if child is None else (child,),
            )
        ),
    )


def test_layout_panel_projects_setup_without_emitting_user_intent(qtbot) -> None:
    child = FigureGridSpecGridState(
        grid_id="child",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=1,
            col_stop=2,
        ),
    )
    panel = FigureLayoutPanel()
    qtbot.addWidget(panel)
    requests: list[FigureSubplotsState] = []
    modes: list[str] = []
    panel.setup_requested.connect(requests.append)
    panel.layout_mode_requested.connect(modes.append)

    setup = _grid_setup(child=child)
    panel.set_setup(setup, reserved_names=("data",))

    assert requests == []
    assert modes == []
    assert panel.layout_mode_combo.currentText() == "gridspec"
    assert panel.nrows_spin.value() == 2
    assert panel.ncols_spin.value() == 2
    assert panel.width_spin.value() == 6.0
    assert panel.height_spin.value() == 4.0
    assert panel.width_mm_spin.value() == pytest.approx(152.4, abs=0.01)
    assert panel.height_mm_spin.value() == pytest.approx(101.6, abs=0.01)
    assert panel.dpi_spin.value() == 150.0

    panel.gridspec_layout_widget.sigNestedGridActivated.emit("child")
    assert panel.nrows_spin.value() == 1
    assert panel.ncols_spin.value() == 1
    assert panel.gridspec_parent_grid_button.isEnabled()

    panel.set_setup(_grid_setup())
    assert panel.nrows_spin.value() == 2
    assert panel.ncols_spin.value() == 2
    assert not panel.gridspec_parent_grid_button.isEnabled()
    assert requests == []


def test_layout_panel_emits_validated_setup_and_mode_requests(qtbot) -> None:
    panel = FigureLayoutPanel()
    qtbot.addWidget(panel)
    requests: list[FigureSubplotsState] = []
    modes: list[str] = []
    panel.setup_requested.connect(requests.append)
    panel.layout_mode_requested.connect(modes.append)
    panel.set_setup(FigureSubplotsState())

    panel.nrows_spin.setValue(3)
    assert requests[-1].nrows == 3
    panel.set_setup(requests[-1])

    request_count = len(requests)
    panel.width_ratios_edit.setText("0")
    panel.width_ratios_edit.editingFinished.emit()
    assert len(requests) == request_count
    panel.set_setup(requests[-1])

    panel.width_mm_spin.setValue(127.0)
    panel.height_mm_spin.setValue(76.2)
    panel.height_mm_spin.editingFinished.emit()
    assert requests[-1].figsize == pytest.approx((5.0, 3.0))
    panel.set_setup(requests[-1])

    panel.layout_mode_combo.setCurrentText("gridspec")
    assert modes == ["gridspec"]
    assert requests[-1].layout_mode == "subplots"


def test_layout_panel_validates_gridspec_gestures_and_preserves_selection(
    qtbot,
) -> None:
    panel = FigureLayoutPanel()
    qtbot.addWidget(panel)
    requests: list[FigureSubplotsState] = []
    panel.setup_requested.connect(requests.append)
    panel.set_setup(_grid_setup())

    occupied = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=0,
        col_stop=1,
    )
    panel.gridspec_layout_widget.sigRegionCreated.emit(occupied, "axes")
    assert requests == []

    available = FigureGridSpecSpanState(
        row_start=0,
        row_stop=1,
        col_start=1,
        col_stop=2,
    )
    panel.gridspec_layout_widget.sigRegionCreated.emit(available, "axes")
    assert len(requests) == 1
    created_setup = requests.pop()
    created_id = created_setup.gridspec.root.axes[-1].axes_id
    panel.set_setup(created_setup)
    assert panel.gridspec_layout_widget.selected_region_id() == created_id

    panel.gridspec_layout_widget.sigRegionSelected.emit(created_id, "axes")
    assert panel.gridspec_delete_region_button.isEnabled()
    panel.gridspec_delete_region_button.click()
    assert len(requests) == 1
    deleted_setup = requests.pop()
    assert tuple(axis.axes_id for axis in deleted_setup.gridspec.root.axes) == ("left",)
    panel.set_setup(deleted_setup)
    assert panel.gridspec_layout_widget.selected_region_id() == "left"


def test_layout_panel_updates_reserved_names_and_restores_status(
    qtbot, monkeypatch
) -> None:
    panel = FigureLayoutPanel()
    qtbot.addWidget(panel)
    requests: list[FigureSubplotsState] = []
    panel.setup_requested.connect(requests.append)
    panel.set_setup(_grid_setup())
    panel.gridspec_layout_widget.set_selected_region("left")
    panel.gridspec_layout_widget.sigRegionSelected.emit("left", "axes")

    panel.set_reserved_names(("live_source",))
    panel.gridspec_region_label_edit.setText("live_source")
    panel.gridspec_region_label_edit.editingFinished.emit()
    assert panel.gridspec_region_label_edit.property("invalid") is True
    assert requests == []

    refreshed_grids: list[str] = []
    refresh_status = panel._refresh_gridspec_status

    def record_status_refresh(grid: FigureGridSpecGridState) -> None:
        refreshed_grids.append(grid.grid_id)
        refresh_status(grid)

    monkeypatch.setattr(panel, "_refresh_gridspec_status", record_status_refresh)
    panel.gridspec_region_label_edit.setText("")
    panel.gridspec_region_label_edit.editingFinished.emit()
    assert panel.gridspec_region_label_edit.property("invalid") is False
    assert refreshed_grids == ["root"]
    assert requests == []

    panel.set_reserved_names(())
    panel.gridspec_region_label_edit.setText("live_source")
    panel.gridspec_region_label_edit.editingFinished.emit()
    assert requests[-1].gridspec.root.axes[0].label == "live_source"


def test_layout_panel_release_detaches_grid_event_filter(qtbot) -> None:
    panel = FigureLayoutPanel()
    qtbot.addWidget(panel)
    panel.set_setup(_grid_setup())
    panel.show()
    qtbot.wait_until(lambda: panel.isVisible())
    qtbot.wait(1)

    widget = panel.gridspec_layout_widget
    assert widget._application_event_filter_installed
    requests: list[FigureSubplotsState] = []
    panel.setup_requested.connect(requests.append)

    panel.release()
    panel.release()
    assert not widget._application_event_filter_installed
    panel.hide()
    panel.show()
    qtbot.wait(1)
    assert not widget._application_event_filter_installed
    panel.nrows_spin.setValue(3)
    assert requests == []

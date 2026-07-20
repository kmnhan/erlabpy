import ast
import enum
import json
import pathlib
import types
import typing
from collections.abc import Callable

import numpy as np
import pydantic
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._code import uses_default_replay_input
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    FileDataSelection,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    full_data,
    script,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    GaussianFilterOperation,
    IselOperation,
    QSelOperation,
    ScriptCodeOperation,
)
from erlab.interactive.imagetool.dialogs import SelectionDialog
from erlab.interactive.imagetool.manager import fetch, replace_data
from erlab.interactive.imagetool.manager._modelview import (
    _NODE_UID_ROLE,
    _ImageToolWrapperItemDelegate,
)
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    child_status_badge,
    click_child_status_badge,
    configure_goldtool_child,
    copy_full_code_for_uid,
    make_fit2d_child,
    menu_map_by_object_name,
    metadata_derivation_texts,
    metadata_detail_map,
    select_child_tool,
    select_metadata_rows,
    select_tools,
    set_transform_launch_mode,
    trigger_menu_action,
)

from ._common import (
    _seed_fit2d_param_results,
    _set_selection_point,
    _set_selection_range,
)


def test_elided_value_label_keeps_full_text_during_resize(qtbot) -> None:
    class _FallbackStyleLabel(manager_widgets._ElidedValueLabel):
        def style(self) -> QtWidgets.QStyle | None:
            return None

    label = _FallbackStyleLabel("/very/long/path/to/data/scan_with_long_name.h5")
    qtbot.addWidget(label)
    label.setMargin(2)
    label.setIndent(4)
    label.resize(90, label.sizeHint().height())
    label.show()

    assert label.text() == label.full_text
    assert label.toolTip() == label.full_text
    assert label.sizeHint().width() > label.minimumSizeHint().width()
    assert label.grab().isNull() is False
    assert label.text() == label.full_text

    label.setText(None)
    assert label.text() == ""
    assert label.full_text == ""
    assert not hasattr(label, "clicked")
    assert label.cursor().shape() != QtCore.Qt.CursorShape.PointingHandCursor
    assert label.textInteractionFlags() == (
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )


def test_manager_metadata_added_label_does_not_force_splitter_width(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    long_time = "2024-01-02 03:04:05 Pacific Daylight Time (-0700)"

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._set_metadata_fields(
            [
                manager_wrapper._MetadataField(
                    "Added",
                    long_time,
                    monospace=True,
                )
            ]
        )
        manager._update_metadata_pane()

        label = manager._metadata_detail_labels["Added"]
        assert isinstance(label, manager_widgets._ElidedValueLabel)
        assert label.text() == long_time
        assert label.full_text == long_time
        assert label.toolTip() == long_time
        assert label.textInteractionFlags() == (
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        assert (
            label.sizePolicy().horizontalPolicy()
            == QtWidgets.QSizePolicy.Policy.Ignored
        )
        key_label = typing.cast(
            "QtWidgets.QLabel",
            manager.metadata_details_layout.itemAtPosition(0, 0).widget(),
        )
        assert key_label.alignment() == (
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        assert label.alignment() == (
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        assert label.sizeHint().width() < label.fontMetrics().horizontalAdvance(
            long_time
        )
        assert manager.metadata_details_widget.minimumSizeHint().width() < (
            label.fontMetrics().horizontalAdvance(long_time)
        )


def test_manager_metadata_derivation_list_has_visible_splitter(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    rows = [
        _ProvenanceDisplayRow(DerivationEntry(f"Step {index}", "derived = data", True))
        for index in range(8)
    ]

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        assert manager.right_splitter.count() == 3
        assert manager.right_splitter.widget(2) is manager.metadata_group
        assert (
            manager.metadata_details_widget.parentWidget()
            is manager.metadata_details_page
        )
        assert (
            manager.metadata_derivation_list.parentWidget()
            is manager.metadata_provenance_page
        )
        assert not isinstance(
            manager.metadata_derivation_list.parentWidget(), QtWidgets.QSplitter
        )

        manager._set_metadata_node(
            typing.cast(
                "typing.Any",
                types.SimpleNamespace(
                    uid="node",
                    display_text="Node",
                    note="",
                    has_note=False,
                    is_imagetool=True,
                    type_badge_text="",
                    displayed_provenance_spec=full_data(),
                    metadata_fields=[
                        manager_wrapper._MetadataField("Kind", "ImageTool")
                    ],
                    derivation_display_rows=rows,
                ),
            )
        )

        assert manager.metadata_group.isVisible()
        assert manager.inspector_tabs.currentWidget() is manager.metadata_details_page
        assert manager.metadata_details_widget.isVisible()
        assert not manager.metadata_derivation_list.isVisible()
        manager.inspector_tabs.setCurrentWidget(manager.metadata_provenance_page)
        QtWidgets.QApplication.processEvents()
        assert manager.metadata_derivation_list.isVisible()
        handle = manager.right_splitter.handle(2)
        assert handle is not None
        qtbot.wait_until(handle.isVisible, timeout=5000)
        assert manager.metadata_derivation_list.minimumHeight() > 0
        assert (
            manager.metadata_derivation_list.maximumHeight()
            == manager_widgets._QWIDGETSIZE_MAX
        )

        manager.resize(640, 700)
        manager.right_splitter.setSizes([200, 160, 260])
        QtWidgets.QApplication.processEvents()
        assert manager.right_splitter.sizes()[2] > (
            manager.metadata_derivation_list.minimumHeight()
        )
        before_right_sizes = manager.right_splitter.sizes()
        before_list_height = manager.metadata_derivation_list.height()
        manager.right_splitter.moveSplitter(
            before_right_sizes[0] + before_right_sizes[1] - 40, 2
        )
        QtWidgets.QApplication.processEvents()
        after_right_sizes = manager.right_splitter.sizes()
        assert after_right_sizes[2] > before_right_sizes[2]
        assert manager.metadata_derivation_list.height() > before_list_height


def test_manager_compact_file_suffix(tmp_path) -> None:
    paths = [
        tmp_path / "scan_a.h5",
        tmp_path / "scan_b.h5",
        tmp_path / "scan_c.h5",
    ]

    assert manager_wrapper._compact_file_suffix(paths) == " (scan_a, scan_b, +1)"


def test_manager_childtool_from_filtered_parent_uses_display_provenance(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})
    expected = operation.apply(data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.apply_filter_operation(operation)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert child.input_provenance_spec is not None
        display_code = child.input_provenance_spec.display_code()
        assert display_code is not None
        assert "gaussian_filter" in display_code
        namespace = {"data": data.copy(deep=True)}
        exec(  # noqa: S102
            display_code,
            {"np": np, "xr": xr, "erlab": erlab, "era": erlab.analysis},
            namespace,
        )
        xr.testing.assert_identical(namespace["derived"], expected)


def test_manager_filtered_parent_updates_source_bound_child(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})
    expected = operation.apply(data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)

        root_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        qtbot.wait_until(
            lambda: (
                child_node.source_state == "fresh"
                and fetch(child_uid).identical(expected)
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(child_uid), expected)
        assert child_node.provenance_spec is not None
        code = child_node.provenance_spec.display_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
        xr.testing.assert_identical(namespace["derived"], expected)


def test_manager_filtered_source_bound_child_refresh_keeps_filter(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    updated = data + 100.0
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})
    expected = operation.apply(updated)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(
            lambda: (
                child_node.source_state == "fresh"
                and fetch(child_uid).identical(expected)
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(child_uid), expected)
        display_spec = child_node.displayed_provenance_spec
        assert display_spec is not None
        display_code = display_spec.display_code()
        assert display_code is not None
        assert "gaussian_filter" in display_code
        namespace = _exec_generated_code(
            display_code, {"data": updated.copy(deep=True)}
        )
        xr.testing.assert_identical(namespace["derived"], expected)


def test_manager_filtered_source_bound_child_failed_refresh_keeps_filter(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    bad_update = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["u", "y"],
        coords={"u": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    operation = GaussianFilterOperation(sigma={"x": 1.0})
    expected = operation.apply(data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, bad_update)

        qtbot.wait_until(
            lambda: child_node.source_state == "unavailable",
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(child_uid), expected)
        assert child_tool.slicer_area._accepted_filter_provenance_operation == operation


def test_manager_duplicate_filtered_child_records_filter_once(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})
    expected = operation.apply(data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        duplicated_uid = manager.duplicate_childtool(child_uid)
        duplicated_node = manager._child_node(duplicated_uid)
        duplicated_tool = manager.get_imagetool(duplicated_uid)

        assert duplicated_node.source_spec is not None
        assert [op.op for op in duplicated_node.source_spec.operations] == []
        displayed_source = duplicated_node.displayed_source_spec
        assert displayed_source is not None
        assert [op.op for op in displayed_source.operations] == ["gaussian_filter"]
        display_code = duplicated_node.displayed_provenance_spec.display_code()
        assert display_code is not None
        assert display_code.count("gaussian_filter") == 1
        xr.testing.assert_identical(duplicated_tool.slicer_area.data, expected)


def test_manager_workspace_roundtrip_filtered_child_records_filter_once(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        tree = manager._workspace_controller.saving._to_datatree()
        saved = typing.cast(
            "xr.DataTree", tree[f"0/childtools/{child_uid}/imagetool"]
        ).to_dataset(inherit=False)
        state = json.loads(saved.attrs["itool_state"])
        assert state["filter_operation"]["op"] == "gaussian_filter"
        source_payload = json.loads(saved.attrs["manager_node_live_source_spec"])
        assert source_payload["operations"] == []

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        loaded_node = manager._child_node(child_uid)
        displayed_source = loaded_node.displayed_source_spec
        assert displayed_source is not None
        assert [op.op for op in displayed_source.operations] == ["gaussian_filter"]

        updated = data + 10.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)
        expected = operation.apply(updated)
        qtbot.wait_until(
            lambda: fetch(child_uid).identical(expected),
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(child_uid), expected)


def test_manager_operation_filter_preserves_output_binding(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    class _OutputToolState(pydantic.BaseModel):
        pass

    class _OutputTool(erlab.interactive.utils.ToolWindow[_OutputToolState]):
        StateModel = _OutputToolState
        tool_name = "output-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _OutputToolState()

        @property
        def tool_status(self) -> _OutputToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _OutputToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def output_imagetool_data(
            self, output_id: str | enum.Enum
        ) -> xr.DataArray | None:
            assert output_id == "out"
            return self._data + 10.0

        def output_imagetool_provenance(
            self, output_id: str | enum.Enum, data: xr.DataArray
        ) -> ToolProvenanceSpec | None:
            assert output_id == "out"
            del data
            return script(
                ScriptCodeOperation(label="Use output", code="result = data + 10"),
                start_label="Start from parent",
                active_name="result",
            )

    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child = _OutputTool(data)
        child_uid = manager.add_childtool(child, 0, show=False)
        child_node = manager._child_node(child_uid)
        assert child_node.displayed_source_spec == child_node.source_spec
        initial_output = typing.cast("xr.DataArray", child.output_imagetool_data("out"))
        output_tool = itool(initial_output, manager=False, execute=False)
        assert isinstance(output_tool, erlab.interactive.imagetool.ImageTool)
        output_uid = manager.add_imagetool_child(
            output_tool,
            child_uid,
            show=False,
            provenance_spec=child.output_imagetool_provenance("out", initial_output),
            source_state="fresh",
            output_id="out",
        )
        operation = GaussianFilterOperation(sigma={"x": 1.0})
        output_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)
        expected = operation.apply(initial_output)

        duplicated_uid = manager.duplicate_childtool(output_uid)
        duplicated_node = manager._child_node(duplicated_uid)
        assert duplicated_node.output_id == "out"
        assert duplicated_node.source_spec is None
        xr.testing.assert_identical(fetch(duplicated_uid), expected)

        tree = manager._workspace_controller.saving._to_datatree()
        saved = typing.cast(
            "xr.DataTree",
            tree[f"0/childtools/{child_uid}/childtools/{output_uid}/imagetool"],
        ).to_dataset(inherit=False)
        assert saved.attrs["manager_node_output_id"] == "out"
        state = json.loads(saved.attrs["itool_state"])
        assert state["filter_operation"]["op"] == "gaussian_filter"
        xr.testing.assert_identical(
            saved[_ITOOL_DATA_NAME].rename(initial_output.name),
            initial_output,
        )


def test_manager_non_imagetool_node_displayed_provenance_uses_tool_provenance(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    class _StaticToolState(pydantic.BaseModel):
        value: int = 0

    class _StaticTool(erlab.interactive.utils.ToolWindow[_StaticToolState]):
        StateModel = _StaticToolState
        tool_name = "static-dummy"

        def __init__(
            self,
            data: xr.DataArray,
            provenance_spec: ToolProvenanceSpec,
        ) -> None:
            super().__init__()
            self._data = data
            self._status = _StaticToolState()
            self._provenance_spec = provenance_spec

        @property
        def tool_status(self) -> _StaticToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _StaticToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> bool:
            self._data = new_data
            return True

        def current_provenance_spec(
            self, *, flush_deferred_restore: bool = True
        ) -> ToolProvenanceSpec | None:
            del flush_deferred_restore
            return self._provenance_spec

    data = xr.DataArray(np.arange(4.0), dims=("x",))
    provenance_spec = script(
        ScriptCodeOperation(label="Double data", code="result = data * 2"),
        start_label="Start from data",
        seed_code="data = source",
        active_name="result",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_uid = manager.add_childtool(
            _StaticTool(data, provenance_spec),
            0,
            show=False,
        )
        child_node = manager._child_node(child_uid)

        assert child_node.displayed_provenance_spec == provenance_spec


def test_manager_goldtool_output_itool_stales_when_fit_results_change(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            GoldTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, GoldTool)
        configure_goldtool_child(child, fitted=True, spline=False)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        before = fetch(output_uid).copy(deep=True)

        child.post_fit(child.edge_center + 1, child.edge_stderr)

        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), before)


def test_manager_ximageitem_open_itool_creates_independent_top_level_window(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        assert child.main_image.data_array is not None

        child.main_image.open_itool()

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        child_node = manager._child_node(child_uid)
        assert child_node._childtool_indices == []
        output_node = manager._tool_graph.root_wrappers[1]
        assert output_node.parent_uid is None
        assert output_node.output_id is None
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        assert output_node.provenance_spec.display_code() is not None
        xr.testing.assert_identical(fetch(1), child.main_image.data_array.T)

        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: pytest.fail("unbound xImageItem opens should not prompt"),
        )
        updated = (child.main_image.data_array * 2).rename(
            child.main_image.data_array.name
        )
        child.main_image.setDataArray(updated)
        child.main_image.open_itool()

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        assert child_node._childtool_indices == []
        second_output_node = manager._tool_graph.root_wrappers[2]
        assert second_output_node.parent_uid is None
        assert second_output_node.output_id is None
        assert second_output_node.source_spec is None
        assert second_output_node.provenance_spec is not None
        assert second_output_node.provenance_spec.display_code() is not None
        xr.testing.assert_identical(fetch(2), updated.T)


def test_manager_workspace_roundtrip_independent_unbound_imagetool(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        expected = child.main_image.data_array.T.copy(deep=True)

        child.main_image.open_itool()
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        tree = manager._workspace_controller.saving._to_datatree()

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        matching_roots = [
            wrapper
            for index, wrapper in manager._tool_graph.root_wrappers.items()
            if wrapper.parent_uid is None
            and wrapper.source_spec is None
            and wrapper.provenance_spec is not None
            and wrapper.output_id is None
            and wrapper._childtool_indices == []
            and fetch(index).identical(expected)
        ]
        assert len(matching_roots) == 1
        assert matching_roots[0].provenance_spec is not None
        assert matching_roots[0].provenance_spec.display_code() is not None


def test_manager_metadata_uses_streamlined_child_derivation(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["alpha", "eV"],
            coords={"alpha": np.arange(5), "eV": np.arange(5)},
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        parent_node = manager._tool_graph.root_wrappers[0]
        original_current_source_data = type(parent_node).current_source_data

        def fail_parent_current_source_data(self):
            if self is parent_node:
                raise AssertionError(
                    "metadata rendering must not compute parent source data"
                )
            return original_current_source_data(self)

        monkeypatch.setattr(
            type(parent_node),
            "current_source_data",
            fail_parent_current_source_data,
        )
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from selected parent ImageTool data"
        assert not any(line == "isel()" for line in derivation)
        assert not any("Sort coordinates" in line for line in derivation)
        assert any(line.startswith("transpose(") for line in derivation)

        monkeypatch.setattr(
            type(parent_node),
            "current_source_data",
            original_current_source_data,
        )
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        namespace = _exec_generated_code(
            copied,
            {"data": parent_tool.slicer_area.data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, manager.get_childtool(child_uid).result)
        assert ".isel()" not in copied
        assert "sort_coord_order" not in copied
        assert ".transpose(" in copied


def test_manager_nested_imagetool_refresh_updates_descendant_dependency(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )
    initial_root_spec = selection(IselOperation(kwargs={"x": slice(0, 2)}))
    updated_root_spec = selection(IselOperation(kwargs={"x": slice(1, 3)}))

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False, provenance_spec=initial_root_spec)

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )

        grandchild_data = root_data.isel(y=slice(0, 2))
        grandchild_tool = itool(grandchild_data, manager=False, execute=False)
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=True,
        )

        root_node = manager._tool_graph.root_wrappers[0]
        grandchild_node = manager._child_node(grandchild_uid)
        assert grandchild_node.provenance_spec is not None
        assert "slice(0, 2)" in typing.cast(
            "str", grandchild_node.provenance_spec.derivation_code()
        )

        root_node.set_detached_provenance(updated_root_spec)
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, base.isel(x=slice(1, 3)))

        qtbot.wait_until(
            lambda: (
                grandchild_node.provenance_spec is not None
                and grandchild_node.provenance_spec.derivation_code() is not None
                and "slice(1, 3)"
                in typing.cast("str", grandchild_node.provenance_spec.derivation_code())
            ),
            timeout=5000,
        )
        code = typing.cast("str", grandchild_node.provenance_spec.derivation_code())
        assert ".isel(x=slice(1, 3))" in code
        assert ".isel(x=slice(0, 2))" not in code
        namespace = {"data": base}
        exec(code, namespace)  # noqa: S102
        xr.testing.assert_identical(
            namespace["derived"],
            base.isel(x=slice(1, 3), y=slice(0, 2)),
        )


def test_manager_nested_imagetool_auto_update_can_be_disabled_from_auto_badge(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=selection(IselOperation(kwargs={"x": slice(0, 2)})),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=False,
        )
        child_node = manager._child_node(child_uid)

        updated = base.isel(x=slice(2, 4))
        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            selection(IselOperation(kwargs={"x": slice(2, 4)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(child_uid), root_data)

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            child_uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: child_node.source_state == "fresh", timeout=5000)
        assert child_node.source_auto_update is True
        xr.testing.assert_identical(fetch(child_uid), updated)
        _, badge_text, _ = child_status_badge(manager, child_uid)
        assert badge_text == "Auto"

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            child_uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )
        assert child_node.source_auto_update is False

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.source_update_action.isVisible()
        assert manager.source_update_action.isEnabled()

        unbound_tool = itool(updated.copy(deep=False), manager=False, execute=False)
        assert isinstance(unbound_tool, erlab.interactive.imagetool.ImageTool)
        unbound_uid = manager.add_imagetool_child(unbound_tool, 0, show=False)
        manager.tree_view.clearSelection()
        select_child_tool(manager, unbound_uid)
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        select_child_tool(manager, unbound_uid)
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.source_update_action.isVisible()
        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        updated2 = base.isel(x=slice(4, 6))
        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            selection(IselOperation(kwargs={"x": slice(4, 6)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated2)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(child_uid), updated)


def test_manager_nested_stale_imagetool_marks_grandchildren_stale(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=selection(IselOperation(kwargs={"x": slice(0, 2)})),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=False,
        )

        grandchild_tool = itool(
            root_data.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=True,
        )

        root_node = manager._tool_graph.root_wrappers[0]
        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)

        root_node.set_detached_provenance(
            selection(IselOperation(kwargs={"x": slice(1, 3)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, base.isel(x=slice(1, 3)))

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "stale", timeout=5000)


def test_manager_manual_nested_refresh_updates_stale_ancestors(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=selection(IselOperation(kwargs={"x": slice(0, 2)})),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=False,
        )

        grandchild_tool = itool(
            root_data.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)
        updated_root = base.isel(x=slice(2, 4))

        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            selection(IselOperation(kwargs={"x": slice(2, 4)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_root)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "stale", timeout=5000)

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            grandchild_uid,
            accept_dialog,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: child_node.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "fresh", timeout=5000)
        assert child_node.source_auto_update is False
        assert grandchild_node.source_auto_update is False
        xr.testing.assert_identical(fetch(child_uid), updated_root)
        xr.testing.assert_identical(
            fetch(grandchild_uid), updated_root.isel(y=slice(0, 2))
        )


def test_manager_manual_nested_refresh_resumes_after_deferred_parent(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    class _DeferredToolState(pydantic.BaseModel):
        value: int = 0

    class _DeferredTool(erlab.interactive.utils.ToolWindow[_DeferredToolState]):
        StateModel = _DeferredToolState
        tool_name = "deferred-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _DeferredToolState()
            self.pending_data: xr.DataArray | None = None

        @property
        def tool_status(self) -> _DeferredToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _DeferredToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> bool:
            self.pending_data = new_data
            self._source_refresh_deferred = self.has_source_binding
            return False

        def finish_deferred_update(self) -> None:
            if self.pending_data is None:
                raise RuntimeError("No deferred data is pending")
            self._data = self.pending_data
            self.pending_data = None
            self.finalize_source_refresh()

    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        parent_tool = _DeferredTool(root_data)
        parent_uid = manager.add_childtool(parent_tool, 0, show=False)
        parent_tool.set_source_binding(full_data(), auto_update=False)

        leaf_tool = itool(root_data.isel(y=slice(0, 2)), manager=False, execute=False)
        assert isinstance(leaf_tool, erlab.interactive.imagetool.ImageTool)
        leaf_uid = manager.add_imagetool_child(
            leaf_tool,
            parent_uid,
            show=False,
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=False,
        )

        parent_node = manager._child_node(parent_uid)
        leaf_node = manager._child_node(leaf_uid)
        updated_root = base.isel(x=slice(2, 4))

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_root)

        qtbot.wait_until(lambda: parent_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: leaf_node.source_state == "stale", timeout=5000)

        assert manager._refresh_source_chain_to_uid(leaf_uid) is False
        assert parent_tool.pending_data is not None
        xr.testing.assert_identical(fetch(leaf_uid), root_data.isel(y=slice(0, 2)))

        parent_tool.finish_deferred_update()

        qtbot.wait_until(lambda: parent_node.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: leaf_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(parent_tool.tool_data, updated_root)
        xr.testing.assert_identical(fetch(leaf_uid), updated_root.isel(y=slice(0, 2)))
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_manager_meshtool_output_itools_use_distinct_output_ids(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: (_ for _ in ()).throw(AssertionError("prompt should not open")),
        )

        child._corrected = child.tool_data.copy(deep=True) + 1
        child._mesh = child.tool_data.copy(deep=True) - 1

        child._corr_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        child._mesh_itool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 2, timeout=5000)

        corr_uid, mesh_uid = child_node._childtool_indices
        corr_node = manager._child_node(corr_uid)
        mesh_node = manager._child_node(mesh_uid)
        assert manager.ntools == 1
        assert corr_node.parent_uid == child_uid
        assert mesh_node.parent_uid == child_uid
        assert corr_node.output_id == "meshtool.corrected_output"
        assert mesh_node.output_id == "meshtool.mesh_output"
        assert corr_node.source_spec is None
        assert corr_node.provenance_spec is not None
        assert mesh_node.source_spec is None
        assert mesh_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(corr_uid), child._corrected)
        xr.testing.assert_identical(fetch(mesh_uid), child._mesh)


def test_manager_selection_dialog_opens_child_with_source_spec(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6)).astype(float),
        dims=["alpha", "eV", "beta", "hv"],
        coords={
            "alpha": np.arange(3, dtype=float),
            "eV": np.arange(4, dtype=float),
            "beta": np.arange(5, dtype=float),
            "hv": np.linspace(20.0, 70.0, 6),
        },
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.set_index(3, 2)
        dialog = SelectionDialog(parent_tool.slicer_area)
        assert (
            dialog.launch_mode_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
            == "replace"
        )
        set_transform_launch_mode(dialog, "nest")

        dialog.accept()

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = child_node.imagetool
        expected = data.qsel(hv=40.0)

        assert child_tool is not None
        assert child_node.source_spec is not None
        assert [op.op for op in child_node.source_spec.operations] == ["qsel"]
        xarray.testing.assert_identical(
            child_node.source_spec.apply(parent_tool.slicer_area.data), expected
        )
        xarray.testing.assert_identical(
            child_tool.slicer_area._data.rename(None), expected.rename(None)
        )


def test_manager_batch_selection_replace_qsel_remains_editable(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 2, dtype=float).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={
            "x": [0.0, 1.0, 2.0],
            "y": np.arange(4, dtype=float),
            "z": np.arange(2, dtype=float),
        },
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        first = itool(data, manager=False, execute=False)
        second = itool(data + 100.0, manager=False, execute=False)
        assert isinstance(first, erlab.interactive.imagetool.ImageTool)
        assert isinstance(second, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(first, show=False)
        manager.add_imagetool(second, show=False)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        select_tools(manager, [0, 1])
        dialog = SelectionDialog(first.slicer_area, batch_manager=manager)
        row = dialog.rows[0]
        row.use_check.setChecked(True)
        row.value_start_spin.setValue(1.0)

        assert manager.apply_batch_transform_dialog(dialog, "replace")
        xarray.testing.assert_identical(
            first.slicer_area._data.rename(None),
            data.qsel(x=1.0).rename(None),
        )

        select_tools(manager, [0, 1])
        dialog = SelectionDialog(first.slicer_area, batch_manager=manager)
        row = dialog.rows[0]
        row.use_check.setChecked(True)
        row.kind_combo.setCurrentIndex(
            row.kind_combo.findData("range", QtCore.Qt.ItemDataRole.UserRole)
        )
        row.value_start_spin.setValue(1.0)
        row.value_stop_spin.setValue(3.0)

        assert manager.apply_batch_transform_dialog(dialog, "replace")
        xarray.testing.assert_identical(
            first.slicer_area._data.rename(None),
            data.qsel(x=1.0).qsel(y=slice(1.0, 3.0)).rename(None),
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        selected_row = manager._selected_derivation_row()
        assert selected_row is not None
        assert manager._provenance_edit_controller.can_edit_row(selected_row) == (
            True,
            "",
        )

        def _edit_qsel(dialog: QtWidgets.QDialog) -> None:
            _set_selection_point(dialog, dim="x", method="qsel", value=2.0)

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_qsel)

        assert first.slicer_area._data.sizes["z"] == data.sizes["z"]
        xarray.testing.assert_identical(
            first.slicer_area._data.rename(None),
            data.qsel(x=2.0).qsel(y=slice(1.0, 3.0)).rename(None),
        )

        manager._update_info()
        select_metadata_rows(manager, [2])
        selected_row = manager._selected_derivation_row()
        assert selected_row is not None
        assert manager._provenance_edit_controller.can_edit_row(selected_row) == (
            True,
            "",
        )

        def _edit_qsel_range(dialog: QtWidgets.QDialog) -> None:
            _set_selection_range(
                dialog,
                dim="y",
                method="qsel",
                start=0.0,
                stop=2.0,
            )

        accept_dialog(
            manager._edit_selected_derivation_step,
            pre_call=_edit_qsel_range,
        )

        xarray.testing.assert_identical(
            first.slicer_area._data.rename(None),
            data.qsel(x=2.0).qsel(y=slice(0.0, 2.0)).rename(None),
        )


@pytest.mark.parametrize(
    ("output_id", "expected_name"),
    [
        ("meshtool.corrected_output", "corrected"),
        ("meshtool.mesh_output", "mesh"),
    ],
)
def test_manager_meshtool_output_child_qsel_copy_code_tracks_selected_output_id(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    output_id: str,
    expected_name: str,
) -> None:

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.order_spin.setValue(1)
        child.n_pad_spin.setValue(0)
        child.roi_hw_spin.setValue(1)
        child.feather_spin.setValue(0.0)
        child.p0_spin0.setValue(1)
        child.p0_spin1.setValue(3)
        child.p1_spin0.setValue(1)
        child.p1_spin1.setValue(1)
        child.update()
        assert child._corrected is not None
        assert child._mesh is not None

        if output_id == "meshtool.corrected_output":
            child._corr_itool()
        else:
            child._mesh_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_data = fetch(output_uid)
        nested_tool = itool(
            output_data.qsel(alpha=1, alpha_width=1), manager=False, execute=False
        )
        nested_uid = manager.add_imagetool_child(
            nested_tool,
            output_uid,
            show=False,
            source_spec=selection(QSelOperation(kwargs={"alpha": 1, "alpha_width": 1})),
            source_auto_update=True,
        )

        copied = copy_full_code_for_uid(monkeypatch, manager, nested_uid)
        assert "corrected, mesh =" in copied
        assert "era.mesh.remove_mesh(" in copied
        assert not any(
            line == f"derived = {expected_name}" for line in copied.splitlines()
        )
        assert ")[0]" not in copied
        assert ")[1]" not in copied
        assert f"derived = {expected_name}.qsel(alpha=1, alpha_width=1)" in copied
        namespace = _exec_generated_code(
            copied,
            {"data": parent_tool.slicer_area.data.copy(deep=True)},
        )
        generated = namespace["derived"]
        assert isinstance(generated, xr.DataArray)
        xr.testing.assert_identical(
            generated,
            output_data.qsel(alpha=1, alpha_width=1),
        )


def test_manager_fit2d_output_itools_use_distinct_output_ids(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        parent_tool = manager.get_imagetool(0)
        parent_tool.set_provenance_spec(
            script(
                ScriptCodeOperation(
                    label="Prepare parent data",
                    code="prepared_parent = data + 1",
                ),
                start_label="Start from test data",
                active_name="prepared_parent",
            )
        )

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: (_ for _ in ()).throw(AssertionError("prompt should not open")),
        )

        first_param_name, second_param_name = list(child._params.keys())[:2]
        params_full = []
        for index in range(len(child._params_full)):
            params = child._params.copy()
            params[first_param_name].set(value=1.0 + index)
            params[first_param_name].stderr = 0.01 + index
            params[second_param_name].set(value=10.0 + index)
            params[second_param_name].stderr = 0.1 + index
            params_full.append(params)
        _seed_fit2d_param_results(child, params_full)

        first_param_index = child.param_plot_combo.findText(first_param_name)
        second_param_index = child.param_plot_combo.findText(second_param_name)
        assert first_param_index >= 0
        assert second_param_index >= 0

        child.param_plot_combo.setCurrentIndex(first_param_index)
        assert child.param_plot_combo.currentText() == first_param_name
        first_values = child._param_plot_dataarray(first_param_name, stderr=False)
        child.param_plot_combo.setCurrentIndex(second_param_index)
        assert child.param_plot_combo.currentText() == second_param_name
        second_values = child._param_plot_dataarray(second_param_name, stderr=False)
        assert not second_values.identical(first_values)

        child.param_plot_combo.setCurrentIndex(first_param_index)
        child.param_plot._show_parameter_values()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        child.param_plot_combo.setCurrentIndex(second_param_index)
        assert child.param_plot_combo.currentText() == second_param_name
        child.param_plot._show_parameter_values()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 2, timeout=5000)

        child.param_plot_combo.setCurrentIndex(first_param_index)
        assert child.param_plot_combo.currentText() == first_param_name
        child.param_plot._show_parameter_stderr()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 3, timeout=5000)

        first_values_uid, second_values_uid, stderr_uid = child_node._childtool_indices
        first_values_node = manager._child_node(first_values_uid)
        second_values_node = manager._child_node(second_values_uid)
        stderr_node = manager._child_node(stderr_uid)
        assert manager.ntools == 1
        assert first_values_node.parent_uid == child_uid
        assert second_values_node.parent_uid == child_uid
        assert stderr_node.parent_uid == child_uid
        assert first_values_node.output_id == Fit2DTool._parameter_output_id(
            Fit2DTool.Output.PARAMETER_VALUES, first_param_name
        )
        assert second_values_node.output_id == Fit2DTool._parameter_output_id(
            Fit2DTool.Output.PARAMETER_VALUES, second_param_name
        )
        assert stderr_node.output_id == Fit2DTool._parameter_output_id(
            Fit2DTool.Output.PARAMETER_STDERR, first_param_name
        )
        assert first_values_node.source_spec is None
        assert first_values_node.provenance_spec is not None
        assert second_values_node.source_spec is None
        assert second_values_node.provenance_spec is not None
        assert stderr_node.source_spec is None
        assert stderr_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(first_values_uid), first_values)
        xr.testing.assert_identical(fetch(second_values_uid), second_values)
        xr.testing.assert_identical(
            fetch(stderr_uid),
            child._param_plot_dataarray(first_param_name, stderr=True),
        )
        child.param_plot_combo.setCurrentIndex(second_param_index)
        assert first_values_node._update_from_parent_source()
        xr.testing.assert_identical(fetch(first_values_uid), first_values)
        assert not fetch(first_values_uid).identical(second_values)

        values_code = copy_full_code_for_uid(monkeypatch, manager, first_values_uid)
        second_values_code = copy_full_code_for_uid(
            monkeypatch, manager, second_values_uid
        )
        stderr_code = copy_full_code_for_uid(monkeypatch, manager, stderr_uid)
        assert "prepared_parent = data + 1" in values_code
        assert "prepared_parent = data + 1" in second_values_code
        assert "prepared_parent = data + 1" in stderr_code

        def selected_fit_output(code: str) -> tuple[str, str]:
            for call in (
                node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
            ):
                if (
                    not isinstance(call.func, ast.Attribute)
                    or call.func.attr != "sel"
                    or not isinstance(call.func.value, ast.Attribute)
                    or call.func.value.attr
                    not in {"modelfit_coefficients", "modelfit_stderr"}
                ):
                    continue
                param_keyword = next(
                    (
                        keyword
                        for keyword in call.keywords
                        if keyword.arg == "param"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ),
                    None,
                )
                if param_keyword is not None:
                    return call.func.value.attr, typing.cast(
                        "str", param_keyword.value.value
                    )
            raise AssertionError("generated fit code does not select a parameter")

        assert selected_fit_output(values_code) == (
            "modelfit_coefficients",
            first_param_name,
        )
        assert selected_fit_output(second_values_code) == (
            "modelfit_coefficients",
            second_param_name,
        )
        assert selected_fit_output(stderr_code) == (
            "modelfit_stderr",
            first_param_name,
        )


def test_manager_output_refresh_updates_stale_parent_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    class _OutputToolState(pydantic.BaseModel):
        value: int = 0

    class _OutputTool(erlab.interactive.utils.ToolWindow[_OutputToolState]):
        StateModel = _OutputToolState
        tool_name = "output-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _OutputToolState()
            self.refreshed_inputs: list[xr.DataArray] = []

        @property
        def tool_status(self) -> _OutputToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _OutputToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> bool:
            self.refreshed_inputs.append(new_data)
            self._data = new_data
            return True

        def output_imagetool_data(
            self, output_id: str | enum.Enum
        ) -> xr.DataArray | None:
            assert output_id == "out"
            return self._data + 10.0

        def output_imagetool_provenance(
            self, output_id: str | enum.Enum, data: xr.DataArray
        ) -> ToolProvenanceSpec | None:
            assert output_id == "out"
            return script(
                ScriptCodeOperation(label="Use output", code="result = data + 10"),
                start_label="Start from parent",
                active_name="result",
            )

    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child = _OutputTool(data)
        child_uid = manager.add_childtool(child, 0, show=False)
        child.set_source_binding(full_data(), auto_update=False)

        initial_output = typing.cast("xr.DataArray", child.output_imagetool_data("out"))
        output_tool = itool(initial_output, manager=False, execute=False)
        assert isinstance(output_tool, erlab.interactive.imagetool.ImageTool)
        output_uid = manager.add_imagetool_child(
            output_tool,
            child_uid,
            show=False,
            provenance_spec=child.output_imagetool_provenance("out", initial_output),
            source_state="fresh",
            output_id="out",
        )

        child_node = manager._child_node(child_uid)
        output_node = manager._child_node(output_uid)
        updated = data * 2.0

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), initial_output)

        assert manager._refresh_source_chain_to_uid(output_uid) is True
        assert child.refreshed_inputs
        assert child.source_state == "fresh"
        assert output_node.source_state == "fresh"
        xr.testing.assert_identical(fetch(output_uid), updated + 10.0)


def test_manager_fit2d_unbound_output_itool_creates_independent_top_level_windows(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        initial = xr.DataArray(
            np.arange(4.0), dims=("x",), coords={"x": np.arange(4)}, name="initial"
        )
        updated = xr.DataArray(
            np.arange(4.0) + 10,
            dims=("x",),
            coords={"x": np.arange(4)},
            name="updated",
        )

        child._show_dataarray_in_itool(initial)
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert child_node._childtool_indices == []
        first_output_node = manager._tool_graph.root_wrappers[1]
        assert first_output_node.parent_uid is None
        assert first_output_node.output_id is None
        assert first_output_node.source_spec is None
        assert first_output_node.provenance_spec is None
        assert not first_output_node.reloadable
        xr.testing.assert_identical(fetch(1), initial)
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: pytest.fail("unbound fit2d opens should not prompt"),
        )

        child._show_dataarray_in_itool(updated)

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        assert child_node._childtool_indices == []
        second_output_node = manager._tool_graph.root_wrappers[2]
        assert second_output_node.parent_uid is None
        assert second_output_node.output_id is None
        assert second_output_node.source_spec is None
        assert second_output_node.provenance_spec is None
        assert not second_output_node.reloadable
        xr.testing.assert_identical(fetch(2), updated)


def test_manager_open_in_new_window_nests_image_tool_children(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        file_dir = tmp_path / ("very_long_directory_name_" * 4)
        file_dir.mkdir(parents=True)
        file_path = file_dir / "scan_with_a_long_name.h5"
        test_data.to_netcdf(file_path, engine="h5netcdf")

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        root_index = manager.tree_view._model._row_index(0)
        assert root_index.data(_NODE_UID_ROLE) == parent.uid
        details = metadata_detail_map(manager)
        assert details["Kind"] == "ImageTool"
        assert details["File"] == str(file_path)
        assert "Chunks" not in details
        assert "Added" in details
        assert metadata_derivation_texts(manager) == [
            "Load data from file 'scan_with_a_long_name.h5'"
        ]
        assert manager._build_metadata_derivation_menu() is not None

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        file_label = manager._metadata_detail_labels["File"]
        assert file_label.toolTip() == str(file_path)
        file_label.setFixedWidth(84)
        qtbot.wait(10)
        assert getattr(file_label, "full_text", file_label.text()) == str(file_path)
        assert file_label.text() == str(file_path)
        assert metadata_detail_map(manager)["File"] == str(file_path)
        details_button = manager.metadata_details_widget.findChild(
            QtWidgets.QToolButton,
            "manager_metadata_file_details_button",
        )
        assert details_button is not None

        def _inspect_source_dialog(dialog: QtWidgets.QDialog) -> None:
            assert not dialog.findChildren(QtWidgets.QLineEdit)
            assert not dialog.findChildren(QtWidgets.QPlainTextEdit)
            assert (
                dialog.findChild(
                    QtWidgets.QLabel, "manager_load_source_path_value_label"
                ).text()  # type: ignore[union-attr]
                == str(file_path)
            )
            assert (
                dialog.findChild(
                    QtWidgets.QLabel, "manager_load_source_loader_value_label"
                ).text()  # type: ignore[union-attr]
            ).endswith("xarray.load_dataarray")
            assert (
                dialog.findChild(
                    QtWidgets.QLabel, "manager_load_source_arguments_value_label"
                ).text()  # type: ignore[union-attr]
                == 'engine="h5netcdf"'
            )
            dialog.copy_code_button.click()  # type: ignore[attr-defined]

        accept_dialog(
            lambda: qtbot.mouseClick(
                details_button,
                QtCore.Qt.MouseButton.LeftButton,
            ),
            pre_call=_inspect_source_dialog,
        )
        assert copied
        load_namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(
            load_namespace["data"],
            xr.load_dataarray(file_path, engine="h5netcdf"),
        )

        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        assert child_node.is_imagetool
        assert child_node.parent_uid == parent.uid
        assert child_node.source_spec is not None
        xr.testing.assert_identical(fetch(child_uid), child_tool.slicer_area._data)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        child_index = manager.tree_view._model._row_index(child_uid)
        assert child_index.data(_NODE_UID_ROLE) == child_uid
        child_details = metadata_detail_map(manager)
        assert child_details["Kind"] == "ImageTool"
        assert "Added" in child_details
        assert child_details["File"] == str(file_path)
        assert "Chunks" not in child_details
        assert metadata_derivation_texts(manager)

        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        nested_uid = child_node._childtool_indices[0]
        nested_tool = manager.get_childtool(nested_uid)
        assert isinstance(nested_tool, DerivativeTool)

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_info(uid=nested_uid)
        nested_details = metadata_detail_map(manager)
        assert nested_details["Kind"] == nested_tool.tool_name
        assert "Added" in nested_details
        assert metadata_derivation_texts(manager)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert copied
        assert not uses_default_replay_input(copied[-1])
        namespace = _exec_generated_code(copied[-1], {})
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, nested_tool.result)


def test_manager_promote_action_enablement_and_menus(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._child_node(child_uid)._childtool_indices) == 1,
            timeout=5000,
        )
        nested_uid = manager._child_node(child_uid)._childtool_indices[0]

        menus = menu_map_by_object_name(manager.menu_bar)
        assert manager.promote_action in menus["manager_edit_menu"].actions()
        assert manager.promote_action in manager.tree_view._menu.actions()

        manager.tree_view.clearSelection()
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.promote_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.promote_action.isEnabled()


def test_manager_rename_action_enablement_for_child_selection(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._child_node(child_uid)._childtool_indices) == 1,
            timeout=5000,
        )
        nested_uid = manager._child_node(child_uid)._childtool_indices[0]

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.rename_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        manager.rename_action.trigger()
        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("renamed_child_tool")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: (
                manager.get_childtool(nested_uid)._tool_display_name
                == "renamed_child_tool"
            ),
            timeout=5000,
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert not manager.rename_action.isEnabled()


def test_manager_promote_selected_cancel_keeps_nested_imagetool(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        select_child_tool(manager, child_uid)

        captured: dict[str, str] = {}

        def _cancel_prompt(
            dialog: QtWidgets.QMessageBox,
        ) -> QtWidgets.QMessageBox.StandardButton:
            captured["text"] = dialog.text()
            captured["info"] = dialog.informativeText()
            return QtWidgets.QMessageBox.StandardButton.Cancel

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _cancel_prompt)

        manager.promote_action.trigger()

        assert captured["text"] == "Promote selected ImageTool to a top-level window?"
        assert "live update linkage" in captured["info"].lower()
        assert "detached history" in captured["info"].lower()
        assert manager.ntools == 1
        assert parent._childtool_indices == [child_uid]
        assert manager._child_node(child_uid).parent_uid == parent.uid


def test_manager_promote_child_imagetool_rehomes_subtree_and_detaches_provenance(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            data,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(parent_tool.mnb._average, pre_call=_nest_average)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)
        child_node.name = "averaged child"
        assert (
            manager_widgets._strip_workspace_modified_placeholder(
                child_tool.windowTitle()
            )
            == "averaged child"
        )
        child_before = fetch(child_uid).copy(deep=True)

        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        nested_uid = child_node._childtool_indices[0]

        select_child_tool(manager, child_uid)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _: QtWidgets.QMessageBox.StandardButton.Yes,
        )

        manager.promote_action.trigger()

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        promoted_index = 1
        promoted = manager._tool_graph.root_wrappers[promoted_index]
        assert promoted.uid == child_uid
        assert child_uid not in parent._childtool_indices
        assert promoted.parent_uid is None
        assert promoted.source_spec is None
        assert promoted.provenance_spec is not None
        assert promoted._childtool_indices == [nested_uid]
        assert manager._child_node(nested_uid).parent_uid == child_uid
        qtbot.wait_until(
            lambda: manager.tree_view.selected_imagetool_indices == [promoted_index],
            timeout=5000,
        )
        assert manager.tree_view.selected_imagetool_indices == [promoted_index]
        assert manager.tree_view.selected_childtool_uids == []
        assert manager._root_wrapper_for_uid(nested_uid).index == promoted_index
        assert (
            manager.get_imagetool(promoted_index).windowTitle()
            == f"{promoted_index}: averaged child (scan)"
        )
        xr.testing.assert_identical(fetch(child_uid), child_before)
        xr.testing.assert_identical(
            manager._parent_source_data_for_uid(nested_uid),
            manager.get_imagetool(promoted_index).slicer_area._data,
        )

        manager._update_info()
        derivation = metadata_derivation_texts(manager)
        assert any("Aggregate" in line for line in derivation)

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) + 10
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        assert promoted.source_state == "fresh"
        xr.testing.assert_identical(fetch(child_uid), child_before)


def test_manager_replace_current_sets_provenance_on_provenance_free_root(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._tool_graph.root_wrappers[0]
        root_tool = manager.get_imagetool(0)
        assert root.provenance_spec is None

        def _replace_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(root_tool.mnb._average, pre_call=_replace_average)

        assert root.source_spec is None
        assert root.provenance_spec is not None
        derivation_code = root.provenance_spec.derivation_code()
        assert derivation_code.count("derived =") == 1
        namespace = _exec_generated_code(
            derivation_code,
            {"data": data.copy(deep=True)},
        )
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(
            derived.rename(None), data.qsel.mean("x").rename(None)
        )
        xr.testing.assert_identical(
            root_tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        derivation = metadata_derivation_texts(manager)
        assert derivation == [
            "Start from current parent ImageTool data",
            'Aggregate(dims=("x",), func="mean")',
        ]


def test_manager_aggregate_child_refreshes_from_parent(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_sum(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            dialog.reducer_combo.setCurrentText("Sum")
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(parent_tool.mnb._aggregate, pre_call=_nest_sum)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)

        assert child_node.source_spec is not None
        assert [op.op for op in child_node.source_spec.operations] == [
            "qsel_aggregate",
        ]
        xr.testing.assert_identical(
            fetch(child_uid).rename(None), data.qsel.sum("x").rename(None)
        )

        updated = data + 10
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            fetch(child_uid).rename(None), updated.qsel.sum("x").rename(None)
        )


def test_manager_replace_transform_on_filtered_source_child_keeps_live_source(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": np.arange(4, dtype=float)},
        name="scan",
    )
    operation = GaussianFilterOperation(sigma={"x": 1.0})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        def _replace_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(child_tool.mnb._average, pre_call=_replace_average)

        filtered = operation.apply(data)
        expected = filtered.qsel.mean("x")
        xr.testing.assert_identical(fetch(child_uid), expected)
        assert child_node.source_spec is not None
        assert child_node.source_spec.is_live_source
        assert [op.op for op in child_node.source_spec.operations] == [
            "gaussian_filter",
            "qsel_aggregate",
        ]

        updated = data + 10.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        updated_filtered = operation.apply(updated)
        updated_expected = updated_filtered.qsel.mean("x")
        qtbot.wait_until(
            lambda: (
                child_node.source_state == "fresh"
                and fetch(child_uid).identical(updated_expected)
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(
            fetch(child_uid),
            updated_expected,
        )


def test_manager_file_backed_replace_current_keeps_file_provenance(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._tool_graph.root_wrappers[0]
        root_tool = manager.get_imagetool(0)
        assert root.provenance_spec is not None
        assert root.provenance_spec.display_entries()[0].label == (
            "Load data from file 'scan.h5'"
        )

        def _replace_average(dialog) -> None:
            dialog.dim_checks["alpha"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(root_tool.mnb._average, pre_call=_replace_average)

        assert root.provenance_spec is not None
        assert root.provenance_spec.kind == "file"
        assert [op.op for op in root.provenance_spec.operations] == ["qsel_aggregate"]
        assert [step.input_policy for step in root.provenance_spec.steps] == ["current"]
        entries = root.provenance_spec.display_entries()
        assert entries[0].label == "Load data from file 'scan.h5'"
        assert any("Aggregate" in entry.label for entry in entries)

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        assert metadata_derivation_texts(manager)[0] == "Load data from file 'scan.h5'"

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert copied
        assert "scan.h5" in copied[-1]

        namespace = _exec_generated_code(copied[-1], {})
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(
            derived.rename(None),
            xr.load_dataarray(file_path, engine="h5netcdf")
            .astype(np.float64)
            .qsel.mean("alpha")
            .rename(None),
        )

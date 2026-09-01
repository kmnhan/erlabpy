import ast
import enum
import json
import pathlib
import types
import typing
from collections.abc import Callable, Mapping

import numpy as np
import pydantic
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._lineage as manager_lineage
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._execution import rebuild_script_inputs
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    FileDataSelection,
    ScriptInput,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    full_data,
    script,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    GaussianFilterOperation,
    IselOperation,
    QSelOperation,
    ScriptCodeOperation,
)
from erlab.interactive.imagetool.dialogs import SelectionDialog
from erlab.interactive.imagetool.manager import fetch, replace_data
from erlab.interactive.imagetool.manager._dialogs import _RenameDialog
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
    _authorize_execution,
    _seed_fit2d_param_results,
    _set_selection_point,
    _set_selection_range,
)


class _ManagedInputTestState(pydantic.BaseModel):
    pass


class _ManagedSumTool(erlab.interactive.utils.ToolWindow[_ManagedInputTestState]):
    StateModel = _ManagedInputTestState
    tool_name = "managed-sum-test"

    def __init__(self, data: xr.DataArray, weights: xr.DataArray | None = None) -> None:
        super().__init__()
        self._input_data = data
        self._input_weights = xr.zeros_like(data) if weights is None else weights
        self._data = data + self._input_weights
        self._status = _ManagedInputTestState()
        self.set_script_inputs(
            (
                ScriptInput(name="data", data_role="source"),
                ScriptInput(name="weights", data_role="source"),
            ),
            primary_input="data",
        )

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> _ManagedInputTestState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _ManagedInputTestState) -> None:
        self._status = status

    def _persistence_data_items(self) -> Mapping[str, xr.DataArray]:
        return {
            "<saved-tool-data>": self._input_data,
            "weights": self._input_weights,
        }

    def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> None:
        self._input_data = inputs["data"]
        self._input_weights = inputs["weights"]
        self._data = self._input_data + self._input_weights


class _ReloadCountingManagedSumTool(_ManagedSumTool):
    def __init__(self, data: xr.DataArray) -> None:
        super().__init__(data)
        self.update_calls = 0

    def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> None:
        self.update_calls += 1
        super().update_inputs(inputs)


def test_multi_input_script_provenance_rejects_mismatched_input_names() -> None:
    controller = manager_lineage._LineageController(types.SimpleNamespace())

    with pytest.raises(ValueError, match="Input names must match"):
        controller._multi_input_script_provenance(
            (0, 1),
            operation_label="Combine inputs",
            operation_code="result = first + second",
            input_names=("first",),
        )


def test_resolve_owner_replay_input_applies_source_spec() -> None:
    data = xr.DataArray(np.arange(4), dims="x")
    source_spec = selection(IselOperation(kwargs={"x": slice(1, 3)}))
    snapshot_token = str(object())
    script_input = ScriptInput(
        name="data",
        node_snapshot_token=snapshot_token,
        source_spec=source_spec.model_dump(mode="json"),
    )
    manager = types.SimpleNamespace(
        _extensions=types.SimpleNamespace(
            execution=types.SimpleNamespace(run_operation=None)
        ),
        _tool_graph=types.SimpleNamespace(nodes={}),
    )
    controller = manager_lineage._LineageController(manager)

    assert (
        controller._resolve_live_script_input_for_reload(
            script_input, target_node_uid="owner"
        )
        is None
    )

    manager._tool_graph.nodes["owner"] = types.SimpleNamespace(
        resolved_replay_source_data=lambda: None
    )
    assert (
        controller._resolve_live_script_input_for_reload(
            script_input, target_node_uid="owner"
        )
        is None
    )

    manager._tool_graph.nodes["owner"] = types.SimpleNamespace(
        resolved_replay_source_data=lambda: data
    )
    resolved = controller._resolve_live_script_input_for_reload(
        script_input, target_node_uid="owner"
    )

    assert resolved is not None
    resolved_data, resolved_input = resolved
    xr.testing.assert_identical(resolved_data, source_spec.apply(data))
    assert resolved_input is script_input


def test_input_plan_rejects_invalid_owner_replay_inputs() -> None:
    owner_input = ScriptInput(name="data", node_snapshot_token=str(object()))
    manager = types.SimpleNamespace(
        _tool_graph=types.SimpleNamespace(
            nodes={"owner": types.SimpleNamespace(has_replay_source=True)}
        )
    )
    controller = manager_lineage._LineageController(manager)

    plan = controller._input_resolution_plan(
        (owner_input, owner_input.model_copy(update={"name": "other"})),
        target_node_uid="owner",
    )
    assert (
        plan.unavailable_reason
        == "A result can use only one stored replay-source input."
    )

    manager._tool_graph.nodes["owner"].has_replay_source = False
    plan = controller._input_resolution_plan((owner_input,), target_node_uid="owner")
    assert plan.unavailable_reason == "data has no recorded reload source."


def test_open_script_input_plan_rejects_invalid_replay() -> None:
    live_input = ScriptInput(name="data", node_uid="source")
    spec = script(
        ScriptCodeOperation(label="Invalid code", code="result = )"),
        start_label="Build result",
        active_name="result",
        script_inputs=(live_input,),
    )
    node = types.SimpleNamespace(
        uid="result",
        is_imagetool=True,
        tool_window=None,
        imagetool=object(),
        slicer_area=types.SimpleNamespace(
            _direct_reloadable=lambda: False,
            _provenance_reloadable=lambda: False,
            _local_reload_unavailable_reason=lambda: "not reloadable",
        ),
        provenance_spec=spec,
    )
    manager = types.SimpleNamespace(
        _extensions=types.SimpleNamespace(
            unavailable_reason_for_node=lambda _uid: None
        ),
        _tool_graph=types.SimpleNamespace(nodes={"result": node}),
    )
    controller = manager_lineage._LineageController(manager)
    controller._live_script_input_node = lambda *_args, **_kwargs: object()

    plan = controller._input_resolution_plan((), reload_node=node)

    assert (
        plan.unavailable_reason == "This result contains code that cannot be replayed."
    )


def test_validate_detached_replacement_rejects_multiple_owner_replay_inputs() -> None:
    controller = manager_lineage._LineageController(types.SimpleNamespace())
    spec = script(
        ScriptCodeOperation(label="Combine inputs", code="result = data + other"),
        start_label="Combine inputs",
        active_name="result",
        script_inputs=(
            ScriptInput(name="data", node_snapshot_token=str(object())),
            ScriptInput(name="other", node_snapshot_token=str(object())),
        ),
    )

    with pytest.raises(ValueError, match="only one stored replay-source input"):
        controller._validate_detached_replacement(
            types.SimpleNamespace(uid="owner"), spec, xr.DataArray([1])
        )


def test_pending_reload_reports_unreplayable_script() -> None:
    spec = script(
        ScriptCodeOperation(label="Invalid code", code="result = )"),
        start_label="Build result",
        active_name="result",
        script_inputs=(ScriptInput(name="data", node_uid="source"),),
    )
    node = types.SimpleNamespace(
        uid="pending",
        provenance_spec=spec,
        has_replay_source=False,
    )
    manager = types.SimpleNamespace(
        _extensions=types.SimpleNamespace(
            unavailable_reason_for_node=lambda _uid: None,
            capability_status=lambda *_args, **_kwargs: None,
        ),
        _tool_graph=types.SimpleNamespace(nodes={"pending": node}),
    )
    controller = manager_lineage._LineageController(manager)

    assert (
        controller._pending_imagetool_reload_unavailable_reason(
            node, resolved_live_uids=frozenset()
        )
        == "The recorded script steps cannot be reloaded."
    )


class _ManagedUnaryTool(erlab.interactive.utils.ToolWindow[_ManagedInputTestState]):
    StateModel = _ManagedInputTestState
    tool_name = "managed-unary-test"

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = data
        self._status = _ManagedInputTestState()
        self.update_calls = 0
        self.set_script_inputs(
            (ScriptInput(name="data", data_role="source"),),
            primary_input="data",
        )

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> _ManagedInputTestState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _ManagedInputTestState) -> None:
        self._status = status

    def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> None:
        self.update_calls += 1
        self._data = inputs["data"]


class _DeferredManagedSumTool(_ManagedSumTool):
    def __init__(self, data: xr.DataArray) -> None:
        super().__init__(data)
        self.pending_inputs: dict[str, xr.DataArray] | None = None
        self.update_calls = 0
        self.defer_updates = True

    def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
        self.update_calls += 1
        if not self.defer_updates:
            return False
        self.pending_inputs = dict(inputs)
        self._defer_source_refresh()
        return False

    def finish_deferred_update(self) -> None:
        if self.pending_inputs is None:
            raise RuntimeError("No deferred inputs are pending")
        self._data = self.pending_inputs["data"] + self.pending_inputs["weights"]
        self.pending_inputs = None
        self.finalize_source_refresh()


def _add_deferred_intermediate_managed_chain(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    data: xr.DataArray,
    weights: xr.DataArray,
) -> tuple[
    _DeferredManagedSumTool,
    _ManagedUnaryTool,
    _ManagedUnaryTool,
    str,
]:
    for value in (data, weights):
        root = itool(value, manager=False, execute=False)
        if not isinstance(root, erlab.interactive.imagetool.ImageTool):
            raise TypeError("Expected ImageTool test input")
        manager.add_imagetool(root, show=False)

    upstream = _DeferredManagedSumTool(data + weights)
    upstream_uid = manager.add_childtool(
        upstream,
        script_inputs={"data": 0, "weights": 1},
        show=False,
    )
    upstream.set_script_inputs(
        upstream.script_inputs,
        primary_input="data",
        auto_update=False,
    )

    intermediate = _ManagedUnaryTool(upstream.tool_data)
    intermediate_uid = manager.add_childtool(
        intermediate,
        script_inputs={"data": upstream_uid},
        show=False,
    )

    target = _ManagedUnaryTool(intermediate.tool_data)
    target_uid = manager.add_childtool(
        target,
        script_inputs={"data": intermediate_uid},
        show=False,
    )
    target.set_script_inputs(
        target.script_inputs,
        primary_input="data",
        auto_update=False,
    )
    return upstream, intermediate, target, target_uid


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
                    tool_window=None,
                    displayed_provenance_spec=full_data(),
                    passive_displayed_provenance_spec=full_data(),
                    metadata_fields=[
                        manager_wrapper._MetadataField("Kind", "ImageTool")
                    ],
                    derivation_display_rows=rows,
                    derivation_display_rows_cache_key=("node", None, None),
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


@pytest.mark.parametrize(
    "path",
    [
        pathlib.Path(r"C:\Users\name\data\scan.h5"),
        pathlib.Path(r"\\server\share\data\scan.h5"),
        pathlib.Path("C:/Users/name/data/scan.h5"),
        pathlib.Path("/Users/name/data/scan.h5"),
    ],
)
def test_manager_compact_file_suffix_accepts_cross_platform_paths(
    path: pathlib.Path,
) -> None:
    assert manager_wrapper._compact_file_suffix([path]) == " (scan)"


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
        assert child.script_inputs[0].data_role == "displayed"
        input_provenance = child.script_inputs[0].parsed_provenance_spec()
        assert input_provenance is not None
        display_code = input_provenance.display_code()
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
            self.set_script_inputs(
                (ScriptInput(name="data", data_role="source"),),
                primary_input="data",
            )

        @property
        def tool_status(self) -> _OutputToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _OutputToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> None:
            self._data = inputs["data"]

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
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
            show=False,
        )
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
            self.set_script_inputs(
                (ScriptInput(name="data", data_role="source"),),
                primary_input="data",
            )

        @property
        def tool_status(self) -> _StaticToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _StaticToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
            self._data = inputs["data"]
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
            script_inputs={"data": 0},
            show=False,
        )
        child_node = manager._child_node(child_uid)

        assert child_node.displayed_provenance_spec == provenance_spec


def test_manager_toolwindow_named_inputs_refresh_atomically(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _MultiInputToolState(pydantic.BaseModel):
        value: int = 0

    class _MultiInputTool(erlab.interactive.utils.ToolWindow[_MultiInputToolState]):
        StateModel = _MultiInputToolState
        tool_name = "multi-input-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _MultiInputToolState()
            self.update_calls = 0
            self.set_script_inputs(
                (
                    ScriptInput(name="data", data_role="source"),
                    ScriptInput(name="weights", data_role="source"),
                ),
                primary_input="data",
            )

        @property
        def tool_status(self) -> _MultiInputToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _MultiInputToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> None:
            self.update_calls += 1
            self._data = inputs["data"] + inputs["weights"]

    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("y", "x"),
        name="data",
    )
    weights = xr.ones_like(data).rename("weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights, data + 100.0):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        rejected_tool = _MultiInputTool(data)
        qtbot.addWidget(rejected_tool)
        with pytest.raises(KeyError):
            manager.add_childtool(
                rejected_tool,
                script_inputs={"data": 0, "weights": "missing"},
                show=False,
            )
        assert all(item.node_uid is None for item in rejected_tool.script_inputs)
        with pytest.raises(ValueError, match="must match the ToolWindow"):
            manager.add_childtool(
                rejected_tool,
                script_inputs={"data": 0},
                show=False,
            )
        with pytest.raises(KeyError):
            manager.add_childtool(
                rejected_tool,
                script_inputs={"data": 0, "weights": 1},
                parent="missing",
                show=False,
            )
        with pytest.raises(ValueError, match="one of the named input targets"):
            manager.add_childtool(
                rejected_tool,
                script_inputs={"data": 0, "weights": 1},
                parent=2,
                show=False,
            )
        assert all(item.node_uid is None for item in rejected_tool.script_inputs)

        tool = _MultiInputTool(data + weights)
        child_uid = manager.add_childtool(
            tool,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        tool.set_script_inputs(
            tool.script_inputs,
            primary_input="data",
            auto_update=True,
        )

        child_node = manager._child_node(child_uid)
        assert child_node.parent_uid == manager._node_for_target(0).uid
        assert {
            ref.name
            for ref in manager._lineage_controller._dependency_refs_for_uid(child_uid)
        } == {
            "data",
            "weights",
        }

        updated_weights = (weights * 3).rename("weights")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(1, updated_weights)
        qtbot.wait_until(lambda: tool.update_calls == 1, timeout=5000)
        xr.testing.assert_identical(tool.tool_data, data + updated_weights)
        assert tool.source_state == "fresh"

        auto_updated_data = (data + 10).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, auto_updated_data)
        qtbot.wait_until(
            lambda: not manager._interaction_gate.pending_keys,
            timeout=5000,
        )
        assert tool.update_calls == 2
        xr.testing.assert_identical(tool.tool_data, auto_updated_data + updated_weights)
        assert tool.source_state == "fresh"

        tool.set_script_inputs(
            tool.script_inputs,
            primary_input="data",
            auto_update=False,
        )

        downstream = _ManagedUnaryTool(tool.tool_data)
        downstream_uid = manager.add_childtool(
            downstream,
            script_inputs={"data": child_uid},
            show=False,
        )
        downstream.set_script_inputs(
            downstream.script_inputs,
            primary_input="data",
            auto_update=True,
        )

        updated_data = (data + 20).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_data)
        qtbot.wait_until(lambda: tool.source_state == "stale", timeout=5000)
        assert tool.update_calls == 2
        assert downstream.source_state == "stale"

        late_downstream = _ManagedUnaryTool(tool.tool_data)
        late_downstream_uid = manager.add_childtool(
            late_downstream,
            script_inputs={"data": child_uid},
            show=False,
        )
        assert late_downstream.source_state == "stale"

        assert manager._lineage_controller._refresh_source_chain_to_uid(downstream_uid)
        assert tool.update_calls == 3
        assert downstream.update_calls == 1
        xr.testing.assert_equal(tool.tool_data, updated_data + updated_weights)
        xr.testing.assert_identical(
            downstream.tool_data,
            updated_data + updated_weights,
        )
        assert downstream.source_state == "fresh"
        qtbot.wait_until(
            lambda: not manager._interaction_gate.pending_keys,
            timeout=5000,
        )
        assert downstream.update_calls == 1

        tree_parent = _ManagedUnaryTool(updated_data)
        tree_parent_uid = manager.add_childtool(
            tree_parent,
            script_inputs={"data": 0},
            show=False,
        )
        nested_tool = _MultiInputTool(updated_data + updated_weights)
        manager.add_childtool(
            nested_tool,
            script_inputs={"data": tree_parent_uid, "weights": 1},
            show=False,
        )
        nested_tool.set_script_inputs(
            nested_tool.script_inputs,
            primary_input="data",
            auto_update=True,
        )
        qtbot.wait_until(
            lambda: not manager._interaction_gate.pending_keys,
            timeout=5000,
        )

        nested_data = (updated_data + 5).rename("data")
        tree_parent._data = nested_data
        tree_parent.sigDataChanged.emit()
        qtbot.wait_until(
            lambda: not manager._interaction_gate.pending_keys,
            timeout=5000,
        )
        assert nested_tool.update_calls == 1
        xr.testing.assert_identical(
            nested_tool.tool_data,
            nested_data + updated_weights,
        )

        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: tool.source_state == "unavailable", timeout=5000)
        qtbot.wait_until(
            lambda: downstream.source_state == "unavailable",
            timeout=5000,
        )
        assert late_downstream.source_state == "unavailable"
        assert child_uid in manager._tool_graph.nodes
        assert downstream_uid in manager._tool_graph.nodes
        assert late_downstream_uid in manager._tool_graph.nodes


def test_managed_input_ancestor_refreshes_descendant_from_current_inputs(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        parent = _ManagedSumTool(data + weights)
        parent_uid = manager.add_childtool(
            parent,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        parent.set_script_inputs(
            parent.script_inputs,
            primary_input="data",
            auto_update=False,
        )

        descendant = _ManagedUnaryTool(parent.tool_data)
        descendant_uid = manager.add_childtool(
            descendant,
            script_inputs={"data": parent_uid},
            show=False,
        )
        descendant_node = manager._child_node(descendant_uid)

        updated_data = (data + 100.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_data)
        qtbot.wait_until(lambda: parent.source_state == "stale", timeout=5000)
        qtbot.wait_until(
            lambda: descendant.source_state == "stale",
            timeout=5000,
        )

        assert manager._reload_target_for_child(descendant_uid) is None
        assert manager._lineage_controller._refresh_source_chain_to_uid(descendant_uid)
        xr.testing.assert_equal(parent.tool_data, updated_data + weights)
        xr.testing.assert_equal(descendant.tool_data, updated_data + weights)
        assert parent.source_state == "fresh"
        assert descendant.source_state == "fresh"

        manager.remove_imagetool(1)
        qtbot.wait_until(
            lambda: parent.source_state == "unavailable",
            timeout=5000,
        )
        assert (
            manager._lineage_controller._reload_boundary_for_child(descendant_uid)
            is descendant_node
        )
        assert manager._reload_target_for_child(descendant_uid) is None


@pytest.mark.parametrize("reload_mode", ["direct", "selected"])
def test_deferred_managed_reload_resumes_selected_descendant(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    reload_mode: str,
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        paths = (tmp_path / "data.h5", tmp_path / "weights.h5")
        for index, (value, path) in enumerate(
            zip((data, weights), paths, strict=True),
            start=1,
        ):
            value.to_netcdf(path, engine="h5netcdf")
            itool(
                value,
                manager=True,
                file_path=path,
                load_func=(
                    xr.load_dataarray,
                    {"engine": "h5netcdf"},
                    FileDataSelection(kind="dataarray"),
                ),
            )
            qtbot.wait_until(
                lambda expected_count=index: manager.ntools == expected_count,
                timeout=5000,
            )

        parent = _DeferredManagedSumTool(data + weights)
        parent_uid = manager.add_childtool(
            parent,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        parent.set_script_inputs(
            parent.script_inputs,
            primary_input="data",
            auto_update=False,
        )
        descendant = _ManagedUnaryTool(parent.tool_data)
        descendant_uid = manager.add_childtool(
            descendant,
            script_inputs={"data": parent_uid},
            show=False,
        )

        updated_data = (data + 100.0).rename("data")
        updated_data.to_netcdf(paths[0], engine="h5netcdf")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_data)
        qtbot.wait_until(lambda: parent.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: descendant.source_state == "stale", timeout=5000)

        if reload_mode == "direct":
            assert not manager._child_node(descendant_uid).reload_source_data()
        else:
            manager.tree_view.selectionModel().clearSelection()
            select_child_tool(manager, descendant_uid)
            manager.reload_selected()

        assert parent.pending_inputs is not None
        assert manager._dependency_tracker.has_pending_source_refreshes()

        parent.finish_deferred_update()

        qtbot.wait_until(lambda: parent.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: descendant.source_state == "fresh", timeout=5000)
        xr.testing.assert_equal(parent.tool_data, updated_data + weights)
        xr.testing.assert_equal(descendant.tool_data, updated_data + weights)
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_unrelated_pending_refresh_does_not_defer_failed_input_plan(
    monkeypatch,
) -> None:
    pending = {("unrelated", "dependent")}
    tracker = types.SimpleNamespace(
        has_pending_source_refreshes=lambda: bool(pending),
        source_refresh_queued=lambda blocker, target: (blocker, target) in pending,
    )
    source = types.SimpleNamespace(
        source_state="stale",
        tool_window=types.SimpleNamespace(_source_refresh_deferred=False),
    )
    manager = types.SimpleNamespace(
        _dependency_tracker=tracker,
        _tool_graph=types.SimpleNamespace(nodes={"source": source}),
    )
    controller = manager_lineage._LineageController(manager)
    monkeypatch.setattr(
        controller,
        "_refresh_tool_inputs",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        controller,
        "_reload_boundary_for_child",
        lambda _uid: None,
    )
    plan = manager_lineage._InputResolutionPlan(
        "target",
        (),
        (("apply", "source"),),
        frozenset(),
        None,
    )

    assert (
        controller._execute_input_resolution_plan(
            plan,
            allow_recorded=True,
            reloaded_uids=set(),
        )
        == "failed"
    )


def test_input_resolution_plan_classifies_each_live_fallback_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_input = ScriptInput(name="live", node_uid="live")
    recorded = script(
        ScriptCodeOperation(
            label="Use trusted code",
            code="import os\nresult = live + int(os.path.exists(os.devnull))",
        ),
        start_label="Build recorded input",
        active_name="result",
        script_inputs=(live_input,),
    )
    controller = manager_lineage._LineageController(types.SimpleNamespace())
    calls: dict[str, int] = {}

    def resolve_live(script_input: ScriptInput, **_kwargs):
        calls[script_input.name] = calls.get(script_input.name, 0) + 1
        return object() if script_input.name == "live" else None

    monkeypatch.setattr(controller, "_live_script_input_node", resolve_live)

    plan = controller._input_resolution_plan(
        (ScriptInput(name="recorded", provenance_spec=recorded),),
        target_node_uid="target",
    )

    assert plan.unavailable_reason is None
    assert calls == {"recorded": 1, "live": 2}


def test_pending_script_reload_rejects_missing_recorded_input() -> None:
    spec = script(
        start_label="Copy missing input",
        seed_code="result = missing",
        active_name="result",
        script_inputs=(ScriptInput(name="missing"),),
    )
    manager = types.SimpleNamespace(
        _extensions=types.SimpleNamespace(
            unavailable_reason_for_node=lambda _uid: None,
        ),
        _tool_graph=types.SimpleNamespace(nodes={}),
    )
    controller = manager_lineage._LineageController(manager)
    node = types.SimpleNamespace(
        uid="pending",
        is_imagetool=True,
        tool_window=None,
        imagetool=None,
        pending_workspace_memory_payload=object(),
        provenance_spec=spec,
    )

    reason = controller._node_reload_unavailable_reason(node)

    assert reason is not None
    assert "no recorded reload source" in reason


def test_pending_script_reload_accepts_owner_replay_source() -> None:
    spec = script(
        start_label="Copy stored input",
        seed_code="result = data",
        active_name="result",
        script_inputs=(ScriptInput(name="data", node_snapshot_token=str(object())),),
    )
    node = types.SimpleNamespace(
        uid="pending",
        is_imagetool=True,
        tool_window=None,
        imagetool=None,
        pending_workspace_memory_payload=object(),
        provenance_spec=spec,
        has_replay_source=True,
    )
    manager = types.SimpleNamespace(
        _extensions=types.SimpleNamespace(
            unavailable_reason_for_node=lambda _uid: None,
            capability_status=lambda *_args, **_kwargs: None,
        ),
        _tool_graph=types.SimpleNamespace(nodes={node.uid: node}),
    )
    controller = manager_lineage._LineageController(manager)

    assert controller._node_reload_unavailable_reason(node) is None


def test_owner_replay_source_does_not_resolve_nested_fallback_input() -> None:
    nested = script(
        start_label="Copy nested input",
        seed_code="nested = inner",
        active_name="nested",
        script_inputs=(ScriptInput(name="inner", node_snapshot_token=str(object())),),
    )
    spec = script(
        start_label="Copy recorded input",
        seed_code="result = data",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="data",
                provenance_spec=nested.model_dump(mode="json"),
            ),
        ),
    )
    node = types.SimpleNamespace(
        uid="pending",
        is_imagetool=True,
        tool_window=None,
        imagetool=None,
        pending_workspace_memory_payload=object(),
        provenance_spec=spec,
        has_replay_source=True,
    )
    manager = types.SimpleNamespace(
        _extensions=types.SimpleNamespace(
            unavailable_reason_for_node=lambda _uid: None,
            capability_status=lambda *_args, **_kwargs: None,
        ),
        _tool_graph=types.SimpleNamespace(nodes={node.uid: node}),
    )
    controller = manager_lineage._LineageController(manager)

    reason = controller._node_reload_unavailable_reason(node)

    assert reason is not None
    assert "no recorded reload source" in reason


def test_deferred_live_managed_input_resumes_downstream_once(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        source = _DeferredManagedSumTool(data + weights)
        source_uid = manager.add_childtool(
            source,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        source.set_script_inputs(
            source.script_inputs,
            primary_input="data",
            auto_update=False,
        )
        downstream = _ManagedUnaryTool(source.tool_data)
        downstream_uid = manager.add_childtool(
            downstream,
            script_inputs={"data": source_uid},
            show=False,
        )
        downstream.set_script_inputs(
            downstream.script_inputs,
            primary_input="data",
            auto_update=True,
        )
        descendant = _ManagedUnaryTool(downstream.tool_data)
        descendant_uid = manager.add_childtool(
            descendant,
            script_inputs={"data": downstream_uid},
            show=False,
        )

        updated_data = (data + 100.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_data)
        qtbot.wait_until(lambda: source.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: downstream.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: descendant.source_state == "stale", timeout=5000)

        assert not manager._lineage_controller._refresh_source_chain_to_uid(
            descendant_uid
        )
        assert source.pending_inputs is not None
        assert manager._dependency_tracker.has_pending_source_refreshes()
        assert source.update_calls == 1
        assert downstream.update_calls == 0

        assert not manager._lineage_controller._refresh_source_chain_to_uid(
            downstream_uid
        )
        assert source.update_calls == 1

        source.finish_deferred_update()

        expected = updated_data + weights
        qtbot.wait_until(lambda: source.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: downstream.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: descendant.source_state == "fresh", timeout=5000)
        qtbot.wait_until(
            lambda: not manager._interaction_gate.pending_keys,
            timeout=5000,
        )
        xr.testing.assert_identical(source.tool_data, expected)
        xr.testing.assert_identical(downstream.tool_data, expected)
        xr.testing.assert_identical(descendant.tool_data, expected)
        assert downstream.update_calls == 1
        assert descendant.update_calls == 1
        assert not manager._dependency_tracker.has_pending_source_refreshes()


@pytest.mark.parametrize("queued_update", ["deferred", "rejected"])
def test_deferred_managed_self_rerun_publication_order(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    queued_update: str,
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        source = _DeferredManagedSumTool(data + weights)
        source_uid = manager.add_childtool(
            source,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        source.set_script_inputs(
            source.script_inputs,
            primary_input="data",
            auto_update=True,
        )
        source_node = manager._child_node(source_uid)
        initial_snapshot = source_node.snapshot_token

        named = _ManagedUnaryTool(source.tool_data)
        named_uid = manager.add_childtool(
            named,
            script_inputs={"data": source_uid},
            show=False,
        )
        named.set_script_inputs(
            named.script_inputs,
            primary_input="data",
            auto_update=True,
        )

        tree_tool = itool(source.tool_data, manager=False, execute=False)
        assert isinstance(tree_tool, erlab.interactive.imagetool.ImageTool)
        tree_uid = manager.add_imagetool_child(
            tree_tool,
            source_uid,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        tree_node = manager._child_node(tree_uid)
        tree_updates: list[xr.DataArray] = []
        replace_tree_data = tree_node.handle_parent_source_replaced

        def record_tree_update(parent_data: xr.DataArray) -> bool:
            tree_updates.append(parent_data.copy(deep=True))
            return replace_tree_data(parent_data)

        monkeypatch.setattr(
            tree_node,
            "handle_parent_source_replaced",
            record_tree_update,
        )

        first = (data + 100.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, first)
        qtbot.wait_until(lambda: source.update_calls == 1, timeout=5000)
        assert source._source_refresh_deferred is True

        latest = (data + 200.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, latest)
        qtbot.wait_until(
            lambda: manager._dependency_tracker.source_refresh_queued(
                source_uid,
                source_uid,
            ),
            timeout=5000,
        )

        dirty_uids: list[str] = []
        mark_node_data_dirty = manager._mark_node_data_dirty

        def record_node_data_dirty(uid: str) -> None:
            dirty_uids.append(uid)
            mark_node_data_dirty(uid)

        monkeypatch.setattr(manager, "_mark_node_data_dirty", record_node_data_dirty)
        if queued_update == "rejected":
            source.defer_updates = False
        source.finish_deferred_update()

        assert source.update_calls == 1
        qtbot.wait_until(lambda: source.update_calls == 2, timeout=5000)
        if queued_update == "rejected":
            qtbot.wait_until(
                lambda: not manager._interaction_gate.pending_keys,
                timeout=5000,
            )
            expected = first + weights
            assert source._source_refresh_deferred is False
            assert source.source_state == "stale"
            assert source_node.snapshot_token != initial_snapshot
            assert dirty_uids.count(source_uid) == 1
            assert named.update_calls == 0
            assert tree_updates == []
            assert manager._dependency_tracker.status_for_uid(named_uid) == "changed"
            xr.testing.assert_identical(source.tool_data, expected)
            xr.testing.assert_identical(named.tool_data, data + weights)
            xr.testing.assert_identical(fetch(tree_uid), data + weights)
            assert not manager._dependency_tracker.has_pending_source_refreshes()
            return

        assert source._source_refresh_deferred is True
        assert source_node.snapshot_token == initial_snapshot
        assert source_uid not in dirty_uids
        assert named.update_calls == 0
        assert tree_updates == []
        xr.testing.assert_identical(named.tool_data, data + weights)
        xr.testing.assert_identical(fetch(tree_uid), data + weights)

        source.finish_deferred_update()

        qtbot.wait_until(lambda: named.update_calls == 1, timeout=5000)
        qtbot.wait_until(lambda: len(tree_updates) == 1, timeout=5000)
        qtbot.wait_until(
            lambda: not manager._interaction_gate.pending_keys,
            timeout=5000,
        )
        expected = latest + weights
        xr.testing.assert_identical(source.tool_data, expected)
        xr.testing.assert_identical(named.tool_data, expected)
        xr.testing.assert_identical(fetch(tree_uid), expected)
        xr.testing.assert_identical(tree_updates[0], expected)
        assert named.update_calls == 1
        assert len(tree_updates) == 1
        assert source_node.snapshot_token != initial_snapshot
        assert dirty_uids.count(source_uid) == 1
        assert not manager._dependency_tracker.has_pending_source_refreshes()


@pytest.mark.parametrize("manual_reload", [False, True])
def test_deferred_managed_self_rerun_respects_disabled_auto_update(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    manual_reload: bool,
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        source, _intermediate, target, target_uid = (
            _add_deferred_intermediate_managed_chain(manager, data, weights)
        )
        source_uid = manager._node_uid_from_window(source)
        assert source_uid is not None
        source.set_script_inputs(
            source.script_inputs,
            primary_input="data",
            auto_update=True,
        )

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, (data + 100.0).rename("data"))
        qtbot.wait_until(lambda: source.update_calls == 1, timeout=5000)
        assert source._source_refresh_deferred

        latest = (data + 200.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, latest)
        qtbot.wait_until(
            lambda: manager._dependency_tracker.source_refresh_queued(
                source_uid,
                source_uid,
            ),
            timeout=5000,
        )

        source._set_source_auto_update(False)
        if manual_reload:
            assert not manager._lineage_controller._reload_target_with_continuations(
                source_uid,
                (target_uid,),
            )

        source.finish_deferred_update()

        if not manual_reload:
            qtbot.wait_until(
                lambda: not manager._interaction_gate.pending_keys,
                timeout=5000,
            )
            assert source.update_calls == 1
            assert source.source_state == "stale"
            assert target.source_state == "stale"
            assert not manager._dependency_tracker.has_pending_source_refreshes()
            return

        expected = latest + weights
        qtbot.wait_until(lambda: source.update_calls == 2, timeout=5000)
        assert source._source_refresh_deferred
        source.finish_deferred_update()
        qtbot.wait_until(lambda: target.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(source.tool_data, expected)
        xr.testing.assert_identical(target.tool_data, expected)
        assert source.update_calls == 2
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_deferred_reload_resumes_managed_target_after_intermediate_update(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        upstream, intermediate, target, target_uid = (
            _add_deferred_intermediate_managed_chain(manager, data, weights)
        )

        updated_data = (data + 100.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_data)
        qtbot.wait_until(lambda: upstream.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: intermediate.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: target.source_state == "stale", timeout=5000)

        assert not manager._lineage_controller._refresh_source_chain_to_uid(target_uid)
        assert upstream.pending_inputs is not None
        assert manager._dependency_tracker.has_pending_source_refreshes()

        upstream.finish_deferred_update()

        qtbot.wait_until(lambda: target.source_state == "fresh", timeout=5000)
        qtbot.wait_until(
            lambda: not manager._interaction_gate.pending_keys,
            timeout=5000,
        )
        expected = updated_data + weights
        xr.testing.assert_identical(upstream.tool_data, expected)
        xr.testing.assert_identical(intermediate.tool_data, expected)
        xr.testing.assert_identical(target.tool_data, expected)
        assert intermediate.update_calls == 1
        assert target.update_calls == 1
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_deferred_abort_runs_newest_input_before_descendant_continuations(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        upstream, intermediate, target, target_uid = (
            _add_deferred_intermediate_managed_chain(manager, data, weights)
        )
        upstream_uid = manager._node_uid_from_window(upstream)
        assert upstream_uid is not None
        upstream.set_script_inputs(
            upstream.script_inputs,
            primary_input="data",
            auto_update=True,
        )
        upstream_node = manager._child_node(upstream_uid)
        initial_snapshot = upstream_node.snapshot_token

        first = (data + 100.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, first)
        qtbot.wait_until(lambda: upstream.update_calls == 1, timeout=5000)
        assert upstream._source_refresh_deferred is True
        assert not manager._lineage_controller._refresh_source_chain_to_uid(target_uid)
        assert manager._dependency_tracker.has_pending_source_refreshes()

        latest = (data + 200.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, latest)
        qtbot.wait_until(
            lambda: (
                upstream_uid
                in manager._dependency_tracker._pending_source_refresh_targets.get(
                    upstream_uid, set()
                )
            ),
            timeout=5000,
        )

        upstream.abort_source_refresh()

        assert upstream.update_calls == 2
        assert upstream._source_refresh_deferred is True
        assert upstream_node.snapshot_token == initial_snapshot
        assert intermediate.update_calls == 0
        assert target.update_calls == 0
        assert manager._dependency_tracker.has_pending_source_refreshes()

        upstream.finish_deferred_update()

        expected = latest + weights
        qtbot.wait_until(lambda: target.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(upstream.tool_data, expected)
        xr.testing.assert_identical(intermediate.tool_data, expected)
        xr.testing.assert_identical(target.tool_data, expected)
        assert intermediate.update_calls == 1
        assert target.update_calls == 1
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_deferred_abort_discards_descendant_continuations_without_rerun(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        upstream, intermediate, target, target_uid = (
            _add_deferred_intermediate_managed_chain(manager, data, weights)
        )
        upstream_uid = manager._node_uid_from_window(upstream)
        assert upstream_uid is not None
        upstream_node = manager._child_node(upstream_uid)
        initial_snapshot = upstream_node.snapshot_token

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, (data + 100.0).rename("data"))
        qtbot.wait_until(lambda: target.source_state == "stale", timeout=5000)
        assert not manager._lineage_controller._refresh_source_chain_to_uid(target_uid)
        assert upstream._source_refresh_deferred is True
        assert manager._dependency_tracker.has_pending_source_refreshes()

        upstream.abort_source_refresh()

        assert upstream_node.snapshot_token == initial_snapshot
        assert upstream.source_state == "stale"
        assert intermediate.source_state == "stale"
        assert target.source_state == "stale"
        assert intermediate.update_calls == 0
        assert target.update_calls == 0
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_deferred_reload_clears_managed_target_after_intermediate_failure(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        upstream, intermediate, target, target_uid = (
            _add_deferred_intermediate_managed_chain(manager, data, weights)
        )

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, data + 100.0)
        qtbot.wait_until(lambda: upstream.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: intermediate.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: target.source_state == "stale", timeout=5000)

        assert not manager._lineage_controller._refresh_source_chain_to_uid(target_uid)
        assert manager._dependency_tracker.has_pending_source_refreshes()

        def fail_update(_inputs: Mapping[str, xr.DataArray]) -> None:
            raise RuntimeError("intermediate update failed")

        monkeypatch.setattr(intermediate, "update_inputs", fail_update)
        upstream.finish_deferred_update()

        qtbot.wait_until(
            lambda: intermediate.source_state == "unavailable", timeout=5000
        )
        assert target.source_state == "unavailable"
        assert target.update_calls == 0
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_deferred_managed_input_failure_clears_descendant_continuation(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        source = _DeferredManagedSumTool(data + weights)
        source_uid = manager.add_childtool(
            source,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        source.set_script_inputs(
            source.script_inputs,
            primary_input="data",
            auto_update=False,
        )
        target = _ManagedUnaryTool(source.tool_data)
        target_uid = manager.add_childtool(
            target,
            script_inputs={"data": source_uid},
            show=False,
        )
        target.set_script_inputs(
            target.script_inputs,
            primary_input="data",
            auto_update=True,
        )
        descendant = _ManagedUnaryTool(target.tool_data)
        descendant_uid = manager.add_childtool(
            descendant,
            script_inputs={"data": target_uid},
            show=False,
        )

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, data + 100.0)
        qtbot.wait_until(lambda: source.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: target.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: descendant.source_state == "stale", timeout=5000)

        assert not manager._lineage_controller._refresh_source_chain_to_uid(
            descendant_uid
        )
        assert manager._dependency_tracker.has_pending_source_refreshes()

        def fail_update(_inputs: Mapping[str, xr.DataArray]) -> None:
            raise RuntimeError("target update failed")

        monkeypatch.setattr(target, "update_inputs", fail_update)
        source.finish_deferred_update()

        qtbot.wait_until(lambda: target.source_state == "unavailable", timeout=5000)
        assert descendant.source_state == "unavailable"
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_failed_live_input_reload_does_not_detach_to_recorded_fallback(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")
    recorded_fallback = script(
        start_label="Create fallback data",
        seed_code="data = xr.DataArray([-1.0, -2.0], dims='x')",
        active_name="data",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        source = _ManagedSumTool(data + weights)
        source_uid = manager.add_childtool(
            source,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        source.set_script_inputs(
            source.script_inputs,
            primary_input="data",
            auto_update=False,
        )
        downstream = _ManagedUnaryTool(source.tool_data)
        downstream_uid = manager.add_childtool(
            downstream,
            script_inputs={"data": source_uid},
            show=False,
        )
        binding = downstream.script_inputs[0].model_copy(
            update={"provenance_spec": recorded_fallback.model_dump(mode="json")}
        )
        downstream.set_script_inputs(
            (binding,),
            primary_input="data",
            auto_update=False,
        )

        updated_data = (data + 100.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_data)
        qtbot.wait_until(lambda: source.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: downstream.source_state == "stale", timeout=5000)
        original = downstream.tool_data
        monkeypatch.setattr(
            manager._child_node(source_uid),
            "reload_source_data",
            lambda: False,
        )

        assert not manager._child_node(downstream_uid).reload_source_data()
        assert downstream.source_state == "stale"
        assert downstream.update_calls == 0
        assert downstream.script_inputs[0].node_uid == source_uid
        xr.testing.assert_identical(downstream.tool_data, original)


def test_managed_input_reload_cycle_fails_without_recursion(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        first = _ManagedUnaryTool(data)
        first_uid = manager.add_childtool(
            first,
            script_inputs={"data": 0},
            show=False,
        )
        second = _ManagedUnaryTool(data)
        second_uid = manager.add_childtool(
            second,
            script_inputs={"data": 0},
            show=False,
        )

        first.set_script_inputs(
            (
                first.script_inputs[0].model_copy(
                    update={
                        "node_uid": second_uid,
                        "node_snapshot_token": "stale",
                        "provenance_spec": None,
                    }
                ),
            ),
            primary_input="data",
            state="stale",
        )
        second.set_script_inputs(
            (
                second.script_inputs[0].model_copy(
                    update={
                        "node_uid": first_uid,
                        "node_snapshot_token": "stale",
                        "provenance_spec": None,
                    }
                ),
            ),
            primary_input="data",
            state="stale",
        )

        assert not manager._child_node(first_uid).reload_source_data()
        assert first.source_state == "stale"
        assert second.source_state == "stale"


def test_managed_input_self_cycle_does_not_use_recorded_fallback(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    fallback = script(
        start_label="Create fallback data",
        seed_code="data = xr.DataArray([-1.0, -2.0], dims='x')",
        active_name="data",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        tool = _ManagedUnaryTool(data)
        uid = manager.add_childtool(
            tool,
            script_inputs={"data": 0},
            show=False,
        )
        tool.set_script_inputs(
            (
                tool.script_inputs[0].model_copy(
                    update={
                        "node_uid": uid,
                        "node_snapshot_token": "stale",
                        "provenance_spec": fallback.model_dump(mode="json"),
                    }
                ),
            ),
            primary_input="data",
            state="stale",
        )
        node = manager._child_node(uid)

        assert "cycle" in (node.reload_unavailable_reason() or "").lower()
        assert not node.reload_source_data()
        assert tool.update_calls == 0
        assert tool.script_inputs[0].node_uid == uid
        xr.testing.assert_identical(tool.tool_data, data)


def test_script_imagetool_self_cycle_is_not_reported_as_reloadable(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    fallback = script(
        start_label="Create fallback data",
        seed_code="data = xr.DataArray([-1.0, -2.0], dims='x')",
        active_name="data",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        target = manager.add_imagetool(tool, show=False)
        node = manager._node_for_target(target)
        node.set_displayed_provenance(
            script(
                start_label="Copy input",
                seed_code="derived = data",
                active_name="derived",
                script_inputs=(
                    ScriptInput(
                        name="data",
                        node_uid=node.uid,
                        provenance_spec=fallback.model_dump(mode="json"),
                    ),
                ),
            )
        )

        reason = manager._lineage_controller._node_reload_unavailable_reason(node)
        assert reason is not None
        assert "cycle" in reason.lower()
        assert node.slicer_area._reload_unavailable_reason() is not None
        assert not node.slicer_area.reloadable
        assert not node.reload_source_data()
        xr.testing.assert_identical(node.current_public_data(), data)

        updated = data + 10.0
        source_path = tmp_path / "source.h5"
        updated.to_netcdf(source_path, engine="h5netcdf")
        node.slicer_area._file_path = source_path
        node.slicer_area._load_func = (
            xr.load_dataarray,
            {"engine": "h5netcdf"},
            FileDataSelection(kind="dataarray"),
        )
        node.set_displayed_provenance(
            script(
                start_label="Copy input",
                seed_code="derived = data",
                active_name="derived",
                script_inputs=(ScriptInput(name="data", node_uid=node.uid),),
            )
        )

        assert manager._lineage_controller._node_reload_unavailable_reason(node) is None
        assert node.slicer_area._reload_unavailable_reason() is None
        assert node.slicer_area.reloadable
        assert node.reload_source_data()
        xr.testing.assert_identical(node.current_public_data(), updated)


def test_managed_input_state_keeps_unavailable_over_stale(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        data_input = _ManagedUnaryTool(data)
        data_input_uid = manager.add_childtool(
            data_input,
            script_inputs={"data": 0},
            show=False,
        )

        dependent = _ManagedSumTool(data + weights)
        dependent_uid = manager.add_childtool(
            dependent,
            script_inputs={"data": data_input_uid, "weights": 1},
            show=False,
        )
        dependent.set_script_inputs(
            dependent.script_inputs,
            primary_input="data",
            auto_update=False,
        )
        downstream = _ManagedUnaryTool(dependent.tool_data)
        manager.add_childtool(
            downstream,
            script_inputs={"data": dependent_uid},
            show=False,
        )

        manager.remove_imagetool(1)
        qtbot.wait_until(
            lambda: dependent.source_state == "unavailable",
            timeout=5000,
        )
        qtbot.wait_until(
            lambda: downstream.source_state == "unavailable",
            timeout=5000,
        )

        updated_data = (data + 100.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_data)
        qtbot.wait_until(lambda: data_input.source_state == "stale", timeout=5000)

        assert dependent.source_state == "unavailable"
        assert downstream.source_state == "unavailable"


def test_managed_inputs_refresh_once_after_sibling_outputs_update(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _State(pydantic.BaseModel):
        pass

    class _OutputTool(erlab.interactive.utils.ToolWindow[_State]):
        StateModel = _State
        tool_name = "multi-output-dummy"

        def __init__(self) -> None:
            super().__init__()
            self._generation = 0
            self._status = _State()
            self.set_script_inputs(
                (ScriptInput(name="data", data_role="source"),),
                primary_input="data",
            )

        @property
        def tool_status(self) -> _State:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _State) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return xr.DataArray([self._generation], dims="x")

        def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> None:
            del inputs

        def output_imagetool_data(
            self, output_id: str | enum.Enum
        ) -> xr.DataArray | None:
            offset = {"values": 0.0, "uncertainty": 10.0}[str(output_id)]
            return xr.DataArray(
                [self._generation + offset], dims="x", name=str(output_id)
            )

        def output_imagetool_provenance(
            self, output_id: str | enum.Enum, data: xr.DataArray
        ) -> ToolProvenanceSpec | None:
            del output_id, data
            return None

    class _ConsumerTool(erlab.interactive.utils.ToolWindow[_State]):
        StateModel = _State
        tool_name = "multi-input-consumer-dummy"

        def __init__(self) -> None:
            super().__init__()
            self._status = _State()
            self._data = xr.DataArray([10.0], dims="x")
            self.update_calls = 0
            self.seen: list[tuple[float, float]] = []
            self.set_script_inputs(
                (
                    ScriptInput(name="values", data_role="source"),
                    ScriptInput(name="uncertainty", data_role="source"),
                ),
                primary_input="values",
            )

        @property
        def tool_status(self) -> _State:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _State) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> None:
            self.update_calls += 1
            self.seen.append(
                (
                    float(inputs["values"].item()),
                    float(inputs["uncertainty"].item()),
                )
            )
            self._data = inputs["values"] + inputs["uncertainty"]

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(xr.DataArray([0.0], dims="x"), manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        output_tool = _OutputTool()
        output_tool_uid = manager.add_childtool(
            output_tool,
            script_inputs={"data": 0},
            show=False,
        )
        output_uids: dict[str, str] = {}
        output_imagetools: list[erlab.interactive.imagetool.ImageTool] = []
        for output_id in ("values", "uncertainty"):
            output_data = typing.cast(
                "xr.DataArray", output_tool.output_imagetool_data(output_id)
            )
            output_imagetool = itool(output_data, manager=False, execute=False)
            assert isinstance(output_imagetool, erlab.interactive.imagetool.ImageTool)
            output_imagetools.append(output_imagetool)
            output_uids[output_id] = manager.add_imagetool_child(
                output_imagetool,
                output_tool_uid,
                show=False,
                source_auto_update=True,
                output_id=output_id,
            )

        consumer = _ConsumerTool()
        manager.add_childtool(
            consumer,
            script_inputs=output_uids,
            parent=output_uids["values"],
            show=False,
        )
        consumer.set_script_inputs(
            consumer.script_inputs,
            primary_input="values",
            auto_update=True,
        )

        output_tool._generation = 1
        output_tool.sigDataChanged.emit()
        qtbot.wait_until(
            lambda: float(fetch(output_uids["uncertainty"]).item()) == 11.0,
            timeout=5000,
        )
        qtbot.wait_until(lambda: consumer.update_calls == 1, timeout=5000)

        assert consumer.update_calls == 1
        assert consumer.seen == [(1.0, 11.0)]


def test_managed_input_reload_uses_live_data_without_code_authorization(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    recorded_spec = script(
        ScriptCodeOperation(
            label="Evaluate recorded expression",
            code=("import os\nwith open(os.devnull):\n    pass\nderived = data"),
        ),
        start_label="Start from data",
        active_name="derived",
        script_inputs=(ScriptInput(name="data", label="Input"),),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False, provenance_spec=recorded_spec)

        tool = _ManagedUnaryTool(data)
        child_uid = manager.add_childtool(
            tool,
            script_inputs={"data": 0},
            show=False,
        )
        tool.set_script_inputs(
            tool.script_inputs,
            primary_input="data",
            state="stale",
        )
        monkeypatch.setattr(
            manager._lineage_controller,
            "_authorize_provenance_execution",
            lambda *_args, **_kwargs: pytest.fail(
                "live managed input requested code authorization"
            ),
        )

        assert manager._lineage_controller._refresh_source_chain_to_uid(child_uid)
        assert tool.source_state == "fresh"


def test_manager_reload_ignores_unreachable_unsafe_live_input_fallback(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    left = xr.DataArray([1.0, 2.0], dims="x")
    right = xr.DataArray([10.0, 20.0], dims="x")
    unsafe_fallback = script(
        ScriptCodeOperation(
            label="Create recorded right input",
            code=("import os\nright = xr.DataArray([100.0, 200.0], dims='x')"),
        ),
        start_label="Create right input",
        active_name="right",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (left, right):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        left_input = manager._lineage_controller._script_input_for_node(
            manager._node_for_target(0)
        ).model_copy(update={"name": "left"})
        right_input = manager._lineage_controller._script_input_for_node(
            manager._node_for_target(1)
        ).model_copy(
            update={
                "name": "right",
                "provenance_spec": unsafe_fallback.model_dump(mode="json"),
            }
        )
        derived_spec = script(
            ScriptCodeOperation(label="Add live inputs", code="result = left + right"),
            start_label="Add live inputs",
            seed_code="result = left",
            active_name="result",
            script_inputs=(left_input, right_input),
        )
        derived = itool(left + right, manager=False, execute=False)
        assert isinstance(derived, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            derived,
            show=False,
            provenance_spec=derived_spec,
        )
        authorized_code: list[str] = []

        def authorize(entries: tuple[typing.Any, ...], **_kwargs: typing.Any):
            authorized_code.extend(entry.code for entry in entries)
            return _authorize_execution(entries)

        monkeypatch.setattr(
            manager._lineage_controller,
            "_authorize_provenance_execution",
            authorize,
        )

        derived_node = manager._node_for_target(2)
        assert manager._lineage_controller._node_can_reload_script_inputs(derived_node)
        updated_left = left + 100.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_left)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_node.uid) == "changed",
            timeout=5000,
        )

        assert derived_node.reload_source_data()
        xr.testing.assert_identical(
            derived_node.current_public_data().rename(None),
            (updated_left + right).rename(None),
        )
        assert authorized_code
        assert not any("import os" in code for code in authorized_code)


def test_managed_input_recorded_fallback_tracks_nested_live_dependency(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root_node = manager._node_for_target(0)

        nested_input = manager._lineage_controller._script_input_for_node(
            root_node
        ).model_copy(update={"name": "source"})
        fallback = script(
            ScriptCodeOperation(
                label="Increment source",
                code="derived = source + 1",
            ),
            start_label="Start from source",
            active_name="derived",
            script_inputs=(nested_input,),
        )

        tool = _ManagedUnaryTool(data)
        child_uid = manager.add_childtool(
            tool,
            script_inputs={"data": 0},
            show=False,
        )
        tool.set_script_inputs(
            (
                ScriptInput(
                    name="data",
                    label="Missing derived input",
                    node_uid="missing-node",
                    data_role="source",
                    provenance_spec=fallback,
                ),
            ),
            primary_input="data",
            auto_update=False,
            state="stale",
        )

        assert {
            ref.node_uid
            for ref in manager._lineage_controller._dependency_refs_for_uid(child_uid)
        } == {"missing-node"}
        assert manager._child_node(child_uid).reload_source_data()
        xr.testing.assert_identical(tool.tool_data, data + 1)

        refreshed_input = tool.script_inputs[0]
        assert refreshed_input.node_uid is None
        refreshed_fallback = refreshed_input.parsed_provenance_spec()
        assert refreshed_fallback is not None
        assert refreshed_fallback.script_inputs[0].node_uid == root_node.uid
        assert {
            ref.node_uid
            for ref in manager._lineage_controller._dependency_refs_for_uid(child_uid)
        } == {root_node.uid}

        updated = (data + 10).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)
        qtbot.wait_until(lambda: tool.source_state == "stale", timeout=5000)


def test_live_managed_input_fallback_includes_source_transform(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(6.0),
        dims="x",
        coords={"x": np.arange(6)},
        name="data",
    )
    path = tmp_path / "transformed-input.h5"
    data.to_netcdf(path, engine="h5netcdf")
    source_spec = selection(IselOperation(kwargs={"x": slice(1, 5, 2)}))

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        itool(
            data,
            manager=True,
            file_path=path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root = manager._tool_graph.root_wrappers[0]
        script_input = manager._lineage_controller._script_input_for_node(
            root
        ).model_copy(
            update={
                "name": "data",
                "source_spec": source_spec.model_dump(mode="json"),
            }
        )

        resolved = manager._lineage_controller._resolve_live_script_input_for_reload(
            script_input
        )
        assert resolved is not None
        live_data, refreshed_input = resolved
        expected = source_spec.apply(data)
        xr.testing.assert_equal(live_data, expected)

        recorded_input = refreshed_input.model_copy(
            update={"node_uid": None, "node_snapshot_token": None}
        )
        rebuilt, _refreshed = rebuild_script_inputs((recorded_input,))
        xr.testing.assert_identical(rebuilt["data"], expected)


def test_detached_managed_input_reloads_from_owner_snapshot(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    shift = xr.DataArray([10.0, 20.0], dims="x")

    with manager_context() as manager:
        manager.show()
        itool([data, shift], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        data_node = manager._node_for_target(0)
        shift_node = manager._node_for_target(1)
        data_input = manager._lineage_controller._script_input_for_node(
            data_node,
            name="data",
            detached_input_uid=data_node.uid,
            data_role="source",
        )
        assert data_input.node_uid is None
        assert data_input.node_snapshot_token == data_node.snapshot_token_for_role(
            "source"
        )
        spec = script(
            ScriptCodeOperation(label="Add inputs", code="result = data + shift"),
            start_label="Add inputs",
            active_name="result",
            script_inputs=(
                data_input,
                manager._lineage_controller._script_input_for_node(
                    shift_node,
                    name="shift",
                    data_role="source",
                ),
            ),
        )
        data_node.replace_with_detached_data(
            data + shift,
            spec,
            replay_source_data=data,
        )
        assert {
            ref.node_uid
            for ref in manager._lineage_controller._dependency_refs_for_uid(
                data_node.uid
            )
        } == {shift_node.uid}

        updated_shift = shift + 100.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(1, updated_shift)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(data_node.uid) == "changed",
            timeout=5000,
        )

        assert data_node.reload_source_data()
        xr.testing.assert_equal(
            data_node.current_public_data().rename(None),
            (data + updated_shift).rename(None),
        )
        refreshed_data_input = {
            item.name: item for item in data_node.provenance_spec.script_inputs
        }["data"]
        assert refreshed_data_input.node_uid is None
        assert (
            refreshed_data_input.node_snapshot_token == data_input.node_snapshot_token
        )


def test_detached_derived_input_ignores_frozen_fallback_dependencies(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray([1.0, 2.0], dims="x")
    shift = xr.DataArray([10.0, 20.0], dims="x")
    derived = source + 1.0

    with manager_context() as manager:
        manager.show()
        itool([source, shift], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        derived_index = manager._lineage_controller._show_multi_input_script_result(
            derived,
            (0,),
            operation_label="Increment data",
            operation_code="derived = data_0 + 1",
            data_role="source",
        )
        assert derived_index == 2
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        source_node = manager._node_for_target(0)
        shift_node = manager._node_for_target(1)
        derived_node = manager._node_for_target(derived_index)
        replacement_spec = manager._multi_input_script_provenance(
            (derived_index, 1),
            operation_label="Add inputs",
            operation_code="result = data + shift",
            active_name="result",
            detached_input_uid=derived_node.uid,
            data_role="source",
            input_names=("data", "shift"),
        )
        derived_node.replace_with_detached_data(
            derived + shift,
            replacement_spec,
            replay_source_data=derived,
        )

        assert {
            ref.node_uid
            for ref in manager._lineage_controller._dependency_refs_for_uid(
                derived_node.uid
            )
        } == {shift_node.uid}

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, source + 100.0)
        assert manager.dependency_status_for_uid(derived_node.uid) == "current"
        assert not manager._dependency_tracker.transitively_depends_on(
            derived_node.uid, source_node.uid
        )

        derived_node._restore_replay_source_data(None)
        assert derived_node.uid in manager._dependency_tracker.dependent_uids(
            source_node.uid
        )
        assert {
            ref.node_uid
            for ref in manager._lineage_controller._dependency_refs_for_uid(
                derived_node.uid
            )
        } == {source_node.uid, shift_node.uid}
        assert manager.dependency_status_for_uid(derived_node.uid) == "changed"
        derived_node._restore_replay_source_data(derived)
        assert derived_node.uid not in manager._dependency_tracker.dependent_uids(
            source_node.uid
        )
        assert {
            ref.node_uid
            for ref in manager._lineage_controller._dependency_refs_for_uid(
                derived_node.uid
            )
        } == {shift_node.uid}
        assert manager.dependency_status_for_uid(derived_node.uid) == "current"

        updated_shift = shift + 100.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(1, updated_shift)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_node.uid) == "changed",
            timeout=5000,
        )
        assert derived_node.reload_source_data()
        xr.testing.assert_equal(
            derived_node.current_public_data().rename(None),
            (derived + updated_shift).rename(None),
        )
        assert manager.dependency_status_for_uid(derived_node.uid) == "current"


def test_detached_replacement_validates_owner_snapshot_before_mutation(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        node = manager._node_for_target(0)
        detached = manager._lineage_controller._script_input_for_node(
            node,
            name="data",
            detached_input_uid=node.uid,
            data_role="source",
        )
        spec = script(
            ScriptCodeOperation(label="Increment data", code="result = data + 1"),
            start_label="Increment data",
            active_name="result",
            script_inputs=(detached,),
        )

        with pytest.raises(ValueError, match="no recorded reload source"):
            node.replace_with_detached_data(
                data + 1,
                spec,
                replay_source_data=None,
            )

        xr.testing.assert_identical(node.current_public_data().rename(None), data)
        assert node.provenance_spec is None


def test_detached_replacement_rejects_indirect_input_cycle(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        dependent_index = manager._lineage_controller._show_multi_input_script_result(
            data + 1.0,
            (0,),
            operation_label="Increment data",
            operation_code="derived = data_0 + 1",
            data_role="source",
        )
        assert dependent_index == 1
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        node = manager._node_for_target(0)
        dependent_node = manager._node_for_target(dependent_index)
        assert manager._dependency_tracker.transitively_depends_on(
            dependent_node.uid, node.uid
        )
        replacement_spec = manager._multi_input_script_provenance(
            (0, dependent_index),
            operation_label="Add inputs",
            operation_code="result = data + other",
            active_name="result",
            detached_input_uid=node.uid,
            data_role="source",
            input_names=("data", "other"),
        )

        with pytest.raises(ValueError, match="dependency cycle"):
            node.replace_with_detached_data(
                data + dependent_node.current_public_data(),
                replacement_spec,
                replay_source_data=data,
            )

        xr.testing.assert_identical(node.current_public_data().rename(None), data)
        assert node.provenance_spec is None
        assert node.replay_source_data is None


def test_failed_detached_replacement_preserves_source_binding(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("y", "x"))
    source_spec = full_data(AssignAttrsOperation(attrs={"source": "child"}))
    child_data = source_spec.apply(data)

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        child = itool(child_data, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_spec=source_spec,
            source_auto_update=True,
        )
        node = manager._child_node(child_uid)
        detached = manager._lineage_controller._script_input_for_node(
            node,
            name="data",
            detached_input_uid=node.uid,
            data_role="source",
        )
        replacement_spec = script(
            ScriptCodeOperation(label="Increment data", code="result = data + 1"),
            start_label="Increment data",
            active_name="result",
            script_inputs=(detached,),
        )
        provenance_before = node.provenance_spec
        snapshot_before = node.snapshot_token
        dependency_refs_before = manager._lineage_controller._dependency_refs_for_uid(
            node.uid
        )

        def fail_replacement(_data: xr.DataArray) -> None:
            raise RuntimeError("replacement failed")

        monkeypatch.setattr(
            node.slicer_area,
            "replace_source_data",
            fail_replacement,
        )

        with pytest.raises(RuntimeError, match="replacement failed"):
            node.replace_with_detached_data(
                child_data + 1,
                replacement_spec,
                replay_source_data=child_data,
            )

        xr.testing.assert_identical(node.current_public_data(), child_data)
        assert node.source_spec == source_spec
        assert node.has_source_binding
        assert node.source_auto_update
        assert node.output_id is None
        assert node.provenance_spec == provenance_before
        assert node.replay_source_data is None
        assert node.snapshot_token == snapshot_before
        assert (
            manager._lineage_controller._dependency_refs_for_uid(node.uid)
            == dependency_refs_before
        )


def test_managed_input_auto_refresh_uses_fresh_live_script_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    other = xr.DataArray([10.0, 20.0], dims="x")

    with manager_context() as manager:
        manager.show()
        itool([data, other], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert (
            manager._lineage_controller._show_multi_input_script_result(
                data + other,
                (0, 1),
                operation_label="Add inputs",
                operation_code="derived = data_0 + data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        tool = _ManagedUnaryTool(data + other)
        tool._set_source_auto_update(True)
        child_uid = manager.add_childtool(
            tool,
            script_inputs={"data": 2},
            show=False,
        )
        binding_before = tool.script_inputs[0]
        source = manager._tool_graph.root_wrappers[2]

        updated = data + 100.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(source.uid) == "changed",
            timeout=5000,
        )

        manager.get_imagetool(2).slicer_area.reload()
        expected = updated + other
        qtbot.wait_until(
            lambda: tool.source_state == "fresh" and tool.tool_data.equals(expected),
            timeout=5000,
        )
        xr.testing.assert_equal(tool.tool_data.rename(None), expected.rename(None))

        binding_after = tool.script_inputs[0]
        assert binding_after.node_snapshot_token == source.snapshot_token
        assert binding_after.node_snapshot_token != binding_before.node_snapshot_token
        fallback = binding_after.parsed_provenance_spec()
        assert fallback is not None
        assert fallback.script_inputs[0].node_snapshot_token == (
            manager._tool_graph.root_wrappers[0].snapshot_token
        )
        assert manager.dependency_status_for_uid(child_uid) == "current"


def test_managed_input_reload_refreshes_script_source_once(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    other = xr.DataArray([10.0, 20.0], dims="x")

    with manager_context() as manager:
        manager.show()
        itool([data, other], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        source_index = manager._lineage_controller._show_multi_input_script_result(
            data + other,
            (0, 1),
            operation_label="Add inputs",
            operation_code="derived = data_0 + data_1",
        )
        assert source_index == 2
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        consumer = _ManagedUnaryTool(data + other)
        consumer_uid = manager.add_childtool(
            consumer,
            script_inputs={"data": source_index},
            show=False,
        )
        consumer.set_script_inputs(
            consumer.script_inputs,
            primary_input="data",
            auto_update=True,
        )

        updated = data + 100.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)
        source = manager._node_for_target(source_index)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(source.uid) == "changed",
            timeout=5000,
        )

        assert manager._child_node(consumer_uid).reload_source_data()
        qtbot.wait_until(
            lambda: not manager._interaction_gate.pending_keys,
            timeout=5000,
        )
        xr.testing.assert_equal(consumer.tool_data, updated + other)
        assert consumer.update_calls == 1


def test_managed_reload_resumes_after_deferred_script_result_input(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="data")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="weights")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        source = _DeferredManagedSumTool(data + weights)
        source_uid = manager.add_childtool(
            source,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        source.set_script_inputs(
            source.script_inputs,
            primary_input="data",
            auto_update=False,
        )
        source_input = manager._lineage_controller._script_input_for_node(
            manager._child_node(source_uid),
            name="data",
        )
        source_input = source_input.model_copy(
            update={
                "provenance_spec": script(
                    start_label="Create recorded fallback",
                    seed_code="data = xr.DataArray([-1.0, -2.0], dims='x')",
                    active_name="data",
                ).model_dump(mode="json")
            }
        )
        derived_spec = script(
            ScriptCodeOperation(
                label="Increment managed input",
                code="derived = data + 1.0",
            ),
            start_label="Start from managed input",
            active_name="derived",
            script_inputs=(source_input,),
        )
        derived = itool(source.tool_data + 1.0, manager=False, execute=False)
        assert isinstance(derived, erlab.interactive.imagetool.ImageTool)
        derived_index = manager.add_imagetool(
            derived,
            show=False,
            provenance_spec=derived_spec,
        )
        derived_node = manager._node_for_target(derived_index)
        consumer = _ManagedUnaryTool(derived_node.current_public_data())
        consumer_uid = manager.add_childtool(
            consumer,
            script_inputs={"data": derived_index},
            show=False,
        )
        consumer.set_script_inputs(
            consumer.script_inputs,
            primary_input="data",
            auto_update=False,
        )

        updated = (data + 100.0).rename("data")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)
        qtbot.wait_until(lambda: source.source_state == "stale", timeout=5000)

        assert not manager._child_node(consumer_uid).reload_source_data()
        assert source.pending_inputs is not None
        assert manager._dependency_tracker.source_refresh_queued(
            source_uid,
            derived_node.uid,
        )
        assert manager._dependency_tracker.source_refresh_queued(
            derived_node.uid,
            consumer_uid,
        )

        source.finish_deferred_update()

        expected = updated + weights + 1.0
        qtbot.wait_until(
            lambda: (
                consumer.source_state == "fresh" and consumer.tool_data.equals(expected)
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(source.tool_data, updated + weights)
        xr.testing.assert_identical(derived_node.current_public_data(), expected)
        xr.testing.assert_identical(consumer.tool_data, expected)
        assert consumer.update_calls == 1
        assert derived_node.provenance_spec is not None
        assert derived_node.provenance_spec.script_inputs[0].node_uid == source_uid
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_managed_multi_input_reload_refreshes_all_file_sources_once(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="value")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="value")
    paths = (tmp_path / "data.h5", tmp_path / "weights.h5")
    for value, path in zip((data, weights), paths, strict=True):
        value.to_netcdf(path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for index, (value, path) in enumerate(
            zip((data, weights), paths, strict=True),
            start=1,
        ):
            itool(
                value,
                manager=True,
                file_path=path,
                load_func=(
                    xr.load_dataarray,
                    {"engine": "h5netcdf"},
                    FileDataSelection(kind="dataarray"),
                ),
            )
            qtbot.wait_until(
                lambda expected_count=index: manager.ntools == expected_count,
                timeout=5000,
            )

        tool = _ReloadCountingManagedSumTool(data + weights)
        child_uid = manager.add_childtool(
            tool,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        tool.set_script_inputs(
            tool.script_inputs,
            primary_input="data",
            auto_update=True,
        )

        updated_data = data + 100.0
        updated_weights = weights + 1000.0
        for value, path in zip((updated_data, updated_weights), paths, strict=True):
            value.to_netcdf(path, engine="h5netcdf")

        assert manager._child_node(child_uid).reload_source_data()

        xr.testing.assert_identical(fetch(0), updated_data)
        xr.testing.assert_identical(fetch(1), updated_weights)
        xr.testing.assert_equal(tool.tool_data, updated_data + updated_weights)
        assert tool.update_calls == 1


def test_managed_input_reload_uses_detached_child_file_source(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="value")
    file_path = tmp_path / "detached-child.h5"
    data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = itool(
            data,
            manager=False,
            execute=False,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)
        child_node = manager._child_node(child_uid)
        assert not child_node.has_source_binding

        consumer = _ManagedUnaryTool(data)
        consumer_uid = manager.add_childtool(
            consumer,
            script_inputs={"data": child_uid},
            show=False,
        )
        updated = data + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        consumer_node = manager._child_node(consumer_uid)
        assert consumer_node.reload_unavailable_reason() is None
        assert consumer_node.reload_source_data()

        xr.testing.assert_identical(fetch(child_uid), updated)
        xr.testing.assert_identical(consumer.tool_data, updated)
        assert consumer.update_calls == 1


def test_selected_script_results_dedupe_nested_file_reload(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(12.0).reshape(6, 2),
        dims=("x", "y"),
        name="value",
    )
    source_path = tmp_path / "source.h5"
    source.to_netcdf(source_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        itool(
            source,
            manager=True,
            file_path=source_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uids: list[str] = []
        derived_targets: list[int] = []
        slices = (slice(0, 3), slice(3, 6))
        for indexer in slices:
            source_spec = selection(IselOperation(kwargs={"x": indexer}))
            child_data = source_spec.apply(source)
            child = itool(child_data, manager=False, execute=False)
            assert isinstance(child, erlab.interactive.imagetool.ImageTool)
            child_uid = manager.add_imagetool_child(
                child,
                0,
                show=False,
                source_spec=source_spec,
                source_auto_update=False,
            )
            child_uids.append(child_uid)

            script_input = manager._lineage_controller._script_input_for_node(
                manager._child_node(child_uid)
            )
            derived = itool(child_data, manager=False, execute=False)
            assert isinstance(derived, erlab.interactive.imagetool.ImageTool)
            derived_targets.append(
                manager.add_imagetool(
                    derived,
                    show=False,
                    provenance_spec=script(
                        ScriptCodeOperation(
                            label="Copy child input",
                            code=f"derived = {script_input.name}",
                        ),
                        start_label="Start from child input",
                        active_name="derived",
                        script_inputs=(script_input,),
                    ),
                )
            )

        root = manager._tool_graph.root_wrappers[0]
        reload_source = root.slicer_area._reload
        reload_calls = 0

        def count_reload() -> bool:
            nonlocal reload_calls
            reload_calls += 1
            return reload_source()

        monkeypatch.setattr(root.slicer_area, "_reload", count_reload)
        updated = source + 100.0
        updated.to_netcdf(source_path, engine="h5netcdf")
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)
        qtbot.wait_until(
            lambda: all(
                manager._child_node(uid).source_state == "stale" for uid in child_uids
            ),
            timeout=5000,
        )

        manager.tree_view.selectionModel().clearSelection()
        select_tools(manager, derived_targets)
        manager.reload_selected()

        assert reload_calls == 1
        for target, indexer in zip(derived_targets, slices, strict=True):
            xr.testing.assert_identical(fetch(target), updated.isel(x=indexer))


def test_selected_managed_tools_dedupe_shared_file_input(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    inputs = (
        xr.DataArray([1.0, 2.0], dims="x", name="value"),
        xr.DataArray([10.0, 20.0], dims="x", name="value"),
        xr.DataArray([100.0, 200.0], dims="x", name="value"),
    )
    paths = tuple(tmp_path / name for name in ("shared.h5", "left.h5", "right.h5"))
    for value, path in zip(inputs, paths, strict=True):
        value.to_netcdf(path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for index, (value, path) in enumerate(
            zip(inputs, paths, strict=True),
            start=1,
        ):
            itool(
                value,
                manager=True,
                file_path=path,
                load_func=(
                    xr.load_dataarray,
                    {"engine": "h5netcdf"},
                    FileDataSelection(kind="dataarray"),
                ),
            )
            qtbot.wait_until(
                lambda expected_count=index: manager.ntools == expected_count,
                timeout=5000,
            )

        first = _ReloadCountingManagedSumTool(inputs[0] + inputs[1])
        first_uid = manager.add_childtool(
            first,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        second = _ReloadCountingManagedSumTool(inputs[0] + inputs[2])
        second_uid = manager.add_childtool(
            second,
            script_inputs={"data": 0, "weights": 2},
            show=False,
        )

        reload_calls: list[int] = []
        for root in manager._tool_graph.root_wrappers.values():
            original_reload = root.slicer_area._reload

            def track_reload(
                *,
                index: int = root.index,
                reload_source: Callable[[], bool] = original_reload,
            ) -> bool:
                reload_calls.append(index)
                return reload_source()

            monkeypatch.setattr(root.slicer_area, "_reload", track_reload)

        updated = tuple(value + 1000.0 for value in inputs)
        for value, path in zip(updated, paths, strict=True):
            value.to_netcdf(path, engine="h5netcdf")

        manager.tree_view.selectionModel().clearSelection()
        select_child_tool(manager, first_uid)
        select_child_tool(manager, second_uid)
        manager.reload_selected()

        assert reload_calls.count(0) == 1
        assert reload_calls.count(1) == 1
        assert reload_calls.count(2) == 1
        xr.testing.assert_equal(first.tool_data, updated[0] + updated[1])
        xr.testing.assert_equal(second.tool_data, updated[0] + updated[2])
        assert first.update_calls == 1
        assert second.update_calls == 1


def test_managed_multi_input_reload_rejects_mixed_file_and_raw_sources(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x", name="value")
    weights = xr.DataArray([10.0, 20.0], dims="x", name="value")
    file_path = tmp_path / "data.h5"
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
        itool(weights, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        tool = _ReloadCountingManagedSumTool(data + weights)
        child_uid = manager.add_childtool(
            tool,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )
        child_node = manager._child_node(child_uid)

        updated_data = data + 100.0
        updated_data.to_netcdf(file_path, engine="h5netcdf")

        assert child_node.reload_unavailable_reason() is not None
        assert not child_node.can_reload_source_data()
        assert not child_node.reload_source_data()
        xr.testing.assert_identical(fetch(0), data)
        xr.testing.assert_identical(tool.tool_data, data + weights)
        assert tool.update_calls == 0


def test_selected_upstream_reload_updates_managed_descendant_once(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    other = xr.DataArray([10.0, 20.0], dims="x")

    with manager_context() as manager:
        manager.show()
        itool([data, other], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        source_index = manager._lineage_controller._show_multi_input_script_result(
            data + other,
            (0, 1),
            operation_label="Add inputs",
            operation_code="derived = data_0 + data_1",
        )
        assert source_index == 2
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        consumer = _ManagedUnaryTool(data + other)
        consumer_uid = manager.add_childtool(
            consumer,
            script_inputs={"data": source_index},
            show=False,
        )
        consumer.set_script_inputs(
            consumer.script_inputs,
            primary_input="data",
            auto_update=True,
        )
        descendant = _ManagedUnaryTool(consumer.tool_data)
        descendant_uid = manager.add_childtool(
            descendant,
            script_inputs={"data": consumer_uid},
            show=False,
        )

        updated = data + 100.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)
        source = manager._node_for_target(source_index)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(source.uid) == "changed",
            timeout=5000,
        )

        manager.tree_view.selectionModel().clearSelection()
        select_tools(manager, [source_index])
        select_child_tool(manager, descendant_uid)
        manager.reload_selected()

        expected = updated + other
        qtbot.wait_until(
            lambda: (
                consumer.source_state == "fresh" and descendant.source_state == "fresh"
            ),
            timeout=5000,
        )
        xr.testing.assert_equal(consumer.tool_data, expected)
        xr.testing.assert_equal(descendant.tool_data, expected)
        assert consumer.update_calls == 1
        assert descendant.update_calls == 1


def test_managed_input_reload_cancellation_preserves_state(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    recorded_spec = script(
        ScriptCodeOperation(
            label="Evaluate recorded expression",
            code=("import os\nwith open(os.devnull):\n    pass\nderived = data"),
        ),
        start_label="Start from data",
        active_name="derived",
        script_inputs=(ScriptInput(name="data", label="Input"),),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False, provenance_spec=recorded_spec)

        tool = _ManagedUnaryTool(data)
        child_uid = manager.add_childtool(
            tool,
            script_inputs={"data": 0},
            show=False,
        )
        tool.set_script_inputs(
            tool.script_inputs,
            primary_input="data",
            state="stale",
        )
        monkeypatch.setattr(
            manager._lineage_controller,
            "_resolve_live_script_input_for_reload",
            lambda *_args, **_kwargs: None,
        )

        def _cancel_trust(*_args, **_kwargs) -> None:
            raise manager_widgets._TrustedProvenanceReplayCancelled

        monkeypatch.setattr(
            manager._lineage_controller,
            "_authorize_provenance_execution",
            _cancel_trust,
        )

        assert not manager._child_node(child_uid).reload_source_data()
        assert tool.source_state == "stale"


def test_duplicate_subtree_rebases_managed_input_dependencies(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    weights = xr.DataArray([3.0, 4.0], dims="x")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in (data, weights):
            root = itool(value, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        tool = _ManagedSumTool(data, weights)
        original_uid = manager.add_childtool(
            tool,
            script_inputs={"data": 0, "weights": 1},
            show=False,
        )

        original_parent = manager._node_for_target(0)
        duplicated_index = manager.duplicate_imagetool(0)
        duplicated_parent = manager._node_for_target(duplicated_index)
        assert duplicated_parent.snapshot_token == original_parent.snapshot_token
        assert (
            duplicated_parent.source_snapshot_token
            == original_parent.source_snapshot_token
        )
        assert len(duplicated_parent._childtool_indices) == 1
        duplicated_uid = duplicated_parent._childtool_indices[0]
        duplicated_tool = manager.get_childtool(duplicated_uid)
        duplicated_inputs = {item.name: item for item in duplicated_tool.script_inputs}

        assert duplicated_inputs["data"].node_uid == duplicated_parent.uid
        assert (
            duplicated_inputs["data"].node_snapshot_token
            == duplicated_parent.snapshot_token
        )
        weights_node = manager._node_for_target(1)
        assert duplicated_inputs["weights"].node_uid == weights_node.uid
        assert (
            duplicated_inputs["weights"].node_snapshot_token
            == weights_node.snapshot_token
        )
        assert manager.dependency_status_for_uid(duplicated_uid) == "current"
        assert duplicated_tool.source_state == "fresh"

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, data + 10.0)
        qtbot.wait_until(lambda: tool.source_state == "stale", timeout=5000)

        assert manager.dependency_status_for_uid(original_uid) == "changed"
        assert manager.dependency_status_for_uid(duplicated_uid) == "current"
        assert duplicated_tool.source_state == "fresh"


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

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
            show=False,
        )
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
        parent_tool.set_provenance_spec(
            script(
                start_label="Start from source data",
                seed_code="derived = source_data",
                active_name="derived",
            )
        )
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

        child_node = manager._child_node(child_uid)
        displayed_spec = child_node.passive_displayed_provenance_spec
        assert displayed_spec is not None
        assert displayed_spec.kind == "script"
        assert len(displayed_spec.script_inputs) == 1
        assert displayed_spec.script_inputs[0].node_uid == parent_node.uid

        start_item = manager.metadata_derivation_list.conceptual_item(0)
        input_item = manager.metadata_derivation_list.conceptual_item(1)
        assert start_item is not None
        assert input_item is not None
        start_row = start_item.data(manager_widgets._METADATA_DERIVATION_ROW_ROLE)
        input_row = input_item.data(manager_widgets._METADATA_DERIVATION_ROW_ROLE)
        assert isinstance(start_row, _ProvenanceDisplayRow)
        assert isinstance(input_row, _ProvenanceDisplayRow)
        assert start_row.replay_ref is not None
        assert start_row.replay_ref.kind == "start"
        assert input_row.replay_ref is not None
        assert input_row.replay_ref.kind == "script_input"
        assert input_row.replay_ref.script_input_index == 0
        assert input_row.script_input_path == ()

        monkeypatch.setattr(
            type(parent_node),
            "current_source_data",
            original_current_source_data,
        )
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        namespace = _exec_generated_code(
            copied,
            {"source_data": parent_tool.slicer_area.data.copy(deep=True)},
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

        root_node.set_detached_provenance(
            updated_root_spec,
            replay_source_data=None,
        )
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
            selection(IselOperation(kwargs={"x": slice(2, 4)})),
            replay_source_data=None,
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
            selection(IselOperation(kwargs={"x": slice(4, 6)})),
            replay_source_data=None,
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
            selection(IselOperation(kwargs={"x": slice(1, 3)})),
            replay_source_data=None,
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
            selection(IselOperation(kwargs={"x": slice(2, 4)})),
            replay_source_data=None,
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
            self.set_script_inputs(
                (ScriptInput(name="data", data_role="source"),),
                primary_input="data",
            )

        @property
        def tool_status(self) -> _DeferredToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _DeferredToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
            self.pending_data = inputs["data"]
            self._defer_source_refresh()
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
        parent_uid = manager.add_childtool(
            parent_tool,
            script_inputs={"data": 0},
            show=False,
        )

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

        assert (
            manager._lineage_controller._refresh_source_chain_to_uid(leaf_uid) is False
        )
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
        parent_tool.set_provenance_spec(
            script(
                start_label="Start from mesh data",
                seed_code="derived = mesh_data",
                active_name="derived",
            )
        )
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
            {"mesh_data": parent_tool.slicer_area.data.copy(deep=True)},
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
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    t = np.linspace(0.0, 4.0, 25)
    y = np.arange(3)
    fit_input = xr.DataArray(
        np.stack([((1.0 + 0.5 * index) * np.exp(-t / 2.0)) for index in y]),
        dims=("y", "t"),
        coords={"y": y, "t": t},
        name="decay2d",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(fit_input, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        parent_tool = manager.get_imagetool(0)
        parent_tool.set_provenance_spec(
            script(
                ScriptCodeOperation(
                    label="Prepare parent data",
                    code="prepared_parent = decay_data + 1",
                ),
                start_label="Start from test data",
                active_name="prepared_parent",
            )
        )

        model = erlab.analysis.fit.models.PolynomialModel(degree=1)
        child_uid, child = make_fit2d_child(manager, 0, model)
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

        def includes_parent_offset(code: str) -> bool:
            return any(
                isinstance(node, ast.BinOp)
                and isinstance(node.left, ast.Name)
                and node.left.id == "decay_data"
                and isinstance(node.op, ast.Add)
                and isinstance(node.right, ast.Constant)
                and node.right.value == 1
                for node in ast.walk(ast.parse(code))
            )

        assert includes_parent_offset(values_code)
        assert includes_parent_offset(second_values_code)
        assert includes_parent_offset(stderr_code)

        for code, active_name in (
            (values_code, "parameter_values"),
            (second_values_code, "parameter_values"),
            (stderr_code, "parameter_stderr"),
        ):
            namespace = _exec_generated_code(
                code,
                {"decay_data": fit_input.copy(deep=True)},
            )
            generated = namespace[active_name]
            assert isinstance(generated, xr.DataArray)
            assert generated.dims == ("y",)

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


def test_manager_file_backed_fit2d_parameter_full_code_is_self_contained(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    t = np.linspace(0.0, 4.0, 25)
    y = np.arange(3)
    fit_input = xr.DataArray(
        np.stack([((1.0 + 0.5 * index) * np.exp(-t / 2.0)) for index in y]),
        dims=("y", "t"),
        coords={"y": y, "t": t},
        name="decay2d",
    )
    source_path = tmp_path / "decay.h5"
    workspace_path = tmp_path / "fit.itws"
    fit_input.to_netcdf(source_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            fit_input,
            manager=True,
            file_path=source_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        model = erlab.analysis.fit.models.PolynomialModel(degree=1)
        child_uid, child = make_fit2d_child(manager, 0, model)
        child.timeout_spin.setValue(30.0)
        child.nfev_spin.setValue(0)
        child.y_index_spin.setValue(child.y_min_spin.value())
        child._run_fit_2d("up")
        qtbot.wait_until(
            lambda: (
                child._fit_thread is None
                and child._fit_2d_total == 0
                and not child._fit_2d_indices
            ),
            timeout=10000,
        )

        parameter_name = next(iter(child._params))
        parameter_index = child.param_plot_combo.findText(parameter_name)
        assert parameter_index >= 0
        child.param_plot_combo.setCurrentIndex(parameter_index)
        expected = child._param_plot_dataarray(parameter_name, stderr=False)
        child.param_plot._show_parameter_values()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]

        manager._workspace_controller.saving._save_workspace_document(workspace_path)
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert output_uid in manager._tool_graph.nodes

        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        assert str(source_path) in copied
        namespace = _exec_generated_code(copied, {})
        generated = namespace["parameter_values"]
        assert isinstance(generated, xr.DataArray)
        xr.testing.assert_identical(generated, expected)


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
            self.set_script_inputs(
                (ScriptInput(name="data", data_role="source"),),
                primary_input="data",
            )

        @property
        def tool_status(self) -> _OutputToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _OutputToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
            self.refreshed_inputs.append(inputs["data"])
            self._data = inputs["data"]
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
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
            show=False,
        )

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

        assert (
            manager._lineage_controller._refresh_source_chain_to_uid(output_uid) is True
        )
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
    accept_dialog,
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
        child_tool.slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(
            lambda: len(manager._child_node(child_uid)._childtool_indices) == 2,
            timeout=5000,
        )
        nested_image_uid = next(
            uid
            for uid in manager._child_node(child_uid)._childtool_indices
            if manager._is_imagetool_target(uid)
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        root_uid = parent.uid

        def _rename_mixed_targets(dialog: _RenameDialog) -> None:
            assert set(dialog._new_name_lines) == {root_uid, child_uid}
            dialog._new_name_lines[root_uid].setText("renamed_root")
            dialog._new_name_lines[child_uid].setText("renamed_child")

        accept_dialog(
            manager.rename_action.trigger,
            pre_call=_rename_mixed_targets,
        )
        assert parent.name == "renamed_root"
        assert manager._child_node(child_uid).name == "renamed_child"

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

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        select_child_tool(manager, nested_image_uid)
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        child_name = manager._child_node(child_uid).name
        nested_image_name = manager._child_node(nested_image_uid).name

        def _prepare_cancelled_rename(dialog: _RenameDialog) -> None:
            assert set(dialog._new_name_lines) == {child_uid, nested_image_uid}
            dialog._new_name_lines[child_uid].setText("cancelled_child")
            dialog._new_name_lines[nested_image_uid].setText("cancelled_nested")

        accept_dialog(
            manager.rename_action.trigger,
            pre_call=_prepare_cancelled_rename,
            accept_call=lambda dialog: dialog.reject(),
        )
        assert manager._child_node(child_uid).name == child_name
        assert manager._child_node(nested_image_uid).name == nested_image_name

        def _rename_nested_targets(dialog: _RenameDialog) -> None:
            dialog._new_name_lines[child_uid].setText("renamed_child_again")
            dialog._new_name_lines[nested_image_uid].setText("renamed_nested")

        accept_dialog(
            manager.rename_action.trigger,
            pre_call=_rename_nested_targets,
        )
        assert manager._child_node(child_uid).name == "renamed_child_again"
        assert manager._child_node(nested_image_uid).name == "renamed_nested"


def test_manager_batch_rename_tracks_uids_across_root_reindex(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        itool([test_data, test_data.copy(deep=True)], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        first = manager._tool_graph.root_wrappers[0]
        second = manager._tool_graph.root_wrappers[1]
        dialog = _RenameDialog(manager)
        qtbot.addWidget(dialog)
        dialog.set_names([first.uid, second.uid], [first.name, second.name])
        dialog._new_name_lines[first.uid].setText("removed")
        dialog._new_name_lines[second.uid].setText("surviving")

        manager.remove_imagetool(0)
        manager.reindex()
        dialog.accept()

        assert manager._tool_graph.root_wrappers[0].uid == second.uid
        assert manager._tool_graph.root_wrappers[0].name == "surviving"


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
            manager._lineage_controller._parent_source_data_for_uid(nested_uid),
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


def test_manager_promote_live_child_retains_replay_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(np.arange(6.0), dims=("x",), name="source")
    source_spec = full_data(
        AssignAttrsOperation(attrs={"order": "first"}),
        AssignAttrsOperation(attrs={"order": "second"}),
    )

    with manager_context() as manager:
        parent_tool = itool(source, manager=False, execute=False)
        assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
        parent_index = manager.add_imagetool(parent_tool, show=False)
        expected_source = (
            manager._node_for_target(parent_index).current_public_data().copy(deep=True)
        )

        child_tool = itool(source_spec.apply(source), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            parent_index,
            show=False,
            source_spec=source_spec,
        )

        promoted_index = manager.promote_child_imagetool(child_uid)
        promoted = manager._tool_graph.root_wrappers[promoted_index]
        assert promoted.replay_source_data is not None
        xr.testing.assert_identical(promoted.replay_source_data, expected_source)

        select_tools(manager, [promoted_index])
        manager._update_info(uid=promoted.uid)
        assert manager._provenance_edit_controller.can_reorder_steps()[0]


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

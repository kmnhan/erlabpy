import pathlib
import sys
import types
import typing
import warnings
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from IPython.core.interactiveshell import InteractiveShell
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._console as manager_console
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.utils
from erlab.interactive.imagetool import (
    _kspace_conversion,
    _provenance_framework,
    itool,
    provenance,
)
from erlab.interactive.imagetool.manager import fetch
from erlab.interactive.imagetool.manager._console import ToolNamespace
from erlab.interactive.imagetool.manager._details_panel import _DetailsPanelController
from erlab.interactive.imagetool.manager._dialogs import _ConcatDialog
from erlab.interactive.imagetool.manager._tool_graph import _ManagerToolGraph

from .helpers import (
    _exec_generated_code,
    click_tree_view_pos,
    console_helper,
    console_helper_dependency,
    dependency_status_badge,
    metadata_detail_labels,
    metadata_detail_map,
    select_child_tool,
    select_tools,
)

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool.manager._modelview import (
        _ImageToolWrapperItemDelegate,
    )


def _record_reload_unavailable_dialog(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reasons: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "_show_reload_unavailable_dialog",
        lambda _parent, reason: reasons.append(reason),
    )
    return reasons


def test_manager_console(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        manager.show()
        manager.activateWindow()

        manager._data_recv([data, data], kwargs={"link": True, "link_colors": True})
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Open console
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        def _get_last_output_contents():
            return manager.console._console_widget.kernel_manager.kernel.shell.user_ns[
                "_"
            ]

        # Test delayed import
        manager.console._console_widget.execute("era")
        assert _get_last_output_contents() == erlab.analysis

        # Loader state should persist across console cells.
        manager.console._console_widget.execute("erlab.io.set_loader('merlin')")
        manager.console._console_widget.execute("erlab.io.loaders.current_loader.name")
        assert _get_last_output_contents() == "merlin"

        # Test repr
        manager.console._console_widget.execute("tools")
        assert str(_get_last_output_contents()) == "0: \n1: "
        manager.console._console_widget.execute("tools[0]")
        manager.console._console_widget.execute(
            "isinstance(tools[0].data, xr.DataArray)"
        )
        assert _get_last_output_contents() is True
        manager.console._console_widget.execute(
            "lst = [1]\nalias = lst\nlst += [2]",
        )
        shell = manager.console._console_widget.kernel_manager.kernel.shell
        assert shell.user_ns["lst"] == [1, 2]
        assert shell.user_ns["alias"] is shell.user_ns["lst"]

        # Select all
        select_tools(manager, list(manager._tool_graph.root_wrappers.keys()))
        manager.console._console_widget.execute("tools.selected_data")
        selected_data = _get_last_output_contents()
        expected_data = [
            wrapper.imagetool.slicer_area._data
            for wrapper in manager._tool_graph.root_wrappers.values()
        ]
        assert len(selected_data) == len(expected_data)
        for result, expected in zip(selected_data, expected_data, strict=True):
            xr.testing.assert_identical(result, expected)

        # Test storing with ipython
        accept_dialog(manager.store_action.trigger)
        manager.console._console_widget.execute(r"%store -d data_0 data_1")

        # Test setting data
        manager.console._console_widget.execute(
            "tools[1].data = xr.DataArray("
            "np.arange(25).reshape((5, 5)) * 2, "
            "dims=['x', 'y'], "
            "coords={'x': np.arange(5), 'y': np.arange(5)}"
            ")",
        )
        xr.testing.assert_identical(manager.get_imagetool(1).slicer_area.data, data * 2)

        manager.console._console_widget.execute("tools[1].data += 1")
        xr.testing.assert_identical(
            manager.get_imagetool(1).slicer_area.data, data * 2 + 1
        )

        manager.console._console_widget.execute("tools[1].data[0, 0] = -5.0")
        assert float(manager.get_imagetool(1).slicer_area._data.values[0, 0]) == -5.0
        assert float(manager.get_imagetool(1).slicer_area.data.values[0, 0]) == -5.0
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, data)

        # Remove all tools
        select_tools(manager, list(manager._tool_graph.root_wrappers.keys()))
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        # Test repr
        manager.console._console_widget.execute("tools")
        assert str(_get_last_output_contents()) == "No tools"

        # Test magic command: itool
        manager.console._console_widget.kernel_manager.kernel.shell.user_ns[
            "example_data"
        ] = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["alpha", "eV"],
            coords={"alpha": np.arange(5), "eV": np.arange(5)},
        )
        manager.console._console_widget.execute(r"%itool example_data --cmap viridis")
        qtbot.wait_until(lambda: manager.ntools == 1)
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        # Destroy console
        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_console_show_event_defers_kernel_initialization(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.ensure_console_initialized()
        scheduled: list[
            tuple[
                QtCore.QObject,
                int,
                Callable[[], None],
                tuple[QtCore.QObject | None, ...],
            ]
        ] = []
        initialized: list[str] = []

        def fake_single_shot(
            receiver: QtCore.QObject,
            msec: int,
            callback: Callable[[], None],
            *guards: QtCore.QObject | None,
        ) -> None:
            scheduled.append((receiver, msec, callback, guards))

        monkeypatch.setattr(erlab.interactive.utils, "single_shot", fake_single_shot)
        monkeypatch.setattr(
            manager.console._console_widget,
            "initialize_kernel",
            lambda: initialized.append("kernel"),
        )
        monkeypatch.setattr(
            manager.console._console_widget,
            "_update_colors",
            lambda: initialized.append("colors"),
        )

        assert not manager.console.eventFilter(
            manager.console._console_widget,
            QtCore.QEvent(QtCore.QEvent.Type.Show),
        )

        assert initialized == []
        assert len(scheduled) == 1
        receiver, msec, callback, guards = scheduled[0]
        assert receiver is manager.console._console_widget
        assert msec == 0
        assert guards == (manager.console,)

        callback()
        assert initialized == ["kernel", "colors"]


def test_manager_console_queued_show_callback_ignores_shutdown_kernel(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.ensure_console_initialized()
        scheduled: list[
            tuple[
                QtCore.QObject,
                int,
                Callable[[], None],
                tuple[QtCore.QObject | None, ...],
            ]
        ] = []
        initialized: list[str] = []

        def fake_single_shot(
            receiver: QtCore.QObject,
            msec: int,
            callback: Callable[[], None],
            *guards: QtCore.QObject | None,
        ) -> None:
            scheduled.append((receiver, msec, callback, guards))

        monkeypatch.setattr(erlab.interactive.utils, "single_shot", fake_single_shot)
        monkeypatch.setattr(
            manager.console._console_widget,
            "initialize_kernel",
            lambda: initialized.append("kernel"),
        )
        monkeypatch.setattr(
            manager.console._console_widget,
            "_update_colors",
            lambda: initialized.append("colors"),
        )

        assert not manager.console.eventFilter(
            manager.console._console_widget,
            QtCore.QEvent(QtCore.QEvent.Type.Show),
        )
        assert len(scheduled) == 1

        manager.console._console_widget.shutdown_kernel()
        scheduled[0][2]()

        assert initialized == []


def test_manager_console_shutdown_blocks_late_kernel_start(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.ensure_console_initialized()
        console_widget = manager.console._console_widget

        console_widget.shutdown_kernel()
        console_widget.initialize_kernel()
        console_widget.execute("late_value = 1")

        assert console_widget.kernel_manager.kernel is None


def test_manager_console_event_filter_ignores_invalid_events(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.ensure_console_initialized()
        scheduled: list[object] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "single_shot",
            lambda *args: scheduled.append(args),
        )

        assert not manager.console.eventFilter(manager.console._console_widget, None)

        other = QtWidgets.QWidget()
        qtbot.addWidget(other)
        assert not manager.console.eventFilter(
            other,
            QtCore.QEvent(QtCore.QEvent.Type.Show),
        )

        assert not manager.console.eventFilter(
            manager.console._console_widget,
            QtCore.QEvent(QtCore.QEvent.Type.Hide),
        )
        assert scheduled == []

        initialized: list[str] = []
        monkeypatch.setattr(
            manager.console._console_widget,
            "initialize_kernel",
            lambda: initialized.append("kernel"),
        )
        monkeypatch.setattr(
            manager.console._console_widget,
            "_update_colors",
            lambda: initialized.append("colors"),
        )
        monkeypatch.setattr(
            erlab.interactive.utils, "qt_is_valid", lambda *_objs: False
        )
        manager.console._initialize_visible_console()
        assert initialized == []

        invalid_event = QtCore.QEvent(QtCore.QEvent.Type.Show)
        original_qt_is_valid = erlab.interactive.utils.qt_is_valid

        def fake_qt_is_valid(*objects: object) -> bool:
            if any(obj is invalid_event for obj in objects):
                return False
            return original_qt_is_valid(*objects)

        monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", fake_qt_is_valid)
        assert not manager.console.eventFilter(
            manager.console._console_widget,
            invalid_event,
        )

        class DeletedEvent:
            def type(self) -> QtCore.QEvent.Type:
                raise RuntimeError("wrapped C/C++ object has been deleted")

        monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_objs: True)
        assert not manager.console.eventFilter(
            manager.console._console_widget,
            typing.cast("QtCore.QEvent", DeletedEvent()),
        )
        assert scheduled == []


def test_manager_console_handles_use_filtered_display_data(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )
    expected = operation.apply(data, parent_data=data)

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tool = manager.get_imagetool(0)
        tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        tools = manager_console.ToolsNamespace(manager)
        handle = tools[0]
        assert handle is not None
        xr.testing.assert_identical(handle.data, expected)
        assert (
            handle._console_provenance_spec(
                active_name="derived",
                label="Assign filtered console result",
            )
            == manager._tool_graph.root_wrappers[0].displayed_provenance_spec
        )

        derived = handle + 1.0
        xr.testing.assert_identical(derived.data, expected + 1.0)
        spec = derived._console_provenance_spec(
            active_name="derived",
            label="Assign filtered console result",
        )
        assert spec is not None
        code = spec.display_code()
        assert code is not None
        namespace = _exec_generated_code(
            code,
            {"data": data.copy(deep=True), "data_0": data.copy(deep=True)},
        )
        xr.testing.assert_identical(namespace["derived"], expected + 1.0)


def test_manager_console_kspace_set_normal_returns_derived_provenance() -> None:
    data = xr.DataArray(
        np.arange(27.0).reshape(3, 3, 3),
        dims=("alpha", "beta", "eV"),
        coords={
            "alpha": [-1.0, 0.0, 1.0],
            "beta": [-1.0, 0.0, 1.0],
            "eV": [-0.2, 0.0, 0.2],
            "xi": 0.0,
            "hv": 21.2,
        },
        attrs={
            "configuration": int(erlab.constants.AxesConfiguration.Type1),
            "sample_workfunction": 4.5,
        },
    )
    handle = manager_console._DerivedDataNamespace(
        None,
        data,
        "data_0",
        (
            provenance.ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=provenance.script(
                    start_label="Use input data",
                    seed_code="data_0 = data",
                    active_name="data_0",
                ).model_dump(mode="json"),
            ),
        ),
        copyable=True,
    )

    original_offsets = dict(data.kspace.offsets.items())

    derived = handle.kspace.set_normal(1.5, -0.5, delta=2.0)

    assert isinstance(derived, manager_console._DerivedDataNamespace)
    assert derived.data.kspace.offsets["delta"] == pytest.approx(2.0)
    for key, value in original_offsets.items():
        assert data.kspace.offsets[key] == pytest.approx(value)
    spec = derived._console_provenance_spec(
        active_name="derived",
        label="Assign kspace result",
    )
    assert spec is not None
    assert [operation.op for operation in spec.operations] == ["kspace_set_normal"]
    assert spec.operations[0].group is None
    code = spec.display_code()
    assert code is not None
    assert "derived = data.copy(deep=False)" in code
    assert "derived.kspace.set_normal(alpha=1.5, beta=-0.5, delta=2.0)" in code
    assert "sample_workfunction" not in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    assert namespace["derived"].kspace.offsets["delta"] == pytest.approx(2.0)
    for key, value in original_offsets.items():
        assert namespace["data"].kspace.offsets[key] == pytest.approx(value)

    converted = derived.kspace.convert()
    grouped_spec = converted._console_provenance_spec(
        active_name="derived",
        label="Assign kspace result",
    )
    assert grouped_spec is not None
    assert [operation.op for operation in grouped_spec.operations] == [
        "kspace_set_normal",
        "kspace_convert",
    ]
    assert provenance.operation_group_range(
        grouped_spec.operations,
        0,
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    ) == (0, 2)

    derived._set_console_name("intermediate")
    separately_converted = derived.kspace.convert()
    separate_spec = separately_converted._console_provenance_spec(
        active_name="derived",
        label="Assign kspace result",
    )
    assert separate_spec is not None
    assert [operation.op for operation in separate_spec.operations] == [
        "kspace_convert",
    ]
    assert separate_spec.operations[0].group is None


def test_macos_matplotlib_cursor_patch_applies_once(monkeypatch) -> None:
    class _Canvas:
        def set_cursor(self, cursor):
            raise AssertionError("original cursor setter should be patched")

    original_set_cursor = _Canvas.set_cursor
    backend_qt = types.SimpleNamespace(FigureCanvasQT=_Canvas)

    monkeypatch.setattr(sys, "platform", "darwin")

    def _import_module(name: str):
        if name == "matplotlib.backends.backend_qt":
            return backend_qt
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(
        erlab.interactive.utils.importlib, "import_module", _import_module
    )

    manager_console._patch_macos_matplotlib_qt_cursor()
    patched_set_cursor = _Canvas.set_cursor

    assert (
        getattr(_Canvas, erlab.interactive.utils._MPL_QT_CURSOR_PATCH_ATTR)
        is original_set_cursor
    )
    assert patched_set_cursor is not original_set_cursor
    assert _Canvas().set_cursor(object()) is None

    manager_console._patch_macos_matplotlib_qt_cursor()

    assert _Canvas.set_cursor is patched_set_cursor
    assert (
        getattr(_Canvas, erlab.interactive.utils._MPL_QT_CURSOR_PATCH_ATTR)
        is original_set_cursor
    )


def test_macos_matplotlib_cursor_patch_skips_non_macos(
    monkeypatch,
) -> None:
    class _Canvas:
        def set_cursor(self, cursor):
            return cursor

    original_set_cursor = _Canvas.set_cursor

    monkeypatch.setattr(sys, "platform", "linux")

    def _import_module(name: str):
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(
        erlab.interactive.utils.importlib, "import_module", _import_module
    )

    manager_console._patch_macos_matplotlib_qt_cursor()

    assert not hasattr(_Canvas, erlab.interactive.utils._MPL_QT_CURSOR_PATCH_ATTR)
    assert _Canvas.set_cursor is original_set_cursor


def test_resolve_console_namespace_patches_before_lazy_import(monkeypatch) -> None:
    events: list[str] = []
    literal = object()

    def _patch_cursor() -> None:
        events.append("patch")

    def _import_module(name: str):
        events.append(name)
        return f"module:{name}"

    monkeypatch.setattr(
        manager_console, "_patch_macos_matplotlib_qt_cursor", _patch_cursor
    )
    monkeypatch.setattr(manager_console.importlib, "import_module", _import_module)

    resolved = manager_console._resolve_console_namespace(
        {
            "literal": literal,
            "plt": "matplotlib.pyplot",
            "era": "erlab.analysis",
        }
    )

    assert events == ["patch", "matplotlib.pyplot", "erlab.analysis"]
    assert resolved == {
        "literal": literal,
        "plt": "module:matplotlib.pyplot",
        "era": "module:erlab.analysis",
    }


def test_tool_namespace_get_data_item(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        namespace = ToolNamespace(manager._tool_graph.root_wrappers[0])
        xr.testing.assert_identical(
            namespace._get_data_item((slice(None), 0)), test_data[:, 0]
        )


def test_tool_namespace_set_data_replaces_source(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        namespace = ToolNamespace(manager._tool_graph.root_wrappers[0])
        child = next(iter(manager._tool_graph.root_wrappers[0]._childtools.values()))
        updated = (test_data * 2).rename(test_data.name)

        namespace.data = updated

        xr.testing.assert_identical(parent_tool.slicer_area.data, updated)
        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)


def test_tool_namespace_set_data_item_marks_child_tools_stale(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        namespace = ToolNamespace(manager._tool_graph.root_wrappers[0])
        child = next(iter(manager._tool_graph.root_wrappers[0]._childtools.values()))

        namespace[(0, 0)] = -5.0

        assert float(parent_tool.slicer_area._data.values[0, 0]) == -5.0
        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)


def test_tool_namespace_set_filtered_data_item_uses_displayed_data(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = test_data.astype(float)
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={data.dims[0]: 1.0}
    )
    filtered = operation.apply(data, parent_data=data)

    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.apply_filter_operation(operation)
        namespace = ToolNamespace(manager._tool_graph.root_wrappers[0])
        expected = filtered.copy(deep=True)
        expected[(0, 0)] = -5.0

        namespace[(0, 0)] = -5.0

        assert parent_tool.slicer_area._accepted_filter_provenance_operation is None
        xr.testing.assert_identical(parent_tool.slicer_area.data, expected)
        provenance_spec = manager._tool_graph.root_wrappers[0].displayed_provenance_spec
        assert provenance_spec is not None
        display_code = provenance_spec.display_code()
        assert display_code is not None
        locals_ns = _exec_generated_code(
            display_code,
            {"data": data.copy(deep=True)},
        )
        xr.testing.assert_identical(
            locals_ns[provenance_spec.active_name or "derived"], expected
        )


def test_tool_namespace_set_unfiltered_data_item_avoids_display_copy(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = test_data.astype(float)

    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        source_data = parent_tool.slicer_area._data
        namespace = ToolNamespace(manager._tool_graph.root_wrappers[0])
        copy_calls = []
        original_copy = xr.DataArray.copy

        def _record_copy(self, *args, **kwargs):
            deep = kwargs.get("deep", args[0] if args else True)
            if self is source_data and deep is True:
                copy_calls.append(self)
            return original_copy(self, *args, **kwargs)

        monkeypatch.setattr(xr.DataArray, "copy", _record_copy)
        namespace[(0, 0)] = -5.0

        assert copy_calls == []
        assert float(parent_tool.slicer_area._data.values[0, 0]) == -5.0
        assert parent_tool.slicer_area._accepted_filter_provenance_operation is None


def test_tool_namespace_set_filtered_data_item_updates_child_provenance(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    operation = provenance.GaussianFilterOperation(sigma={"x": 1.0})
    filtered = operation.apply(data, parent_data=data)
    expected = filtered.copy(deep=True)
    expected[(0, 0)] = -5.0

    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)
        qtbot.wait_until(lambda: fetch(child_uid).identical(filtered), timeout=5000)

        namespace = ToolNamespace(manager._tool_graph.root_wrappers[0])
        namespace[(0, 0)] = -5.0

        qtbot.wait_until(
            lambda: (
                child_node.source_state == "fresh"
                and fetch(child_uid).identical(expected)
            ),
            timeout=5000,
        )
        child_spec = child_node.displayed_provenance_spec
        assert child_spec is not None
        display_code = child_spec.display_code()
        assert display_code is not None
        locals_ns = _exec_generated_code(
            display_code,
            {"data": data.copy(deep=True), "data_0": data.copy(deep=True)},
        )
        xr.testing.assert_identical(
            locals_ns[child_spec.active_name or "derived"], expected
        )


def test_tool_namespace_set_data_item_failure_keeps_child_tools_fresh(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        namespace = ToolNamespace(manager._tool_graph.root_wrappers[0])
        child = next(iter(manager._tool_graph.root_wrappers[0]._childtools.values()))

        with pytest.raises(IndexError, match="too many indices"):
            namespace[(0, 0, 0)] = -5.0

        assert child.source_state == "fresh"
        assert float(parent_tool.slicer_area._data.values[0, 0]) == 0.0
        assert manager._tool_graph.root_wrappers[0].provenance_spec is None


def test_manager_console_bare_expression_opens_provenance_root(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(9, dtype=float).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
        name="left",
    )
    data1 = data0 + 1.0
    data1.name = "right"

    with manager_context() as manager:
        manager.show()
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.console._console_widget.execute(
            "raw_diff = tools[0].data - tools[1].data"
        )
        assert manager.ntools == 2
        shell = manager.console._console_widget.kernel_manager.kernel.shell
        assert isinstance(shell.user_ns["raw_diff"], xr.DataArray)

        manager.console._console_widget.execute("tools[0] - tools[1]")
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data, data0 - data1
        )
        provenance = manager._tool_graph.root_wrappers[2].provenance_spec
        assert provenance is not None
        assert [source.name for source in provenance.script_inputs] == [
            "data_0",
            "data_1",
        ]
        assert [source.node_uid for source in provenance.script_inputs] == [
            manager._tool_graph.root_wrappers[0].uid,
            manager._tool_graph.root_wrappers[1].uid,
        ]
        assert [source.node_snapshot_token for source in provenance.script_inputs] == [
            manager._tool_graph.root_wrappers[0].snapshot_token,
            manager._tool_graph.root_wrappers[1].snapshot_token,
        ]
        assert (
            manager.dependency_status_for_uid(manager._tool_graph.root_wrappers[2].uid)
            == "current"
        )
        assert provenance.operations
        assert "data_0 - data_1" in typing.cast(
            "str", provenance.operations[-1].derivation_entry().code
        )

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        loaded = [
            wrapper.provenance_spec
            for wrapper in manager._tool_graph.root_wrappers.values()
            if wrapper.provenance_spec is not None
            and len(wrapper.provenance_spec.script_inputs) == 2
        ]
        assert len(loaded) == 1
        assert [source.name for source in loaded[0].script_inputs] == [
            "data_0",
            "data_1",
        ]
        derived_uid = manager._tool_graph.root_wrappers[2].uid
        assert manager.dependency_status_for_uid(derived_uid) == "current"

        original_difference = manager.get_imagetool(2).slicer_area.data.copy()
        manager.get_imagetool(0).slicer_area.replace_source_data(data0 + 10.0)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "changed",
            timeout=5000,
        )
        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            original_difference,
        )

        manager.remove_imagetool(1)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "missing",
            timeout=5000,
        )
        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            original_difference,
        )

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_console_child_window_divide_uses_public_nonuniform_dims(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60.0).reshape(3, 4, 5),
        dims=("sample_temp", "eV", "alpha"),
        coords={
            "sample_temp": [10.0, 20.5, 22.0],
            "eV": np.arange(4.0),
            "alpha": np.arange(5.0),
        },
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_tool = manager.get_imagetool(child_uid)
        tools = manager_console.ToolsNamespace(manager)

        assert tools[0].data.dims == ("sample_temp", "eV", "alpha")
        assert tools[0].children[0].data.dims == ("sample_temp", "eV")
        assert "sample_temp_idx" not in tools[0].data.dims
        assert "sample_temp_idx" not in tools[0].children[0].data.dims

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = tools[0] / tools[0].children[0]

        assert isinstance(result, manager_console._DerivedDataNamespace)
        assert not any(
            "Duplicate dimension names" in str(warning.message) for warning in caught
        )
        assert result.data.dims == ("sample_temp", "eV", "alpha")
        xr.testing.assert_identical(
            result.data,
            data / child_tool.slicer_area.displayed_data,
        )


@pytest.mark.parametrize("reserved_name", ["data", "derived", "tools", "data_0"])
def test_manager_console_rejects_reserved_result_names_for_provenance(
    reserved_name: str,
) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",))
    result = manager_console._DerivedDataNamespace(
        None,
        data + 1.0,
        "data_1 + 1.0",
        (provenance.ScriptInput(name="data_1", label="ImageTool 1"),),
        copyable=True,
    )

    result._set_console_name(reserved_name)

    operand = result._console_operand()
    assert [script_input.name for script_input in operand.script_inputs] == ["data_1"]


def test_manager_console_reserved_result_name_replays_without_shadowing() -> None:
    data0 = xr.DataArray(np.array([10.0]), dims=("x",))
    data1 = xr.DataArray(np.array([1.0]), dims=("x",))
    source = manager_console._DerivedDataNamespace(
        None,
        data0,
        "data_0",
        (provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
        copyable=True,
    )
    result = manager_console._DerivedDataNamespace(
        None,
        data1 + 1.0,
        "data_1 + 1.0",
        (provenance.ScriptInput(name="data_1", label="ImageTool 1"),),
        copyable=True,
    )
    result._set_console_name("data_0")

    combined = result + source
    spec = combined._console_provenance_spec(
        active_name="derived", label="Evaluate console expression"
    )

    assert spec is not None
    assert [script_input.name for script_input in spec.script_inputs] == [
        "data_1",
        "data_0",
    ]
    xr.testing.assert_identical(
        provenance.replay_script_provenance(spec, {"data_0": data0, "data_1": data1}),
        combined.data,
    )


def test_manager_console_helpers_preserve_nested_operands() -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    handle = manager_console._DerivedDataNamespace(
        None,
        data,
        "data_0",
        (provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
        copyable=True,
    )

    namespace = {"np": np}
    for value in (
        np.nan,
        np.inf,
        -np.inf,
        complex(np.nan, np.inf),
        (1, np.nan),
        [1, np.inf],
        {"offset": -np.inf},
    ):
        code, copyable = manager_console._literal_code(value)
        assert copyable
        evaluated = eval(code, namespace)  # noqa: S307
        np.testing.assert_equal(evaluated, erlab.utils.misc._convert_to_native(value))

    assert manager_console._literal_code(object())[1] is False
    assert manager_console._derived_operand_code("data_0 + data_1").startswith("(")
    assert manager_console._derived_operand_code("data_0 == data_1").startswith("(")
    assert manager_console._derived_operand_code("data_0.sel(x=0)") == "data_0.sel(x=0)"

    nested = {"left": [handle, slice(0, 1)], "right": (handle,)}
    assert manager_console._first_console_handle(nested) is handle
    unwrapped = manager_console._unwrap_console_value(nested)
    xr.testing.assert_identical(unwrapped["left"][0], data)
    xr.testing.assert_identical(unwrapped["right"][0], data)

    operand = manager_console._operand_from_value(nested)
    assert [script_input.name for script_input in operand.script_inputs] == ["data_0"]
    assert operand.copyable
    assert operand.value["left"][1] == slice(0, 1)


def test_manager_console_namespace_protocols_and_proxies() -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    handle = manager_console._DerivedDataNamespace(
        None,
        data,
        "data_0",
        (provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
        copyable=True,
    )

    for result, expected in (
        (-handle, -data),
        (+handle, +data),
        (abs(handle - 2.0), abs(data - 2.0)),
        (np.sin(handle), np.sin(data)),
        (handle[{"x": 0}], data[{"x": 0}]),
        (handle.x, data.x),
        (handle.coarsen(y=2).mean(), data.coarsen(y=2).mean()),
    ):
        assert isinstance(result, manager_console._DerivedDataNamespace)
        xr.testing.assert_identical(result.data, expected)

    bool_handle = manager_console._DerivedDataNamespace(
        None,
        xr.DataArray([True, False], dims=("x",)),
        "data_1",
        (provenance.ScriptInput(name="data_1", label="ImageTool 1"),),
        copyable=True,
    )
    inverted = ~bool_handle
    assert isinstance(inverted, manager_console._DerivedDataNamespace)
    xr.testing.assert_identical(inverted.data, xr.DataArray([False, True], dims=("x",)))

    assert handle.__array_ufunc__(np.add, "reduce", handle) is NotImplemented
    np.testing.assert_array_equal(np.asarray(handle), data.values)
    np.testing.assert_array_equal(handle.__array__(copy=True), data.values)

    coarsened = handle.coarsen(y=2).mean()
    spec = coarsened._console_provenance_spec(
        active_name="derived",
        label="Evaluate console expression",
    )
    assert spec is not None
    xr.testing.assert_identical(
        provenance.replay_script_provenance(spec, {"data_0": data}),
        coarsened.data,
    )

    module_proxy = manager_console._ConsoleModuleProxy(np, "np")
    assert isinstance(module_proxy.linalg, manager_console._ConsoleModuleProxy)
    assert module_proxy.float64 is np.float64
    assert module_proxy.pi == np.pi
    np.testing.assert_allclose(module_proxy.sin(np.array([0.0, np.pi / 2])), [0.0, 1.0])
    module_result = module_proxy.sin(handle)
    assert isinstance(module_result, manager_console._DerivedDataNamespace)
    xr.testing.assert_allclose(module_result.data, np.sin(data))

    def add_one(value):
        return value + 1

    function_proxy = manager_console._ConsoleFunctionProxy(add_one)
    xr.testing.assert_identical(function_proxy(data), data + 1)
    function_proxy.__name__ = ""
    function_result = function_proxy(handle)
    assert isinstance(function_result, manager_console._DerivedDataNamespace)
    xr.testing.assert_identical(function_result.data, data + 1)


def test_manager_console_operator_and_proxy_branches() -> None:
    data = xr.DataArray(
        np.arange(1, 5).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [0, 1]},
    )
    handle = manager_console._DerivedDataNamespace(
        None,
        data,
        "data_0",
        (provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
        copyable=True,
    )

    operations = (
        (lambda: 10 + handle, 10 + data),
        (lambda: 10 - handle, 10 - data),
        (lambda: 2 * handle, 2 * data),
        (lambda: handle @ data, data @ data),
        (lambda: handle.__rmatmul__(data), data @ data),
        (lambda: handle / 2, data / 2),
        (lambda: 10 / handle, 10 / data),
        (lambda: handle // 2, data // 2),
        (lambda: 10 // handle, 10 // data),
        (lambda: handle % 2, data % 2),
        (lambda: 10 % handle, 10 % data),
        (lambda: handle**2, data**2),
        (lambda: 2**handle, 2**data),
        (lambda: handle < 3, data < 3),
        (lambda: handle <= 3, data <= 3),
        (lambda: handle > 3, data > 3),
        (lambda: handle >= 3, data >= 3),
        (lambda: handle == 3, data == 3),
        (lambda: handle != 3, data != 3),
    )
    for operation, expected in operations:
        result = operation()
        assert isinstance(result, manager_console._DerivedDataNamespace)
        xr.testing.assert_identical(result.data, expected)

    int_handle = manager_console._DerivedDataNamespace(
        None,
        data.astype(int),
        "data_1",
        (provenance.ScriptInput(name="data_1", label="ImageTool 1"),),
        copyable=True,
    )
    for operation, expected in (
        (lambda: int_handle & 1, data.astype(int) & 1),
        (lambda: 1 & int_handle, 1 & data.astype(int)),
        (lambda: int_handle | 1, data.astype(int) | 1),
        (lambda: 1 | int_handle, 1 | data.astype(int)),
        (lambda: int_handle ^ 1, data.astype(int) ^ 1),
        (lambda: 1 ^ int_handle, 1 ^ data.astype(int)),
    ):
        result = operation()
        assert isinstance(result, manager_console._DerivedDataNamespace)
        xr.testing.assert_identical(result.data, expected)

    assert handle.qsel.__doc__
    assert handle._wrap_console_result(1, "one", (), copyable=True) == 1
    with pytest.raises(TypeError, match="not callable"):
        manager_console._ConsoleAccessorProxy(
            handle,
            types.SimpleNamespace(value=10),
            ("fake",),
            "data_0.fake",
        )()
    accessor_proxy = manager_console._ConsoleAccessorProxy(
        handle,
        types.SimpleNamespace(array=data, value=10),
        ("fake",),
        "data_0.fake",
    )
    accessor_array = accessor_proxy.array
    assert isinstance(accessor_array, manager_console._DerivedDataNamespace)
    xr.testing.assert_identical(accessor_array.data, data)
    assert accessor_proxy.value == 10
    coarsened_with_kwargs = handle.coarsen(y=2).mean(skipna=True)
    assert isinstance(coarsened_with_kwargs, manager_console._DerivedDataNamespace)
    xr.testing.assert_identical(
        coarsened_with_kwargs.data,
        data.coarsen(y=2).mean(skipna=True),
    )
    assert "mean" in dir(handle.coarsen(y=2))

    custom_module = types.ModuleType("custom_console_module")
    custom_module.__file__ = "custom_console_module.py"

    def scale(value, *, offset=0):
        return value + offset

    custom_module.scale = scale
    module_proxy = manager_console._ConsoleModuleProxy(custom_module, "custom")
    assert module_proxy.__file__ == "custom_console_module.py"
    module_result = module_proxy.scale(handle, offset=2)
    assert isinstance(module_result, manager_console._DerivedDataNamespace)
    xr.testing.assert_identical(module_result.data, data + 2)
    assert module_proxy == custom_module
    assert module_proxy == manager_console._ConsoleModuleProxy(custom_module, "alias")
    assert hash(module_proxy) == hash(custom_module)

    class ParentWithoutTools:
        _console_label = "Parent"

        @property
        def _console_tools(self):
            return None

    children = manager_console._ToolChildren(ParentWithoutTools())
    assert len(children) == 0
    assert list(children) == []
    assert "No child" in repr(children)
    with pytest.raises(LookupError):
        children[0]


def test_manager_console_tools_namespace_helper_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))

    class FakeManager:
        manager_index = 3

        def __init__(self) -> None:
            self._tool_graph = _ManagerToolGraph()
            self.added: list[tuple[xr.DataArray, typing.Any]] = []

        def add_imagetool(self, tool, **kwargs) -> None:
            self.added.append((tool.slicer_area.data, kwargs["provenance_spec"]))

    fake_manager = FakeManager()
    tools = manager_console.ToolsNamespace(fake_manager)
    assert repr(tools) == "No tools"
    assert tools._manager_argument_targets_this_manager(None)
    assert tools._manager_argument_targets_this_manager(True)
    assert tools._manager_argument_targets_this_manager(3)
    assert not tools._manager_argument_targets_this_manager(False)
    assert not tools._manager_argument_targets_this_manager(4)
    assert not tools._manager_argument_targets_this_manager("manager-3")
    tools.unbind_shell()
    tools._pre_run_cell()

    result_with_error = types.SimpleNamespace(
        error_before_exec=RuntimeError("before"),
        error_in_exec=None,
        result=None,
    )
    tools._post_run_cell(result_with_error)

    handle_without_provenance = manager_console._DerivedDataNamespace(
        None,
        data,
        "data_0",
        (),
        copyable=False,
    )
    monkeypatch.setattr(
        handle_without_provenance, "_console_provenance_spec", lambda **_kwargs: None
    )
    assert not tools._show_handle(
        handle_without_provenance,
        active_name="derived",
        label="Evaluate console expression",
    )

    monkeypatch.setattr(erlab.interactive, "itool", lambda *_args, **_kwargs: object())
    shown = tools._show_dataarray_with_provenance(
        data,
        provenance.script(
            provenance.ScriptCodeOperation(label="Copy", code="derived = data_0"),
            start_label="Run script",
            active_name="derived",
            script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
        ),
        execute=True,
    )
    assert not shown
    assert not fake_manager.added


def test_manager_console_tree_namespace_helper_branches() -> None:
    class FakeNode:
        def __init__(
            self,
            uid: str,
            *,
            parent_uid: str | None = None,
            is_imagetool: bool = False,
            name: str = "",
            kind: str | None = "tool",
        ) -> None:
            self.uid = uid
            self.parent_uid = parent_uid
            self.is_imagetool = is_imagetool
            self.name = name
            self.type_badge_text = kind
            self._childtool_indices: list[str] = []
            self.added_time_display = "2026-06-17 12:00:00 UTC (+0000)"
            self.imagetool = object()
            self.window = object()
            self.calls: list[str] = []
            self.non_callable = "value"
            self.manager = None

        def show(self) -> None:
            self.calls.append("show")

        def hide(self) -> None:
            self.calls.append("hide")

        def dispose(self) -> None:
            self.calls.append("dispose")

        def callable_method(self) -> str:
            return "called"

    class FakeManager:
        def __init__(self) -> None:
            self._tool_graph = types.SimpleNamespace(root_wrappers={}, nodes={})

        def get_imagetool(self, uid: str) -> object:
            return self._tool_graph.nodes[uid].imagetool

    manager = FakeManager()
    root = FakeNode("root", is_imagetool=True, name="root-data")
    tool = FakeNode("tool", parent_uid="root", name="Tool Row", kind="customtool")
    child = FakeNode("child", parent_uid="tool", is_imagetool=True, name="child-data")
    unnamed_tool = FakeNode("unnamed", parent_uid="tool", name="", kind=None)
    wrong_parent = FakeNode("wrong", parent_uid="other", is_imagetool=True)
    skip_parent = FakeNode("skip", is_imagetool=True)
    skip_child = FakeNode("skip-child", parent_uid="skip", is_imagetool=True)
    detached = FakeNode("detached", is_imagetool=True, name="detached")
    orphan = FakeNode("orphan", parent_uid="missing", is_imagetool=True)
    omitted = FakeNode("omitted", parent_uid="root", is_imagetool=True)
    root._childtool_indices = ["tool"]
    tool._childtool_indices = ["child", "unnamed"]
    skip_parent._childtool_indices = ["missing-node", "wrong", "skip-child"]
    manager._tool_graph.root_wrappers[0] = root
    manager._tool_graph.nodes.update(
        {
            "root": root,
            "tool": tool,
            "child": child,
            "unnamed": unnamed_tool,
            "wrong": wrong_parent,
            "skip": skip_parent,
            "skip-child": skip_child,
            "detached": detached,
            "orphan": orphan,
            "omitted": omitted,
        }
    )
    for node in manager._tool_graph.nodes.values():
        node.manager = manager
    tools = manager_console.ToolsNamespace(manager)

    root_handle = manager_console.ToolNamespace(root, tools)
    assert root_handle.window is root.imagetool
    assert root_handle.uid == "root"
    assert root_handle.name == "root-data"
    assert root_handle.kind == "ImageTool"
    assert root_handle.is_imagetool

    tool_handle = typing.cast(
        "manager_console._ManagedToolNamespace",
        root_handle.children[0],
    )
    assert tool_handle.window is tool.window
    assert tool_handle.uid == "tool"
    assert tool_handle.name == "Tool Row"
    assert tool_handle.kind == "customtool"
    assert not tool_handle.is_imagetool
    assert tool_handle.callable_method() == "called"
    with pytest.raises(AttributeError):
        _ = tool_handle.non_callable
    with pytest.raises(AttributeError):
        _ = tool_handle.missing_attribute

    tool_handle.show()
    tool_handle.hide()
    tool_handle.dispose()
    assert tool.calls == ["show", "hide", "dispose"]

    rejected_operations = (
        lambda: tool_handle.data,
        lambda: tool_handle[0],
        lambda: tool_handle + 1,
        lambda: 1 + tool_handle,
        lambda: tool_handle - 1,
        lambda: 1 - tool_handle,
        lambda: tool_handle * 1,
        lambda: 1 * tool_handle,
        lambda: tool_handle @ 1,
        lambda: 1 @ tool_handle,
        lambda: tool_handle / 1,
        lambda: 1 / tool_handle,
        lambda: tool_handle // 1,
        lambda: 1 // tool_handle,
        lambda: tool_handle % 1,
        lambda: 1 % tool_handle,
        lambda: tool_handle**1,
        lambda: 1**tool_handle,
        lambda: -tool_handle,
        lambda: +tool_handle,
        lambda: abs(tool_handle),
    )
    for operation in rejected_operations:
        with pytest.raises(TypeError, match="not an ImageTool"):
            operation()

    child_handles = tool_handle.children
    assert len(child_handles) == 2
    assert [handle.uid for handle in child_handles[:]] == ["child", "unnamed"]
    assert [handle.uid for handle in child_handles] == ["child", "unnamed"]
    assert (
        repr(child_handles) == "Children for customtool 0.0:\n"
        "├─ [0] ImageTool 0.0.0: child-data\n"
        "└─ [1] tool"
    )
    assert repr(typing.cast("typing.Any", child_handles[0]).children) == (
        "No children for ImageTool 0.0.0"
    )

    no_tools_handle = manager_console._ManagedToolNamespace(tool)
    assert no_tools_handle._console_tools is None
    assert "customtool: Tool Row" in repr(no_tools_handle)
    assert repr(no_tools_handle.children) == "No children"
    with pytest.raises(LookupError):
        no_tools_handle.children[0]
    no_tools_handle.children._append_tree_lines([], tool, "")

    assert tools._node_path(orphan) is None
    assert tools._node_path(omitted) is None
    assert tools._node_path(detached) is None
    assert tools._node_data_name(detached).startswith("data_detached")
    assert tools._child_nodes(root) == [tool]
    assert tools._child_nodes(skip_parent) == [skip_child]
    assert isinstance(tools._node_handle(child), manager_console.ToolNamespace)
    assert isinstance(tools._node_handle(tool), manager_console._ManagedToolNamespace)
    assert tools._node_kind(root) == "ImageTool"
    assert tools._node_kind(unnamed_tool) == "tool"
    assert tools._tree_node_label(unnamed_tool) == "tool"


def test_manager_console_callable_operand_captures_replayable_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    monkeypatch.setattr(console_helper, "__module__", "__main__")
    monkeypatch.setattr(console_helper_dependency, "__module__", "__main__")

    operand = manager_console._callable_operand(console_helper)

    assert operand is not None
    assert operand.copyable
    namespace: dict[str, typing.Any] = {
        "data": data,
        "np": np,
        "xr": xr,
        "erlab": erlab,
        "era": erlab.analysis,
    }
    exec(  # noqa: S102
        manager_console._script_code(
            operand.code_prelude,
            f"derived = {operand.code}(data)",
        ),
        namespace,
        namespace,
    )
    xr.testing.assert_identical(namespace["derived"], console_helper(data))

    assert manager_console._function_global_names("def f(:", "f") is None
    assert (
        manager_console._function_global_names("def f():\n    return missing", "g")
        is None
    )

    with monkeypatch.context() as patch:
        patch.setattr(console_helper, "__name__", "data")
        patch.setattr(console_helper, "__qualname__", "data")
        assert manager_console._callable_operand(console_helper) is None

    with monkeypatch.context() as patch:
        patch.setattr(console_helper, "__module__", "__main__")

        def raise_oserror(_value):
            raise OSError

        patch.setattr(manager_console.inspect, "getsource", raise_oserror)
        assert manager_console._callable_operand(console_helper) is None


def test_manager_console_callable_operand_rejects_unreplayable_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(console_helper, "__module__", "__main__")
    monkeypatch.setattr(console_helper_dependency, "__module__", "__main__")

    source_cases = (
        "def console_helper(value):\n"
        "    return value\n\n"
        "def extra():\n"
        "    return value",
        "@decorator\ndef console_helper(value):\n    return value",
    )
    for source in source_cases:
        with monkeypatch.context() as patch:
            patch.setattr(
                manager_console.inspect,
                "getsource",
                lambda _value, source=source: source,
            )
            assert manager_console._callable_operand(console_helper) is None

    with monkeypatch.context() as patch:
        patch.setattr(
            manager_console, "_function_global_names", lambda _source, _name: None
        )
        assert manager_console._callable_operand(console_helper) is None

    for global_names in ({"data_0"}, {"missing_global"}):
        with monkeypatch.context() as patch:
            patch.setattr(
                manager_console,
                "_function_global_names",
                lambda _source, _name, names=global_names: names,
            )
            assert manager_console._callable_operand(console_helper) is None

    def local_dependency(value):
        return value

    with monkeypatch.context() as patch:
        patch.setitem(console_helper.__globals__, "local_dependency", local_dependency)
        patch.setattr(
            manager_console,
            "_function_global_names",
            lambda _source, _name: {"local_dependency"},
        )
        assert manager_console._callable_operand(console_helper) is None

    with monkeypatch.context() as patch:
        patch.setitem(console_helper.__globals__, "uncopyable_global", object())
        patch.setattr(
            manager_console,
            "_function_global_names",
            lambda _source, _name: {"uncopyable_global"},
        )
        assert manager_console._callable_operand(console_helper) is None

    with monkeypatch.context() as patch:
        patch.setattr(
            erlab.interactive.imagetool.provenance,
            "_validate_script_replay_code",
            lambda _code: (_ for _ in ()).throw(ValueError),
        )
        assert manager_console._callable_operand(console_helper) is None


def test_manager_console_assignment_tracks_until_explicit_show(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 2.0

    with manager_context() as manager:
        manager.show()
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.console._console_widget.execute("diff = tools[0] - tools[1]")
        assert manager.ntools == 2

        manager.console._console_widget.execute(
            "diff.qshow(manager=True, execute=False)"
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data, data0 - data1
        )
        qshow_provenance = manager._tool_graph.root_wrappers[2].provenance_spec
        assert qshow_provenance is not None
        assert qshow_provenance.active_name == "diff"

        manager.console._console_widget.execute(
            "non_manager = itool(diff, manager=False, execute=False)"
        )
        assert manager.ntools == 3

        manager.console._console_widget.execute(
            "itool(diff, manager=True, execute=False)"
        )
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(3).slicer_area.data, data0 - data1
        )
        itool_provenance = manager._tool_graph.root_wrappers[3].provenance_spec
        assert itool_provenance == qshow_provenance

        manager.console._console_widget.execute("diff + tools[0]")
        qtbot.wait_until(lambda: manager.ntools == 5, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(4).slicer_area.data,
            data0 - data1 + data0,
        )
        nested_provenance = manager._tool_graph.root_wrappers[4].provenance_spec
        assert nested_provenance is not None
        assert [source.name for source in nested_provenance.script_inputs] == [
            "diff",
            "data_0",
        ]
        nested_uid = manager._tool_graph.root_wrappers[4].uid
        assert manager.dependency_status_for_uid(nested_uid) == "current"
        nested_data = manager.get_imagetool(4).slicer_area.data.copy()
        manager.get_imagetool(1).slicer_area.replace_source_data(data1 + 5.0)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(nested_uid) == "changed",
            timeout=5000,
        )
        xr.testing.assert_identical(
            manager.get_imagetool(4).slicer_area.data,
            nested_data,
        )

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_console_selected_expression_opens_provenance_root(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 3.0

    with manager_context() as manager:
        manager.show()
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.console._console_widget.execute("xr.DataArray([1.0], dims=['x'])")
        assert manager.ntools == 2

        select_tools(manager, [0, 1])
        manager.console._console_widget.execute("tools.selected[0] - tools.selected[1]")
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            data0 - data1,
        )
        provenance = manager._tool_graph.root_wrappers[2].provenance_spec
        assert provenance is not None
        assert [source.name for source in provenance.script_inputs] == [
            "data_0",
            "data_1",
        ]
        assert (
            provenance.operations[-1].derivation_entry().code
            == "derived = data_0 - data_1"
        )

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_console_child_imagetool_access_tracks_provenance(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _ConsoleTreeTool(erlab.interactive.utils.ToolWindow):
        tool_name = "dtool"

    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
        name="parent",
    )
    data1 = data0 + 2.0
    data1.name = "right"
    child_data = data0 + 10.0
    child_data.name = "child"
    second_child_data = data0 + 20.0
    second_child_data.name = "second"

    with manager_context() as manager:
        manager.show()
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        intermediate_tool = _ConsoleTreeTool()
        intermediate_uid = manager.add_childtool(intermediate_tool, 0, show=False)
        intermediate_node = manager._child_node(intermediate_uid)
        intermediate_node.name = "Derivative"
        child_tool = itool(child_data, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool, intermediate_uid, show=False
        )
        child_node = manager._child_node(child_uid)
        second_child_tool = itool(second_child_data, manager=False, execute=False)
        assert isinstance(second_child_tool, erlab.interactive.imagetool.ImageTool)
        second_child_uid = manager.add_imagetool_child(
            second_child_tool, intermediate_uid, show=False
        )

        shell = manager.console._console_widget.kernel_manager.kernel.shell

        manager.console._console_widget.execute("child_handles = tools[0].children")
        child_handles = shell.user_ns["child_handles"]
        assert len(child_handles) == 1
        assert (
            repr(child_handles) == "Children for ImageTool 0:\n"
            "└─ [0] dtool: Derivative\n"
            "   ├─ [0] ImageTool 0.0.0: child\n"
            "   └─ [1] ImageTool 0.0.1: second"
        )
        intermediate_handle = child_handles[0]
        assert not intermediate_handle.is_imagetool
        assert intermediate_handle.uid == intermediate_uid
        assert intermediate_handle.kind == "dtool"
        assert intermediate_handle.name == "Derivative"
        assert intermediate_handle.window is intermediate_tool
        assert len(intermediate_handle.children) == 2

        with pytest.raises(TypeError, match="not an ImageTool"):
            _ = intermediate_handle.data

        tools_namespace = manager_console.ToolsNamespace(manager)
        top_left = tools_namespace[0]
        top_right = tools_namespace[1]
        assert top_left is not None
        assert top_right is not None
        with pytest.raises(TypeError, match="not an ImageTool"):
            top_left.children[0] - top_right

        manager.console._console_widget.execute(
            "isinstance(tools[0].children[0].children[0].data, xr.DataArray)"
        )
        assert shell.user_ns["_"] is True
        first_nested_child = intermediate_handle.children[0]
        assert first_nested_child.is_imagetool
        xr.testing.assert_identical(first_nested_child.data, child_data)
        xr.testing.assert_identical(
            top_left.children[0].children[1].data,
            second_child_data,
        )

        manager.console._console_widget.execute(
            "tools[0].children[0].children[0] - tools[1]"
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            child_data - data1,
        )
        provenance = manager._tool_graph.root_wrappers[2].provenance_spec
        assert provenance is not None
        assert [source.name for source in provenance.script_inputs] == [
            "data_0_0_0",
            "data_1",
        ]
        assert [source.node_uid for source in provenance.script_inputs] == [
            child_uid,
            manager._tool_graph.root_wrappers[1].uid,
        ]
        assert [source.node_snapshot_token for source in provenance.script_inputs] == [
            child_node.snapshot_token,
            manager._tool_graph.root_wrappers[1].snapshot_token,
        ]

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager.console._console_widget.execute("tools.selected[0].data")
        xr.testing.assert_identical(shell.user_ns["_"], child_data)
        manager.console._console_widget.execute("tools.selected_data")
        assert len(shell.user_ns["_"]) == 1
        xr.testing.assert_identical(shell.user_ns["_"][0], child_data)

        derived_uid = manager._tool_graph.root_wrappers[2].uid
        updated_child = child_data + 5.0
        child_tool.slicer_area.replace_source_data(updated_child)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "changed",
            timeout=5000,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [2])
        manager.reload_selected()
        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            updated_child - data1,
        )
        assert manager.dependency_status_for_uid(derived_uid) == "current"

        manager._remove_childtool(child_uid)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "missing",
            timeout=5000,
        )
        assert intermediate_node._childtool_indices == [second_child_uid]

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_reload_script_input_uses_public_nested_1d_child_data(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _ConsoleTreeTool(erlab.interactive.utils.ToolWindow):
        tool_name = "ftool"

    x = np.arange(3.0)
    eV = np.arange(5.0)
    data = xr.DataArray(
        np.arange(15.0).reshape(3, 5),
        dims=("x", "eV"),
        coords={"x": x, "eV": eV},
        name="cut",
    )
    shift = xr.DataArray(
        [0.1, -0.1, 0.0],
        dims=("x",),
        coords={"x": x},
        name="fit_shift",
    )
    updated_shift = shift + xr.DataArray(
        [0.1, 0.1, -0.1],
        dims=("x",),
        coords={"x": x},
    )

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        intermediate_tool = _ConsoleTreeTool()
        intermediate_uid = manager.add_childtool(intermediate_tool, 0, show=False)
        manager._child_node(intermediate_uid).name = "Fit"
        child_tool = itool(shift, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool, intermediate_uid, show=False
        )
        manager._child_node(child_uid).name = "fit_shift"

        tools = manager_console.ToolsNamespace(manager)
        source = tools[0]
        shift_handle = source.children[0].children[0]
        assert shift_handle.data.dims == ("x",)
        assert child_tool.slicer_area.data.dims == ("x", "stack_dim")

        era_proxy = manager_console._ConsoleModuleProxy(erlab.analysis, "era")
        shifted = era_proxy.transform.shift(
            source,
            -shift_handle,
            along="eV",
            shift_coords=True,
        )
        assert isinstance(shifted, manager_console._DerivedDataNamespace)
        spec = shifted._console_provenance_spec(
            active_name="derived",
            label="Shift with fit result",
        )
        assert spec is not None

        shifted_tool = itool(shifted.data, manager=False, execute=False)
        assert isinstance(shifted_tool, erlab.interactive.imagetool.ImageTool)
        shifted_index = manager.add_imagetool(
            shifted_tool,
            show=False,
            provenance_spec=spec,
        )

        child_tool.slicer_area.replace_source_data(updated_shift)

        rebuilt = manager._rebuild_script_provenance(
            spec,
            target_node_uid=manager._tool_graph.root_wrappers[shifted_index].uid,
        )
        assert "stack_dim" not in rebuilt.data.dims
        xr.testing.assert_identical(
            rebuilt.data,
            erlab.analysis.transform.shift(
                data,
                -updated_shift,
                along="eV",
                shift_coords=True,
            ),
        )


def test_manager_console_structures_erlab_and_xarray_calls(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("alpha", "eV"),
        coords={"alpha": np.arange(3.0), "eV": np.linspace(-1.0, 1.0, 4)},
    )
    data1 = data0 + 10.0

    with manager_context() as manager:
        manager.show()
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        shell = manager.console._console_widget.kernel_manager.kernel.shell

        manager.console._console_widget.execute("tools[0].qsel(alpha=slice(0.0, 1.0))")
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        qsel_spec = manager._tool_graph.root_wrappers[2].provenance_spec
        assert qsel_spec is not None
        assert [operation.op for operation in qsel_spec.operations] == ["qsel"]
        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            data0.qsel(alpha=slice(0.0, 1.0)),
        )

        manager.console._console_widget.execute("avg = tools[0].qsel.average('alpha')")
        assert manager.ntools == 3
        manager.console._console_widget.execute(
            "itool(avg, manager=True, execute=False)"
        )
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=5000)
        avg_spec = manager._tool_graph.root_wrappers[3].provenance_spec
        assert avg_spec is not None
        assert [operation.op for operation in avg_spec.operations] == ["average"]
        avg_data = manager.get_imagetool(3).slicer_area.data
        xr.testing.assert_identical(
            avg_data.isel(stack_dim=0, drop=True),
            data0.qsel.average("alpha"),
        )

        manager.console._console_widget.execute("summed = tools[0].qsel.sum('alpha')")
        summed = shell.user_ns["summed"]
        assert [operation.op for operation in summed._operations] == ["qsel_aggregate"]
        assert summed._operations[0].func == "sum"
        xr.testing.assert_identical(summed.data, data0.qsel.sum("alpha"))

        manager.console._console_widget.execute(
            "rot = era.transform.rotate("
            "tools[0], 5.0, axes=('alpha', 'eV'), center=(1.0, 0.0), "
            "reshape=False)"
        )
        manager.console._console_widget.execute(
            "itool(rot, manager=True, execute=False)"
        )
        qtbot.wait_until(lambda: manager.ntools == 5, timeout=5000)
        rotate_spec = manager._tool_graph.root_wrappers[4].provenance_spec
        assert rotate_spec is not None
        assert [operation.op for operation in rotate_spec.operations] == ["rotate"]
        xr.testing.assert_identical(
            manager.get_imagetool(4).slicer_area.data,
            erlab.analysis.transform.rotate(
                data0,
                5.0,
                axes=("alpha", "eV"),
                center=(1.0, 0.0),
                reshape=False,
            ),
        )

        manager.console._console_widget.execute(
            "rot_kw = era.transform.rotate("
            "darr=tools[0], angle=0.0, axes=('alpha', 'eV'), "
            "center=(1.0, 0.0), reshape=False)"
        )
        manager.console._console_widget.execute(
            "itool(rot_kw, manager=True, execute=False)"
        )
        qtbot.wait_until(lambda: manager.ntools == 6, timeout=5000)
        rotate_kw_spec = manager._tool_graph.root_wrappers[5].provenance_spec
        assert rotate_kw_spec is not None
        assert [operation.op for operation in rotate_kw_spec.operations] == ["rotate"]
        xr.testing.assert_identical(
            manager.get_imagetool(5).slicer_area.data,
            erlab.analysis.transform.rotate(
                data0,
                0.0,
                axes=("alpha", "eV"),
                center=(1.0, 0.0),
                reshape=False,
            ),
        )

        manager.console._console_widget.execute(
            "cat = xr.concat([tools[0], tools[1]], dim='source')"
        )
        assert manager.ntools == 6
        manager.console._console_widget.execute(
            "itool(cat, manager=True, execute=False)"
        )
        qtbot.wait_until(lambda: manager.ntools == 7, timeout=5000)
        concat_spec = manager._tool_graph.root_wrappers[6].provenance_spec
        assert concat_spec is not None
        assert [source.name for source in concat_spec.script_inputs] == [
            "data_0",
            "data_1",
        ]
        assert [operation.op for operation in concat_spec.operations] == ["script_code"]
        xr.testing.assert_identical(
            manager.get_imagetool(6).slicer_area.data,
            xr.concat([data0, data1], dim=xr.IndexVariable("source", [0, 1])),
        )

        manager.console._console_widget.execute(
            "chain = tools[0].interp(eV=np.array([-0.5, 0.5])).transpose('eV', 'alpha')"
        )
        manager.console._console_widget.execute(
            "itool(chain, manager=True, execute=False)"
        )
        qtbot.wait_until(lambda: manager.ntools == 8, timeout=5000)
        chain_spec = manager._tool_graph.root_wrappers[7].provenance_spec
        assert chain_spec is not None
        assert chain_spec.seed_code == "chain = data_0"
        assert [operation.op for operation in chain_spec.operations] == [
            "interpolate",
            "transpose",
        ]
        expected_chain = data0.interp(eV=np.array([-0.5, 0.5])).transpose("eV", "alpha")
        xr.testing.assert_identical(
            manager.get_imagetool(7).slicer_area.data,
            expected_chain,
        )
        xr.testing.assert_identical(
            erlab.interactive.imagetool.provenance.replay_script_provenance(
                chain_spec, {"data_0": data0}
            ),
            expected_chain,
        )

        manager.console._console_widget.execute(
            "import importlib, inspect\n"
            "real_rotate = importlib.import_module("
            "'erlab.analysis.transform').rotate\n"
            "signature_ok = str(inspect.signature(era.transform.rotate)) "
            "== str(inspect.signature(real_rotate))\n"
            "wrapped_ok = era.transform.rotate.__wrapped__ is real_rotate\n"
            "dir_ok = 'rotate' in dir(era.transform)\n"
            "qsel_signature_ok = str(inspect.signature(tools[0].qsel)) "
            "== str(inspect.signature(tools[0].data.qsel))\n"
            "qsel_wrapped_signature_ok = str("
            "inspect.signature(tools[0].qsel.__wrapped__)) "
            "== str(inspect.signature(tools[0].data.qsel))\n"
            "qsel_doc_ok = tools[0].qsel.__doc__ == tools[0].data.qsel.__doc__\n"
            "interp_signature_ok = str(inspect.signature(tools[0].interp)) "
            "== str(inspect.signature(tools[0].data.interp))\n"
            "interp_wrapped_signature_ok = str("
            "inspect.signature(tools[0].interp.__wrapped__)) "
            "== str(inspect.signature(tools[0].data.interp))\n"
            "interp_doc_ok = tools[0].interp.__doc__ == tools[0].data.interp.__doc__\n"
            "interp_dir_ok = 'interp' in dir(tools[0])\n"
            "assign_coords_dir_ok = 'assign_coords' in dir(tools[0])\n"
            "qsel_dir_ok = 'average' in dir(tools[0].qsel)\n"
            "qsel_sum_dir_ok = 'sum' in dir(tools[0].qsel)\n"
            "leading_edge_dir_ok = 'leading_edge' in dir(era.interpolate)"
        )
        assert shell.user_ns["signature_ok"]
        assert shell.user_ns["wrapped_ok"]
        assert shell.user_ns["dir_ok"]
        assert shell.user_ns["qsel_signature_ok"]
        assert shell.user_ns["qsel_wrapped_signature_ok"]
        assert shell.user_ns["qsel_doc_ok"]
        assert shell.user_ns["interp_signature_ok"]
        assert shell.user_ns["interp_wrapped_signature_ok"]
        assert shell.user_ns["interp_doc_ok"]
        assert shell.user_ns["interp_dir_ok"]
        assert shell.user_ns["assign_coords_dir_ok"]
        assert shell.user_ns["qsel_dir_ok"]
        assert shell.user_ns["qsel_sum_dir_ok"]
        assert shell.user_ns["leading_edge_dir_ok"]

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_console_captures_self_contained_function_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
    )

    with manager_context() as manager:
        manager.show()
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        shell = manager.console._console_widget.kernel_manager.kernel.shell

        manager.console._console_widget.execute(
            "scale = 2.0\n"
            "def add_scale(data):\n"
            "    return data + scale\n"
            "\n"
            "def offset_data(data):\n"
            "    return add_scale(data)\n"
        )
        manager.console._console_widget.execute(
            "import inspect\n"
            "function_help_ok = offset_data.__wrapped__.__name__ == 'offset_data'\n"
            "function_signature_ok = str(inspect.signature(offset_data)) == '(data)'"
        )
        assert shell.user_ns["function_help_ok"]
        assert shell.user_ns["function_signature_ok"]

        manager.console._console_widget.execute("shifted = offset_data(tools[0])")
        shifted = shell.user_ns["shifted"]
        assert shifted._copyable is True

        manager.console._console_widget.execute(
            "itool(shifted, manager=True, execute=False)"
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        shifted_spec = manager._tool_graph.root_wrappers[1].provenance_spec
        assert shifted_spec is not None
        shifted_code = shifted_spec.derivation_entries()[-1].code
        assert shifted_code is not None
        assert "def add_scale(data):" in shifted_code
        assert "def offset_data(data):" in shifted_code
        xr.testing.assert_identical(
            erlab.interactive.imagetool.provenance.replay_script_provenance(
                shifted_spec, {"data_0": data}
            ),
            data + 2.0,
        )
        manager.console._console_widget.execute("piped = tools[0].pipe(offset_data)")
        piped = shell.user_ns["piped"]
        assert piped._copyable is True

        manager.console._console_widget.execute(
            "runtime_object = object()\n"
            "def uses_runtime_object(data):\n"
            "    _ = runtime_object\n"
            "    return data\n"
        )
        manager.console._console_widget.execute(
            "opaque = tools[0].pipe(uses_runtime_object)"
        )
        opaque = shell.user_ns["opaque"]
        assert opaque._copyable is False

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_console_derived_tool_reload_action_routes_to_manager(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
    )
    data1 = data0 + 1.0

    with manager_context() as manager:
        manager.show()
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.console._console_widget.execute("tools[0] - tools[1]")
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        derived = manager.get_imagetool(2)
        derived_uid = manager._tool_graph.root_wrappers[2].uid
        updated0 = data0 + 10.0
        manager.get_imagetool(0).slicer_area.replace_source_data(updated0)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "changed",
            timeout=5000,
        )

        assert derived.slicer_area.reloadable
        menu_bar = typing.cast("typing.Any", derived.menuBar())
        menu_bar._file_menu_visibility()
        assert derived.slicer_area.reload_act.isVisible()
        derived.slicer_area.reload_act.trigger()

        xr.testing.assert_identical(
            derived.slicer_area.data,
            updated0 - data1,
        )
        assert manager.dependency_status_for_uid(derived_uid) == "current"

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_console_derived_reload_reapplies_filter(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
    )
    data1 = data0 + 1.0
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 1.0}
    )

    with manager_context() as manager:
        manager.show()
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.console._console_widget.execute("tools[0] + tools[1]")
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        derived = manager.get_imagetool(2)
        derived_uid = manager._tool_graph.root_wrappers[2].uid
        derived.slicer_area.apply_filter_operation(operation, emit_edited=True)

        updated0 = data0 + 10.0
        manager.get_imagetool(0).slicer_area.replace_source_data(updated0)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "changed",
            timeout=5000,
        )

        derived.slicer_area.reload()
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "current",
            timeout=5000,
        )

        raw_expected = updated0 + data1
        expected = operation.apply(raw_expected, parent_data=raw_expected)
        xr.testing.assert_identical(derived.slicer_area.data, expected)
        xr.testing.assert_identical(derived.slicer_area.displayed_data, expected)
        display_spec = manager._tool_graph.root_wrappers[2].displayed_provenance_spec
        assert display_spec is not None
        assert any(
            entry.label.startswith("Gaussian Filter")
            for entry in display_spec.display_entries()
        )

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_manager_concat_records_dependencies_and_handles_removed_inputs(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 10.0

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        expected = xr.concat(
            [
                manager.get_imagetool(0).slicer_area.data,
                manager.get_imagetool(1).slicer_area.data,
            ],
            dim="concat_dim",
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="override",
        )
        expected = expected.assign_coords(concat_dim=np.arange(2))

        select_tools(manager, [0, 1])
        accept_dialog(manager.concat_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        xr.testing.assert_identical(manager.get_imagetool(2).slicer_area.data, expected)
        concat_wrapper = manager._tool_graph.root_wrappers[2]
        provenance = concat_wrapper.provenance_spec
        assert provenance is not None
        assert [source.name for source in provenance.script_inputs] == [
            "data_0",
            "data_1",
        ]
        assert [source.node_uid for source in provenance.script_inputs] == [
            manager._tool_graph.root_wrappers[0].uid,
            manager._tool_graph.root_wrappers[1].uid,
        ]
        assert [source.node_snapshot_token for source in provenance.script_inputs] == [
            manager._tool_graph.root_wrappers[0].snapshot_token,
            manager._tool_graph.root_wrappers[1].snapshot_token,
        ]
        operation_entry = provenance.operations[-1].derivation_entry()
        assert operation_entry.label == "Concatenate selected ImageTools"
        assert operation_entry.code is not None
        assert "xr.concat([data_0, data_1]" in operation_entry.code
        assert manager.dependency_status_for_uid(concat_wrapper.uid) == "current"

        manager.get_imagetool(0).slicer_area.replace_source_data(data0 + 1.0)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(concat_wrapper.uid) == "changed",
            timeout=5000,
        )
        xr.testing.assert_identical(manager.get_imagetool(2).slicer_area.data, expected)

        select_tools(manager, [0, 1])
        expected_removed_inputs = xr.concat(
            [
                manager.get_imagetool(0).slicer_area.data,
                manager.get_imagetool(1).slicer_area.data,
            ],
            dim="concat_dim",
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="override",
        )
        expected_removed_inputs = expected_removed_inputs.assign_coords(
            concat_dim=np.arange(2)
        )

        def _remove_originals(dialog: _ConcatDialog) -> None:
            dialog._sources_combo.setCurrentIndex(
                dialog._sources_combo.findData(_ConcatDialog._SOURCES_REMOVE)
            )

        accept_dialog(manager.concat_action.trigger, pre_call=_remove_originals)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        removed_inputs_wrapper = manager._tool_graph.root_wrappers[3]
        assert (
            manager.dependency_status_for_uid(removed_inputs_wrapper.uid) == "missing"
        )
        xr.testing.assert_identical(
            manager.get_imagetool(3).slicer_area.data,
            expected_removed_inputs,
        )


def test_manager_concat_can_replace_source_tool_and_preserve_children(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 10.0
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )
    expected = xr.concat(
        [data0, data1],
        dim="concat_dim",
        coords="minimal",
        compat="override",
        join="outer",
        combine_attrs="override",
    ).assign_coords(concat_dim=np.arange(2))

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        old_root_provenance = provenance.full_data()
        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            old_root_provenance
        )
        manager.get_imagetool(0).slicer_area.apply_filter_operation(
            operation,
            emit_edited=True,
        )

        compatible_child = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(data0.copy(deep=True), manager=False, execute=False),
        )
        incompatible_child = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(data0.transpose("y", "x"), manager=False, execute=False),
        )
        compatible_uid = manager.add_imagetool_child(
            compatible_child,
            0,
            source_spec=provenance.full_data(),
            source_auto_update=True,
            show=False,
        )
        incompatible_uid = manager.add_imagetool_child(
            incompatible_child,
            0,
            source_spec=provenance.full_data(
                provenance.TransposeOperation(dims=("y", "x"))
            ),
            source_auto_update=True,
            show=False,
        )

        select_tools(manager, [0, 1])

        def _replace_first_source(dialog: _ConcatDialog) -> None:
            assert dialog.result_mode() == _ConcatDialog._RESULT_NEW
            assert dialog.sources_mode() == _ConcatDialog._SOURCES_KEEP
            assert not dialog._replace_target_combo.isEnabled()

            dialog._result_combo.setCurrentIndex(
                dialog._result_combo.findData(_ConcatDialog._RESULT_REPLACE)
            )
            assert dialog._replace_target_combo.isEnabled()
            dialog._replace_target_combo.setCurrentIndex(
                dialog._replace_target_combo.findData(0)
            )

        accept_dialog(manager.concat_action.trigger, pre_call=_replace_first_source)
        qtbot.wait_until(
            lambda: manager.get_imagetool(compatible_uid).slicer_area.data.identical(
                expected
            ),
            timeout=5000,
        )

        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, expected)
        assert not manager.get_imagetool(0).slicer_area.has_active_filter
        xr.testing.assert_identical(manager.get_imagetool(1).slicer_area.data, data1)
        assert compatible_uid in manager._tool_graph.root_wrappers[0]._childtool_indices
        assert (
            incompatible_uid in manager._tool_graph.root_wrappers[0]._childtool_indices
        )

        compatible_node = manager._child_node(compatible_uid)
        incompatible_node = manager._child_node(incompatible_uid)
        assert compatible_node.source_state == "fresh"
        assert incompatible_node.source_state == "unavailable"
        xr.testing.assert_identical(
            manager.get_imagetool(incompatible_uid).slicer_area.data,
            data0.transpose("y", "x"),
        )

        replacement_wrapper = manager._tool_graph.root_wrappers[0]
        replacement_provenance = replacement_wrapper.provenance_spec
        assert replacement_provenance is not None
        assert manager.dependency_status_for_uid(replacement_wrapper.uid) == "current"
        assert [source.node_uid for source in replacement_provenance.script_inputs] == [
            None,
            manager._tool_graph.root_wrappers[1].uid,
        ]
        assert [
            source.node_snapshot_token
            for source in replacement_provenance.script_inputs
        ] == [
            None,
            manager._tool_graph.root_wrappers[1].snapshot_token,
        ]
        assert (
            replacement_provenance.script_inputs[0].parsed_provenance_spec()
            == old_root_provenance
        )

        dialog = manager._actions_controller._concat_dialog
        dialog._result_combo.setCurrentIndex(
            dialog._result_combo.findData(_ConcatDialog._RESULT_REPLACE)
        )
        dialog._sources_combo.setCurrentIndex(
            dialog._sources_combo.findData(_ConcatDialog._SOURCES_REMOVE)
        )
        select_tools(manager, [0, 1])
        dialog.open()
        assert dialog.result_mode() == _ConcatDialog._RESULT_NEW
        assert dialog.sources_mode() == _ConcatDialog._SOURCES_KEEP
        assert not dialog._replace_target_combo.isEnabled()
        dialog.reject()


def test_manager_concat_replace_can_remove_unpreserved_sources(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 10.0
    expected = xr.concat(
        [data0, data1],
        dim="concat_dim",
        coords="minimal",
        compat="override",
        join="outer",
        combine_attrs="override",
    ).assign_coords(concat_dim=np.arange(2))

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        child = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(data0.copy(deep=True), manager=False, execute=False),
        )
        child_uid = manager.add_imagetool_child(
            child,
            0,
            source_spec=provenance.full_data(),
            source_auto_update=True,
            show=False,
        )

        select_tools(manager, [0, 1])

        def _replace_first_source_and_remove_others(dialog: _ConcatDialog) -> None:
            dialog._result_combo.setCurrentIndex(
                dialog._result_combo.findData(_ConcatDialog._RESULT_REPLACE)
            )
            dialog._replace_target_combo.setCurrentIndex(
                dialog._replace_target_combo.findData(0)
            )
            dialog._sources_combo.setCurrentIndex(
                dialog._sources_combo.findData(_ConcatDialog._SOURCES_REMOVE)
            )

        accept_dialog(
            manager.concat_action.trigger,
            pre_call=_replace_first_source_and_remove_others,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        assert list(manager._tool_graph.root_wrappers) == [0]
        assert child_uid in manager._tool_graph.nodes
        assert child_uid in manager._tool_graph.root_wrappers[0]._childtool_indices
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, expected)
        xr.testing.assert_identical(
            manager.get_imagetool(child_uid).slicer_area.data,
            expected,
        )


def test_manager_concat_uses_unfiltered_source_data(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 10.0
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.get_imagetool(0).slicer_area.apply_filter_operation(
            operation,
            emit_edited=True,
        )
        expected = xr.concat(
            [data0, data1],
            dim="concat_dim",
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="override",
        ).assign_coords(concat_dim=np.arange(2))

        select_tools(manager, [0, 1])
        accept_dialog(manager.concat_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        xr.testing.assert_identical(manager.get_imagetool(2).slicer_area.data, expected)
        provenance = manager._tool_graph.root_wrappers[2].provenance_spec
        assert provenance is not None
        assert provenance.script_inputs[0].parsed_provenance_spec() is None


def test_manager_reload_script_inputs_replaces_compatible_and_preserves_cursor(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
    )
    data1 = data0 + 1.0

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                data0 - data1,
                (0, 1),
                operation_label="Subtract inputs",
                operation_code="derived = data_0 - data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        derived = manager.get_imagetool(2)
        derived.slicer_area.array_slicer.set_indices(0, [2, 1], update=False)
        before_token = manager._tool_graph.root_wrappers[2].snapshot_token
        manager.get_imagetool(0).slicer_area.replace_source_data(data0 + 10.0)
        derived_uid = manager._tool_graph.root_wrappers[2].uid
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "changed",
            timeout=5000,
        )

        manager.tree_view.deselect_all()
        select_tools(manager, [2])
        manager._update_actions()
        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()
        assert manager.tree_view._menu.actions().count(manager.reload_action) == 1
        assert not hasattr(manager, "rebuild_inputs_action")
        badge_rect, badge_index = dependency_status_badge(manager, 2)
        delegate = typing.cast(
            "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
        )
        badge = delegate._badge_at(
            delegate._option_for_index(manager.tree_view, badge_index),
            badge_index,
            badge_rect.center(),
        )
        assert badge is not None
        assert badge.kind == "dependency_status"
        manager.tree_view.deselect_all()
        select_tools(manager, [0])

        def _cancel_reload(dialog: QtWidgets.QDialog) -> None:
            assert isinstance(dialog, QtWidgets.QMessageBox)
            cancel_button = dialog.button(QtWidgets.QMessageBox.StandardButton.Cancel)
            assert cancel_button is not None
            cancel_button.click()

        accept_dialog(
            lambda: click_tree_view_pos(manager.tree_view, badge_rect.center()),
            accept_call=_cancel_reload,
        )
        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            data0 - data1,
        )
        assert manager.dependency_status_for_uid(derived_uid) == "changed"
        assert manager._tool_graph.root_wrappers[2].snapshot_token == before_token

        accept_dialog(
            lambda: click_tree_view_pos(manager.tree_view, badge_rect.center())
        )

        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            (data0 + 10.0) - data1,
        )
        assert manager.get_imagetool(2).slicer_area.array_slicer.get_indices(0) == [
            2,
            1,
        ]
        assert manager.dependency_status_for_uid(derived_uid) == "current"
        assert manager._tool_graph.root_wrappers[2].snapshot_token != before_token


def test_manager_reload_script_inputs_normalizes_derived_1d_stack_dim(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3.0), "y": np.arange(3.0)},
    )

    with manager_context() as manager:
        manager.show()
        itool([data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                data.mean("y"),
                (0,),
                operation_label="Average input",
                operation_code='derived = data_0.mean("y")',
            )
            == 1
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        derived = manager.get_imagetool(1)
        assert derived.slicer_area.data.dims == ("x", "stack_dim")
        derived.slicer_area.array_slicer.set_indices(0, [2, 0], update=False)
        before_token = manager._tool_graph.root_wrappers[1].snapshot_token
        updated = data + 10.0
        manager.get_imagetool(0).slicer_area.replace_source_data(updated)
        derived_uid = manager._tool_graph.root_wrappers[1].uid
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "changed",
            timeout=5000,
        )

        monkeypatch.setattr(
            manager,
            "_prompt_incompatible_reload_commit",
            lambda _details: pytest.fail("compatible 1D reload prompted"),
        )
        manager.tree_view.deselect_all()
        select_tools(manager, [1])
        manager.reload_selected()

        xr.testing.assert_identical(
            manager.get_imagetool(1).slicer_area.data.squeeze("stack_dim", drop=True),
            updated.mean("y"),
        )
        assert manager.get_imagetool(1).slicer_area.array_slicer.get_indices(0) == [
            2,
            0,
        ]
        assert manager.dependency_status_for_uid(derived_uid) == "current"
        assert manager._tool_graph.root_wrappers[1].snapshot_token != before_token


def test_manager_reload_script_derived_target_reports_runtime_error(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3.0), "y": np.arange(3.0)},
    )
    critical_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _critical(*args: object, **kwargs: object) -> int:
        critical_calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_critical),
    )

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                data.copy(deep=True),
                (0,),
                operation_label="Runtime failure",
                operation_code="derived = data_0.isel(not_a_dim=0)",
            )
            == 1
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        assert not manager._reload_script_derived_target(1)

    assert len(critical_calls) == 1
    assert "ValueError" in str(critical_calls[0][1].get("detailed_text", ""))


def test_manager_reload_script_inputs_normalizes_nonuniform_idx_dims(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(4, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 0.8, 1.5], "y": np.arange(3.0)},
    )
    operation_code = (
        "derived = "
        "erlab.interactive.imagetool.slicer.restore_nonuniform_dims(data_0) + 1"
    )

    with manager_context() as manager:
        manager.show()
        itool([data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.get_imagetool(0).slicer_area.data.dims == ("x_idx", "y")
        assert (
            manager._show_multi_input_script_result(
                data + 1.0,
                (0,),
                operation_label="Offset restored input",
                operation_code=operation_code,
            )
            == 1
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert manager.get_imagetool(1).slicer_area.data.dims == ("x_idx", "y")

        updated = data + 10.0
        manager.get_imagetool(0).slicer_area.replace_source_data(updated)
        derived_uid = manager._tool_graph.root_wrappers[1].uid
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "changed",
            timeout=5000,
        )

        monkeypatch.setattr(
            manager,
            "_prompt_incompatible_reload_commit",
            lambda _details: pytest.fail("compatible nonuniform reload prompted"),
        )
        manager.tree_view.deselect_all()
        select_tools(manager, [1])
        manager.reload_selected()

        xr.testing.assert_identical(
            erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
                manager.get_imagetool(1).slicer_area.data
            ),
            updated + 1.0,
        )
        assert manager.dependency_status_for_uid(derived_uid) == "current"


def test_manager_reload_script_inputs_rebuilds_live_nested_input(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
    )
    data1 = data0 + 1.0

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                data0 - data1,
                (0, 1),
                operation_label="Subtract inputs",
                operation_code="derived = data_0 - data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                (data0 - data1) + data0,
                (2, 0),
                operation_label="Add derived input",
                operation_code="derived = data_2 + data_0",
            )
            == 3
        )
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=5000)

        final_uid = manager._tool_graph.root_wrappers[3].uid
        updated0 = data0 + 10.0
        manager.get_imagetool(0).slicer_area.replace_source_data(updated0)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(final_uid) == "changed",
            timeout=5000,
        )

        manager.tree_view.deselect_all()
        select_tools(manager, [3])
        manager.reload_selected()

        xr.testing.assert_identical(
            manager.get_imagetool(3).slicer_area.data,
            (updated0 - data1) + updated0,
        )
        assert manager.dependency_status_for_uid(final_uid) == "current"
        rebuilt_spec = manager._tool_graph.root_wrappers[3].provenance_spec
        assert rebuilt_spec is not None
        assert rebuilt_spec.script_inputs[0].node_uid is None
        nested_spec = rebuilt_spec.script_inputs[0].parsed_provenance_spec()
        assert nested_spec is not None
        assert [source.node_uid for source in nested_spec.script_inputs] == [
            manager._tool_graph.root_wrappers[0].uid,
            manager._tool_graph.root_wrappers[1].uid,
        ]


def test_manager_reload_script_inputs_after_workspace_roundtrip(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
    )
    data1 = data0 + 1.0

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                data0 - data1,
                (0, 1),
                operation_label="Subtract inputs",
                operation_code="derived = data_0 - data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        workspace_path = tmp_path / "script-derived-reload.itws"
        manager._save_workspace_document(workspace_path, force_full=True)
        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        loaded_result = manager.get_imagetool(2).slicer_area.data.copy()
        xr.testing.assert_identical(loaded_result, data0 - data1)
        derived_uid = manager._tool_graph.root_wrappers[2].uid
        provenance = manager._tool_graph.root_wrappers[2].provenance_spec
        assert provenance is not None
        assert [source.node_uid for source in provenance.script_inputs] == [
            manager._tool_graph.root_wrappers[0].uid,
            manager._tool_graph.root_wrappers[1].uid,
        ]
        assert manager.dependency_status_for_uid(derived_uid) == "current"

        updated0 = data0 + 10.0
        manager.get_imagetool(0).slicer_area.replace_source_data(updated0)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "changed",
            timeout=5000,
        )
        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data, loaded_result
        )

        before_token = manager._tool_graph.root_wrappers[2].snapshot_token
        manager.tree_view.deselect_all()
        select_tools(manager, [2])
        manager._update_actions()
        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()
        manager.reload_selected()

        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            updated0 - data1,
        )
        assert manager.dependency_status_for_uid(derived_uid) == "current"
        assert manager._tool_graph.root_wrappers[2].snapshot_token != before_token


def test_manager_reload_script_inputs_incompatible_prompt_paths(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    data1 = data0 + 1.0
    shifted0 = data0.assign_coords(x=[10.0, 11.0]) + 10.0
    shifted1 = data1.assign_coords(x=[10.0, 11.0])
    expected = shifted0 - shifted1

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                data0 - data1,
                (0, 1),
                operation_label="Subtract inputs",
                operation_code="derived = data_0 - data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        original = manager.get_imagetool(2).slicer_area.data.copy()
        manager.get_imagetool(0).slicer_area.replace_source_data(shifted0)
        manager.get_imagetool(1).slicer_area.replace_source_data(shifted1)
        manager.tree_view.deselect_all()
        select_tools(manager, [2])

        monkeypatch.setattr(
            manager, "_prompt_incompatible_reload_commit", lambda _details: "cancel"
        )
        manager.reload_selected()
        assert manager.ntools == 3
        xr.testing.assert_identical(manager.get_imagetool(2).slicer_area.data, original)

        monkeypatch.setattr(
            manager, "_prompt_incompatible_reload_commit", lambda _details: "new"
        )
        manager.reload_selected()
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=5000)
        xr.testing.assert_identical(manager.get_imagetool(2).slicer_area.data, original)
        xr.testing.assert_identical(manager.get_imagetool(3).slicer_area.data, expected)

        manager.tree_view.deselect_all()
        select_tools(manager, [2])
        monkeypatch.setattr(
            manager, "_prompt_incompatible_reload_commit", lambda _details: "replace"
        )
        manager.reload_selected()
        xr.testing.assert_identical(manager.get_imagetool(2).slicer_area.data, expected)


def test_manager_reload_script_inputs_uses_recorded_file_for_removed_parent(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 5.0
    path1 = tmp_path / "right.nc"
    data1.to_netcdf(path1)
    file_spec = provenance.file_load(
        start_label="Load right",
        seed_code=f"derived = xr.load_dataarray({str(path1)!r})",
        file_load_source=provenance.FileLoadSource(
            path=str(path1),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        manager._tool_graph.root_wrappers[1].set_detached_provenance(file_spec)
        assert (
            manager._show_multi_input_script_result(
                data0 - data1,
                (0, 1),
                operation_label="Subtract inputs",
                operation_code="derived = data_0 - data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        manager.remove_imagetool(1)
        derived_uid = manager._tool_graph.root_wrappers[2].uid
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(derived_uid) == "missing",
            timeout=5000,
        )
        manager.tree_view.deselect_all()
        select_tools(manager, [2])
        manager._update_info()
        details = metadata_detail_map(manager)
        assert metadata_detail_labels(manager).count("Inputs") == 1
        assert manager._metadata_detail_labels["Inputs"].wordWrap()
        assert (
            manager.metadata_details_widget.sizePolicy().verticalPolicy()
            == QtWidgets.QSizePolicy.Policy.Maximum
        )
        assert (
            manager._metadata_detail_labels["Inputs"].sizePolicy().verticalPolicy()
            == QtWidgets.QSizePolicy.Policy.Preferred
        )
        assert (
            manager.metadata_group.sizePolicy().verticalPolicy()
            == QtWidgets.QSizePolicy.Policy.Preferred
        )
        assert manager.metadata_group.parentWidget() is manager.right_splitter
        assert (
            manager.metadata_details_widget.parentWidget()
            is manager.metadata_details_page
        )
        assert (
            manager.metadata_derivation_list.parentWidget()
            is manager.metadata_provenance_page
        )
        assert not isinstance(
            manager.metadata_details_widget.parentWidget(), QtWidgets.QSplitter
        )
        handle = manager.right_splitter.handle(2)
        assert handle is not None
        qtbot.wait_until(handle.isVisible, timeout=5000)
        assert (
            manager.metadata_group.maximumHeight() == manager_widgets._QWIDGETSIZE_MAX
        )
        assert (
            manager.metadata_details_widget.maximumHeight()
            == manager_widgets._QWIDGETSIZE_MAX
        )
        manager.resize(640, 700)
        manager.right_splitter.setSizes([180, 120, 280])
        manager.inspector_tabs.setCurrentWidget(manager.metadata_provenance_page)
        QtWidgets.QApplication.processEvents()
        before_right_sizes = manager.right_splitter.sizes()
        before_list_height = manager.metadata_derivation_list.height()
        manager.right_splitter.moveSplitter(
            before_right_sizes[0] + before_right_sizes[1] - 40, 2
        )
        QtWidgets.QApplication.processEvents()
        after_right_sizes = manager.right_splitter.sizes()
        assert after_right_sizes[2] > before_right_sizes[2]
        assert manager.metadata_derivation_list.height() > before_list_height
        assert "\n" in details["Inputs"]
        assert "\n\n" not in details["Inputs"]
        assert all(line.strip() for line in details["Inputs"].splitlines())
        assert "recorded source file found" in details["Inputs"]

        manager.reload_selected()

        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            data0 - data1,
        )
        rebuilt_spec = manager._tool_graph.root_wrappers[2].provenance_spec
        assert rebuilt_spec is not None
        assert (
            rebuilt_spec.script_inputs[0].node_uid
            == manager._tool_graph.root_wrappers[0].uid
        )
        assert rebuilt_spec.script_inputs[1].node_uid is None
        assert rebuilt_spec.script_inputs[1].parsed_provenance_spec() == file_spec


def test_manager_reload_script_inputs_reuses_shared_recorded_file_prefix(
    qtbot,
    tmp_path,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(24.0).reshape(2, 2, 3, 2),
        dims=("pol", "energy", "k", "beta"),
        coords={
            "pol": ["LH", "LV"],
            "energy": [0.0, 1.0],
            "k": [0, 1, 2],
            "beta": [10.0, 20.0],
        },
    )
    path = tmp_path / "polarization.nc"
    source.to_netcdf(path)
    file_spec = provenance.file_load(
        start_label="Load both polarizations",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=provenance.FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )
    shared_stage = provenance.full_data(provenance.AverageOperation(dims=("k",)))
    left_stage = provenance.selection(
        provenance.SelOperation(kwargs={"pol": "LH"}),
        provenance.SqueezeOperation(),
    )
    right_stage = provenance.selection(
        provenance.SelOperation(kwargs={"pol": "LV"}),
        provenance.SqueezeOperation(),
    )
    left_data = left_stage.apply(shared_stage.apply(source))
    right_data = right_stage.apply(shared_stage.apply(source))
    left_spec = provenance.compose_full_provenance(
        provenance.compose_full_provenance(file_spec, shared_stage),
        left_stage,
    )
    right_spec = provenance.compose_full_provenance(
        provenance.compose_full_provenance(file_spec, shared_stage),
        right_stage,
    )
    assert left_spec is not None
    assert right_spec is not None
    load_count = 0

    def _load_shared_source(_load_source: typing.Any) -> xr.DataArray:
        nonlocal load_count
        load_count += 1
        return source

    monkeypatch.setattr(
        _provenance_framework,
        "_load_file_source_data",
        _load_shared_source,
    )

    with manager_context() as manager:
        manager.show()
        itool([left_data, right_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        manager._tool_graph.root_wrappers[0].set_detached_provenance(left_spec)
        manager._tool_graph.root_wrappers[1].set_detached_provenance(right_spec)
        assert (
            manager._show_multi_input_script_result(
                left_data - right_data,
                (0, 1),
                operation_label="Subtract inputs",
                operation_code="derived = data_0 - data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        workspace_path = tmp_path / "shared-replay.itws"
        manager._save_workspace_document(workspace_path, force_full=True)
        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        load_count = 0

        manager.remove_imagetool(1)
        manager.remove_imagetool(0)
        monkeypatch.setattr(
            manager,
            "_prompt_incompatible_reload_commit",
            lambda _details: "replace",
        )
        assert manager._reload_script_derived_target(2)

        xr.testing.assert_identical(
            manager.get_imagetool(2).slicer_area.data,
            left_data - right_data,
        )
        assert load_count == 1


def test_manager_reload_script_inputs_missing_parent_without_source_noops(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 1.0
    errors: list[tuple[str, str, str, str]] = []
    unavailable_reasons = _record_reload_unavailable_dialog(monkeypatch)

    def _critical(
        parent, title, text, informative_text="", detailed_text=None, buttons=None
    ):
        errors.append((title, text, informative_text, detailed_text or ""))
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                data0 - data1,
                (0, 1),
                operation_label="Subtract inputs",
                operation_code="derived = data_0 - data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        original = manager.get_imagetool(2).slicer_area.data.copy()

        manager.remove_imagetool(1)
        manager.tree_view.deselect_all()
        select_tools(manager, [2])
        manager._update_actions()

        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()
        manager.reload_selected()

        xr.testing.assert_identical(manager.get_imagetool(2).slicer_area.data, original)
        assert unavailable_reasons
        assert not errors


def test_manager_reload_helper_status_dialog_and_workspace_branches(
    qtbot,
    tmp_path,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    data1 = data0 + 1.0
    path = tmp_path / "recorded.nc"
    data0.to_netcdf(path)
    file_spec = provenance.file_load(
        start_label="Load recorded",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=provenance.FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )

    class FakeMessageBox:
        Icon = types.SimpleNamespace(Warning=object(), Question=object())
        ButtonRole = types.SimpleNamespace(AcceptRole=object(), ActionRole=object())
        StandardButton = types.SimpleNamespace(Cancel=object(), Close=object())
        clicked_index: int | None = 0
        instances: typing.ClassVar[list["FakeMessageBox"]] = []

        def __init__(self, _parent=None) -> None:
            self.buttons: list[object] = []
            self.clicked: object | None = None
            self.standard_buttons = None
            FakeMessageBox.instances.append(self)

        def setWindowTitle(self, _value) -> None: ...

        def setDetailedText(self, _value) -> None: ...

        def setIcon(self, _value) -> None: ...

        def setText(self, _value) -> None: ...

        def setInformativeText(self, _value) -> None: ...

        def setStandardButtons(self, value) -> None:
            self.standard_buttons = value

        def addButton(self, *_args) -> object:
            button = object()
            self.buttons.append(button)
            return button

        def setDefaultButton(self, _value) -> None: ...

        def exec(self) -> None:
            if (
                FakeMessageBox.clicked_index is not None
                and FakeMessageBox.clicked_index < len(self.buttons)
            ):
                self.clicked = self.buttons[FakeMessageBox.clicked_index]

        def clickedButton(self) -> object | None:
            return self.clicked

    with manager_context() as manager:
        manager.show()
        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert (
            manager._show_multi_input_script_result(
                data0 - data1,
                (0, 1),
                operation_label="Subtract inputs",
                operation_code="derived = data_0 - data_1",
            )
            == 2
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        derived_wrapper = manager._tool_graph.root_wrappers[2]
        derived_uid = derived_wrapper.uid
        script_spec = derived_wrapper.provenance_spec
        assert script_spec is not None

        assert manager.dependency_status_label_for_uid(derived_uid) is not None
        manager.dependency_status_badge_for_uid(derived_uid)
        assert manager.dependency_status_tooltip_for_uid(derived_uid) is not None
        assert manager.dependency_input_summary_for_uid(derived_uid) is not None
        assert manager.dependency_status_label_for_uid("missing") is None
        assert manager.dependency_status_badge_for_uid("missing") is None
        assert manager.dependency_status_tooltip_for_uid("missing") is None
        assert manager.dependency_input_summary_for_uid("missing") is None
        assert not manager._missing_dependencies_have_recorded_file("missing")
        assert manager._script_provenance_inputs_current(script_spec)
        stale_input = script_spec.script_inputs[0].model_copy(
            update={"node_snapshot_token": "stale"}
        )
        stale_spec = script_spec.model_copy(
            update={"script_inputs": (stale_input, *script_spec.script_inputs[1:])}
        )
        missing_input = script_spec.script_inputs[0].model_copy(
            update={"node_uid": "missing-parent"}
        )
        missing_spec = script_spec.model_copy(
            update={"script_inputs": (missing_input, *script_spec.script_inputs[1:])}
        )
        assert not manager._script_provenance_inputs_current(stale_spec)
        assert not manager._script_provenance_inputs_current(missing_spec)

        full_data_input = provenance.ScriptInput(
            name="full",
            label="Full data",
            provenance_spec=provenance.full_data().model_dump(mode="json"),
        )
        assert not manager._script_input_can_reload(full_data_input)
        assert manager._script_input_unavailable_reason(full_data_input) is not None
        file_without_source_input = types.SimpleNamespace(
            node_uid=None,
            label="File without source",
            parsed_provenance_spec=lambda: file_spec.model_copy(
                update={"file_load_source": None}
            ),
        )
        assert not manager._script_input_can_reload(file_without_source_input)
        assert (
            manager._script_input_unavailable_reason(file_without_source_input)
            is not None
        )
        missing_file_input = provenance.ScriptInput(
            name="missing_file",
            label="Missing file",
            provenance_spec=file_spec.model_copy(
                update={
                    "file_load_source": file_spec.file_load_source.model_copy(
                        update={"path": str(tmp_path / "missing.nc")}
                    )
                }
            ).model_dump(mode="json"),
        )
        assert not manager._script_input_has_recorded_file(missing_file_input)
        missing_file_reason = manager._script_input_unavailable_reason(
            missing_file_input
        )
        assert missing_file_reason is not None
        assert str(tmp_path / "missing.nc") in missing_file_reason
        load_source = file_spec.file_load_source
        assert load_source is not None
        replay_call = load_source.replay_call
        assert replay_call is not None
        no_replay_call_input = types.SimpleNamespace(
            node_uid=None,
            label="No replay call",
            parsed_provenance_spec=lambda: file_spec.model_copy(
                update={
                    "file_load_source": load_source.model_copy(
                        update={"replay_call": None}
                    )
                }
            ),
        )
        assert not manager._script_input_can_reload(no_replay_call_input)
        assert manager._script_input_unavailable_reason(no_replay_call_input)
        valid_file_input = provenance.ScriptInput(
            name="valid_file",
            label="Valid file",
            provenance_spec=file_spec.model_dump(mode="json"),
        )
        assert manager._script_input_can_reload(valid_file_input)
        assert manager._script_input_unavailable_reason(valid_file_input) is None

        script_file_spec = provenance.script(
            start_label="Load recorded",
            seed_code=typing.cast("str", file_spec.seed_code),
            active_name="derived",
            file_load_source=load_source,
        ).append_replay_stage(
            provenance.full_data(provenance.AverageOperation(dims=("x",)))
        )
        script_file_input = provenance.ScriptInput(
            name="script_file",
            label="Script-backed file",
            provenance_spec=script_file_spec.model_dump(mode="json"),
        )
        assert manager._script_input_has_recorded_file(script_file_input)
        assert manager._script_input_can_reload(script_file_input)
        assert manager._script_input_unavailable_reason(script_file_input) is None

        missing_script_file = tmp_path / "missing-script.nc"
        missing_script_file_input = provenance.ScriptInput(
            name="missing_script_file",
            label="Missing script-backed file",
            provenance_spec=script_file_spec.model_copy(
                update={
                    "file_load_source": load_source.model_copy(
                        update={"path": str(missing_script_file)}
                    )
                }
            ).model_dump(mode="json"),
        )
        assert not manager._script_input_has_recorded_file(missing_script_file_input)
        assert not manager._script_input_can_reload(missing_script_file_input)
        missing_script_reason = manager._script_input_unavailable_reason(
            missing_script_file_input
        )
        assert missing_script_reason is not None
        assert str(missing_script_file) in missing_script_reason

        missing_loader = "definitely-missing-erlab-loader"
        missing_loader_input = provenance.ScriptInput(
            name="missing_loader",
            label="Missing loader",
            provenance_spec=file_spec.model_copy(
                update={
                    "file_load_source": load_source.model_copy(
                        update={
                            "loader_label": "Loader",
                            "loader_text": missing_loader,
                            "replay_call": replay_call.model_copy(
                                update={
                                    "kind": "erlab_loader",
                                    "target": missing_loader,
                                }
                            ),
                        }
                    )
                }
            ).model_dump(mode="json"),
        )
        assert not manager._script_input_can_reload(missing_loader_input)
        reason = manager._script_input_unavailable_reason(missing_loader_input)
        assert reason is not None
        assert missing_loader in reason

        nonreplayable_script_input = provenance.ScriptInput(
            name="nonreplayable",
            label="Nonreplayable script",
            provenance_spec=provenance.script(
                provenance.ScriptCodeOperation(label="Opaque", code=None),
                start_label="Run script",
                active_name="derived",
                script_inputs=(valid_file_input,),
            ).model_dump(mode="json"),
        )
        assert not manager._script_input_can_reload(nonreplayable_script_input)
        assert (
            manager._script_input_unavailable_reason(nonreplayable_script_input)
            is not None
        )
        nested_missing_input = provenance.ScriptInput(
            name="nested_missing",
            label="Nested missing file",
            provenance_spec=provenance.script(
                provenance.ScriptCodeOperation(
                    label="Use missing", code="derived = missing_file"
                ),
                start_label="Run script",
                active_name="derived",
                script_inputs=(missing_file_input,),
            ).model_dump(mode="json"),
        )
        nested_reason = manager._script_input_unavailable_reason(nested_missing_input)
        assert nested_reason is not None
        assert str(tmp_path / "missing.nc") in nested_reason
        nested_valid_input = provenance.ScriptInput(
            name="nested_valid",
            label="Nested valid file",
            provenance_spec=provenance.script(
                provenance.ScriptCodeOperation(
                    label="Use valid", code="derived = valid_file"
                ),
                start_label="Run script",
                active_name="derived",
                script_inputs=(valid_file_input,),
            ).model_dump(mode="json"),
        )
        assert manager._script_input_unavailable_reason(nested_valid_input) is None

        file_marker = "file-marker"
        child_marker = "child-marker"
        saved_marker = "saved-marker"
        file_input = provenance.ScriptInput(
            name="file_input",
            label="File input",
            node_uid="missing-file-node",
            node_snapshot_token=file_marker,
            provenance_spec=file_spec.model_dump(mode="json"),
        )
        nested_spec = provenance.script(
            provenance.ScriptCodeOperation(
                label="Use file", code="derived = file_input"
            ),
            start_label="Run script",
            active_name="derived",
            script_inputs=(file_input,),
        )
        nested_input = provenance.ScriptInput(
            name="nested",
            label="Nested input",
            provenance_spec=nested_spec.model_dump(mode="json"),
        )
        ref = provenance.ScriptInputDependencyRef(
            name="file_input",
            label="File input",
            node_uid="missing-file-node",
            node_snapshot_token=file_marker,
        )
        assert manager._script_input_has_recorded_file(file_input)
        assert manager._script_input_has_recorded_file(nested_input)
        assert manager._dependency_ref_has_recorded_file(nested_spec, ref)
        assert not manager._dependency_ref_has_recorded_file(None, ref)
        derived_wrapper.set_displayed_provenance(nested_spec)
        assert manager._missing_dependencies_have_recorded_file(derived_uid)

        fake_child = types.SimpleNamespace(
            uid="1 child-node",
            provenance_spec=file_spec,
            displayed_provenance_spec=file_spec,
            display_text="Child node",
            is_imagetool=False,
            snapshot_token=child_marker,
            type_badge_text="tool",
        )
        script_input = manager._script_input_for_node(fake_child)
        assert script_input.name.startswith("data__1_child")
        assert script_input.parsed_provenance_spec() == file_spec

        monkeypatch.setattr(manager_mainwindow.QtWidgets, "QMessageBox", FakeMessageBox)
        FakeMessageBox.instances.clear()
        manager._show_dependency_reload_dialog(0)
        assert not FakeMessageBox.instances

        with monkeypatch.context() as patch:
            patch.setattr(manager, "dependency_status_for_uid", lambda _uid: "current")
            patch.setattr(
                manager, "dependency_input_summary_for_uid", lambda _uid: None
            )
            patch.setattr(
                manager, "_node_can_reload_script_inputs", lambda _node: False
            )
            manager._show_dependency_reload_dialog(0)
        assert (
            FakeMessageBox.instances[-1].standard_buttons
            is FakeMessageBox.StandardButton.Close
        )

        reload_targets: list[int | str] = []
        with monkeypatch.context() as patch:
            patch.setattr(manager, "dependency_status_for_uid", lambda _uid: "changed")
            patch.setattr(
                manager, "dependency_input_summary_for_uid", lambda _uid: "details"
            )
            patch.setattr(manager, "_node_can_reload_script_inputs", lambda _node: True)
            patch.setattr(
                manager, "_reload_script_derived_target", reload_targets.append
            )
            FakeMessageBox.clicked_index = 0
            manager._show_dependency_reload_dialog(0)
            FakeMessageBox.clicked_index = 1
            manager._show_dependency_reload_dialog(0)
        assert reload_targets == [0]

        for clicked_index, expected in (
            (0, "replace"),
            (1, "new"),
            (2, "cancel"),
            (None, "cancel"),
        ):
            FakeMessageBox.clicked_index = clicked_index
            assert manager._prompt_incompatible_reload_commit("details") == expected

        details = manager._reload_incompatibility_details(
            data0,
            data0.rename({"x": "energy"}).assign_coords(y=[10.0, 11.0]),
        )
        assert "Current dims" in details
        assert "Reloaded dims" in details

        with monkeypatch.context() as patch:
            patch.setattr(
                erlab.interactive, "itool", lambda *_args, **_kwargs: object()
            )
            assert (
                manager._show_multi_input_script_result(
                    data0,
                    (0,),
                    operation_label="No tool",
                    operation_code="derived = data_0",
                )
                is None
            )

        with monkeypatch.context() as patch:
            patch.setattr(manager, "target_from_slicer_area", lambda _area: None)
            assert not manager._script_reload_from_slicer_area(object(), execute=False)
        with monkeypatch.context() as patch:
            patch.setattr(manager, "target_from_slicer_area", lambda _area: 0)
            patch.setattr(manager, "_node_can_reload_script_inputs", lambda _node: True)
            assert manager._script_reload_from_slicer_area(object(), execute=False)

        lineage = manager._lineage_controller
        non_imagetool_node = types.SimpleNamespace(is_imagetool=False)
        assert lineage._node_reload_unavailable_reason(non_imagetool_node)
        closed_imagetool_node = types.SimpleNamespace(
            is_imagetool=True,
            imagetool=None,
            pending_workspace_memory_payload=None,
        )
        assert lineage._node_reload_unavailable_reason(closed_imagetool_node)
        pending_memory_node = types.SimpleNamespace(
            uid="pending-memory",
            is_imagetool=True,
            imagetool=None,
            pending_workspace_memory_payload=(
                tmp_path / "workspace.itws",
                "0/imagetool",
            ),
            provenance_spec=provenance.full_data(),
        )
        assert lineage._node_reload_unavailable_reason(pending_memory_node)
        pending_file_node = types.SimpleNamespace(
            uid="pending-file",
            is_imagetool=True,
            imagetool=None,
            pending_workspace_memory_payload=(
                tmp_path / "workspace.itws",
                "0/imagetool",
            ),
            provenance_spec=file_spec,
        )
        assert lineage._node_reload_unavailable_reason(pending_file_node) is None
        script_no_inputs_node = types.SimpleNamespace(
            uid="script-no-inputs",
            is_imagetool=True,
            imagetool=object(),
            slicer_area=types.SimpleNamespace(
                _direct_reloadable=lambda: False,
                _provenance_reloadable=lambda: False,
                _local_reload_unavailable_reason=lambda: "local reason",
            ),
            provenance_spec=provenance.script(
                provenance.ScriptCodeOperation(
                    label="Use input", code="derived = data"
                ),
                start_label="Run script",
                active_name="derived",
            ),
        )
        assert lineage._node_reload_unavailable_reason(script_no_inputs_node)
        script_file_input_node = types.SimpleNamespace(
            uid="script-file-input",
            is_imagetool=True,
            imagetool=object(),
            slicer_area=script_no_inputs_node.slicer_area,
            provenance_spec=provenance.script(
                provenance.ScriptCodeOperation(
                    label="Use file", code="derived = valid_file"
                ),
                start_label="Run script",
                active_name="derived",
                script_inputs=(valid_file_input,),
            ),
        )
        assert lineage._node_reload_unavailable_reason(script_file_input_node) is None

        with monkeypatch.context() as patch:
            patch.setattr(manager, "_selected_reload_candidates", lambda: None)
            assert manager._selected_reload_targets() is None
            manager.reload_selected()
        with monkeypatch.context() as patch:
            patch.setattr(
                manager,
                "_selected_reload_candidates",
                lambda: ([0], {}, "blocked"),
            )
            assert manager._selected_reload_targets() is None
        with monkeypatch.context() as patch:
            patch.setattr(
                manager,
                "_selected_reload_candidates",
                lambda: ([0], {}, None),
            )
            assert manager._selected_reload_targets() == ([0], {})
        with monkeypatch.context() as patch:
            patch.setattr(
                manager,
                "_node_for_target",
                lambda _target: (_ for _ in ()).throw(KeyError("missing")),
            )
            assert manager._reload_unavailable_reason_for_target(0)
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_node_for_target", lambda _target: object())
            patch.setattr(manager, "_reload_target_for_child", lambda _uid: 0)
            assert manager._reload_unavailable_reason_for_target("child") is None
            patch.setattr(manager, "_reload_target_for_child", lambda _uid: None)
            patch.setattr(
                manager,
                "_reload_unavailable_reason_for_child",
                lambda _uid: "child reason",
            )
            assert manager._reload_unavailable_reason_for_target("child")

        old_parent_uid = "saved-parent"
        derived_wrapper.set_displayed_provenance(
            provenance.script(
                provenance.ScriptCodeOperation(label="Copy", code="derived = data_0"),
                start_label="Run script",
                active_name="derived",
                script_inputs=(
                    provenance.ScriptInput(
                        name="data_0",
                        label="Saved parent",
                        node_uid=old_parent_uid,
                        node_snapshot_token=saved_marker,
                    ),
                ),
            )
        )
        assert manager._workspace_loaded_uid_map(
            {old_parent_uid: 0, "missing": "missing"}
        ) == {old_parent_uid: manager._tool_graph.root_wrappers[0].uid}
        manager._rebase_loaded_workspace_dependency_refs(
            {old_parent_uid: 0, derived_wrapper.uid: 2, "missing": "missing"}
        )
        rebased_spec = derived_wrapper.provenance_spec
        assert rebased_spec is not None
        assert (
            rebased_spec.script_inputs[0].node_uid
            == manager._tool_graph.root_wrappers[0].uid
        )


def test_manager_reload_self_replacement_uses_recorded_source(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
        name="source",
    )
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    file_spec = provenance.file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=provenance.FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        wrapper = manager._tool_graph.root_wrappers[0]
        wrapper.set_detached_provenance(file_spec)

        tools = manager_console.ToolsNamespace(manager)
        tool = tools[0]
        assert tool is not None
        tool.data = tool + 1.0
        expected = data + 1.0
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, expected)

        manager.tree_view.deselect_all()
        select_tools(manager, [0])
        manager._update_actions()
        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()

        manager.reload_selected()

        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, expected)
        rebuilt_spec = wrapper.provenance_spec
        assert rebuilt_spec is not None
        assert rebuilt_spec.script_inputs[0].node_uid is None
        assert rebuilt_spec.script_inputs[0].node_snapshot_token is None
        assert rebuilt_spec.script_inputs[0].parsed_provenance_spec() == file_spec


def test_manager_reload_raw_self_replacement_unavailable(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    unavailable_reasons = _record_reload_unavailable_dialog(monkeypatch)
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tools = manager_console.ToolsNamespace(manager)
        tool = tools[0]
        assert tool is not None
        tool.data = tool + 1.0
        expected = data + 1.0
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, expected)

        wrapper = manager._tool_graph.root_wrappers[0]
        manager.tree_view.deselect_all()
        select_tools(manager, [0])
        manager._update_actions()
        assert not manager._node_can_reload_script_inputs(wrapper)
        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()

        manager.reload_selected()

        assert unavailable_reasons
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, expected)


def test_unavailable_replay_code_details_lists_unique_labels_and_fallback() -> None:
    controller = object.__new__(_DetailsPanelController)
    start_entry = provenance.DerivationEntry("Start from data", None, False)
    unavailable_entry = provenance.DerivationEntry("Opaque step", None, False)

    details = controller._unavailable_replay_code_details(
        types.SimpleNamespace(
            derivation_entries=(start_entry, unavailable_entry, unavailable_entry)
        )
    )
    assert details.count("Opaque step") == 1

    fallback = controller._unavailable_replay_code_details(
        types.SimpleNamespace(derivation_entries=(start_entry,))
    )
    assert fallback

    no_spec_node = types.SimpleNamespace(
        derivation_entries=(start_entry, unavailable_entry),
        displayed_provenance_spec=None,
    )
    assert controller._unavailable_replay_code_traceback(no_spec_node) is None
    dialog_details = controller._unavailable_replay_code_dialog_details(no_spec_node)
    assert "Opaque step" in dialog_details
    assert "Traceback" not in dialog_details


def test_unavailable_replay_code_traceback_ignores_successful_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object.__new__(_DetailsPanelController)
    spec = types.SimpleNamespace(
        kind="script",
        operations=(object(),),
        active_name="derived",
    )
    node = types.SimpleNamespace(displayed_provenance_spec=spec)
    calls: list[tuple[typing.Any, str]] = []

    monkeypatch.setattr(
        manager_details_panel._replay_graph,
        "compile_replay_graph",
        lambda received_spec, *, display: ("graph", received_spec, display),
    )

    def _emit_replay_code(graph, *, output_name: str) -> str:
        calls.append((graph, output_name))
        return "derived = data"

    monkeypatch.setattr(
        manager_details_panel._replay_graph,
        "emit_replay_code",
        _emit_replay_code,
    )

    assert controller._unavailable_replay_code_traceback(node) is None
    assert calls == [(("graph", spec, True), "derived")]


def test_manager_reload_data_explains_non_replayable_script_provenance(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    dialogs: list[erlab.interactive.utils.MessageDialog] = []

    class _RecordingMessageDialog(erlab.interactive.utils.MessageDialog):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            dialogs.append(self)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _RecordingMessageDialog,
    )

    with manager_context() as manager:
        manager.show()
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        wrapper = manager._tool_graph.root_wrappers[0]
        wrapper.set_detached_provenance(
            provenance.script(
                provenance.ScriptCodeOperation(
                    label="Run opaque code",
                    code=None,
                    copyable=False,
                ),
                start_label="Run opaque code",
                active_name="derived",
                script_inputs=(manager._script_input_for_node(wrapper),),
            )
        )

        manager.tree_view.deselect_all()
        select_tools(manager, [0])
        manager._update_actions()

        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()
        copied: list[str] = []
        monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", copied.append)
        manager._set_metadata_node(wrapper)
        manager._copy_full_derivation_code()
        assert not copied
        assert len(dialogs) == 1
        assert dialogs[0].windowTitle() == "Replay Code Unavailable"
        assert dialogs[0].text()
        assert dialogs[0].informativeText()
        details = dialogs[0].detailedText()
        assert "Run opaque code" in details
        assert "ReplayGraphError" in details
        assert "non-replayable code" in details


def test_manager_console_replacement_updates_provenance_and_descendants(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = xr.DataArray(
        np.arange(25, dtype=float).reshape(5, 5),
        dims=("x", "y"),
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    data1 = data0.assign_coords(time=("x", np.linspace(10.0, 50.0, 5)))

    with manager_context() as manager:
        manager.show()

        itool([data0, data1], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )
        child = next(iter(manager._tool_graph.root_wrappers[0]._childtools.values()))

        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        manager.console._console_widget.execute(
            "tools[0].data = tools[0].assign_coords(time=tools[1].time)"
        )
        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)

        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area.data,
            data0.assign_coords(time=data1.time),
        )
        provenance = manager._tool_graph.root_wrappers[0].provenance_spec
        assert provenance is not None
        assert manager._tool_graph.root_wrappers[0].source_spec is None
        assert [source.name for source in provenance.script_inputs] == [
            "data_0",
            "data_1",
        ]
        assert (
            provenance.operations[-1].derivation_entry().code
            == "data_0 = data_0.assign_coords(time=data_1.time)"
        )

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        loaded_provenance = manager._tool_graph.root_wrappers[0].provenance_spec
        assert loaded_provenance is not None
        assert [source.name for source in loaded_provenance.script_inputs] == [
            "data_0",
            "data_1",
        ]

        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()

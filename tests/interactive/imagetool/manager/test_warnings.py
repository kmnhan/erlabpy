import logging
import pathlib
import sys
import tempfile
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._actions as manager_actions
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace as manager_workspace
import erlab.interactive.imagetool.manager._workspace_io as manager_workspace_io
import erlab.interactive.imagetool.manager._xarray as manager_xarray
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool.manager import ImageToolManager

from .helpers import _UnserializableChildTool, select_child_tool


def test_warning_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        warning_logger = logging.getLogger("test.warning.history")
        warning_logger.warning("First warning")
        warning_logger.warning("Second warning")

        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 2)
        qtbot.wait_until(
            lambda: all(warning.isVisible() for warning in manager._alert_dialogs)
        )

        texts = [notification.text() for notification in manager._alert_dialogs]
        assert any("First warning" in text for text in texts)
        assert any("Second warning" in text for text in texts)

        clear_all_button = manager._alert_dialogs[-1].findChild(
            QtWidgets.QPushButton, "warningDismissAllButton"
        )
        assert clear_all_button is not None

        qtbot.mouseClick(clear_all_button, QtCore.Qt.MouseButton.LeftButton, delay=10)
        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 0)


def test_warning_alert_suppressed_by_log_flag(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        suppressed = logging.LogRecord(
            name="test.warning.suppressed",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="suppressed warning",
            args=(),
            exc_info=None,
        )
        suppressed.suppress_ui_alert = True
        manager._warning_handler.emit(suppressed)
        QtWidgets.QApplication.processEvents()
        assert manager._alert_dialogs == []

        regular = logging.LogRecord(
            name="test.warning.regular",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="regular warning",
            args=(),
            exc_info=None,
        )
        manager._warning_handler.emit(regular)
        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 1)

        manager._clear_all_alerts()
        QtWidgets.QApplication.processEvents()


def test_warning_handler_ignores_deleted_emitter(qtbot) -> None:
    emitter = manager_widgets._WarningEmitter()
    handler = manager_widgets._WarningNotificationHandler(emitter)

    emitter.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, 0)
    QtWidgets.QApplication.processEvents()
    qtbot.wait_until(
        lambda: not erlab.interactive.utils.qt_is_valid(emitter),
        timeout=1000,
    )

    record = logging.LogRecord(
        name="test.warning.deleted",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg="deleted warning emitter",
        args=(),
        exc_info=None,
    )
    handler.emit(record)


def test_error_creating_imagetool_does_not_duplicate_alert_dialog(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )

    with manager_context() as manager:
        try:
            raise RuntimeError("boom")  # noqa: TRY301
        except RuntimeError:
            manager._error_creating_imagetool()

        QtWidgets.QApplication.processEvents()

        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_manager_duplicate_unserializable_child_shows_error(
    qtbot,
    gold,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[tuple[typing.Any, ...], dict[str, typing.Any]]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            _UnserializableChildTool(gold.copy(deep=True)),
            0,
            show=False,
        )

        select_child_tool(manager, child_uid)
        manager.duplicate_selected()

        assert len(manager._tool_graph.root_wrappers[0]._childtools) == 1
        assert len(critical_calls) == 1
        assert critical_calls[0][0][2] == (
            "An error occurred while duplicating the selected window."
        )
        assert "data_corr" in critical_calls[0][1]["detailed_text"]


def test_manager_save_unserializable_child_shows_error(
    qtbot,
    accept_dialog,
    gold,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[tuple[typing.Any, ...], dict[str, typing.Any]]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.add_childtool(
            _UnserializableChildTool(gold.copy(deep=True)),
            0,
            show=False,
        )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = f"{tmp_dir_name}/workspace.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(tmp_dir_name)
                dialog.selectFile(filename)
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText("workspace.itws")

            accept_dialog(lambda: manager.save(native=False), pre_call=_go_to_file)

            assert len(critical_calls) == 1
            assert critical_calls[0][0][2] == (
                "An error occurred while saving the workspace file."
            )
            assert "data_corr" in critical_calls[0][1]["detailed_text"]
            assert not pathlib.Path(filename).exists()


def test_data_recv_dataset_creation_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _raise_from_dataset(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(
        manager_workspace_io.ImageTool,
        "from_dataset",
        staticmethod(_raise_from_dataset),
    )

    with manager_context() as manager:
        ds = xr.Dataset({"v": xr.DataArray(np.ones((2, 2)), dims=("x", "y"))})
        flags = manager._data_recv([ds], {})

        QtWidgets.QApplication.processEvents()

        assert flags == [False]
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_data_recv_dataarray_creation_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _raise_imagetool(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(
        manager_workspace_io,
        "ImageTool",
        _raise_imagetool,
    )

    with manager_context() as manager:
        flags = manager._data_recv([xr.DataArray(np.ones((2, 2)), dims=("x", "y"))], {})

        QtWidgets.QApplication.processEvents()

        assert flags == [False]
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_load_workspace_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    exec_calls = {"count": 0}
    native_opt_calls = {"count": 0}

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _fake_exec(self):
        exec_calls["count"] += 1
        return exec_calls["count"] == 1

    def _raise_open_datatree(*args, **kwargs):
        raise RuntimeError("broken workspace")

    def _track_set_option(self, option, on=True):
        if option == QtWidgets.QFileDialog.Option.DontUseNativeDialog and on:
            native_opt_calls["count"] += 1
        return original_set_option(self, option, on)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    original_set_option = QtWidgets.QFileDialog.setOption
    monkeypatch.setattr(QtWidgets.QFileDialog, "setOption", _track_set_option)
    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", _fake_exec)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "selectedFiles",
        lambda self: ["broken_workspace.itws"],
    )
    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _raise_open_datatree)

    with manager_context() as manager:
        ImageToolManager.load(manager, native=False)

        QtWidgets.QApplication.processEvents()

        assert exec_calls["count"] == 1
        assert native_opt_calls["count"] == 1
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_workspace_file_lock_error_detects_nested_blocking_io() -> None:
    err = RuntimeError("open failed")
    err.__cause__ = BlockingIOError(35, "unable to lock file")

    assert manager_workspace._is_workspace_file_lock_error(err)
    assert not manager_workspace._is_workspace_file_lock_error(
        RuntimeError("broken workspace")
    )


def test_import_workspace_locked_file_shows_specific_error(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    lock_calls: list[pathlib.Path] = []
    generic_calls: list[tuple[typing.Any, ...]] = []
    fname = tmp_path / "locked.itws"

    def _raise_locked(*args, **kwargs):
        raise BlockingIOError(35, "unable to lock file")

    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", lambda self: True)
    monkeypatch.setattr(QtWidgets.QFileDialog, "selectedFiles", lambda self: [fname])
    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _raise_locked)
    monkeypatch.setattr(
        manager_workspace_io,
        "_show_workspace_file_lock_error",
        lambda _parent, locked_fname: lock_calls.append(pathlib.Path(locked_fname)),
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args, **kwargs: generic_calls.append(args)),
    )

    with manager_context() as manager:
        assert not manager.import_workspace(native=False)

    assert lock_calls == [fname]
    assert generic_calls == []


def test_open_multiple_files_locked_workspace_does_not_fall_through_to_loaders(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    lock_calls: list[pathlib.Path] = []
    fname = tmp_path / "locked.itws"

    def _raise_locked(*args, **kwargs):
        raise BlockingIOError(35, "unable to lock file")

    def _file_loaders_should_not_run(*args, **kwargs):
        raise AssertionError("locked workspace should not fall through to loaders")

    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _raise_locked)
    monkeypatch.setattr(
        manager_actions,
        "_show_workspace_file_lock_error",
        lambda _parent, locked_fname: lock_calls.append(pathlib.Path(locked_fname)),
    )
    monkeypatch.setattr(
        erlab.interactive.utils, "file_loaders", _file_loaders_should_not_run
    )

    with manager_context() as manager:
        manager.open_multiple_files([fname], try_workspace=True)

    assert lock_calls == [fname]


@pytest.mark.parametrize("suffix", [".ITWS", ".ItWs"])
def test_handle_dropped_files_treats_itws_suffix_case_insensitively(
    monkeypatch,
    tmp_path,
    suffix: str,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    calls: list[tuple[list[pathlib.Path], bool]] = []
    fname = tmp_path / f"workspace{suffix}"
    fname.write_bytes(b"")

    def _record_open(paths, try_workspace: bool = False) -> None:
        calls.append((paths, try_workspace))

    with manager_context() as manager:
        monkeypatch.setattr(manager, "open_multiple_files", _record_open)
        manager._handle_dropped_files([fname])

    assert calls == [([fname], True)]


def test_open_multiple_files_dropped_itws_error_does_not_fall_through_to_loaders(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    fname = tmp_path / "broken.itws"
    fname.write_text("not a workspace", encoding="utf-8")

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _raise_open_datatree(*args, **kwargs):
        raise RuntimeError("cannot read workspace")

    def _file_loaders_should_not_run(*args, **kwargs):
        raise AssertionError("broken .itws should not fall through to loaders")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _raise_open_datatree)
    monkeypatch.setattr(
        erlab.interactive.utils, "file_loaders", _file_loaders_should_not_run
    )

    with manager_context() as manager:
        manager.open_multiple_files([fname], try_workspace=True)

    assert len(critical_calls) == 1
    assert critical_calls[0][0][2] == (
        "An error occurred while loading the workspace file."
    )
    assert "cannot read workspace" in critical_calls[0][1]["detailed_text"]


def test_open_multiple_files_h5_workspace_probe_failure_falls_back_to_loaders(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_loader_calls: list[list[pathlib.Path]] = []
    add_calls: list[tuple[list[pathlib.Path], Callable]] = []
    fname = tmp_path / "data.h5"
    fname.write_bytes(b"not a workspace")

    def _raise_open_datatree(*args, **kwargs):
        raise RuntimeError("not a workspace")

    def _fake_file_loaders(paths):
        file_loader_calls.append(list(paths))
        return {"Fake HDF5 (*.h5)": (lambda *_args, **_kwargs: None, {})}

    def _fake_add_from_multiple_files(
        loaded, queued, failed, func, kwargs, retry_callback
    ) -> None:
        add_calls.append((queued, func))

    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _raise_open_datatree)
    monkeypatch.setattr(erlab.interactive.utils, "file_loaders", _fake_file_loaders)

    with manager_context() as manager:
        monkeypatch.setattr(
            manager, "_add_from_multiple_files", _fake_add_from_multiple_files
        )
        manager.open_multiple_files([fname], try_workspace=True)

    assert file_loader_calls == [[fname]]
    assert add_calls[0][0] == [fname]


def test_open_multiple_files_h5_non_workspace_falls_back_to_loaders(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_loader_calls: list[list[pathlib.Path]] = []
    closed = {"value": False}
    fname = tmp_path / "data.h5"
    fname.write_bytes(b"hdf5 but not a workspace")

    class _FakeDataTree:
        attrs: typing.ClassVar[dict[str, typing.Any]] = {}

        def close(self) -> None:
            closed["value"] = True

    def _fake_file_loaders(paths):
        file_loader_calls.append(list(paths))
        return {"Fake HDF5 (*.h5)": (lambda *_args, **_kwargs: None, {})}

    monkeypatch.setattr(
        manager_xarray,
        "open_workspace_datatree",
        lambda *args, **kwargs: _FakeDataTree(),
    )
    monkeypatch.setattr(erlab.interactive.utils, "file_loaders", _fake_file_loaders)

    with manager_context() as manager:
        monkeypatch.setattr(
            manager,
            "_add_from_multiple_files",
            lambda *args, **kwargs: None,
        )
        manager.open_multiple_files([fname], try_workspace=True)

    assert closed["value"]
    assert file_loader_calls == [[fname]]


def test_open_multiple_files_dropped_itws_non_workspace_does_not_fall_through(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    closed = {"value": False}
    fname = tmp_path / "not-workspace.itws"
    fname.write_text("hdf5 but not a workspace", encoding="utf-8")

    class _FakeDataTree:
        attrs: typing.ClassVar[dict[str, typing.Any]] = {}

        def close(self) -> None:
            closed["value"] = True

    def _file_loaders_should_not_run(*args, **kwargs):
        raise AssertionError("non-workspace .itws should not fall through to loaders")

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(
        manager_xarray,
        "open_workspace_datatree",
        lambda *args, **kwargs: _FakeDataTree(),
    )
    monkeypatch.setattr(
        erlab.interactive.utils, "file_loaders", _file_loaders_should_not_run
    )

    with manager_context() as manager:
        manager.open_multiple_files([fname], try_workspace=True)

    assert closed["value"]
    assert len(critical_calls) == 1
    assert critical_calls[0][0][2] == (
        "An error occurred while loading the workspace file."
    )


def test_open_retry_preserves_non_native_dialog(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    exec_calls = {"count": 0}
    native_opt_calls = {"count": 0}
    add_calls = {"count": 0}

    def _fake_exec(self):
        exec_calls["count"] += 1
        return exec_calls["count"] == 1

    def _track_set_option(self, option, on=True):
        if option == QtWidgets.QFileDialog.Option.DontUseNativeDialog and on:
            native_opt_calls["count"] += 1
        return original_set_option(self, option, on)

    original_set_option = QtWidgets.QFileDialog.setOption
    monkeypatch.setattr(QtWidgets.QFileDialog, "setOption", _track_set_option)
    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", _fake_exec)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "selectedFiles",
        lambda self: ["fake_data.h5"],
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "selectedNameFilter",
        lambda self: "Fake Loader (*.h5)",
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda: {"Fake Loader (*.h5)": (lambda *_a, **_k: None, {})},
    )

    with manager_context() as manager:

        def _fake_add_from_multiple_files(
            *,
            loaded,
            queued,
            failed,
            func,
            kwargs,
            retry_callback,
        ):
            add_calls["count"] += 1
            if add_calls["count"] == 1:
                retry_callback(None)

        monkeypatch.setattr(
            manager, "_add_from_multiple_files", _fake_add_from_multiple_files
        )
        ImageToolManager.open(manager, native=False)

    assert exec_calls["count"] >= 2
    assert native_opt_calls["count"] >= 2


def test_open_multiple_files_workspace_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    class _FakeDataTree:
        def __init__(self) -> None:
            self.attrs = {"imagetool_workspace_schema_version": 4}

        def close(self) -> None:
            pass

    def _fake_open_datatree(*args, **kwargs):
        return _FakeDataTree()

    def _raise_from_datatree(*args, **kwargs):
        raise RuntimeError("cannot restore workspace")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _fake_open_datatree)

    with manager_context() as manager:
        monkeypatch.setattr(manager, "_is_datatree_workspace", lambda *args: True)
        monkeypatch.setattr(manager, "_from_datatree", _raise_from_datatree)

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = pathlib.Path(tmp_dir) / "workspace.itws"
            p.write_text("placeholder", encoding="utf-8")
            manager.open_multiple_files([p], try_workspace=True)

        QtWidgets.QApplication.processEvents()

        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_manager_context_starts_cleanly_back_to_back(
    qtbot,
    caplog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with caplog.at_level(logging.ERROR):
        for _ in range(2):
            with manager_context() as manager:
                manager.show()
                qtbot.wait_until(
                    lambda: (
                        manager.server.isRunning()
                        and manager.watcher_server.isRunning()
                    ),
                    timeout=5000,
                )
                qtbot.wait(100)
                QtWidgets.QApplication.processEvents()
                assert manager._alert_dialogs == []

    assert "Address already in use" not in caplog.text
    assert not any(
        record.name == "erlab.interactive.imagetool.manager._server"
        and record.levelno >= logging.ERROR
        for record in caplog.records
    )


def test_manager_progressbar_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        message = "Load data:   8%|8         | 1/12 [00:00<00:07,  1.54it/s]"
        manager._show_alert("INFO", logging.INFO, message, "")

        assert 12 in manager._progress_bars
        pbar = manager._progress_bars[12]
        assert pbar.labelText() == "Load data"
        assert pbar.value() == 1
        assert manager._alert_dialogs == []

        manager._show_alert(
            "INFO",
            logging.INFO,
            "Load data:  50%|##        | 6/12 [00:00<00:07,  1.54it/s]",
            "",
        )
        assert manager._progress_bars[12] is pbar
        assert pbar.value() == 6

        pbar.close()


def test_manager_alert_icons(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    recorded_icons: list[QtWidgets.QStyle.StandardPixmap | None] = []

    class _RecordingMessageDialog(erlab.interactive.utils.MessageDialog):
        def __init__(self, *args, icon_pixmap=None, **kwargs):
            recorded_icons.append(icon_pixmap)
            super().__init__(*args, icon_pixmap=icon_pixmap, **kwargs)

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )

    with manager_context() as manager:
        manager._show_alert("INFO", logging.INFO, "info", "")
        manager._show_alert("WARNING", logging.WARNING, "warning", "")
        manager._show_alert("ERROR", logging.ERROR, "error", "")

        assert recorded_icons == [
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation,
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical,
        ]

        manager._clear_all_alerts()
        QtWidgets.QApplication.processEvents()


def test_uncaught_exception_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        assert sys.excepthook == manager._handle_uncaught_exception
        manager._previous_excepthook = lambda *exc: None

        try:
            raise RuntimeError("boom")  # noqa: TRY301
        except RuntimeError:
            exc_info = sys.exc_info()

        sys.excepthook(*exc_info)

        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 1)
        qtbot.wait_until(manager._alert_dialogs[0].isVisible)

        text = manager._alert_dialogs[0].text()
        detailed_text = manager._alert_dialogs[0].detailedText()

        assert "ERROR" in manager._alert_dialogs[0].windowTitle()
        assert "An unexpected error occurred" in text
        assert detailed_text is not None
        assert "boom" in detailed_text
        assert "RuntimeError" in detailed_text

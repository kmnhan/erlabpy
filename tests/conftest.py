import contextlib
import csv
import datetime
import functools
import importlib.util
import logging
import os
import pathlib
import re
import sys
import tempfile
import threading
import time
import typing
import uuid
from collections.abc import Callable, Iterator, Sequence

import dask
import dask.distributed
from dask.distributed import Client, LocalCluster

if "QT_QPA_PLATFORM" not in os.environ:
    qt_api = os.environ.get("QT_API") or os.environ.get("PYTEST_QT_API")
    if qt_api is not None and qt_api.lower() == "pyside6":
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    elif sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    else:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

import lmfit
import numexpr
import numpy as np
import pooch
import pytest
import requests
import xarray as xr
from numpy.testing import assert_almost_equal
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager as imagetool_manager
import erlab.interactive.imagetool.manager._registry as imagetool_manager_registry
import erlab.interactive.imagetool.manager._server as imagetool_manager_server
from erlab.interactive.utils import (
    _is_deleted_qt_wrapper_error,
    _WaitDialog,
    qt_is_valid,
)
from erlab.io.dataloader import LoaderBase
from erlab.io.exampledata import generate_data_angles, generate_gold_edge

_TEST_DATA_CONFIG = dict(
    line.split("=", maxsplit=1)
    for line in pathlib.Path(__file__)
    .with_name("test-data.env")
    .read_text()
    .splitlines()
    if line
)

DATA_COMMIT_HASH = _TEST_DATA_CONFIG["ERLAB_TEST_DATA_COMMIT"]
"""The commit hash of the commit to retrieve from `kmnhan/erlabpy-data`."""

DATA_KNOWN_HASH = _TEST_DATA_CONFIG["ERLAB_TEST_DATA_ARCHIVE_SHA256"]
"""The SHA-256 checksum of the `.tar.gz` file."""

DATA_RETRIEVE_ATTEMPTS = 4
"""Maximum attempts for transient test data download failures."""

DATA_DOWNLOAD_LOCK_TIMEOUT_SECONDS = 20 * 60
"""Maximum time to wait for another worker to finish downloading test data."""

log = logging.getLogger(__name__)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TEST_OPTIONS_ENV_VAR = "ERLAB_INTERACTIVE_OPTIONS_PATH"
_TEST_OPTIONS_MANAGED_ENV_VAR = "ERLAB_INTERACTIVE_OPTIONS_PATH_TEST_MANAGED"
_TEST_INTERACTIVE_OPTIONS_PATHS: list[pathlib.Path] = []


def pytest_configure(config: pytest.Config) -> None:
    if (
        _TEST_OPTIONS_ENV_VAR not in os.environ
        or os.environ.get(_TEST_OPTIONS_MANAGED_ENV_VAR) == "1"
    ):
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        settings_path = pathlib.Path(tempfile.gettempdir()) / (
            "erlabpy-test-interactive-options-"
            f"{worker_id}-{os.getpid()}-{uuid.uuid4().hex}.ini"
        )
        os.environ[_TEST_OPTIONS_ENV_VAR] = str(settings_path)
        os.environ[_TEST_OPTIONS_MANAGED_ENV_VAR] = "1"
        _TEST_INTERACTIVE_OPTIONS_PATHS.append(settings_path)


def _load_ci_test_groups_module():
    module_path = REPO_ROOT / "scripts" / "_ci_test_groups.py"
    spec = importlib.util.spec_from_file_location("_ci_test_groups", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load CI test group definitions from {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CI_TEST_GROUPS = _load_ci_test_groups_module()
is_compat_nodeid = _CI_TEST_GROUPS.is_compat_nodeid
is_compat_path = _CI_TEST_GROUPS.is_compat_path
is_gui_path = _CI_TEST_GROUPS.is_gui_path
serial_xdist_group = _CI_TEST_GROUPS.serial_xdist_group


def _qt_msg_filter(msg_type, context, message):
    """Filter out some Qt warnings related to offscreen mode."""
    if (
        "This plugin does not support raise()" in message
        or "This plugin does not support propagateSizeHints()" in message
        or "This plugin does not support grabbing the keyboard" in message
        or "Populating font family aliases took" in message
    ):
        return  # swallow
    # forward others to stderr
    sys.__stderr__.write(message + "\n")


QtCore.qInstallMessageHandler(_qt_msg_filter)

# Limit numexpr to a single thread; this reduces probability of segfaults
numexpr.set_num_threads(1)

# Limit dask to a single thread and disable worker profiling to reduce segfaults
dask.config.set(
    {"scheduler": "synchronous", "distributed.worker.profile.enabled": False}
)


def _coverage_is_active(pytestconfig: pytest.Config) -> bool:
    """Return whether pytest-cov is actively collecting coverage."""
    return bool(getattr(pytestconfig.option, "cov_source", None))


@contextlib.contextmanager
def _test_data_download_lock(lock_path: pathlib.Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + DATA_DOWNLOAD_LOCK_TIMEOUT_SECONDS
    lock_fd: int | None = None
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except OSError:
                continue
            if lock_age > DATA_DOWNLOAD_LOCK_TIMEOUT_SECONDS:
                with contextlib.suppress(OSError):
                    lock_path.unlink()
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for erlabpy test data lock {lock_path}"
                ) from None
            time.sleep(0.25)
    try:
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        os.close(lock_fd)
        with contextlib.suppress(OSError):
            lock_path.unlink()


def _test_data_dir_ready(path: pathlib.Path) -> bool:
    return all((path / dirname).is_dir() for dirname in ("da30", "erpes", "merlin"))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        rel_path = pathlib.Path(item.path).resolve().relative_to(REPO_ROOT).as_posix()

        if is_gui_path(rel_path):
            item.add_marker(pytest.mark.gui)
        serial_group = serial_xdist_group(rel_path, item.nodeid)
        if serial_group is not None:
            item.add_marker(pytest.mark.serial)
            item.add_marker(pytest.mark.xdist_group(serial_group))
        if is_compat_path(rel_path) or is_compat_nodeid(item.nodeid):
            item.add_marker(pytest.mark.compat)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    for settings_path in _TEST_INTERACTIVE_OPTIONS_PATHS:
        with contextlib.suppress(OSError):
            settings_path.unlink()

    qapp = QtWidgets.QApplication.instance()
    if qapp is None:
        return

    # pytest-qt closes registered widgets with deleteLater(), but processEvents()
    # does not guarantee that DeferredDelete events run before interpreter shutdown.
    for _ in range(2):
        QtWidgets.QApplication.sendPostedEvents(
            None, int(QtCore.QEvent.Type.DeferredDelete.value)
        )
        QtWidgets.QApplication.sendPostedEvents(None, 0)
        qapp.processEvents()


@pytest.fixture(autouse=True)
def _restore_interactive_options_between_tests() -> Iterator[None]:
    erlab.interactive.options.restore()
    try:
        yield
    finally:
        erlab.interactive.options.restore()


@pytest.fixture(scope="session")
def cluster():
    with LocalCluster(
        processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None
    ) as dask_cluster:
        yield dask_cluster


@pytest.fixture
def client(cluster):
    try:
        dask_client = dask.distributed.default_client()
    except ValueError:
        with Client(cluster, direct_to_workers=True, asynchronous=True) as dask_client:
            yield dask_client
    else:
        yield dask_client


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    path = os.getenv("ERLAB_TEST_DATA_DIR", None)
    if path is None:
        cache_folder = pooch.os_cache("erlabpy")
        path = cache_folder / f"kmnhan-erlabpy-data-{DATA_COMMIT_HASH[:7]}"
        data_url = (
            "https://codeload.github.com/kmnhan/erlabpy-data/legacy.tar.gz/"
            + DATA_COMMIT_HASH
        )
        if not _test_data_dir_ready(path):
            lock_path = cache_folder / f".erlabpy-data-{DATA_COMMIT_HASH[:7]}.lock"
            with _test_data_download_lock(lock_path):
                if not _test_data_dir_ready(path):
                    for attempt in range(1, DATA_RETRIEVE_ATTEMPTS + 1):
                        try:
                            pooch.retrieve(
                                data_url,
                                known_hash=DATA_KNOWN_HASH,
                                path=cache_folder,
                                processor=pooch.Untar(extract_dir=""),
                            )
                            break
                        except (
                            requests.exceptions.ConnectionError,
                            requests.exceptions.HTTPError,
                            requests.exceptions.Timeout,
                        ) as exc:
                            if isinstance(exc, requests.exceptions.HTTPError):
                                response = exc.response
                                if response is not None and response.status_code < 500:
                                    raise
                            if attempt == DATA_RETRIEVE_ATTEMPTS:
                                raise
                            delay = 2 ** (attempt - 1)
                            log.warning(
                                "Failed to retrieve erlabpy test data from %s on "
                                "attempt %d/%d; retrying in %d seconds.",
                                data_url,
                                attempt,
                                DATA_RETRIEVE_ATTEMPTS,
                                delay,
                                exc_info=True,
                            )
                            time.sleep(delay)
                    if not _test_data_dir_ready(path):
                        raise FileNotFoundError(
                            f"Retrieved erlabpy test data is incomplete: {path}"
                        )

    return pathlib.Path(path)


def _exp_decay(t, n0, tau=1):
    return n0 * np.exp(-t / tau)


@pytest.fixture
def exp_decay_model():
    return lmfit.Model(_exp_decay)


@pytest.fixture(scope="session")
def fit_test_darr():
    t = np.arange(0, 5, 0.5)
    da = xr.DataArray(
        np.stack([_exp_decay(t, 3, 3), _exp_decay(t, 5, 4), np.nan * t], axis=-1),
        dims=("t", "x"),
        coords={"t": t, "x": [0, 1, 2]},
    )
    da[0, 0] = np.nan
    return da


@pytest.fixture(scope="session")
def anglemap():
    return generate_data_angles(shape=(10, 10, 10), assign_attributes=True)


@pytest.fixture(scope="session")
def cut():
    return generate_data_angles(
        (300, 1, 500),
        angrange={"alpha": (-15, 15), "beta": (4.5, 4.5)},
        assign_attributes=True,
    ).T


@pytest.fixture(scope="session")
def gold() -> xr.DataArray:
    return generate_gold_edge(
        (15, 150), temp=100, Eres=1e-2, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )


@pytest.fixture(scope="session")
def gold_fit_res(gold) -> xr.Dataset:
    return erlab.analysis.gold.poly(
        gold,
        angle_range=(-13.5, 13.5),
        eV_range=(-0.204, 0.276),
        fast=True,
        parallel_kw={"backend": "threading"},
    )


@pytest.fixture(scope="session")
def gold_fit_res_fd(gold) -> xr.Dataset:
    return erlab.analysis.gold.poly(
        gold,
        angle_range=(-13.5, 13.5),
        eV_range=(-0.204, 0.276),
        fast=False,
        parallel_kw={"backend": "threading"},
    )


@pytest.fixture(scope="session")
def gold_fine():
    return generate_gold_edge(
        (400, 500), temp=100, Eres=1e-2, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )


@pytest.fixture(scope="session")
def manager_context() -> Callable[
    ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
]:
    def _drain_qt_events(iterations: int = 1) -> None:
        for _ in range(iterations):
            QtWidgets.QApplication.sendPostedEvents(
                None, int(QtCore.QEvent.Type.DeferredDelete.value)
            )
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents()

    @contextlib.contextmanager
    def _ctx(
        use_socket: bool = False,
    ):
        original_port = imagetool_manager.PORT
        original_port_watch = imagetool_manager.PORT_WATCH
        original_server_port = imagetool_manager_server.PORT
        original_server_port_watch = imagetool_manager_server.PORT_WATCH
        original_registry_path = imagetool_manager_registry._REGISTRY_PATH
        original_lock_path = imagetool_manager_registry._LOCK_PATH
        original_default_manager = imagetool_manager_registry.get_default_manager(
            validate=False
        )

        registry_path = (
            pathlib.Path(tempfile.gettempdir())
            / f"erlab-test-manager-{os.getpid()}-{time.time_ns()}.json"
        )
        imagetool_manager_registry._REGISTRY_PATH = registry_path
        imagetool_manager_registry._LOCK_PATH = registry_path.with_suffix(
            registry_path.suffix + ".lock"
        )
        imagetool_manager_registry.clear_default_manager()
        imagetool_manager.PORT = 0
        imagetool_manager.PORT_WATCH = 0
        imagetool_manager_server.PORT = 0
        imagetool_manager_server.PORT_WATCH = 0
        imagetool_manager._always_use_socket = use_socket

        _drain_qt_events()
        imagetool_manager.main(execute=False)

        try:
            yield imagetool_manager._manager_instance
        finally:
            manager = imagetool_manager._manager_instance
            if manager is not None:
                manager._workspace_state.loading_depth += 1
                server = manager.server
                watcher_server = manager.watcher_server
                try:

                    def _thread_is_running(thread: object) -> bool:
                        with contextlib.suppress(RuntimeError):
                            return qt_is_valid(thread) and thread.isRunning()
                        return False

                    def _stop_thread(thread: object) -> None:
                        if qt_is_valid(thread):
                            thread.stop(timeout_ms=1000)

                    qapp = QtWidgets.QApplication.instance()
                    if (
                        isinstance(qapp, QtWidgets.QApplication)
                        and manager._application_quit_filter is not None
                    ):
                        qapp.removeEventFilter(manager._application_quit_filter)
                        manager._application_quit_filter = None
                    manager._registry_heartbeat_timer.stop()
                    manager._registry_heartbeat.stop()
                    _stop_thread(server)
                    _stop_thread(watcher_server)
                    _drain_qt_events()
                    clipboard = QtWidgets.QApplication.clipboard()
                    if clipboard is not None:
                        clipboard.clear()
                    manager._close_standalone_apps()
                    _drain_qt_events()
                    manager.remove_all_tools()
                    manager._workspace_controller._mark_workspace_clean()
                    manager.close()
                    deadline = time.perf_counter() + 5.0
                    while (
                        _thread_is_running(server) or _thread_is_running(watcher_server)
                    ) and time.perf_counter() < deadline:
                        _drain_qt_events()
                        time.sleep(0.01)
                    if _thread_is_running(server):
                        _stop_thread(server)
                    if _thread_is_running(watcher_server):
                        _stop_thread(watcher_server)
                finally:
                    if qt_is_valid(manager):
                        manager._workspace_state.loading_depth -= 1
                        manager._workspace_controller._mark_workspace_clean()
                if qt_is_valid(manager):
                    manager.deleteLater()
                    delete_deadline = time.perf_counter() + 1.0
                    while (
                        qt_is_valid(manager) and time.perf_counter() < delete_deadline
                    ):
                        _drain_qt_events()
                        time.sleep(0.01)
                _drain_qt_events(iterations=3)
            imagetool_manager._manager_instance = None
            imagetool_manager._always_use_socket = False
            imagetool_manager.PORT = original_port
            imagetool_manager.PORT_WATCH = original_port_watch
            imagetool_manager_server.PORT = original_server_port
            imagetool_manager_server.PORT_WATCH = original_server_port_watch
            imagetool_manager_registry._REGISTRY_PATH = original_registry_path
            imagetool_manager_registry._LOCK_PATH = original_lock_path
            if original_default_manager is None:
                imagetool_manager_registry.clear_default_manager()
            else:
                imagetool_manager_registry._default_manager_index = (
                    original_default_manager
                )
            for path in (registry_path, registry_path.with_suffix(".json.lock")):
                with contextlib.suppress(OSError):
                    path.unlink()

    return _ctx


class _DialogDetectionThread(QtCore.QThread):
    sigUpdate = QtCore.Signal(object)
    sigTimeout = QtCore.Signal(int)
    sigTrigger = QtCore.Signal(int, object)
    sigPreCall = QtCore.Signal(int, object)

    def __init__(
        self,
        parent: QtCore.QObject | None,
        index: int,
        pre_call: Callable | None,
        timeout: float,
        ignored_dialog: QtWidgets.QDialog | None = None,
    ) -> None:
        super().__init__(parent)
        self.pre_call = pre_call
        self.index = index
        self.timeout = timeout
        self.ignored_dialog = ignored_dialog
        self._precall_called = threading.Event()

    def precall_called(self):
        if self.isRunning():
            self.mutex.lock()
        self._precall_called.set()
        if self.isRunning():
            self.mutex.unlock()

    def run(self):
        self.mutex = QtCore.QMutex()
        time.sleep(0.001)
        start_time = time.perf_counter()

        dialog = None

        log.debug("looking for dialog %d...", self.index)
        while dialog is None and time.perf_counter() - start_time < self.timeout:
            candidate = QtWidgets.QApplication.activeModalWidget()
            # The next detector starts before accepting the current modal so
            # synchronous child dialogs are not missed; ignore that modal
            # while it closes.
            if (
                candidate is not None
                and candidate is not self.ignored_dialog
                and not isinstance(candidate, _WaitDialog)
            ):
                dialog = candidate
            time.sleep(0.01)

        if dialog is None or isinstance(dialog, _WaitDialog):
            log.debug("emitting timeout %d", self.index)
            self.sigTimeout.emit(self.index)
            return

        log.debug("dialog %d detected: %s", self.index, dialog)

        if self.pre_call is not None:
            log.debug("pre_call %d...", self.index)
            self.sigPreCall.emit(self.index, dialog)
            while not self._precall_called.is_set():
                time.sleep(0.01)
            log.debug("pre_call %d done", self.index)

        log.debug("emitting trigger for %d", self.index + 1)
        self.sigTrigger.emit(self.index + 1, dialog)


class _DialogHandler(QtCore.QObject):
    """Accept a dialog during testing.

    If there is no dialog, it waits until one is created for a maximum of 5 seconds (by
    default). Adapted from `this issue comment on pytest-qt
    <https://github.com/pytest-dev/pytest-qt/issues/256#issuecomment-1915675942>`_.

    Parameters
    ----------
    dialog_trigger
        Callable that triggers the dialog creation. Takes no arguments.
    timeout
        Maximum time in seconds to wait for the dialog creation.
    pre_call
        Callable that takes the dialog as a single argument. If provided, it is executed
        prior to calling ``.accept()`` on the dialog. This is useful if some dialog
        elements need to be interacted with before accepting the dialog.

        If a sequence of callables of length equal to ``chained_dialogs`` is provided,
        each callable will be called before each dialog is accepted.
    accept_call
        If provided, it is called instead of ``.accept()`` on the dialog. If a sequence
        of callables of length equal to ``chained_dialogs`` is provided, each callable
        will be called instead of ``.accept()`` on each dialog. When a sequence is
        provided, the first callable is called for the second dialog, and so on.
        Elements of the sequence can be ``None``, in which case ``.accept()`` is called
        for that dialog.

        If the dialog is a ``QMessageBox``, ``.defaultButton().click()`` is called
        instead of ``.accept()``.
    chained_dialogs
        Number of dialogs expected to be created in a chain. The first dialog is created
        by calling ``dialog_trigger``. When that dialog is accepted, a second dialog is
        expected to be created, and so on. For example, if 2, a second dialog is
        expected to be created right after the first dialog is accepted. The new dialog
        will also be automatically accepted.
    """

    sigFinished = QtCore.Signal()

    def __init__(self, qtbot):
        super().__init__()
        self._qtbot = qtbot

    def __call__(
        self,
        dialog_trigger: Callable,
        timeout: float = 5.0,
        pre_call: Callable | Sequence[Callable | None] | None = None,
        accept_call: Callable | Sequence[Callable | None] | None = None,
        chained_dialogs: int = 1,
    ):
        self.timeout: float = timeout
        self._timed_out = False

        if not isinstance(pre_call, Sequence):
            pre_call = [pre_call] + [None] * (chained_dialogs - 1)

        if not isinstance(accept_call, Sequence):
            accept_call = [accept_call] + [None] * (chained_dialogs - 1)
        self._pre_call_list = pre_call
        self._accept_call_list = accept_call
        self._max_index = chained_dialogs - 1

        with self._qtbot.wait_signal(self.sigFinished, timeout=round(timeout * 1e3)):
            self.trigger_index(0, dialog_trigger)

    @QtCore.Slot(int)
    def _timeout(self, index: int) -> None:
        log.debug("timeout %d", index)
        self._timed_out = True
        if hasattr(self, "_handler") and self._handler.isRunning():
            self._handler.wait()
            self._handler = None

        pytest.fail(
            f"No dialog for index {index} was created after {self.timeout} seconds."
        )

    @QtCore.Slot(int, object)
    def trigger_index(
        self, index: int, dialog_or_trigger: QtWidgets.QDialog | Callable
    ) -> None:
        """Trigger the dialog creation.

        Parameters
        ----------
        index
            The index of the dialog to trigger, starting from 0. If the index is greater
            than 0, ``dialog_or_trigger`` should be a dialog.
        dialog_or_trigger
            The callable that triggers the dialog creation or a previously created
            dialog which will create the next dialog upon acceptance.
        """
        log.debug("index %d triggered", index)

        if index <= self._max_index:
            if hasattr(self, "_handler") and self._handler.isRunning():
                self._handler.wait()
                self._handler = None

            self._handler = _DialogDetectionThread(
                self,
                index,
                self._pre_call_list[index],
                self.timeout,
                (
                    dialog_or_trigger
                    if isinstance(dialog_or_trigger, QtWidgets.QDialog)
                    else None
                ),
            )
            self._handler.sigTimeout.connect(self._timeout)
            self._handler.sigTrigger.connect(self.trigger_index)
            self._handler.sigPreCall.connect(self.handle_pre_call)
            self._handler.start()

        if isinstance(dialog_or_trigger, QtWidgets.QDialog):
            accept_call = self._accept_call_list[index - 1]

            if accept_call is not None:
                accept_call(dialog_or_trigger)
            else:
                if (
                    isinstance(dialog_or_trigger, QtWidgets.QMessageBox)
                    and dialog_or_trigger.defaultButton() is not None
                ):
                    dialog_or_trigger.defaultButton().click()
                else:
                    dialog_or_trigger.accept()
            log.debug("finished %d", index - 1)

            if index > self._max_index:
                log.debug("all dialogs finished, emitting sigFinished")
                self.sigFinished.emit()

        else:
            dialog_or_trigger()

    @QtCore.Slot(int, object)
    def handle_pre_call(self, index: int, dialog: QtWidgets.QDialog) -> None:
        log.debug("pre-call callable received")
        self._pre_call_list[index](dialog)
        log.debug("pre-call successfully called")
        self._handler.precall_called()


@pytest.fixture
def accept_dialog(qtbot):
    return _DialogHandler(qtbot=qtbot)


def _move_and_compare_values(bot, win, expected, cursor=0, target_win=None):
    if target_win is None:
        target_win = win
    with bot.waitExposed(win):
        target_win.show()
        target_win.activateWindow()
        target_win.setFocus()

    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])

    x_ax = win.slicer_area.main_image.display_axis[0]
    y_ax = win.slicer_area.main_image.display_axis[1]

    x0, y0 = (
        win.slicer_area.get_current_index(x_ax),
        win.slicer_area.get_current_index(y_ax),
    )

    # Move left
    win.slicer_area.step_index(x_ax, -1)
    bot.waitUntil(
        lambda: win.slicer_area.get_current_index(x_ax) == x0 - 1, timeout=2000
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[1])

    # Move down
    win.slicer_area.step_index(y_ax, -1)
    bot.waitUntil(
        lambda: win.slicer_area.get_current_index(y_ax) == y0 - 1, timeout=2000
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[2])

    # Move right
    win.slicer_area.step_index(x_ax, 1)
    bot.waitUntil(lambda: win.slicer_area.get_current_index(x_ax) == x0, timeout=2000)
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[3])

    # Move up
    win.slicer_area.step_index(y_ax, 1)
    bot.waitUntil(lambda: win.slicer_area.get_current_index(y_ax) == y0, timeout=2000)
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])


@pytest.fixture
def move_and_compare_values():
    return _move_and_compare_values


@pytest.fixture(autouse=True)
def cover_qthreads(monkeypatch, qtbot, pytestconfig):
    # https://github.com/nedbat/coveragepy/issues/686#issuecomment-2286288111
    from qtpy.QtCore import QThread

    coverage_active = _coverage_is_active(pytestconfig)
    if not coverage_active:
        return

    base_constructor = QThread.__init__

    def run_with_trace(self):  # pragma: no cover
        # https://github.com/nedbat/coveragepy/issues/686#issuecomment-634932753
        sys.settrace(threading._trace_hook)
        self._base_run()

    def init_with_trace(self, *args, **kwargs):
        base_constructor(self, *args, **kwargs)
        self._base_run = self.run
        self.run = functools.partial(run_with_trace, self)

    monkeypatch.setattr(QThread, "__init__", init_with_trace)


@pytest.fixture(autouse=True)
def cover_qthreadpool(monkeypatch, qtbot, pytestconfig):
    # https://github.com/nedbat/coveragepy/issues/686#issuecomment-2435049275
    from qtpy.QtCore import QThreadPool

    base_start = QThreadPool.start
    QThreadPool.globalInstance().setMaxThreadCount(1)
    coverage_active = _coverage_is_active(pytestconfig)
    if not coverage_active:
        return

    def start_with_trace(self, runnable, *args, **kwargs):
        original_run = runnable.run

        def wrapped_run(*a, **kw):
            sys.settrace(threading._trace_hook)
            return original_run(*a, **kw)

        runnable.run = wrapped_run
        return base_start(self, runnable, *args, **kwargs)

    monkeypatch.setattr(QThreadPool, "start", start_with_trace)


@pytest.fixture(scope="session", autouse=True)
def serialize_hdf5_loads():
    """Prevent non-threadsafe HDF5 wheels from crashing during threaded loads."""
    mp = pytest.MonkeyPatch()
    from erlab.interactive.explorer import _base_explorer
    from erlab.interactive.imagetool.manager import _io

    lock = threading.Lock()
    original_data_loader_run = _io._DataLoader.run
    original_repr_fetcher_run = _base_explorer._ReprFetcher.run

    def locked_data_loader_run(self):
        with lock:
            return original_data_loader_run(self)

    def locked_repr_fetcher_run(self):
        with lock:
            return original_repr_fetcher_run(self)

    mp.setattr(_io._DataLoader, "run", locked_data_loader_run)
    mp.setattr(_base_explorer._ReprFetcher, "run", locked_repr_fetcher_run)
    try:
        yield
    finally:
        mp.undo()


@pytest.fixture(scope="session", autouse=True)
def patch_pyqtgraph_boundingrect():
    """Guard pyqtgraph InfiniteLine boundingRect during Qt teardown."""
    mp = pytest.MonkeyPatch()
    import pyqtgraph as pg
    from qtpy import QtCore

    original_br = pg.InfiniteLine.boundingRect

    def safe_br(self):
        if not qt_is_valid(self):
            return QtCore.QRectF()
        try:
            vb = self.getViewBox()
        except RuntimeError as exc:
            if _is_deleted_qt_wrapper_error(exc):
                return QtCore.QRectF()
            raise
        if vb is None:
            return QtCore.QRectF()
        try:
            return original_br(self)
        except RuntimeError as exc:
            if _is_deleted_qt_wrapper_error(exc):
                return QtCore.QRectF()
            raise

    mp.setattr(pg.InfiniteLine, "boundingRect", safe_br, raising=False)
    try:
        yield
    finally:
        mp.undo()


@pytest.fixture
def ip_shell():
    """IPython shell with the interactive extension loaded."""
    from IPython.testing.globalipapp import start_ipython

    ip_session = start_ipython()
    ip_session.run_line_magic("load_ext", "erlab.interactive")

    yield ip_session

    ip_session.run_line_magic("unload_ext", "erlab.interactive")
    ip_session.user_ns.clear()
    ip_session.clear_instance()
    with contextlib.suppress(AttributeError):
        del start_ipython.already_called


def make_data(beta=5.0, temp=20.0, hv=50.0, bandshift=0.0):
    data = generate_data_angles(
        shape=(250, 1, 300),
        angrange={"alpha": (-15, 15), "beta": (beta, beta)},
        hv=hv,
        configuration=1,
        temp=temp,
        bandshift=bandshift,
        assign_attributes=False,
        seed=1,
    ).T

    # Rename coordinates. The loader must rename them back to the original names.
    data = data.rename(
        {
            "alpha": "ThetaX",
            "beta": "Polar",
            "eV": "BindingEnergy",
            "hv": "PhotonEnergy",
            "xi": "Tilt",
            "delta": "Azimuth",
        }
    )
    dt = datetime.datetime.now()

    # Assign some attributes that real data would have
    return data.assign_attrs(
        {
            "LensMode": "Angular30",  # Lens mode of the analyzer
            "SpectrumType": "Fixed",  # Acquisition mode of the analyzer
            "PassEnergy": 10,  # Pass energy of the analyzer
            "UndPol": 0,  # Undulator polarization
            "Date": dt.strftime(r"%d/%m/%Y"),  # Date of the measurement
            "Time": dt.strftime("%I:%M:%S %p"),  # Time of the measurement
            "TB": temp,
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }
    )


@pytest.fixture(scope="session")
def example_data_dir(tmp_path_factory) -> pathlib.Path:
    tmp_dir: pathlib.Path = tmp_path_factory.mktemp("example_data")

    # Generate a map
    beta_coords = np.linspace(2, 7, 10)

    # Generate and save cuts with different beta values
    data_2d = []
    for i, beta in enumerate(beta_coords):
        data = make_data(beta=beta, temp=20.0 + i, hv=50.0)
        filename = tmp_dir / f"data_001_S{str(i + 1).zfill(3)}.h5"
        data.to_netcdf(filename, engine="h5netcdf")
        data_2d.append(data)

    data_2d = xr.concat(data_2d, dim="Polar")

    # Write scan coordinates to a csv file
    with open(tmp_dir / "data_001_axis.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Polar"])

        for i, beta in enumerate(beta_coords):
            writer.writerow([i + 1, beta])

    # Generate some cuts with different band shifts
    for i in range(4):
        data = make_data(beta=5.0, temp=20.0, hv=50.0, bandshift=-i * 0.05)
        filename = tmp_dir / f"data_{str(i + 2).zfill(3)}.h5"
        data.to_netcdf(filename, engine="h5netcdf")

    # Save map data
    data_2d.to_netcdf(tmp_dir / "data_006.h5", engine="h5netcdf")

    # Save XPS data
    data_2d.isel(Polar=0, ThetaX=0).to_netcdf(
        tmp_dir / "data_007.h5", engine="h5netcdf"
    )

    # Save data with wrong file extension
    wrong_file = tmp_dir / "data_010.nc"
    data.to_netcdf(wrong_file, engine="h5netcdf")

    return tmp_dir


@pytest.fixture(scope="session")
def example_loader():
    def _format_polarization(val) -> str:
        val = round(float(val))
        return {0: "LH", 2: "LV", -1: "RC", 1: "LC"}.get(val, str(val))

    def _parse_time(darr: xr.DataArray) -> datetime.datetime:
        return datetime.datetime.strptime(
            f"{darr.attrs['Date']} {darr.attrs['Time']}", r"%d/%m/%Y %I:%M:%S %p"
        )

    def _determine_kind(darr: xr.DataArray) -> str:
        if "scan_type" in darr.attrs and darr.attrs["scan_type"] == "live":
            return "LP" if "beta" in darr.dims else "LXY"

        data_type = "xps"
        if "alpha" in darr.dims:
            data_type = "cut"
        if "beta" in darr.dims:
            data_type = "map"
        if "hv" in darr.dims:
            data_type = "hvdep"
        return data_type

    class ExampleLoader(LoaderBase):
        name = "example"
        description = "Example loader for testing purposes"
        extensions: typing.ClassVar[set[str]] = {".h5"}

        name_map: typing.ClassVar[dict] = {
            "eV": "BindingEnergy",
            "alpha": "ThetaX",
            "beta": [
                "Polar",
                "Polar Compens",
            ],  # Can have multiple names assigned to the same name
            # If both are present in the data, a ValueError will be raised
            "delta": "Azimuth",
            "xi": "Tilt",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "hv": "PhotonEnergy",
            "polarization": "UndPol",
            "sample_temp": "TB",
        }

        coordinate_attrs = (
            "beta",
            "delta",
            "xi",
            "hv",
            "x",
            "y",
            "z",
            "polarization",
            "photon_flux",
            "sample_temp",
        )
        # Attributes to be used as coordinates. Place all attributes that we don't want
        # to lose when merging multiple file scans here.

        additional_attrs: typing.ClassVar[dict] = {
            "configuration": 1,  # Experimental geometry required for kspace conversion
            "sample_workfunction": 4.3,
        }  # Any additional metadata you want to add to the data

        formatters: typing.ClassVar[dict] = {
            "polarization": _format_polarization,
            "LensMode": lambda x: x.replace("Angular", "A"),
        }

        summary_attrs: typing.ClassVar[dict] = {
            "Time": _parse_time,
            "Type": _determine_kind,
            "Lens Mode": "LensMode",
            "Scan Type": "SpectrumType",
            "T(K)": "sample_temp",
            "Pass E": "PassEnergy",
            "Polarization": "polarization",
            "hv": "hv",
            "x": "x",
            "y": "y",
            "z": "z",
            "polar": "beta",
            "tilt": "xi",
            "azi": "delta",
        }

        summary_sort = "File Name"

        skip_validate = False

        always_single = False

        def identify(self, num, data_dir):
            coord_dict = {}
            data_dir = pathlib.Path(data_dir)

            # Look for scans with data_###_S###.h5, and sort them
            files = list(data_dir.glob(f"data_{str(num).zfill(3)}_S*.h5"))
            files.sort()

            if len(files) == 0:
                # If no files found, look for data_###.h5
                files = list(data_dir.glob(f"data_{str(num).zfill(3)}.h5"))
            else:
                # If files found, extract coordinate values from the filenames
                axis_file = data_dir / f"data_{str(num).zfill(3)}_axis.csv"
                with axis_file.open("r", encoding="locale") as f:
                    header = f.readline().strip().split(",")

                coord_arr = np.loadtxt(axis_file, delimiter=",", skiprows=1)

                for i, hdr in enumerate(header[1:]):
                    coord_dict[hdr] = coord_arr[: len(files), i + 1].astype(np.float64)

            if len(files) == 0:
                # If no files found up to this point, return None
                return None

            return files, coord_dict

        def load_single(self, file_path, without_values=False):
            darr = xr.open_dataarray(file_path, engine="h5netcdf")

            if without_values:
                # Do not load the data into memory
                return xr.DataArray(
                    np.zeros(darr.shape, darr.dtype),
                    coords=darr.coords,
                    dims=darr.dims,
                    attrs=darr.attrs,
                    name=darr.name,
                )

            return darr

        def post_process(self, data: xr.DataArray) -> xr.DataArray:
            data = super().post_process(data)

            if "sample_temp" in data.coords:
                # Add temperature to attributes, for backwards compatibility
                temp = float(data.sample_temp.mean())
                data = data.assign_attrs(sample_temp=temp)

            return data

        def infer_index(self, name):
            # Get the scan number from file name
            try:
                scan_num: str = re.match(r".*?(\d{3})(?:_S\d{3})?", name).group(1)
            except (AttributeError, IndexError):
                return None, None

            if scan_num.isdigit():
                return int(scan_num), {}
            return None, None

        def files_for_summary(self, data_dir):
            return erlab.io.utils.get_files(data_dir, extensions=[".h5"])

        @property
        def file_dialog_methods(self):
            return {"Example Raw Data (*.h5)": (self.load, {})}

    return ExampleLoader

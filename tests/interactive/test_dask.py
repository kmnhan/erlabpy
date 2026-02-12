import contextlib

import dask.distributed
import pytest
from qtpy import QtWidgets

import erlab
from erlab.interactive._dask import ClientSetupDialog, DaskMenu, _IntOrNoneWidget


@pytest.fixture
def main_window(qtbot):
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    win.show()
    return win


@pytest.fixture
def cleanup_dask_client():
    try:
        yield
    finally:
        try:
            client = dask.distributed.default_client()
        except ValueError:
            pass
        else:
            client.close()


def test_actions_visibility(main_window, monkeypatch):
    menu = DaskMenu(main_window)

    # No default client
    menu.update_actions_visibility()
    assert menu.connect_existing_action.isEnabled()
    assert menu.create_local_action.isEnabled()
    assert not menu.close_client_action.isEnabled()
    assert not menu.open_dashboard_action.isEnabled()
    assert not menu.about_client_action.isEnabled()

    # Simulate a default client without launching a real distributed cluster.
    monkeypatch.setattr(dask.distributed, "default_client", lambda: object())
    menu.update_actions_visibility()
    assert not menu.connect_existing_action.isEnabled()
    assert not menu.create_local_action.isEnabled()
    assert menu.close_client_action.isEnabled()
    assert menu.open_dashboard_action.isEnabled()
    assert menu.about_client_action.isEnabled()


def test_create_local_cluster(main_window, accept_dialog, monkeypatch):
    menu = DaskMenu(main_window)
    created_kwargs: dict[str, object] = {}
    fake_client: object | None = None

    class _FakeClient:
        def close(self) -> None:
            nonlocal fake_client
            fake_client = None

    def _default_client():
        if fake_client is None:
            raise ValueError("No default client")
        return fake_client

    def _client_ctor(**kwargs):
        nonlocal fake_client
        created_kwargs.update(kwargs)
        fake_client = _FakeClient()
        return fake_client

    @contextlib.contextmanager
    def _wait_stub(*args, **kwargs):
        yield None

    monkeypatch.setattr(dask.distributed, "Client", _client_ctor)
    monkeypatch.setattr(dask.distributed, "default_client", _default_client)
    monkeypatch.setattr(erlab.interactive.utils, "wait_dialog", _wait_stub)
    monkeypatch.setattr(menu, "show_client_info", lambda: None)

    def _configure_local_cluster(dialog):
        if hasattr(dialog, "_processes_checkbox"):
            dialog._processes_checkbox.setChecked(False)
            if dialog._n_workers_widget._auto_check.isChecked():
                dialog._n_workers_widget._auto_check.setChecked(False)
            dialog._n_workers_widget._spin.setValue(1)
            if dialog._threads_per_worker_widget._auto_check.isChecked():
                dialog._threads_per_worker_widget._auto_check.setChecked(False)
            dialog._threads_per_worker_widget._spin.setValue(1)

    accept_dialog(menu.create_local_cluster, pre_call=_configure_local_cluster)
    assert menu.default_client is fake_client
    assert created_kwargs["set_as_default"] is True
    assert created_kwargs["processes"] is False
    assert created_kwargs["n_workers"] == 1
    assert created_kwargs["threads_per_worker"] == 1


def test_connect_existing_cluster(main_window, accept_dialog, monkeypatch):
    menu = DaskMenu(main_window)
    fake_client: object | None = None
    errors: list[tuple[str, str]] = []

    class _Scheduler:
        def __init__(self, address: str) -> None:
            self.address = address

    class _FakeClient:
        def __init__(self, address: str) -> None:
            self.scheduler = _Scheduler(address)

        def close(self) -> None:
            nonlocal fake_client
            fake_client = None

    def _default_client():
        if fake_client is None:
            raise ValueError("No default client")
        return fake_client

    def _client_ctor(*, address: str, timeout: int, set_as_default: bool, **kwargs):
        del timeout, kwargs
        nonlocal fake_client
        if address == "nonexistent:8786":
            raise RuntimeError("Connection failed")
        if set_as_default:
            fake_client = _FakeClient(address)
        return fake_client

    @contextlib.contextmanager
    def _wait_stub(*args, **kwargs):
        yield None

    monkeypatch.setattr(dask.distributed, "Client", _client_ctor)
    monkeypatch.setattr(dask.distributed, "default_client", _default_client)
    monkeypatch.setattr(erlab.interactive.utils, "wait_dialog", _wait_stub)
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda parent, title, text: errors.append((title, text)),
    )

    def populate_dialog(dialog: ClientSetupDialog):
        dialog._scheduler_address_edit.setText("tcp://host:8786")

    accept_dialog(menu.connect_existing_cluster, pre_call=populate_dialog)
    assert menu.default_client is not None
    assert menu.default_client.scheduler.address == "tcp://host:8786"

    # Close the client for next part of the test
    menu.close_client()
    assert menu.default_client is None

    def populate_dialog_fail(dialog: ClientSetupDialog):
        dialog._scheduler_address_edit.setText("nonexistent:8786")
        dialog._timeout_spin.setValue(1)

    accept_dialog(menu.connect_existing_cluster, pre_call=populate_dialog_fail)
    assert menu.default_client is None
    assert errors


def test_int_or_none_widget(qapp):
    w = _IntOrNoneWidget()
    # Default is Auto -> None
    assert w.value() is None

    # Uncheck auto to enable spin
    w._auto_check.setChecked(False)
    w._spin.setValue(7)
    assert w.value() == 7

    # Re-enable auto
    w._auto_check.setChecked(True)
    assert w.value() is None


def test_client_setup_dialog_getters(qapp):
    dlg = ClientSetupDialog()
    dlg._scheduler_address_edit.setText("  tcp://host:8786  ")
    dlg._timeout_spin.setValue(12)
    assert dlg.scheduler_address() == "tcp://host:8786"
    assert dlg.timeout() == 12

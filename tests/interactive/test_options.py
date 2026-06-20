from __future__ import annotations

import typing

import pytest
from qtpy import QtWidgets

from erlab.interactive._options import OptionDialog, options
from erlab.interactive._options.core import (
    OptionManager,
    model_with_workspace_overrides,
    normalize_workspace_option_overrides,
    option_paths,
    workspace_overridable_option_paths,
)
from erlab.interactive._options.parameters import StylesheetListWidget
from erlab.interactive._options.schema import AppOptions


@pytest.fixture(autouse=True)
def isolated_interactive_options(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ERLAB_INTERACTIVE_OPTIONS_PATH",
        str(tmp_path / "interactive-options.ini"),
    )
    options.restore()


@pytest.fixture
def dialog(qtbot):
    dlg = OptionDialog()
    qtbot.addWidget(dlg)
    return dlg


class _WorkspaceManagerStub(QtWidgets.QWidget):
    def __init__(
        self,
        overrides: dict[str, typing.Any] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.overrides = normalize_workspace_option_overrides(overrides)
        self.dirty_marks: list[bool] = []

    def workspace_option_overrides(self) -> dict[str, typing.Any]:
        return dict(self.overrides)

    def _set_workspace_option_overrides(
        self,
        overrides: typing.Mapping[str, typing.Any],
        *,
        mark_dirty: bool = True,
    ) -> None:
        self.overrides = normalize_workspace_option_overrides(overrides)
        self.dirty_marks.append(mark_dirty)

    def set_workspace_option_override(self, path: str, value: typing.Any) -> None:
        overrides = self.workspace_option_overrides()
        overrides[path] = value
        self._set_workspace_option_overrides(overrides)

    def clear_workspace_option_override(self, path: str) -> None:
        overrides = self.workspace_option_overrides()
        overrides.pop(path, None)
        self._set_workspace_option_overrides(overrides)

    @property
    def effective_interactive_options(self) -> AppOptions:
        return model_with_workspace_overrides(options.model, self.overrides)


def _control(
    dialog: OptionDialog, scope: str, path: str, cls: type[QtWidgets.QWidget]
) -> QtWidgets.QWidget:
    widget = dialog.findChild(cls, f"settingsControl_{scope}_{path.replace('/', '__')}")
    if widget is None:
        raise AssertionError(f"Missing control for {scope}:{path}")
    return widget


def _button(dialog: OptionDialog, scope: str, path: str) -> QtWidgets.QToolButton:
    widget = dialog.findChild(
        QtWidgets.QToolButton,
        f"settingsReset_{scope}_{path.replace('/', '__')}",
    )
    if widget is None:
        raise AssertionError(f"Missing action for {scope}:{path}")
    return widget


def _override(dialog: OptionDialog, path: str) -> QtWidgets.QCheckBox:
    widget = dialog.findChild(
        QtWidgets.QCheckBox,
        f"settingsOverride_{path.replace('/', '__')}",
    )
    if widget is None:
        raise AssertionError(f"Missing override switch for {path}")
    return widget


def test_dialog_initial_settings(dialog: OptionDialog):
    assert dialog.current_options.model_dump() == options.model.model_dump()
    assert not dialog.modified


def test_dialog_native_structure(dialog: OptionDialog):
    assert dialog.findChild(QtWidgets.QDialogButtonBox) is None
    assert dialog.findChild(QtWidgets.QTabBar, "settingsScopeTabs").count() == 1
    assert dialog.findChild(QtWidgets.QListWidget, "settingsCategoryList").count() == 4
    assert dialog.findChild(QtWidgets.QStackedWidget, "settingsPageStack").count() == 4
    assert dialog.findChild(QtWidgets.QPushButton, "settingsRevertButton") is not None
    page = dialog.findChild(QtWidgets.QScrollArea, "settingsPage_user_colors")
    if page is None:
        raise AssertionError("Missing settings page")
    assert page.frameShape() == QtWidgets.QFrame.Shape.NoFrame
    container = dialog.findChild(QtWidgets.QWidget, "settingsPageContainer_user_colors")
    if container is None or container.layout() is None:
        raise AssertionError("Missing settings page container")
    assert container.layout().contentsMargins().right() > 0


def test_dialog_close_button_closes_window(qtbot):
    dlg = OptionDialog()
    qtbot.addWidget(dlg)
    dlg.show()
    assert dlg.isVisible()

    close_button = dlg.findChild(QtWidgets.QPushButton, "settingsCloseButton")
    if close_button is None:
        raise AssertionError("Missing close button")
    close_button.click()

    qtbot.waitUntil(lambda: not dlg.isVisible())


def test_stylesheet_editor_fits_settings_page(dialog: OptionDialog, qtbot):
    category_list = dialog.findChild(QtWidgets.QListWidget, "settingsCategoryList")
    if category_list is None:
        raise AssertionError("Missing settings category list")
    category_list.setCurrentRow(3)
    dialog.resize(dialog.minimumSize())
    dialog.show()

    page = dialog.findChild(QtWidgets.QScrollArea, "settingsPage_user_figure")
    if page is None:
        raise AssertionError("Missing figure settings page")
    _control(
        dialog,
        "user",
        "figure/stylesheets",
        StylesheetListWidget,
    )

    qtbot.waitUntil(lambda: page.horizontalScrollBar().maximum() == 0)


def test_user_edit_saves_immediately(dialog: OptionDialog):
    combo = typing.cast(
        "QtWidgets.QComboBox",
        _control(dialog, "user", "colors/cmap/name", QtWidgets.QComboBox),
    )
    combo.setCurrentText("bwr")

    assert options.model.colors.cmap.name == "bwr"
    assert dialog.modified


def test_session_revert_restores_user_baseline(dialog: OptionDialog):
    combo = typing.cast(
        "QtWidgets.QComboBox",
        _control(dialog, "user", "colors/cmap/name", QtWidgets.QComboBox),
    )
    combo.setCurrentText("bwr")

    dialog.revert_changes()

    assert options.model.colors.cmap.name == AppOptions().colors.cmap.name
    assert not dialog.modified


def test_reused_dialog_revert_uses_reopen_baseline(qtbot):
    dlg = OptionDialog()
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitUntil(dlg.isVisible)
    combo = typing.cast(
        "QtWidgets.QComboBox",
        _control(dlg, "user", "colors/cmap/name", QtWidgets.QComboBox),
    )
    combo.setCurrentText("bwr")
    dlg.close()
    qtbot.waitUntil(lambda: not dlg.isVisible())

    dlg.show()
    qtbot.waitUntil(dlg.isVisible)
    assert not dlg.modified
    combo.setCurrentText("viridis")

    dlg.revert_changes()

    assert options.model.colors.cmap.name == "bwr"
    assert not dlg.modified


def test_user_row_reset_restores_default(qtbot):
    options.model = AppOptions().model_copy(
        update={
            "colors": AppOptions().colors.model_copy(
                update={
                    "cmap": AppOptions().colors.cmap.model_copy(update={"name": "bwr"})
                }
            )
        }
    )
    dlg = OptionDialog()
    qtbot.addWidget(dlg)

    _button(dlg, "user", "colors/cmap/name").click()

    assert options.model.colors.cmap.name == AppOptions().colors.cmap.name


def test_workspace_scope_shows_only_overridable_settings(qtbot):
    manager = _WorkspaceManagerStub()
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)

    assert dlg.findChild(QtWidgets.QTabBar, "settingsScopeTabs").count() == 2
    workspace_paths = {
        row.path for (scope, _path), row in dlg._rows.items() if scope == "workspace"
    }
    assert workspace_paths == set(workspace_overridable_option_paths())


def test_workspace_override_switch_saves_sparse_override(qtbot):
    manager = _WorkspaceManagerStub()
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    path = "colors/cmap/name"
    combo = typing.cast(
        "QtWidgets.QComboBox", _control(dlg, "workspace", path, QtWidgets.QComboBox)
    )
    override = _override(dlg, path)

    assert not combo.isEnabled()
    override.setChecked(True)
    combo.setCurrentText("bwr")

    assert manager.overrides[path] == "bwr"
    assert combo.isEnabled()


def test_workspace_row_action_removes_override(qtbot):
    path = "colors/cmap/name"
    manager = _WorkspaceManagerStub({path: "bwr"})
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    _button(dlg, "workspace", path).click()

    assert path not in manager.overrides


def test_session_revert_restores_user_and_workspace(qtbot):
    path = "colors/cmap/name"
    manager = _WorkspaceManagerStub({path: "viridis"})
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)

    user_combo = typing.cast(
        "QtWidgets.QComboBox",
        _control(dlg, "user", "colors/cmap/name", QtWidgets.QComboBox),
    )
    user_combo.setCurrentText("bwr")
    dlg.scope_tabs.setCurrentIndex(1)
    workspace_combo = typing.cast(
        "QtWidgets.QComboBox", _control(dlg, "workspace", path, QtWidgets.QComboBox)
    )
    workspace_combo.setCurrentText("magma")

    dialog_workspace_before_revert = manager.workspace_option_overrides()
    assert options.model.colors.cmap.name == "bwr"
    assert dialog_workspace_before_revert[path] == "magma"

    dlg.revert_changes()

    assert options.model.colors.cmap.name == AppOptions().colors.cmap.name
    assert manager.overrides == {path: "viridis"}
    assert not dlg.modified


def test_workspace_override_helpers_filter_to_curated_subset() -> None:
    paths = set(workspace_overridable_option_paths())

    assert "colors/cmap/name" in paths
    assert "colors/cmap/packages" not in paths
    assert "io/workspace/compress" not in paths
    assert normalize_workspace_option_overrides(
        {
            "colors/cmap/name": "bwr",
            "io/workspace/compress": False,
        }
    ) == {"colors/cmap/name": "bwr"}


def test_options_get_set():
    options.restore()

    assert options["colors/cmap/name"] == AppOptions().colors.cmap.name
    assert options["io/workspace/compress"] is True
    assert options["io/workspace/use_incremental"] is True
    assert options["io/workspace/incremental_save_on_remote"] is False

    options["colors/cmap/name"] = "viridis"
    options["io/workspace/compress"] = False
    options["io/workspace/use_incremental"] = False
    options["io/workspace/incremental_save_on_remote"] = True
    options["figure/stylesheets"] = ["classic", "missing-style"]

    assert options["colors/cmap/name"] == "viridis"
    assert options["io/workspace/compress"] is False
    assert options["io/workspace/use_incremental"] is False
    assert options["io/workspace/incremental_save_on_remote"] is True
    assert options["figure/stylesheets"] == ["classic", "missing-style"]
    assert not options.model.io.workspace.compress
    assert not options.model.io.workspace.use_incremental
    assert options.model.io.workspace.incremental_save_on_remote
    assert options.model.figure.stylesheets == ["classic", "missing-style"]

    options.restore()


def test_option_manager_uses_configured_settings_path(monkeypatch, tmp_path):
    settings_path = tmp_path / "interactive-options.ini"
    monkeypatch.setenv("ERLAB_INTERACTIVE_OPTIONS_PATH", str(settings_path))

    isolated_options = OptionManager()
    isolated_options["figure/stylesheets"] = ["classic", "missing-style"]

    assert isolated_options["figure/stylesheets"] == ["classic", "missing-style"]
    assert settings_path.exists()

    other_settings_path = tmp_path / "other-options.ini"
    monkeypatch.setenv("ERLAB_INTERACTIVE_OPTIONS_PATH", str(other_settings_path))

    assert OptionManager()["figure/stylesheets"] == AppOptions().figure.stylesheets


def test_workspace_overridable_paths_are_existing_option_paths() -> None:
    assert set(workspace_overridable_option_paths()) <= set(option_paths())

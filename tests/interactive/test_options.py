from __future__ import annotations

import typing

import pydantic
import pytest
from qtpy import QtWidgets

import erlab.interactive._options.ui as options_ui
from erlab.interactive._options import OptionDialog, options
from erlab.interactive._options.core import (
    OptionManager,
    model_with_workspace_overrides,
    normalize_workspace_option_overrides,
    option_paths,
    workspace_overridable_option_paths,
)
from erlab.interactive._options.parameters import (
    ColorListWidget,
    FigureDpiOverrideWidget,
    StylesheetListWidget,
)
from erlab.interactive._options.schema import AppOptions
from erlab.interactive.colors import ColorMapComboBox


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


def test_dialog_rebuild_pages_replaces_existing_pages(dialog: OptionDialog):
    old_pages = dialog.findChild(QtWidgets.QStackedWidget, "settingsPageStack").count()

    dialog._build_pages()

    assert (
        dialog.findChild(QtWidgets.QStackedWidget, "settingsPageStack").count()
        == old_pages
    )
    assert ("user", "colors/cmap/name") in dialog._rows


def test_dialog_empty_category_page(
    monkeypatch: pytest.MonkeyPatch, dialog: OptionDialog
):
    monkeypatch.setattr(
        options_ui,
        "_leaf_paths_for_category",
        lambda _category, *, workspace_only: (),
    )

    container = dialog._make_category_page("workspace", "empty")

    assert (
        container.findChild(QtWidgets.QLabel, "settingsEmpty_workspace_empty")
        is not None
    )


def test_dialog_finds_workspace_manager_through_parent(qtbot):
    manager = _WorkspaceManagerStub()
    intermediate = QtWidgets.QWidget(manager)
    qtbot.addWidget(manager)
    qtbot.addWidget(intermediate)

    dlg = OptionDialog(intermediate)
    qtbot.addWidget(dlg)

    assert dlg.findChild(QtWidgets.QTabBar, "settingsScopeTabs").count() == 2


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


def test_dialog_update_visible_page_ignores_missing_selection(dialog: OptionDialog):
    dialog.category_list.setCurrentRow(-1)

    dialog._update_visible_page()

    assert dialog.category_list.currentItem() is None


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
    _control(
        dialog,
        "user",
        "figure/dpi",
        FigureDpiOverrideWidget,
    )

    qtbot.waitUntil(lambda: page.horizontalScrollBar().maximum() == 0)


def test_stylesheet_names_normalize_saved_values() -> None:
    assert options_ui._stylesheet_names(None) == []
    assert options_ui._stylesheet_names("classic, ggplot, classic") == [
        "classic",
        "ggplot",
    ]
    assert options_ui._stylesheet_names(42) == ["42"]


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


def test_dialog_control_value_helpers(dialog: OptionDialog, qtbot):
    checkbox = QtWidgets.QCheckBox()
    qtbot.addWidget(checkbox)
    checkbox.setChecked(True)
    assert dialog._control_value(checkbox, "colors/cmap/reverse") is True

    spin = QtWidgets.QSpinBox()
    qtbot.addWidget(spin)
    spin.setValue(42)
    assert dialog._control_value(spin, "io/dask/compute_threshold") == 42

    color_combo = ColorMapComboBox()
    qtbot.addWidget(color_combo)
    color_combo.ensure_populated()
    color_combo.setCurrentText("bwr")
    assert dialog._control_value(color_combo, "colors/cmap/name") == "bwr"

    combo = QtWidgets.QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Visible text")
    assert dialog._control_value(combo, "io/default_loader") == "Visible text"

    combo_with_data = QtWidgets.QComboBox()
    qtbot.addWidget(combo_with_data)
    combo_with_data.addItem("Visible text", "stored-value")
    assert dialog._control_value(combo_with_data, "io/default_loader") == "stored-value"

    colors = ColorListWidget()
    qtbot.addWidget(colors)
    colors.set_colors(["#ff0000", "#00ff00"])
    assert colors.get_colors() == ["#ff0000", "#00ff00"]
    assert dialog._control_value(colors, "colors/cursors") == [
        "#ff0000",
        "#00ff00",
    ]

    stylesheets = StylesheetListWidget(["classic", "ggplot"])
    qtbot.addWidget(stylesheets)
    assert dialog._control_value(stylesheets, "figure/stylesheets") == [
        "classic",
        "ggplot",
    ]

    figure_dpi = FigureDpiOverrideWidget()
    qtbot.addWidget(figure_dpi)
    assert dialog._control_value(figure_dpi, "figure/dpi") is None
    figure_dpi.override_check.setChecked(True)
    figure_dpi.dpi_spin.setValue(180.0)
    assert dialog._control_value(figure_dpi, "figure/dpi") == pytest.approx(180.0)

    list_line = QtWidgets.QLineEdit("one, two,, ")
    qtbot.addWidget(list_line)
    assert dialog._control_value(list_line, "colors/cmap/exclude") == ["one", "two"]

    text_line = QtWidgets.QLineEdit("example")
    qtbot.addWidget(text_line)
    assert dialog._control_value(text_line, "io/default_loader") == "example"
    unknown_widget = QtWidgets.QWidget()
    qtbot.addWidget(unknown_widget)
    assert dialog._control_value(unknown_widget, "io/default_loader") is None


def test_dialog_set_control_value_helpers(dialog: OptionDialog, qtbot):
    combo = QtWidgets.QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Known", "known")

    dialog._set_control_value(combo, "io/default_loader", "missing")

    assert combo.currentData() == "missing"
    assert combo.currentText() == "missing (unavailable)"

    list_line = QtWidgets.QLineEdit()
    qtbot.addWidget(list_line)
    dialog._set_control_value(list_line, "colors/cmap/exclude", ["one", "two"])
    assert list_line.text() == "one, two"

    text_line = QtWidgets.QLineEdit()
    qtbot.addWidget(text_line)
    dialog._set_control_value(text_line, "io/default_loader", "example")
    assert text_line.text() == "example"

    figure_dpi = FigureDpiOverrideWidget()
    qtbot.addWidget(figure_dpi)
    dialog._set_control_value(figure_dpi, "figure/dpi", 150.0)
    assert figure_dpi.override_check.isChecked()
    assert figure_dpi.dpi_spin.isEnabled()
    assert figure_dpi.get_dpi() == pytest.approx(150.0)
    dialog._set_control_value(figure_dpi, "figure/dpi", None)
    assert not figure_dpi.override_check.isChecked()
    assert not figure_dpi.dpi_spin.isEnabled()
    assert figure_dpi.get_dpi() is None


def test_dialog_spinbox_constraint_variants(
    monkeypatch: pytest.MonkeyPatch, dialog: OptionDialog, qtbot
):
    int_spin = QtWidgets.QSpinBox()
    qtbot.addWidget(int_spin)
    monkeypatch.setattr(options_ui, "_field_constraints", lambda _field: {"gt": 2})
    dialog._configure_spinbox(int_spin, "io/dask/compute_threshold")
    assert int_spin.minimum() == 2

    double_spin = QtWidgets.QDoubleSpinBox()
    qtbot.addWidget(double_spin)
    monkeypatch.setattr(options_ui, "_field_constraints", lambda _field: {"lt": 3.5})
    dialog._configure_spinbox(double_spin, "colors/cmap/gamma")
    assert double_spin.maximum() == pytest.approx(3.5)

    unconstrained_int = QtWidgets.QSpinBox()
    qtbot.addWidget(unconstrained_int)
    monkeypatch.setattr(options_ui, "_field_constraints", lambda _field: {})
    dialog._configure_spinbox(unconstrained_int, "io/dask/compute_threshold")
    assert unconstrained_int.minimum() == -2147483648
    assert unconstrained_int.maximum() == 2147483647


def test_dialog_workspace_helpers_without_manager_are_noops(dialog: OptionDialog):
    dialog._set_workspace_override("colors/cmap/name", "bwr")
    dialog._clear_workspace_override("colors/cmap/name")

    assert dialog._workspace_overrides() == {}
    assert dialog._effective_options().model_dump() == options.model.model_dump()


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


def test_workspace_stylesheet_override_keeps_raw_saved_names(qtbot):
    path = "figure/stylesheets"
    saved = ["classic", "missing-style"]
    manager = _WorkspaceManagerStub({path: saved})
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)

    row = dlg._rows[("workspace", path)]

    assert dlg._keeps_raw_workspace_value(row.control)
    assert dlg._value_for_row(row) == saved


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


def test_workspace_figure_dpi_override_supports_unset_and_numeric(qtbot):
    path = "figure/dpi"
    manager = _WorkspaceManagerStub()
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    control = typing.cast(
        "FigureDpiOverrideWidget",
        _control(dlg, "workspace", path, FigureDpiOverrideWidget),
    )
    override = _override(dlg, path)

    assert not control.isEnabled()
    override.setChecked(True)
    assert manager.overrides[path] is None
    assert control.isEnabled()

    control.override_check.setChecked(True)
    control.dpi_spin.setValue(180.0)
    assert manager.overrides[path] == pytest.approx(180.0)

    control.override_check.setChecked(False)
    assert manager.overrides[path] is None


def test_workspace_control_change_without_override_is_ignored(qtbot):
    manager = _WorkspaceManagerStub()
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    row = dlg._rows[("workspace", "colors/cmap/name")]
    dlg._control_changed(row)

    assert manager.overrides == {}


def test_workspace_override_switch_can_clear_existing_override(qtbot):
    path = "colors/cmap/name"
    manager = _WorkspaceManagerStub({path: "bwr"})
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    _override(dlg, path).setChecked(False)

    assert path not in manager.overrides


def test_workspace_override_changed_ignores_user_rows(qtbot):
    manager = _WorkspaceManagerStub()
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)

    dlg._override_changed(dlg._rows[("user", "colors/cmap/name")])

    assert manager.overrides == {}


def test_refresh_all_reraises_invalid_user_values(
    monkeypatch: pytest.MonkeyPatch, dialog: OptionDialog
) -> None:
    def raise_value_error(*_args, **_kwargs) -> None:
        raise ValueError("invalid user setting")

    monkeypatch.setattr(dialog, "_set_control_value", raise_value_error)

    with pytest.raises(ValueError, match="invalid user setting"):
        dialog._refresh_all()


def test_workspace_row_action_removes_override(qtbot):
    path = "colors/cmap/name"
    manager = _WorkspaceManagerStub({path: "bwr"})
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    _button(dlg, "workspace", path).click()

    assert path not in manager.overrides


def test_workspace_invalid_numeric_override_remains_removable(qtbot):
    path = "colors/max_rendered_abs_value"
    manager = _WorkspaceManagerStub({path: "not-a-number"})
    qtbot.addWidget(manager)

    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    spin = typing.cast(
        "QtWidgets.QDoubleSpinBox",
        _control(dlg, "workspace", path, QtWidgets.QDoubleSpinBox),
    )
    assert spin.value() == pytest.approx(options.model.colors.max_rendered_abs_value)
    assert _override(dlg, path).isChecked()
    assert manager.overrides[path] == "not-a-number"

    _button(dlg, "workspace", path).click()

    assert path not in manager.overrides


def test_workspace_invalid_color_list_override_remains_removable(qtbot):
    path = "colors/cursors"
    manager = _WorkspaceManagerStub({path: ["not-a-color"]})
    qtbot.addWidget(manager)

    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    control = typing.cast(
        "ColorListWidget",
        _control(dlg, "workspace", path, ColorListWidget),
    )
    assert control.get_colors() == options.model.colors.cursors
    assert _override(dlg, path).isChecked()
    assert manager.overrides[path] == ["not-a-color"]

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


def test_reset_current_scope_confirms_user_reset(
    monkeypatch: pytest.MonkeyPatch, dialog: OptionDialog
):
    combo = typing.cast(
        "QtWidgets.QComboBox",
        _control(dialog, "user", "colors/cmap/name", QtWidgets.QComboBox),
    )
    combo.setCurrentText("bwr")
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
    )

    dialog._reset_current_scope()

    assert options.model.colors.cmap.name == "bwr"

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    dialog._reset_current_scope()

    assert options.model.colors.cmap.name == AppOptions().colors.cmap.name


def test_reset_current_scope_confirms_workspace_clear(
    monkeypatch: pytest.MonkeyPatch, qtbot
):
    path = "colors/cmap/name"
    manager = _WorkspaceManagerStub({path: "bwr"})
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    dlg._reset_current_scope()

    assert manager.overrides == {}


def test_reset_current_scope_workspace_without_overrides_is_noop(qtbot):
    manager = _WorkspaceManagerStub()
    qtbot.addWidget(manager)
    dlg = OptionDialog(manager)
    qtbot.addWidget(dlg)
    dlg.scope_tabs.setCurrentIndex(1)

    dlg._reset_current_scope()

    assert manager.dirty_marks == []


def test_dialog_compatibility_slots(dialog: OptionDialog):
    combo = typing.cast(
        "QtWidgets.QComboBox",
        _control(dialog, "user", "colors/cmap/name", QtWidgets.QComboBox),
    )
    combo.setCurrentText("bwr")

    dialog.apply()
    dialog.update()
    dialog.restore()

    assert options.model.colors.cmap.name == AppOptions().colors.cmap.name

    dialog.accept()
    dialog.reject()


def test_workspace_override_helpers_filter_to_curated_subset() -> None:
    paths = set(workspace_overridable_option_paths())

    assert "colors/cmap/name" in paths
    assert "colors/cmap/packages" not in paths
    assert "io/workspace/compress" not in paths
    assert "figure/dpi" in paths
    assert normalize_workspace_option_overrides(
        {
            "colors/cmap/name": "bwr",
            "io/workspace/compress": False,
            "figure/dpi": 180.0,
        }
    ) == {"colors/cmap/name": "bwr", "figure/dpi": 180.0}


def test_options_get_set():
    options.restore()

    assert options["colors/cmap/name"] == AppOptions().colors.cmap.name
    assert options["io/workspace/compress"] is True
    assert options["io/workspace/use_incremental"] is True
    assert options["io/workspace/incremental_save_on_remote"] is False
    assert options["figure/dpi"] is None

    options["colors/cmap/name"] = "viridis"
    options["io/workspace/compress"] = False
    options["io/workspace/use_incremental"] = False
    options["io/workspace/incremental_save_on_remote"] = True
    options["figure/stylesheets"] = ["classic", "missing-style"]
    options["figure/dpi"] = 150.0

    assert options["colors/cmap/name"] == "viridis"
    assert options["io/workspace/compress"] is False
    assert options["io/workspace/use_incremental"] is False
    assert options["io/workspace/incremental_save_on_remote"] is True
    assert options["figure/stylesheets"] == ["classic", "missing-style"]
    assert options["figure/dpi"] == pytest.approx(150.0)
    assert not options.model.io.workspace.compress
    assert not options.model.io.workspace.use_incremental
    assert options.model.io.workspace.incremental_save_on_remote
    assert options.model.figure.stylesheets == ["classic", "missing-style"]
    assert options.model.figure.dpi == pytest.approx(150.0)

    options["figure/dpi"] = None
    assert options["figure/dpi"] is None

    options.restore()
    assert options["figure/dpi"] is None


@pytest.mark.parametrize("value", [0.0, -1.0, "not-a-number"])
def test_figure_dpi_option_validates(value: object) -> None:
    with pytest.raises(pydantic.ValidationError):
        AppOptions.model_validate({"figure": {"dpi": value}})


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

import warnings

import pytest
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._stylesheets
from erlab.interactive._options.parameters import (
    _STYLESHEET_AVAILABLE_ROLE,
    _STYLESHEET_NAME_ROLE,
    ColorListParameter,
    ColorListWidget,
    FigureDpiOverrideParameter,
    FigureDpiOverrideWidget,
    StylesheetListParameter,
    StylesheetListWidget,
    _stylesheet_names,
)


def test_colorlistwidget_initialization(qtbot):
    widget = ColorListWidget(colors=["#ff0000", "#00ff00", "#0000ff"])
    qtbot.addWidget(widget)
    assert len(widget.colors) == 3
    assert all(isinstance(c, QtGui.QColor) for c in widget.colors)
    assert widget.get_colors() == ["#ff0000", "#00ff00", "#0000ff"]


def test_colorlistwidget_add_color(qtbot, monkeypatch):
    widget = ColorListWidget(colors=[])
    qtbot.addWidget(widget)
    # Simulate QColorDialog.getColor returning a valid color
    monkeypatch.setattr(
        QtWidgets.QColorDialog, "getColor", lambda *a, **k: QtGui.QColor("#123456")
    )
    widget.add_color()
    assert widget.get_colors() == ["#123456"]


def test_colorlistwidget_edit_color(qtbot, monkeypatch):
    widget = ColorListWidget(colors=["#ff0000"])
    qtbot.addWidget(widget)
    # Simulate QColorDialog.getColor returning a valid color
    monkeypatch.setattr(
        QtWidgets.QColorDialog, "getColor", lambda *a, **k: QtGui.QColor("#abcdef")
    )
    widget.edit_color(0)
    assert widget.get_colors() == ["#abcdef"]


def test_colorlistwidget_remove_color(qtbot):
    widget = ColorListWidget(colors=["#ff0000", "#00ff00"])
    qtbot.addWidget(widget)
    widget.remove_color(0)
    assert widget.get_colors() == ["#00ff00"]


def test_colorlistwidget_set_colors(qtbot):
    widget = ColorListWidget(colors=["#ff0000"])
    qtbot.addWidget(widget)
    widget.set_colors(["#111111", "#222222"])
    assert widget.get_colors() == ["#111111", "#222222"]


def test_colorlistwidget_sigColorChanged_emitted(qtbot):
    widget = ColorListWidget(colors=["#ff0000"])
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.sigColorChanged, timeout=1000) as blocker:
        widget.set_colors(["#123456"])
    assert blocker.args[0][0].name() == "#123456"


def test_colorlistparameter_save_state() -> None:
    param = ColorListParameter(name="colors", value=[QtGui.QColor("#010203")])
    state = param.saveState()
    assert state["value"][0][:3] == (1, 2, 3)


def test_figure_dpi_override_parameter_item_widget(qtbot) -> None:
    param = FigureDpiOverrideParameter(name="dpi", value=150.0)
    item = param.makeTreeItem(0)
    widget = item.widget
    qtbot.addWidget(widget)

    assert isinstance(widget, FigureDpiOverrideWidget)
    assert not item.hideWidget
    assert widget.value() == pytest.approx(150.0)

    widget.override_check.setChecked(False)
    assert param.value() is None


def test_style_library_paths_falls_back_to_matplotlib_core(monkeypatch) -> None:
    paths = ["stylelib"]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The matplotlib.style.core module was deprecated",
            category=DeprecationWarning,
        )
        from matplotlib.style import core as mpl_style_core

    monkeypatch.delattr(
        erlab.interactive._stylesheets.mpl_style,
        "USER_LIBRARY_PATHS",
        raising=False,
    )
    monkeypatch.setattr(mpl_style_core, "USER_LIBRARY_PATHS", paths)

    assert erlab.interactive._stylesheets._style_library_paths() is paths


def test_generic_data_directory_uses_qstandardpaths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        erlab.interactive._stylesheets.QtCore.QStandardPaths,
        "writableLocation",
        lambda location: "",
    )

    assert erlab.interactive._stylesheets._generic_data_directory() is None

    monkeypatch.setattr(
        erlab.interactive._stylesheets.QtCore.QStandardPaths,
        "writableLocation",
        lambda location: str(tmp_path),
    )

    assert (
        erlab.interactive._stylesheets._generic_data_directory()
        == tmp_path / "erlabpy" / "ImageTool Manager"
    )


def test_user_stylesheet_directory_uses_generic_data_fallback(
    tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "erlabpy" / "ImageTool Manager"
    monkeypatch.setattr(
        erlab.interactive._stylesheets, "_app_data_directory", lambda: None
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_generic_data_directory",
        lambda: data_dir,
    )

    style_dir = erlab.interactive._stylesheets.user_stylesheet_directory()

    assert style_dir == data_dir / "stylelib"
    assert style_dir.is_dir()


def test_user_stylesheet_directory_raises_without_qt_data_path(monkeypatch) -> None:
    monkeypatch.setattr(
        erlab.interactive._stylesheets, "_app_data_directory", lambda: None
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_generic_data_directory",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="custom Matplotlib stylesheets"):
        erlab.interactive._stylesheets.user_stylesheet_directory()


def test_user_stylesheet_directory_reports_create_failure(
    tmp_path, monkeypatch
) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("")
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_app_data_directory",
        lambda: file_path,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_generic_data_directory",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="Could not create"):
        erlab.interactive._stylesheets.user_stylesheet_directory()


def test_user_stylesheet_directory_can_skip_creation(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "app-data"
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_app_data_directory",
        lambda: data_dir,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_generic_data_directory",
        lambda: None,
    )

    style_dir = erlab.interactive._stylesheets.user_stylesheet_directory(create=False)

    assert style_dir == data_dir / "stylelib"
    assert not style_dir.exists()


def test_stylesheet_names_in_directory_handles_missing_directory(tmp_path) -> None:
    assert (
        erlab.interactive._stylesheets._stylesheet_names_in_directory(
            tmp_path / "missing"
        )
        == frozenset()
    )


def test_reload_stylesheets_loads_bundled_and_user_styles(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_erlab_plotting_stylesheets",
        lambda: calls.append("erlab"),
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *, reload=False: calls.append(reload),
    )

    erlab.interactive._stylesheets.reload_stylesheets()

    assert calls == ["erlab", True]


def test_stylesheets_require_user_stylesheets_ignores_empty_values() -> None:
    assert not erlab.interactive._stylesheets.stylesheets_require_user_stylesheets([])


def test_load_user_stylesheets_tracks_added_and_removed_files(
    tmp_path, monkeypatch
) -> None:
    library_paths = erlab.interactive._stylesheets._style_library_paths()
    old_library_paths = list(library_paths)
    user_registered = erlab.interactive._stylesheets._USER_REGISTERED_STYLESHEETS
    old_registered = set(user_registered)
    old_names = erlab.interactive._stylesheets._USER_STYLESHEET_NAMES
    user_registered.clear()
    erlab.interactive._stylesheets._USER_STYLESHEET_NAMES = frozenset()
    data_dir = tmp_path / "app-data"
    style_dir = data_dir / "stylelib"
    style_name = "erlab-test-user-style"

    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_app_data_directory",
        lambda: data_dir,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "_generic_data_directory",
        lambda: None,
    )
    try:
        (style_dir / f"{style_name}.mplstyle").parent.mkdir(parents=True)
        (style_dir / f"{style_name}.mplstyle").write_text("axes.facecolor: white\n")

        erlab.interactive._stylesheets.load_user_stylesheets(reload=True)

        assert str(style_dir) in library_paths
        assert style_name in erlab.interactive._stylesheets.available_stylesheets()
        assert erlab.interactive._stylesheets.stylesheets_require_user_stylesheets(
            [style_name]
        )

        (style_dir / f"{style_name}.mplstyle").unlink()
        erlab.interactive._stylesheets.load_user_stylesheets(reload=True)

        assert style_name not in erlab.interactive._stylesheets.available_stylesheets()
        assert not erlab.interactive._stylesheets.stylesheets_require_user_stylesheets(
            [style_name]
        )
    finally:
        library_paths[:] = old_library_paths
        erlab.interactive._stylesheets.mpl_style.reload_library()
        user_registered.clear()
        user_registered.update(old_registered)
        erlab.interactive._stylesheets._USER_STYLESHEET_NAMES = old_names


def test_stylesheetlistwidget_preserves_unavailable_styles(qtbot, monkeypatch):
    monkeypatch.setattr(
        "erlab.interactive._stylesheets.mpl_style.available",
        ["classic", "ggplot"],
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_erlab_plotting_stylesheets",
        lambda: None,
    )
    widget = StylesheetListWidget(stylesheets=["classic", "missing-style"])
    qtbot.addWidget(widget)

    assert widget.get_stylesheets() == ["classic", "missing-style"]
    assert widget.list_widget.item(1).data(_STYLESHEET_AVAILABLE_ROLE) is False
    assert "unavailable" in widget.list_widget.item(1).toolTip()
    widget.list_widget.setCurrentRow(1)
    assert widget.remove_button.isEnabled()
    assert widget.up_button.isEnabled()
    assert not widget.down_button.isEnabled()
    widget.remove_selected_stylesheet()
    assert widget.get_stylesheets() == ["classic"]


def test_stylesheet_names_normalizes_and_deduplicates_values() -> None:
    assert _stylesheet_names(None) == []
    assert _stylesheet_names(" classic, ggplot, classic ,, ") == [
        "classic",
        "ggplot",
    ]
    assert _stylesheet_names(("classic", "ggplot", "classic", 2)) == [
        "classic",
        "ggplot",
        "2",
    ]
    assert _stylesheet_names(3) == ["3"]


def test_stylesheetlistwidget_add_remove_and_reorder(qtbot, monkeypatch):
    monkeypatch.setattr(
        "erlab.interactive._stylesheets.mpl_style.available",
        ["classic", "ggplot", "bmh"],
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    widget = StylesheetListWidget(stylesheets=["classic"])
    qtbot.addWidget(widget)

    widget.add_combo.setCurrentText("ggplot")
    widget.add_stylesheet()
    assert widget.get_stylesheets() == ["classic", "ggplot"]

    widget.list_widget.setCurrentRow(1)
    widget.move_selected_stylesheet(-1)
    assert widget.get_stylesheets() == ["ggplot", "classic"]

    widget.remove_selected_stylesheet()
    assert widget.get_stylesheets() == ["classic"]


def test_stylesheetlistwidget_ignores_invalid_actions_and_preserves_roles(
    qtbot,
    monkeypatch,
):
    monkeypatch.setattr(
        "erlab.interactive._stylesheets.mpl_style.available",
        ["classic", "ggplot"],
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    widget = StylesheetListWidget(stylesheets=["classic", "ggplot"])
    qtbot.addWidget(widget)

    widget.list_widget.clearSelection()
    widget.list_widget.setCurrentRow(-1)
    assert widget.current_stylesheet() is None
    widget.remove_selected_stylesheet()
    widget.move_selected_stylesheet(-1)
    assert widget.get_stylesheets() == ["classic", "ggplot"]
    assert not widget.remove_button.isEnabled()
    assert not widget.up_button.isEnabled()
    assert not widget.down_button.isEnabled()

    widget.add_combo.setCurrentText("")
    widget.add_stylesheet()
    widget.add_combo.setCurrentText("classic")
    widget.add_stylesheet()
    assert widget.get_stylesheets() == ["classic", "ggplot"]

    widget.list_widget.setCurrentRow(0)
    widget.move_selected_stylesheet(-1)
    assert widget.get_stylesheets() == ["classic", "ggplot"]
    widget.list_widget.setCurrentRow(1)
    widget.move_selected_stylesheet(1)
    assert widget.get_stylesheets() == ["classic", "ggplot"]

    first_item = widget.list_widget.item(0)
    assert first_item.data(_STYLESHEET_NAME_ROLE) == "classic"
    assert first_item.data(_STYLESHEET_AVAILABLE_ROLE) is True

    widget.set_stylesheets(["ggplot", "classic", "ggplot"])
    assert widget.get_stylesheets() == ["ggplot", "classic"]


def test_stylesheetlistwidget_loads_erlab_styles_on_popup(qtbot, monkeypatch):
    calls: list[None] = []
    monkeypatch.setattr(
        "erlab.interactive._stylesheets.mpl_style.available",
        ["classic"],
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "reload_stylesheets",
        lambda: calls.append(None),
    )
    widget = StylesheetListWidget(stylesheets=[])
    qtbot.addWidget(widget)

    widget.add_combo.showPopup()
    widget.add_combo.hidePopup()

    assert calls == [None]


def test_stylesheetlistwidget_rechecks_saved_styles_after_erlab_import(
    qtbot, monkeypatch
):
    available = ["classic"]
    monkeypatch.setattr("erlab.interactive._stylesheets.mpl_style.available", available)
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_erlab_plotting_stylesheets",
        lambda: available.append("nature"),
    )

    widget = StylesheetListWidget(stylesheets=["nature"])
    qtbot.addWidget(widget)

    assert widget.list_widget.item(0).data(_STYLESHEET_AVAILABLE_ROLE) is True


def test_stylesheetlistwidget_reload_updates_saved_style_availability(
    qtbot, monkeypatch
) -> None:
    available = ["classic"]
    monkeypatch.setattr("erlab.interactive._stylesheets.mpl_style.available", available)
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_erlab_plotting_stylesheets",
        lambda: None,
    )

    def reload_stylesheets() -> None:
        available.append("restored-style")

    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "reload_stylesheets",
        reload_stylesheets,
    )
    widget = StylesheetListWidget(stylesheets=["restored-style"])
    qtbot.addWidget(widget)

    assert widget.get_stylesheets() == ["restored-style"]
    assert widget.list_widget.item(0).data(_STYLESHEET_AVAILABLE_ROLE) is False

    widget.reload_button.click()

    assert widget.get_stylesheets() == ["restored-style"]
    assert widget.list_widget.item(0).data(_STYLESHEET_AVAILABLE_ROLE) is True


def test_stylesheetlistwidget_open_folder_uses_custom_style_directory(
    qtbot, monkeypatch, tmp_path
) -> None:
    opened_urls: list[QtCore.QUrl] = []
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "user_stylesheet_directory",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        QtGui.QDesktopServices,
        "openUrl",
        lambda url: opened_urls.append(url) or True,
    )
    widget = StylesheetListWidget(stylesheets=[])
    qtbot.addWidget(widget)

    widget.open_folder_button.click()

    assert opened_urls[0].toLocalFile() == str(tmp_path)


def test_stylesheetlistwidget_open_folder_reports_unavailable_directory(
    qtbot, monkeypatch
) -> None:
    warnings: list[tuple[str, str]] = []
    opened_urls: list[QtCore.QUrl] = []
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_user_stylesheets",
        lambda *_, **__: None,
    )

    def raise_no_style_directory():
        raise RuntimeError("no stylesheet directory")

    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "user_stylesheet_directory",
        raise_no_style_directory,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )
    monkeypatch.setattr(
        QtGui.QDesktopServices,
        "openUrl",
        lambda url: opened_urls.append(url) or True,
    )
    widget = StylesheetListWidget(stylesheets=[])
    qtbot.addWidget(widget)

    widget.open_folder_button.click()

    assert warnings == [("Stylesheet folder unavailable", "no stylesheet directory")]
    assert opened_urls == []


def test_stylesheetlistparameter_value_roundtrip() -> None:
    param = StylesheetListParameter(
        name="stylesheets", value=["classic", "missing-style"]
    )
    assert param.value() == ["classic", "missing-style"]

from qtpy import QtGui, QtWidgets

import erlab.interactive._stylesheets
from erlab.interactive._options.parameters import (
    _STYLESHEET_AVAILABLE_ROLE,
    ColorListParameter,
    ColorListWidget,
    StylesheetListParameter,
    StylesheetListWidget,
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


def test_stylesheetlistwidget_preserves_unavailable_styles(qtbot, monkeypatch):
    monkeypatch.setattr(
        "erlab.interactive._stylesheets.mpl_style.available",
        ["classic", "ggplot"],
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


def test_stylesheetlistwidget_add_remove_and_reorder(qtbot, monkeypatch):
    monkeypatch.setattr(
        "erlab.interactive._stylesheets.mpl_style.available",
        ["classic", "ggplot", "bmh"],
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


def test_stylesheetlistwidget_loads_erlab_styles_on_popup(qtbot, monkeypatch):
    calls: list[None] = []
    monkeypatch.setattr(
        "erlab.interactive._stylesheets.mpl_style.available",
        ["classic"],
    )
    monkeypatch.setattr(
        erlab.interactive._stylesheets,
        "load_erlab_plotting_stylesheets",
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
        "load_erlab_plotting_stylesheets",
        lambda: available.append("nature"),
    )

    widget = StylesheetListWidget(stylesheets=["nature"])
    qtbot.addWidget(widget)

    assert widget.list_widget.item(0).data(_STYLESHEET_AVAILABLE_ROLE) is True


def test_stylesheetlistparameter_value_roundtrip() -> None:
    param = StylesheetListParameter(
        name="stylesheets", value=["classic", "missing-style"]
    )
    assert param.value() == ["classic", "missing-style"]

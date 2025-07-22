from qtpy import QtGui, QtWidgets

from erlab.interactive._options.parameters import ColorListWidget


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

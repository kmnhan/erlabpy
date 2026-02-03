import tempfile

import lmfit
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.utils import (
    ChunkEditDialog,
    IconActionButton,
    IdentifierValidator,
    MessageDialog,
    PythonHighlighter,
    SingleLinePlainTextEdit,
    array_rect,
    generate_code,
    load_fit_ui,
    save_fit_ui,
)


@pytest.fixture
def action():
    action = QtGui.QAction("Test Action")
    action.setCheckable(True)
    return action


def test_icon_action_button_initialization(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    assert button._action == action
    assert button.icon_key_on == "mdi6.plus"
    assert button.icon_key_off == "mdi6.minus"
    assert button.isCheckable()


def test_icon_action_button_set_action(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    new_action = QtGui.QAction("New Action")
    new_action.setCheckable(True)
    button.setAction(new_action)
    assert button._action == new_action


def test_icon_action_button_update_from_action(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    action.setText("Updated Action")
    action.setEnabled(False)
    action.setChecked(True)
    action.setToolTip("Updated Tooltip")
    assert not button.isEnabled()
    assert button.isChecked()
    assert button.toolTip() == "Updated Tooltip"


def test_icon_action_button_click(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    qtbot.addWidget(button)
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    assert action.isChecked()


def test_icon_action_button_toggle(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    qtbot.addWidget(button)
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    assert button.isChecked()
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    assert not button.isChecked()


@pytest.fixture
def fit_result_ds():
    rng = np.random.default_rng(1)
    xvals = np.linspace(0, 20, 100)
    yvals = 2 * np.sin(xvals) + 0.1 * rng.normal(size=xvals.shape)
    test_data = xr.DataArray(yvals, dims=["x"], coords={"x": xvals})

    def model_func(x, a, b):
        return a * np.sin(x) + b

    model = lmfit.Model(model_func, independent_vars=["x"])
    return test_data.xlm.modelfit("x", model=model, params={"a": 2, "b": 0})


def test_save_fit_ui(qtbot, accept_dialog, fit_result_ds):
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/fit_save.h5"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("fit_save.h5")

    # Save fit
    accept_dialog(lambda: save_fit_ui(fit_result_ds), pre_call=_go_to_file)

    # Load fit
    accept_dialog(lambda: load_fit_ui(), pre_call=_go_to_file)

    tmp_dir.cleanup()


@pytest.mark.parametrize(
    ("input_str", "expected_state"),
    [
        ("valid_name", QtGui.QValidator.State.Acceptable),
        ("", QtGui.QValidator.State.Intermediate),
        ("1invalid", QtGui.QValidator.State.Invalid),
        ("with space", QtGui.QValidator.State.Invalid),
        ("for", QtGui.QValidator.State.Intermediate),  # Python keyword
        ("_hidden", QtGui.QValidator.State.Acceptable),
        ("invalid-char!", QtGui.QValidator.State.Invalid),
        (None, QtGui.QValidator.State.Intermediate),
    ],
)
def test_identifier_validator_validate(input_str, expected_state):
    validator = IdentifierValidator()
    state, _, _ = validator.validate(input_str, 0)
    assert state == expected_state


def test_array_rect_handles_singleton_dimension() -> None:
    data = xr.DataArray(
        np.zeros((1, 5)),
        dims=("y", "x"),
        coords={"y": np.array([2.0]), "x": np.arange(5, dtype=float)},
    )

    rect = array_rect(data)

    assert rect.height() == pytest.approx(1.0)
    assert rect.width() == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("valid_name", "valid_name"),
        ("1invalid", "_1invalid"),
        ("with space", "with_space"),
        ("for", "for_"),
        ("", "var"),
        ("___", "var"),
        ("!@#", "var"),
        ("class", "class_"),
        (None, "var"),
    ],
)
def test_identifier_validator_fixup(input_str, expected):
    validator = IdentifierValidator()
    assert validator.fixup(input_str) == expected


class CustomException(Exception):
    pass


@pytest.fixture
def raise_exception():
    def _func():
        raise CustomException("Dialog test error")

    return _func


@pytest.mark.parametrize(
    "informative_text", ["Additional info", ""], ids=["with_info", "no_info"]
)
def test_message_dialog_critical(
    qtbot, raise_exception, accept_dialog, informative_text
):
    try:
        raise_exception()
    except CustomException:
        parent = QtWidgets.QWidget()
        qtbot.addWidget(parent)
        parent.show()

        def _call_messagebox_critical():
            MessageDialog.critical(
                parent=parent,
                title="Show Traceback",
                text="An error occurred",
                informative_text=informative_text,
            )

        accept_dialog(_call_messagebox_critical)


def test_message_dialog_with_details_toggle(qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    parent.show()

    dialog = MessageDialog(
        parent=parent,
        title="Title",
        text="Main text",
        informative_text="Info text",
        detailed_text="<b>Details</b>",
        buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.windowTitle() == "Title"
    assert dialog.text() == "Main text"
    assert dialog.informativeText() == "Info text"
    assert dialog.detailedText() == "<b>Details</b>"

    assert dialog._details_toggle.isVisible()
    assert not dialog._details_container.isVisible()

    dialog._details_toggle.setChecked(True)
    qtbot.wait_until(lambda: dialog._details_container.isVisible(), timeout=200)
    assert dialog._details_toggle.arrowType() == QtCore.Qt.ArrowType.DownArrow
    assert dialog._details_toggle.text() == "Hide Details"

    dialog._details_toggle.setChecked(False)
    qtbot.wait_until(lambda: not dialog._details_container.isVisible(), timeout=200)
    assert dialog._details_toggle.arrowType() == QtCore.Qt.ArrowType.RightArrow
    assert dialog._details_toggle.text() == "Show Details…"

    ok_btn = dialog._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    ok_btn.click()
    assert dialog.result() == QtWidgets.QDialog.Accepted


def test_single_line_plain_text_edit_newlines(qtbot) -> None:
    widget = SingleLinePlainTextEdit()
    qtbot.addWidget(widget)

    widget.setText("alpha\nbeta")
    assert widget.text() == "alpha beta"
    assert "\n" not in widget.text()

    mime = QtCore.QMimeData()
    mime.setText("a\nb\r\nc")
    widget.clear()
    widget.insertFromMimeData(mime)
    assert widget.text() == "abc"


def test_single_line_plain_text_edit_block_return(qtbot) -> None:
    widget = SingleLinePlainTextEdit()
    qtbot.addWidget(widget)
    widget.setText("abc")

    qtbot.keyPress(widget, QtCore.Qt.Key_Return)
    assert widget.text() == "abc"


def test_python_highlighter_formats_operator(qtbot) -> None:
    doc = QtGui.QTextDocument("a+b")
    highlighter = PythonHighlighter(doc, style="default")
    qtbot.addWidget(QtWidgets.QWidget())

    highlighter._relex_document_if_needed()
    spans = highlighter._block_spans.get(0, [])

    has_operator_format = any(
        start <= 1 < (start + length) and not fmt.isEmpty()
        for start, length, fmt in spans
    )
    assert has_operator_format


def test_generate_code_multiple_assignment() -> None:
    def _dummy(a, b=1):
        return a + b

    code = generate_code(
        _dummy, args=(1,), kwargs={"b": 2}, module="dummy", assign=("x", "y")
    )

    assert code.startswith("x, y = dummy._dummy(1")
    assert "b=2" in code


def test_message_dialog_without_details(qtbot):
    dialog = MessageDialog(
        parent=None,
        title="No Details",
        text="Text",
        informative_text="",
        detailed_text="",
        buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog._details_toggle.isVisible()
    assert not dialog._details_container.isVisible()

    dialog._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
    assert dialog.result() == QtWidgets.QDialog.Accepted


def test_message_dialog_custom_buttons_and_default(qtbot):
    dialog = MessageDialog(
        parent=None,
        title="Custom Buttons",
        text="Choose",
        informative_text="",
        detailed_text="<p>detail</p>",
        buttons=(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        ),
        default_button=QtWidgets.QDialogButtonBox.StandardButton.Cancel,
    )
    qtbot.addWidget(dialog)
    dialog.show()

    ok_btn = dialog._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    cancel_btn = dialog._button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    assert ok_btn is not None
    assert cancel_btn is not None
    assert cancel_btn.isDefault() or cancel_btn.autoDefault()

    cancel_btn.click()
    assert dialog.result() == QtWidgets.QDialog.Rejected


def test_message_dialog_setters_update_labels(qtbot):
    dialog = MessageDialog(parent=None)
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.setText("Hello")
    dialog.setInformativeText("Info")
    dialog.setDetailedText("<i>Details</i>")

    assert dialog.text() == "Hello"
    assert dialog.informativeText() == "Info"
    assert dialog.detailedText() == "<i>Details</i>"
    assert dialog._details_toggle.isVisible()

    dialog._button_box.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    dialog._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    assert dialog.result() == QtWidgets.QDialog.Accepted


@pytest.mark.parametrize(
    ("chunks", "inputs", "expected"),
    [
        (
            {"x": (10, 10, 10), "y": (30, 30)},
            ["auto", "None"],
            {"x": "auto"},
        ),
        (
            {"x": (15, 15), "y": (60,)},
            ["15, 15", "60"],
            {"x": (15, 15), "y": 60},
        ),
        (
            {"x": (20, 20, 20), "y": (30, 30)},
            ["10, 10, 10", "30, 30"],
            {},  # invalid: sum does not match dimension size
        ),
        (
            {"x": (10, 10, 10), "y": (30, 30)},
            ["", ""],
            {},  # empty input, should skip
        ),
    ],
)
def test_chunk_edit_dialog_accept(qtbot, chunks, inputs, expected, accept_dialog):
    arr = xr.DataArray(
        np.zeros((sum(chunks["x"]), sum(chunks["y"]))),
        dims=("x", "y"),
    ).chunk(chunks)
    chunk_dialog = ChunkEditDialog(arr)
    qtbot.addWidget(chunk_dialog)
    chunk_dialog.show()
    qtbot.waitExposed(chunk_dialog)

    # Set user inputs
    for row, val in enumerate(inputs):
        item = chunk_dialog.table.item(row, 2)
        item.setText(val)

    # Patch QMessageBox to avoid actual dialogs
    if expected:
        chunk_dialog.accept()
        assert chunk_dialog.result_chunks == expected
    else:
        accept_dialog(chunk_dialog.accept)
        assert chunk_dialog.result_chunks == {}


def test_chunk_edit_dialog_populate_and_parse(qtbot):
    arr = xr.DataArray(np.zeros((20, 60)), dims=("x", "y")).chunk(
        {"x": (10, 10), "y": (30, 30)}
    )
    dialog = ChunkEditDialog(arr)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)

    # Check table population
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).text() == "x"
    assert dialog.table.item(1, 0).text() == "y"
    assert dialog.table.item(0, 1).text() == "10"
    assert dialog.table.item(1, 1).text() == "30"

    # Test _parse_chunk_string
    assert dialog._parse_chunk_string("auto") == "auto"
    assert dialog._parse_chunk_string("None") is None
    assert dialog._parse_chunk_string("  42 ") == 42
    assert dialog._parse_chunk_string("5, 6, 7") == (5, 6, 7)


def test_chunk_edit_dialog_invalid_input(qtbot, accept_dialog):
    arr = xr.DataArray(np.zeros((10,)), dims=("x",)).chunk({"x": (10,)})
    dialog = ChunkEditDialog(arr)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)

    # Set invalid input
    dialog.table.item(0, 2).setText("notanumber")
    accept_dialog(dialog.accept)
    assert dialog.result_chunks == {}


def test_chunk_edit_dialog_cancel(qtbot):
    arr = xr.DataArray(np.zeros((10,)), dims=("x",)).chunk({"x": (10,)})
    dialog = ChunkEditDialog(arr)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    # Simulate cancel
    dialog.reject()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected

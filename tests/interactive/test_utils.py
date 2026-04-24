import enum
import importlib
import sys
import tempfile
import types
import typing

import lmfit
import numpy as np
import pydantic
import pytest
import qtpy
import xarray as xr
from qtpy import PYQT6, QtCore, QtGui, QtWidgets

import erlab.interactive.utils
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
    load_ui,
    qt_is_valid,
    save_fit_ui,
    xImageItem,
)


def _exec_generated_code(
    code: str, namespace: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    locals_ns = dict(namespace)
    exec(code, {"__builtins__": {"slice": slice}}, locals_ns)  # noqa: S102
    return locals_ns


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


def test_icon_action_button_ignores_deleted_action(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    qtbot.addWidget(button)

    action.deleteLater()
    qtbot.waitUntil(lambda: not qt_is_valid(action))

    button.refresh_icons()

    assert button._action is None


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


def test_load_ui_temporarily_disables_autoconnect(monkeypatch) -> None:
    restored_calls: list[object] = []

    def _original_autoconnect(obj) -> None:
        restored_calls.append(obj)

    def _fake_load_ui(_uifile, baseinstance=None):
        QtCore.QMetaObject.connectSlotsByName(baseinstance)
        return baseinstance

    monkeypatch.setattr(
        erlab.interactive.utils.QtCore.QMetaObject,
        "connectSlotsByName",
        _original_autoconnect,
    )
    monkeypatch.setattr(erlab.interactive.utils.uic, "loadUi", _fake_load_ui)

    widget = QtWidgets.QWidget()
    loaded = load_ui("dummy.ui", widget)
    assert loaded is widget

    if PYQT6:
        # The autoconnect callback should be suppressed inside load_ui.
        assert restored_calls == []
    else:
        assert restored_calls == [widget]

    # Outside the context manager, the original callback should be restored.
    QtCore.QMetaObject.connectSlotsByName(widget)
    expected_restored = [widget]
    if not PYQT6:
        expected_restored = [widget, widget]
    assert restored_calls == expected_restored


def test_load_ui_restores_autoconnect_after_error(monkeypatch) -> None:
    restored_calls: list[object] = []

    def _original_autoconnect(obj) -> None:
        restored_calls.append(obj)

    def _failing_load_ui(*_args, **_kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        erlab.interactive.utils.QtCore.QMetaObject,
        "connectSlotsByName",
        _original_autoconnect,
    )
    monkeypatch.setattr(erlab.interactive.utils.uic, "loadUi", _failing_load_ui)

    with pytest.raises(RuntimeError, match="load failed"):
        load_ui("dummy.ui", QtWidgets.QWidget())

    QtCore.QMetaObject.connectSlotsByName(object())
    assert len(restored_calls) == 1


def test_qt_is_valid_ignores_none() -> None:
    assert qt_is_valid(None)


def test_qt_object_is_valid_fallback() -> None:
    assert erlab.interactive.utils._qt_object_is_valid_fallback(object())
    assert not erlab.interactive.utils._qt_object_is_valid_fallback(None)


def test_qt_is_valid_rejects_deleted_widget(qtbot) -> None:
    widget = QtWidgets.QWidget()
    widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    widget.show()

    assert qt_is_valid(widget, None)

    widget.close()
    QtWidgets.QApplication.sendPostedEvents(None, 0)
    QtWidgets.QApplication.processEvents()

    qtbot.wait_until(lambda: not qt_is_valid(widget), timeout=1000)


def test_qt_object_is_valid_uses_shiboken_when_available(monkeypatch) -> None:
    sentinel = object()
    other = object()
    calls: list[object] = []
    fake_shiboken6 = types.ModuleType("shiboken6")

    def _fake_is_valid(obj: object) -> bool:
        calls.append(obj)
        return obj is sentinel

    fake_shiboken6.isValid = _fake_is_valid

    try:
        with monkeypatch.context() as context:
            context.setattr(qtpy, "PYSIDE6", True, raising=False)
            context.setattr(qtpy, "PYQT6", False, raising=False)
            context.setitem(sys.modules, "shiboken6", fake_shiboken6)

            reloaded = importlib.reload(erlab.interactive.utils)

            assert reloaded._qt_object_is_valid(sentinel)
            assert not reloaded._qt_object_is_valid(other)
            assert not reloaded._qt_object_is_valid(None)
            assert calls == [sentinel, other]
    finally:
        importlib.reload(erlab.interactive.utils)


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


def test_ximageitem_set_cut_tolerance_scalar(qtbot) -> None:
    item = xImageItem()
    item.set_cut_tolerance(10)
    assert item.cut_tolerance == [10, 10]


def test_ximageitem_set_cut_tolerance_iterable(qtbot) -> None:
    item = xImageItem()
    item.set_cut_tolerance([5, 15])
    assert item.cut_tolerance == [5, 15]


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


def test_tool_window_declared_output_dispatch_and_validation(qtbot) -> None:
    class _DummyState(pydantic.BaseModel):
        value: int = 0

    class _DummyTool(erlab.interactive.utils.ToolWindow[_DummyState]):
        StateModel = _DummyState
        tool_name = "dummy"
        COPY_PROVENANCE: typing.ClassVar = (
            erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from current dummy input data",
                label_method="_result_output_label",
                expression_method="_result_output_expression",
                assign="result",
            )
        )

        class Output(enum.StrEnum):
            RESULT = "dummy.result"

        IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
            Output.RESULT: erlab.interactive.utils.ToolImageOutputDefinition(
                data_method="_result_output_data",
                provenance=erlab.interactive.utils.ToolScriptProvenanceDefinition(
                    start_label="Start from current dummy input data",
                    label_method="_result_output_label",
                    expression_method="_result_output_expression",
                    assign="result",
                ),
            )
        }

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data

        @property
        def tool_status(self) -> _DummyState:
            return _DummyState()

        @tool_status.setter
        def tool_status(self, status: _DummyState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

        def _result_output_data(self) -> xr.DataArray:
            return self._data + 1

        def _result_output_label(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            return "Compute dummy output"

        def _result_output_expression(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del data
            return f"{input_name or 'data'} + 1"

    tool = _DummyTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)

    expected = tool.tool_data + 1
    xr.testing.assert_identical(
        tool.output_imagetool_data(_DummyTool.Output.RESULT),
        expected,
    )

    spec = tool.output_imagetool_provenance(_DummyTool.Output.RESULT, expected)
    assert spec is not None
    assert spec.active_name == "result"
    spec_code = spec.display_code()
    assert spec_code is not None
    spec_namespace = _exec_generated_code(
        spec_code,
        {"data": tool.tool_data.copy(deep=True)},
    )
    spec_result = spec_namespace["result"]
    assert isinstance(spec_result, xr.DataArray)
    xr.testing.assert_identical(spec_result, expected)
    assert tool.current_provenance_spec() is not None
    assert tool.current_provenance_spec().active_name == "result"
    current_code = tool.current_provenance_spec().display_code()
    assert current_code is not None
    current_namespace = _exec_generated_code(
        current_code,
        {"data": tool.tool_data.copy(deep=True)},
    )
    current_result = current_namespace["result"]
    assert isinstance(current_result, xr.DataArray)
    xr.testing.assert_identical(current_result, expected)

    with pytest.raises(ValueError, match="does not define ImageTool output"):
        tool.output_imagetool_data("dummy.unknown")


def test_tool_script_provenance_definition_validates_assignments() -> None:
    with pytest.raises(ValueError, match="tuple `assign` must define `active_name`"):
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from current dummy input data",
            label="Compute dummy output",
            expression_method="_result_output_expression",
            assign=("result", "other"),
        )


def test_tool_script_provenance_rejects_invalid_expression(qtbot) -> None:
    class _DummyState(pydantic.BaseModel):
        value: int = 0

    class _DummyTool(erlab.interactive.utils.ToolWindow[_DummyState]):
        StateModel = _DummyState
        tool_name = "dummy"
        COPY_PROVENANCE: typing.ClassVar = (
            erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from current dummy input data",
                label="Compute dummy output",
                expression_method="_invalid_expression",
                assign="result",
            )
        )

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data

        @property
        def tool_status(self) -> _DummyState:
            return _DummyState()

        @tool_status.setter
        def tool_status(self, status: _DummyState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

        def _invalid_expression(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del input_name, data
            return "result = data + 1"

    tool = _DummyTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)

    with pytest.raises(ValueError, match="must return a valid Python expression"):
        tool.current_provenance_spec()


def test_tool_window_launch_paths_keep_declared_outputs_and_unbound_windows_separate(
    qtbot, monkeypatch
) -> None:
    prov = erlab.interactive.imagetool.provenance

    class _DummyState(pydantic.BaseModel):
        value: int = 0

    class _DummyTool(erlab.interactive.utils.ToolWindow[_DummyState]):
        StateModel = _DummyState
        tool_name = "dummy"
        COPY_PROVENANCE: typing.ClassVar = (
            erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from current dummy input data",
                label_method="_result_output_label",
                expression_method="_result_output_expression",
                assign="result",
            )
        )

        class Output(enum.StrEnum):
            RESULT = "dummy.result"

        IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
            Output.RESULT: erlab.interactive.utils.ToolImageOutputDefinition(
                data_method="_result_output_data",
                provenance=erlab.interactive.utils.ToolScriptProvenanceDefinition(
                    start_label="Start from current dummy input data",
                    label_method="_result_output_label",
                    expression_method="_result_output_expression",
                    assign="result",
                ),
            )
        }

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data

        @property
        def tool_status(self) -> _DummyState:
            return _DummyState()

        @tool_status.setter
        def tool_status(self, status: _DummyState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

        def _result_output_data(self) -> xr.DataArray:
            return self._data + 1

        def _result_output_label(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            return "Compute dummy output"

        def _result_output_expression(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del data
            return f"{input_name or 'data'} + 1"

    tool = _DummyTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)

    calls: list[dict[str, object]] = []

    def _open_stub(
        data: xr.DataArray,
        *,
        output_id: str | None,
        provenance_spec: object,
        prompt_on_reuse: bool,
    ) -> None:
        calls.append(
            {
                "data": data.copy(deep=True),
                "output_id": output_id,
                "provenance_spec": provenance_spec,
                "prompt_on_reuse": prompt_on_reuse,
            }
        )

    monkeypatch.setattr(tool, "_open_output_imagetool", _open_stub)

    live_data = tool.output_imagetool_data(_DummyTool.Output.RESULT)
    assert live_data is not None
    tool._launch_output_imagetool(live_data, output_id=_DummyTool.Output.RESULT)

    detached_data = tool.tool_data * 2
    detached_spec = prov.script(
        prov.ScriptCodeOperation(
            label="Compute detached dummy output",
            code="result = data * 2",
        ),
        start_label="Start from current dummy input data",
        active_name="result",
    )
    tool._launch_detached_output_imagetool(
        detached_data,
        provenance_spec=detached_spec,
    )

    assert len(calls) == 2
    xr.testing.assert_identical(
        typing.cast("xr.DataArray", calls[0]["data"]),
        live_data,
    )
    assert calls[0]["output_id"] == _DummyTool.Output.RESULT.value
    assert calls[0]["prompt_on_reuse"] is True
    assert calls[0]["provenance_spec"] is not None

    xr.testing.assert_identical(
        typing.cast("xr.DataArray", calls[1]["data"]),
        detached_data,
    )
    assert calls[1]["output_id"] is None
    assert calls[1]["prompt_on_reuse"] is False
    assert calls[1]["provenance_spec"] is detached_spec


def test_tool_window_unbound_launches_open_distinct_windows(qtbot) -> None:
    class _DummyState(pydantic.BaseModel):
        value: int = 0

    class _DummyTool(erlab.interactive.utils.ToolWindow[_DummyState]):
        StateModel = _DummyState
        tool_name = "dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data

        @property
        def tool_status(self) -> _DummyState:
            return _DummyState()

        @tool_status.setter
        def tool_status(self, status: _DummyState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

    tool = _DummyTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)

    first = tool._launch_detached_output_imagetool(tool.tool_data)
    second = tool._launch_detached_output_imagetool(tool.tool_data + 10)

    assert isinstance(first, erlab.interactive.imagetool.ImageTool)
    assert isinstance(second, erlab.interactive.imagetool.ImageTool)
    assert first is not second
    xr.testing.assert_identical(first.slicer_area._data, tool.tool_data)
    xr.testing.assert_identical(second.slicer_area._data, tool.tool_data + 10)

    first.close()
    second.close()

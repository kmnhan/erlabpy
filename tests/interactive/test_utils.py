import enum
import importlib
import json
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
from qtpy import PYQT6, QtCore, QtGui, QtTest, QtWidgets

import erlab.interactive.utils
from erlab.interactive.imagetool.manager._modelview import (
    _MIME,
    _NODE_UID_ROLE,
    _TOOL_TYPE_ROLE,
    _ImageToolWrapperItemModel,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)
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


class _PersistentToolState(pydantic.BaseModel):
    value: int = 0


class _PersistentTool(erlab.interactive.utils.ToolWindow[_PersistentToolState]):
    StateModel = _PersistentToolState
    tool_name = "persistent-dummy"

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = data
        self._status = _PersistentToolState()

    @property
    def tool_status(self) -> _PersistentToolState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _PersistentToolState) -> None:
        self._status = status

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    def update_data(self, new_data: xr.DataArray) -> None:
        self._data = new_data


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


def test_qt_object_is_valid_shiboken_helper_ignores_none() -> None:
    sentinel = object()
    checker = erlab.interactive.utils._make_qt_object_is_valid_from_shiboken(
        lambda obj: obj is sentinel
    )

    assert checker(sentinel)
    assert not checker(object())
    assert not checker(None)


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


def test_toolwindow_subclass_checks_survive_utils_reload(qtbot) -> None:
    class _ReloadedBaseTool(erlab.interactive.utils.ToolWindow):
        pass

    try:
        reloaded = importlib.reload(erlab.interactive.utils)

        tool = _ReloadedBaseTool()
        qtbot.addWidget(tool)

        assert issubclass(_ReloadedBaseTool, reloaded.ToolWindow)
        assert isinstance(tool, reloaded.ToolWindow)
    finally:
        importlib.reload(erlab.interactive.utils)


def test_format_kwargs_treats_python_keywords_as_mapping_keys() -> None:
    assert erlab.interactive.utils.format_kwargs({"for": 1}) == '{"for": 1}'
    assert erlab.interactive.utils.format_call_kwargs({"for": 1}) == '**{"for": 1}'


def test_generate_code_expands_python_keyword_argument_names() -> None:
    def _dummy(**kwargs):
        return kwargs

    code = generate_code(_dummy, args=(), kwargs={"for": 1, "value": 2})

    assert code == '_dummy(value=2, **{"for": 1})'
    compile(code, "<generated>", "eval")


def test_generate_code_handles_invalid_kwargs_for_empty_signature() -> None:
    def _dummy():
        return None

    code = generate_code(_dummy, args=(), kwargs={"bad key": 1})

    assert code == '_dummy(**{"bad key": 1})'
    compile(code, "<generated>", "eval")


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "label": "Compute dummy output",
                "expression_method": "_result_output_expression",
                "operations_method": "_result_output_operations",
                "assign": "result",
            },
            "must define exactly one",
        ),
        (
            {
                "expression_method": "_result_output_expression",
                "assign": "result",
            },
            "`label` or `label_method`",
        ),
        (
            {
                "label": "Compute dummy output",
                "expression_method": "_result_output_expression",
            },
            "`assign` or `assign_method`",
        ),
        (
            {
                "operations_method": "_result_output_operations",
                "label": "Compute dummy output",
            },
            "must not define `label`",
        ),
        (
            {
                "label": "Compute dummy output",
                "label_method": "_result_output_label",
                "expression_method": "_result_output_expression",
                "assign": "result",
            },
            "both `label` and `label_method`",
        ),
        (
            {
                "label": "Compute dummy output",
                "expression_method": "_result_output_expression",
                "assign": "result",
                "active_name": "other",
            },
            "single-target `assign`",
        ),
        (
            {
                "label": "Compute dummy output",
                "expression_method": "_result_output_expression",
                "assign": ("result", "other"),
                "active_name": "missing",
            },
            "tuple `assign` must include",
        ),
    ],
)
def test_tool_script_provenance_definition_rejects_invalid_configurations(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from current dummy input data",
            **kwargs,
        )


def test_tool_script_helper_validation_branches() -> None:
    with pytest.raises(TypeError, match="field must be a string"):
        erlab.interactive.utils._validate_script_identifier(1, field_name="field")
    with pytest.raises(ValueError, match="field must be a valid"):
        erlab.interactive.utils._validate_script_identifier("for", field_name="field")
    with pytest.raises(TypeError, match="field must be a string"):
        erlab.interactive.utils._normalize_script_assign(1, field_name="field")
    with pytest.raises(ValueError, match="field must not be empty"):
        erlab.interactive.utils._normalize_script_assign([], field_name="field")
    assert (
        erlab.interactive.utils._validate_script_expression("   ", field_name="expr")
        == ""
    )
    with pytest.raises(ValueError, match="valid Python statements"):
        erlab.interactive.utils._validate_script_prelude("for", field_name="prelude")

    assert erlab.interactive.utils._normalize_tool_output_id("out") == "out"
    with pytest.raises(TypeError, match="output_id must be a string"):
        erlab.interactive.utils._normalize_tool_output_id(1)
    with pytest.raises(ValueError, match="must not be empty"):
        erlab.interactive.utils._normalize_tool_output_id("")


def test_tool_window_output_definition_rejects_duplicate_normalized_ids() -> None:
    class _DuplicateOutputTool(_PersistentTool):
        class Output(enum.Enum):
            RESULT = "duplicate.result"

        IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
            Output.RESULT: erlab.interactive.utils.ToolImageOutputDefinition(
                data_method="_result_output_data"
            ),
            "duplicate.result": erlab.interactive.utils.ToolImageOutputDefinition(
                data_method="_result_output_data"
            ),
        }

        def _result_output_data(self) -> xr.DataArray:
            return self.tool_data

    with pytest.raises(ValueError, match="duplicate ImageTool output"):
        _DuplicateOutputTool._image_output_definitions()


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


def test_tool_window_dynamic_expression_provenance_uses_input_lineage(qtbot) -> None:
    prov = erlab.interactive.imagetool.provenance

    class _DynamicTool(erlab.interactive.utils.ToolWindow[_PersistentToolState]):
        StateModel = _PersistentToolState
        tool_name = "dynamic-dummy"
        COPY_PROVENANCE: typing.ClassVar = (
            erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from dynamic input",
                label_method="_dynamic_label",
                expression_method="_dynamic_expression",
                assign_method="_dynamic_assign",
                prelude_method="_dynamic_prelude",
                active_name_method="_dynamic_active_name",
                seed_code_method="_dynamic_seed_code",
            )
        )

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data

        @property
        def tool_status(self) -> _PersistentToolState:
            return _PersistentToolState()

        @tool_status.setter
        def tool_status(self, status: _PersistentToolState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

        def _dynamic_label(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del data
            assert input_name == "watched"
            return "Build dynamic outputs"

        def _dynamic_expression(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del data
            assert input_name == "watched"
            return "(watched * scale, watched + 1)"

        def _dynamic_assign(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> tuple[str, str]:
            del input_name, data
            return ("left", "right")

        def _dynamic_prelude(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del input_name, data
            return "scale = 2"

        def _dynamic_active_name(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del input_name, data
            return "right"

        def _dynamic_seed_code(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del input_name, data
            return "derived = watched"

    tool = _DynamicTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)
    tool.set_input_provenance_spec(
        prov.script(
            start_label="Start from watched data",
            seed_code="derived = watched",
            active_name="derived",
        )
    )

    spec = tool.current_provenance_spec()

    assert spec is not None
    assert spec.start_label == "Start from watched data"
    assert spec.active_name == "right"
    code = spec.display_code()
    assert code is not None
    assert "derived = watched" in code
    assert "scale = 2" in code
    assert "left, right = (watched * scale, watched + 1)" in code


def test_tool_window_operations_provenance_methods_normalize_results(qtbot) -> None:
    prov = erlab.interactive.imagetool.provenance

    class _OperationsTool(erlab.interactive.utils.ToolWindow[_PersistentToolState]):
        StateModel = _PersistentToolState
        tool_name = "operations-dummy"
        COPY_PROVENANCE: typing.ClassVar = (
            erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from operations input",
                operations_method="_dynamic_operations",
                active_name_method="_dynamic_active_name",
                seed_code_method="_dynamic_seed_code",
            )
        )

        def __init__(
            self,
            data: xr.DataArray,
            operations: object,
        ) -> None:
            super().__init__()
            self._data = data
            self._operations = operations

        @property
        def tool_status(self) -> _PersistentToolState:
            return _PersistentToolState()

        @tool_status.setter
        def tool_status(self, status: _PersistentToolState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

        def _dynamic_operations(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> object:
            del input_name, data
            return self._operations

        def _dynamic_active_name(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del input_name, data
            return "derived"

        def _dynamic_seed_code(
            self,
            *,
            input_name: str | None = None,
            data: xr.DataArray | None = None,
        ) -> str:
            del input_name, data
            return "derived = data"

    operation = prov.ScriptCodeOperation(
        label="Scale values",
        code="derived = data * 2",
    )
    tool = _OperationsTool(
        xr.DataArray(np.arange(4.0), dims=("x",), name="data"),
        operation,
    )
    qtbot.addWidget(tool)

    single_spec = tool.current_provenance_spec()
    assert single_spec is not None
    assert single_spec.operations == (operation,)

    tool._operations = [operation]
    sequence_spec = tool.current_provenance_spec()
    assert sequence_spec is not None
    assert sequence_spec.operations == (operation,)
    assert sequence_spec.seed_code == "derived = data"

    tool._operations = None
    assert tool.current_provenance_spec() is None


def test_tool_window_copy_provenance_code_handles_empty_specs(
    qtbot,
    monkeypatch,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    copied: list[str] = []
    tool = _PersistentTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda content: copied.append(content) or content,
    )
    spec = prov.script(
        prov.ScriptCodeOperation(label="Compute output", code="result = data + 1"),
        start_label="Start from current data",
        active_name="result",
    )

    assert tool._copy_provenance_code(None) == ""
    assert copied == []

    expected_code = spec.display_code()
    assert expected_code is not None
    assert tool._copy_provenance_code(spec) == expected_code
    assert copied == [expected_code]


def test_tool_window_output_target_cleanup_branches(qtbot, monkeypatch) -> None:
    manager_module = erlab.interactive.imagetool.manager
    tool = _PersistentTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)

    tool._output_imagetool_targets["out"] = "child"
    monkeypatch.setattr(manager_module, "_manager_instance", None)
    assert tool._output_imagetool_target("out") is None
    assert "out" not in tool._output_imagetool_targets

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    tool._output_imagetool_targets["out"] = widget
    monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_args: False)
    assert tool._output_imagetool_target("out") is None

    class _FakeManager:
        def __init__(self) -> None:
            self._all_nodes: dict[str, object] = {}

        def _node_uid_from_window(self, window: object) -> str:
            assert window is tool
            return "parent"

    fake_manager = _FakeManager()
    monkeypatch.setattr(manager_module, "_manager_instance", fake_manager)
    monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_args: True)

    stale_widget = QtWidgets.QWidget()
    qtbot.addWidget(stale_widget)
    tool._output_imagetool_targets["out"] = stale_widget
    assert tool._output_imagetool_target("out") is None
    assert "out" not in tool._output_imagetool_targets

    tool._output_imagetool_targets["out"] = "missing"
    assert tool._output_imagetool_target("out") is None

    fake_manager._all_nodes["wrong-parent"] = types.SimpleNamespace(
        is_imagetool=True,
        parent_uid="other",
    )
    tool._output_imagetool_targets["out"] = "wrong-parent"
    assert tool._output_imagetool_target("out") is None

    fake_manager._all_nodes["child"] = types.SimpleNamespace(
        is_imagetool=True,
        parent_uid="parent",
    )
    tool._output_imagetool_targets["out"] = "child"
    assert tool._output_imagetool_target("out") == "child"


def test_tool_window_prompt_existing_output_choices(qtbot, monkeypatch) -> None:
    tool = _PersistentTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)

    class _FakeMessageBox:
        class Icon:
            Question = object()

        class ButtonRole:
            AcceptRole = object()
            ActionRole = object()

        class StandardButton:
            Cancel = object()

        next_choice = "update"

        def __init__(self, parent: QtWidgets.QWidget) -> None:
            assert parent is tool
            self._buttons: dict[str, object] = {}

        def setIcon(self, icon: object) -> None:
            assert icon is self.Icon.Question

        def setWindowTitle(self, _title: str) -> None:
            return

        def setText(self, _text: str) -> None:
            return

        def setInformativeText(self, _text: str) -> None:
            return

        def addButton(self, *args: object) -> object:
            button = object()
            if args[0] == "Update Existing":
                self._buttons["update"] = button
            elif args[0] == "Open New":
                self._buttons["new"] = button
            else:
                self._buttons["cancel"] = button
            return button

        def setDefaultButton(self, _button: QtWidgets.QPushButton) -> None:
            return

        def setEscapeButton(self, _button: object) -> None:
            return

        def exec(self) -> int:
            return 0

        def clickedButton(self) -> object:
            return self._buttons[self.next_choice]

    monkeypatch.setattr(
        erlab.interactive.utils.QtWidgets, "QMessageBox", _FakeMessageBox
    )

    for choice in ("update", "new", "cancel"):
        _FakeMessageBox.next_choice = choice
        assert tool._prompt_existing_output_imagetool() == choice


def test_tool_window_dataset_roundtrips_source_and_input_provenance(qtbot) -> None:
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4)})
    tool = _PersistentTool(data)
    qtbot.addWidget(tool)

    source_spec = prov.full_data(prov.IselOperation(kwargs={"x": slice(0, 2)}))
    input_spec = prov.script(
        prov.ScriptCodeOperation(label="Use watched data", code="derived = watched"),
        start_label="Start from watched data",
        seed_code="derived = watched",
        active_name="derived",
    )
    tool.set_source_binding(source_spec, auto_update=True, state="stale")
    tool.set_input_provenance_spec(input_spec)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)

    assert isinstance(restored, _PersistentTool)
    assert restored.source_spec == source_spec
    assert restored.source_auto_update is True
    assert restored.source_state == "stale"
    assert restored.input_provenance_spec == input_spec.to_replay_spec()


def test_tool_window_dataset_ignores_invalid_saved_provenance(
    qtbot,
    caplog,
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4)})
    tool = _PersistentTool(data)
    qtbot.addWidget(tool)
    ds = tool.to_dataset()
    ds.attrs["tool_source_spec"] = "{not-json"
    ds.attrs["tool_input_provenance_spec"] = "{not-json"

    with caplog.at_level("WARNING", logger="erlab.interactive.utils"):
        restored = erlab.interactive.utils.ToolWindow.from_dataset(ds)
    qtbot.addWidget(restored)

    assert isinstance(restored, _PersistentTool)
    assert restored.source_spec is None
    assert restored.input_provenance_spec is None
    assert "Ignoring invalid saved tool source provenance" in caplog.text
    assert "Ignoring invalid saved tool input provenance" in caplog.text


def test_managed_tool_window_node_source_binding_branches(qtbot, monkeypatch) -> None:
    prov = erlab.interactive.imagetool.provenance

    class _FakeTreeView:
        def __init__(self) -> None:
            self.refreshed: list[str] = []

        def refresh(self, uid: str) -> None:
            self.refreshed.append(uid)

    class _FakeManager(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.tree_view = _FakeTreeView()
            self.updated: list[str] = []
            self.propagated: list[tuple[str, object | None]] = []
            self.marked: list[tuple[str, str]] = []
            self.unavailable: list[str] = []
            self.removed: list[str] = []

        def _update_info(self, *, uid: str) -> None:
            self.updated.append(uid)

        def _propagate_source_change_from_uid(
            self,
            uid: str,
            parent_data: object | None = None,
        ) -> None:
            self.propagated.append((uid, parent_data))

        def _mark_descendants_source_state(self, uid: str, state: str) -> None:
            self.marked.append((uid, state))

        def _mark_descendants_source_unavailable(self, uid: str) -> None:
            self.unavailable.append(uid)

        def _remove_childtool(self, uid: str) -> None:
            self.removed.append(uid)

    manager = _FakeManager()
    qtbot.addWidget(manager)
    tool = _PersistentTool(xr.DataArray(np.arange(4.0), dims=("x",), name="data"))
    qtbot.addWidget(tool)
    node = _ManagedWindowNode(
        typing.cast("erlab.interactive.imagetool.manager.ImageToolManager", manager),
        "child",
        None,
        tool,
    )

    with pytest.raises(TypeError, match="source_spec must be"):
        node.set_source_binding(typing.cast("object", {"kind": "full_data"}))
    with pytest.raises(ValueError, match="output_id must not be None"):
        node.set_output_binding(typing.cast("str", None))
    with pytest.raises(TypeError, match="output_id must be a string"):
        node.set_output_binding(typing.cast("str", 1))
    with pytest.raises(ValueError, match="output_id must not be empty"):
        node.set_output_binding("")
    with pytest.raises(TypeError, match="provenance_spec must be"):
        node.set_output_binding(
            "out",
            provenance_spec=typing.cast("object", {"kind": "script"}),
        )

    node.set_output_binding("out", auto_update=True, state="stale")
    assert node.output_id == "out"
    assert node._source_state == "stale"
    assert node._source_auto_update is True
    assert node._output_id == "out"
    assert manager.tree_view.refreshed[-1] == "child"
    assert manager.updated[-1] == "child"

    source_spec = prov.full_data()
    tool.set_source_binding(source_spec, auto_update=True, state="stale")
    assert node.source_spec == source_spec
    assert node.source_state == "stale"
    assert node.source_auto_update is True
    assert node.has_source_binding is True
    assert node.tree_uid_text == "child"
    with pytest.raises(TypeError, match="Window transfer"):
        node.take_window()

    xr.testing.assert_identical(node.current_source_data(), tool.tool_data)

    node._handle_tool_data_changed()
    assert manager.marked[-1] == ("child", "stale")

    tool.set_source_binding(source_spec, state="fresh")
    node._handle_tool_data_changed()
    assert manager.propagated[-1] == ("child", None)

    node._suspend_descendant_signal_propagation = True
    node._handle_tool_data_changed()
    assert manager.propagated[-1] == ("child", None)
    node._suspend_descendant_signal_propagation = False

    unbound_tool = _PersistentTool(tool.tool_data)
    qtbot.addWidget(unbound_tool)
    unbound_node = _ManagedWindowNode(
        typing.cast("erlab.interactive.imagetool.manager.ImageToolManager", manager),
        "unbound",
        None,
        unbound_tool,
    )
    assert not unbound_node.handle_parent_source_replaced(tool.tool_data)

    updated = xr.DataArray(np.arange(4.0) + 10.0, dims=("x",), name="updated")
    tool.set_source_binding(source_spec, state="stale")
    assert not node._update_from_parent_source()
    assert manager.marked[-1] == ("child", "stale")

    tool.set_source_parent_fetcher(lambda: updated)
    assert node._update_from_parent_source()
    xr.testing.assert_identical(tool.tool_data, updated)
    assert manager.propagated[-1] == ("child", None)

    node.window = None
    assert node._detach_imagetool() is None
    assert node._metadata_data() is None
    with pytest.raises(ValueError, match="ImageTool is not available"):
        _ = node.slicer_area
    with pytest.raises(ValueError, match="Managed node is not available"):
        node.current_source_data()
    node._handle_source_data_replaced(object())
    assert manager.unavailable[-1] == "child"

    stale_tool = _PersistentTool(tool.tool_data)
    stale_imagetool = QtWidgets.QWidget()
    qtbot.addWidget(stale_tool)
    qtbot.addWidget(stale_imagetool)
    node._tool_window = stale_tool
    node._imagetool = typing.cast(
        "erlab.interactive.imagetool.ImageTool", stale_imagetool
    )
    monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_args: False)
    assert node.tool_window is None
    assert node.imagetool is None


def test_managed_tool_window_node_detached_update_branches(
    qtbot,
    monkeypatch,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    parent_data = xr.DataArray(np.arange(4.0), dims=("x",), name="parent")

    class _FakeTreeView:
        def __init__(self) -> None:
            self.refreshed: list[str] = []

        def refresh(self, uid: str) -> None:
            self.refreshed.append(uid)

    class _OutputTool(_PersistentTool):
        def __init__(self, data: xr.DataArray) -> None:
            super().__init__(data)
            self.output_data: xr.DataArray | None = None
            self.output_provenance: (
                erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None
            ) = None

        def output_imagetool_data(
            self, output_id: str | enum.Enum
        ) -> xr.DataArray | None:
            assert output_id == "out"
            return self.output_data

        def output_imagetool_provenance(
            self,
            output_id: str | enum.Enum,
            data: xr.DataArray,
        ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
            assert output_id == "out"
            assert data is self.output_data
            return self.output_provenance

    class _FakeManager(QtWidgets.QWidget):
        def __init__(self, parent_tool: _OutputTool) -> None:
            super().__init__()
            self.tree_view = _FakeTreeView()
            self.updated: list[str] = []
            self.marked: list[tuple[str, str]] = []
            self.unavailable: list[str] = []
            self.removed: list[str] = []
            self.parent_node = types.SimpleNamespace(
                tool_window=parent_tool,
                provenance_spec=None,
                current_source_data=lambda: parent_tool.tool_data,
            )

        def _update_info(self, *, uid: str) -> None:
            self.updated.append(uid)

        def _parent_node(self, _node: _ManagedWindowNode) -> object:
            return self.parent_node

        def _parent_source_data_for_uid(self, uid: str) -> xr.DataArray:
            assert uid == "child"
            return typing.cast("_OutputTool", self.parent_node.tool_window).tool_data

        def _mark_descendants_source_state(self, uid: str, state: str) -> None:
            self.marked.append((uid, state))

        def _mark_descendants_source_unavailable(self, uid: str) -> None:
            self.unavailable.append(uid)

        def _remove_childtool(self, uid: str) -> None:
            self.removed.append(uid)

    parent_tool = _OutputTool(parent_data)
    qtbot.addWidget(parent_tool)
    manager = _FakeManager(parent_tool)
    qtbot.addWidget(manager)
    tool = _PersistentTool(parent_data)
    qtbot.addWidget(tool)
    node = _ManagedWindowNode(
        typing.cast("erlab.interactive.imagetool.manager.ImageToolManager", manager),
        "child",
        "parent",
        tool,
    )

    with pytest.raises(TypeError, match="provenance_spec must be"):
        node.set_source_binding(
            prov.full_data(),
            provenance_spec=typing.cast("object", {"kind": "full_data"}),
        )
    with pytest.raises(TypeError, match="provenance_spec must be"):
        node.set_detached_provenance(typing.cast("object", {"kind": "script"}))

    display_spec = prov.script(
        prov.ScriptCodeOperation(label="Use output", code="result = data + 1"),
        start_label="Start from data",
        active_name="result",
    )
    node.set_detached_provenance(display_spec)
    assert node._provenance_spec == display_spec

    assert node._resolved_output_payload() is None
    node.set_output_binding("out", state="fresh")
    manager.parent_node.tool_window = None
    assert node._resolved_output_payload() is None
    manager.parent_node.tool_window = parent_tool
    assert node._resolved_output_payload() is None

    parent_tool.output_data = xr.DataArray(np.arange(4.0) + 1.0, dims=("x",))
    payload = node._resolved_output_payload()
    assert payload is not None
    xr.testing.assert_identical(payload[0], parent_tool.output_data)

    node.window = None
    node.set_detached_provenance(display_spec)
    assert node.derivation_lines == ["Start from data", "Use output"]
    node._discard_archived_file()
    node._handle_tool_data_changed()
    node.show()
    node.archive()

    node.set_output_binding("out", state="fresh")
    parent_tool.set_source_binding(prov.full_data(), state="stale")
    assert not node._update_from_parent_source()
    assert node.source_state == "stale"
    assert manager.marked[-1] == ("child", "stale")

    parent_tool.set_source_binding(prov.full_data(), state="fresh")
    parent_tool.output_data = None
    assert not node._update_from_parent_source()
    assert node.source_state == "unavailable"

    node.set_detached_provenance(None)
    assert not node._update_from_parent_source()

    parent_tool.output_data = xr.DataArray(np.arange(4.0) + 2.0, dims=("x",))
    node.set_output_binding("out", auto_update=True, state="fresh")
    assert not node._update_from_parent_source()
    assert manager.unavailable[-1] == "child"

    node.set_output_binding("out", auto_update=False, state="fresh")
    assert not node.handle_parent_source_replaced(parent_data)
    assert node.source_state == "stale"

    node.set_output_binding("out", auto_update=True, state="fresh")
    parent_tool.output_data = None
    assert not node.handle_parent_source_replaced(parent_data)
    assert node.source_state == "unavailable"

    node.set_detached_provenance(None)
    assert not node.handle_parent_source_replaced(parent_data)

    node.set_source_binding(
        prov.full_data(prov.IselOperation(kwargs={"missing": 0})),
        state="stale",
    )
    assert not node.handle_parent_source_replaced(parent_data)
    assert node.source_state == "unavailable"

    node.set_source_binding(prov.full_data(), auto_update=True, state="stale")
    assert not node.handle_parent_source_replaced(parent_data)
    assert node.source_state == "unavailable"

    node.set_detached_provenance(None)
    assert node.show_source_update_dialog(parent=manager) == int(
        QtWidgets.QDialog.DialogCode.Rejected
    )

    class _FakeSourceUpdateDialog:
        def __init__(
            self,
            parent: QtWidgets.QWidget,
            *,
            state: str,
            auto_update: bool,
        ) -> None:
            assert parent is manager
            assert state == "stale"
            assert auto_update is False
            self.auto_update_check = types.SimpleNamespace(isChecked=lambda: True)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "_ToolSourceUpdateDialog",
        _FakeSourceUpdateDialog,
    )
    node.set_output_binding("out", auto_update=False, state="stale")
    assert node.show_source_update_dialog(parent=manager) == int(
        QtWidgets.QDialog.DialogCode.Accepted
    )
    assert node.source_auto_update is True
    assert manager.updated[-1] == "child"


def test_imagetool_wrapper_item_model_child_edge_branches(qtbot, monkeypatch) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")

    class _FakeTreeView:
        def refresh(self, _uid: str) -> None:
            return

    class _FakeManager(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.tree_view = _FakeTreeView()
            self._displayed_indices: list[int] = []
            self._imagetool_wrappers: list[object] = []
            self._all_nodes: dict[str, object] = {}
            self.updated: list[str] = []
            self.removed: list[str] = []
            self.renamed: list[tuple[int, object]] = []

        def _update_info(self, *, uid: str) -> None:
            self.updated.append(uid)

        def _remove_childtool(self, uid: str) -> None:
            self.removed.append(uid)

        def _child_node(self, uid: str) -> _ManagedWindowNode:
            return typing.cast("_ManagedWindowNode", self._all_nodes[uid])

        def rename_imagetool(self, index: int, value: object) -> None:
            self.renamed.append((index, value))

        def label_of_imagetool(self, index: int) -> str:
            return f"label {index}"

        def name_of_imagetool(self, index: int) -> str:
            return f"name {index}"

    manager = _FakeManager()
    qtbot.addWidget(manager)
    parent_tool = _PersistentTool(data)
    child_tool = _PersistentTool(data + 1)
    orphan_tool = _PersistentTool(data + 2)
    qtbot.addWidget(parent_tool)
    qtbot.addWidget(child_tool)
    qtbot.addWidget(orphan_tool)

    parent_node = _ImageToolWrapper(
        typing.cast("erlab.interactive.imagetool.manager.ImageToolManager", manager),
        0,
        "parent",
        parent_tool,
    )
    child_node = _ManagedWindowNode(
        typing.cast("erlab.interactive.imagetool.manager.ImageToolManager", manager),
        "child",
        "parent",
        child_tool,
    )
    orphan_node = _ManagedWindowNode(
        typing.cast("erlab.interactive.imagetool.manager.ImageToolManager", manager),
        "orphan",
        None,
        orphan_tool,
    )
    parent_node.add_child_reference("child", child_tool)
    manager._displayed_indices.append(0)
    manager._imagetool_wrappers.append(parent_node)
    manager._all_nodes.update(
        {
            "parent": parent_node,
            "child": child_node,
            "orphan": orphan_node,
        }
    )

    model = _ImageToolWrapperItemModel(
        typing.cast("erlab.interactive.imagetool.manager.ImageToolManager", manager)
    )
    model_tester = QtTest.QAbstractItemModelTester(
        model,
        QtTest.QAbstractItemModelTester.FailureReportingMode.Fatal,
        model,
    )
    assert model_tester.model() is model
    wrong_pointer = object()
    missing_index = model.createIndex(0, 0, "missing")
    object_index = model.createIndex(0, 0, wrong_pointer)
    child_index = model._row_index("child")
    orphan_index = model.createIndex(0, 0, "orphan")
    nonzero_column_index = model.createIndex(0, 1, "child")

    assert model.data(object_index, QtCore.Qt.ItemDataRole.DisplayRole) is None
    assert model.data(missing_index, QtCore.Qt.ItemDataRole.DisplayRole) is None
    assert model.data(nonzero_column_index, QtCore.Qt.ItemDataRole.DisplayRole) is None
    assert model.flags(QtCore.QModelIndex()) == QtCore.Qt.ItemFlag.ItemIsDropEnabled
    assert model.flags(object_index) == QtCore.Qt.ItemFlag.NoItemFlags
    assert model.flags(nonzero_column_index) == QtCore.Qt.ItemFlag.NoItemFlags
    assert not model.setData(missing_index, "unused")
    assert not model.setData(nonzero_column_index, "unused")
    assert not model.parent(missing_index).isValid()
    assert not model.parent(orphan_index).isValid()
    assert isinstance(model.parent(child_index), QtCore.QModelIndex)
    assert model.parent(child_index).internalPointer() is parent_node
    assert model.rowCount(nonzero_column_index) == 0
    assert not model.hasChildren(nonzero_column_index)
    assert model.mimeTypes() == [_MIME]
    with pytest.raises(KeyError):
        model._childtool_uid(0, "missing-parent")

    assert model._childtool(model.createIndex(0, 0, "child"), "parent") is child_node
    assert not model._row_index("orphan").isValid()
    assert model.data(child_index, QtCore.Qt.ItemDataRole.DisplayRole) == ""
    assert model.data(child_index, _TOOL_TYPE_ROLE) == "persistent-dummy"
    assert model.data(child_index, _NODE_UID_ROLE) == "child"
    assert model.data(child_index, QtCore.Qt.ItemDataRole.SizeHintRole) == QtCore.QSize(
        100, 25
    )
    assert model.data(child_index, QtCore.Qt.ItemDataRole.UserRole) is None

    child_node.window = None
    child_node.name = "Closed Child"
    assert model.data(child_index, QtCore.Qt.ItemDataRole.EditRole) == "Closed Child"
    assert model.flags(child_index) & QtCore.Qt.ItemFlag.ItemIsEditable

    model.remove_rows(0, 1, missing_index)
    with monkeypatch.context() as patch:
        patch.setattr(model, "_row_index", lambda _idx: missing_index)
        model._insert_childtool("child", "missing-parent")
    mime_data = model.mimeData([model._row_index("child")])
    payload = json.loads(bytes(mime_data.data(_MIME)).decode())
    assert payload == {"parent_id": "parent", "rows": [0]}
    assert _ImageToolWrapperItemModel._decode_mime(QtCore.QMimeData()) is None


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

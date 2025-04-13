"""Various helper functions and extensions to Qt and pyqtgraph.

This module contains various helper functions and classes that extend the functionality
of pyqtgraph and Qt.
"""

from __future__ import annotations

import contextlib
import fnmatch
import inspect
import itertools
import pathlib
import re
import sys
import types
import typing
import warnings
import weakref
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Callable, Collection, Iterator, Mapping

    import pyperclip
    import qtawesome
    from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
else:
    import lazy_loader as _lazy

    pyperclip = _lazy.load("pyperclip")
    qtawesome = _lazy.load("qtawesome")

__all__ = [
    "AnalysisWidgetBase",
    "AnalysisWindow",
    "BetterAxisItem",
    "BetterSpinBox",
    "DictMenuBar",
    "ExclusiveComboGroup",
    "IconActionButton",
    "IconButton",
    "KeyboardEventFilter",
    "ParameterGroup",
    "RotatableLine",
    "copy_to_clipboard",
    "file_loaders",
    "format_kwargs",
    "generate_code",
    "make_crosshairs",
    "parse_data",
    "wait_dialog",
    "xImageItem",
]


def parse_data(data) -> xr.DataArray:
    if isinstance(data, xr.Dataset):
        raise TypeError(
            "input argument data must be a xarray.DataArray or a "
            "numpy.ndarray. Create an xarray.DataArray "
            "first, either with indexing on the Dataset or by "
            "invoking the `to_array()` method."
        ) from None
    if isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    return data  # .astype(float, order="C")


@contextlib.contextmanager
def setup_qapp(execute: bool | None = None) -> Iterator[bool]:
    """Set up a Qt application instance and manage its execution.

    This function initializes a Qt application instance if one does not already exist.
    It sets the application style to "Fusion" and determines whether to execute the
    application based on the environment (interactive or not). The function yields a
    boolean indicating whether the application is executed.

    Generally, a Qt application in a python script is executed like this:

    .. code-block:: python

        qapp = QtWidgets.QApplication.instance()

        if not qapp:
            qapp = QtWidgets.QApplication(sys.argv)

        win = MyMainWindow()

        win.show()

        qapp.exec()

    In an interactive environment like IPython and jupyter, the event loop is handled by
    IPython, so ``qapp.exec()`` changes to:

    .. code-block:: python

        from IPython.lib.guisupport import start_event_loop_qt4

        start_event_loop_qt4(qapp)

    This function combines the two approaches and determines whether to execute the
    event loop based on the environment. The resulting code is:

    .. code-block:: python

        with setup_qapp():
            win = MyMainWindow() win.show()

    Parameters
    ----------
    execute : bool or None, optional
        If True, the application will be executed. If False, it will not be executed. If
        None, the function will determine the value based on the environment. See notes.
        Default is None.

    Yields
    ------
    bool
        A boolean indicating whether the application should be executed. If ``execute``
        is provided, the value will be the same as the input. Otherwise, the
        automatically determined value will be returned.

    Notes
    -----
    If the environment is interactive (e.g., IPython), the application will not be
    executed, and the event loop will be handled by IPython. If the environment is not
    interactive, the application will be executed unless `execute` is explicitly set to
    False.

    """
    try:
        qapp = QtWidgets.QApplication.instance()
        if not qapp:
            qapp = QtWidgets.QApplication(sys.argv)
        if isinstance(qapp, QtWidgets.QApplication):
            qapp.setStyle("Fusion")

        is_ipython: bool = False

        if execute is None:
            execute = True
            is_ipython = erlab.utils.misc.is_interactive()
            if is_ipython:
                execute = False
        yield execute

    finally:
        if is_ipython:
            from IPython.lib.guisupport import start_event_loop_qt4

            start_event_loop_qt4(qapp)

        elif execute and (qapp is not None):
            qapp.exec()


class _WaitDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, message: str) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel(message))
        self.setLayout(layout)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)


@contextlib.contextmanager
def wait_dialog(parent: QtWidgets.QWidget, message: str) -> Iterator[_WaitDialog]:
    """Show a wait dialog while executing a block of code.

    This context manager creates a simple dialog with a message while the block of code
    is being executed. The dialog is closed when the block is done.

    Parameters
    ----------
    parent
        Parent widget.
    message
        Message to display in the dialog.

    Example
    -------
    >>> with wait_dialog(self, "Processing data..."):
    >>>    some_long_running_code()

    """
    dialog = _WaitDialog(parent, message)
    try:
        dialog.open()
        yield dialog
    finally:
        dialog.close()


def array_rect(data):
    data_coords = tuple(data[dim].values for dim in data.dims)
    data_incs = tuple(coord[1] - coord[0] for coord in data_coords)
    data_lims = tuple((coord[0], coord[-1]) for coord in data_coords)
    y, x = data_lims[0][0] - data_incs[0], data_lims[1][0] - data_incs[1]
    h, w = data_lims[0][-1] - y, data_lims[1][-1] - x
    y += 0.5 * data_incs[0]
    x += 0.5 * data_incs[1]
    return QtCore.QRectF(x, y, w, h)


def copy_to_clipboard(content: str | list[str]) -> str:
    """Copy content to the clipboard.

    Parameters
    ----------
    content
        The content to be copied. If a list of strings is passed, the strings are joined
        by newlines.

    Returns
    -------
    str
        The copied content.
    """
    if isinstance(content, list):
        content = "\n".join(content)
    pyperclip.copy(str(content))
    return content


def format_kwargs(d: dict[str, typing.Any]) -> str:
    """Format a dictionary of keyword arguments for a function call.

    If the keys are valid Python identifiers, the output will be formatted as keyword
    arguments. Otherwise, the output will be formatted as a dictionary.

    Parameters
    ----------
    d
        Dictionary of keyword arguments.

    """
    if all(s.isidentifier() for s in d):
        return ", ".join(f"{k}={_parse_single_arg(v)!s}" for k, v in d.items())
    out = ", ".join(f'"{k}": {_parse_single_arg(v)!s}' for k, v in d.items())
    return "{" + out + "}"


def _parse_single_arg(arg):
    arg = erlab.utils.misc._convert_to_native(arg)

    if isinstance(arg, str):
        # If the string is surrounded by vertical bars, remove them
        # Otherwise, quote the string
        if arg.startswith("|") and arg.endswith("|"):
            arg = arg[1:-1]
        elif '"' in arg:
            arg = f"'{arg}'"
        else:
            arg = f'"{arg}"'

    elif isinstance(arg, dict):
        # If the argument is a dict, convert to string
        arg = {k: erlab.utils.misc._convert_to_native(v) for k, v in arg.items()}
        arg = (
            "{"
            + ", ".join([f'"{k}": {_parse_single_arg(v)}' for k, v in arg.items()])
            + "}"
        )

    return arg


# @functools.cache
def _filter_to_patterns(name_filter: str) -> list[str]:
    """Extract a list of patterns from a name filter."""
    split = name_filter.split("(", 1)
    return (split[0] if len(split) == 1 else split[1].rstrip(")")).split(" ")


def file_loaders(
    file_name: str | os.PathLike | None | Iterable[str | os.PathLike] = None,
) -> dict[str, tuple[Callable, dict]]:
    """Generate a dictionary of namefilters and loader functions for file dialogs.

    Parameters
    ----------
    file_name
        Name of the file to load. If provided, only the loaders that match the file name
        are returned. If an iterable of file names is provided, the loaders that match
        all file names are returned.

    Returns
    -------
    dict
        Dictionary of file loaders. The keys are name filters(argument to
        :meth:`QtWidgets.QFileDialog.setNameFilter`), and the values are tuples of the
        loader function and additional keyword arguments.
    """
    valid_loaders: dict[str, tuple[Callable, dict]] = {
        "xarray HDF5 Files (*.h5)": (xr.load_dataarray, {"engine": "h5netcdf"}),
        "NetCDF Files (*.nc *.nc4 *.cdf)": (xr.load_dataarray, {}),
        "Igor Binary Waves (*.ibw)": (xr.load_dataarray, {"engine": "erlab-igor"}),
        "Igor Packed Experiment Templates (*.pxt)": (
            xr.load_dataarray,
            {"engine": "erlab-igor"},
        ),
    }

    additional_loaders: dict[str, tuple[Callable, dict]] = {}
    for k in erlab.io.loaders:
        additional_loaders = (
            additional_loaders | erlab.io.loaders[k].file_dialog_methods
        )

    valid_loaders = valid_loaders | dict(sorted(additional_loaders.items()))

    if not file_name:
        return valid_loaders

    if not isinstance(file_name, Iterable):
        file_name = [file_name]

    file_name = [pathlib.Path(f) for f in file_name]

    valid_keys: list[str] = []
    for name_filter in valid_loaders:
        for pattern in _filter_to_patterns(name_filter):
            if all(fnmatch.fnmatch(p.name, pattern) for p in file_name):
                valid_keys.append(name_filter)
                break

    return {k: valid_loaders[k] for k in valid_keys}


def generate_code(
    func: Callable,
    args: Collection[typing.Any],
    kwargs: dict[str, typing.Any],
    module: str | None = None,
    name: str | None = None,
    assign: str | None = None,
    prefix: str | None = None,
    remove_defaults: bool = True,
    line_length: int = 88,
    copy: bool = False,
) -> str:
    r"""Generate Python code for a function call.

    The result can be copied to your clipboard in a form that can be pasted into an
    interactive Python session or Jupyter notebook cell.

    Parameters
    ----------
    func
        Function to generate code for.
    args
        Positional arguments passed onto the function. Non-string arguments are
        converted to strings. If given a string, quotes are added. If surrounded by
        vertical bars (``|``), the string is not quoted.
    kwargs
        Keyword arguments passed onto the function.
    module
        Prefix to add to the function name. For example, ``"scipy.ndimage"``.
    name
        Name of the function. If `None`, `func.__name__` is used.
    assign
        If provided, the return value will be assigned to this variable.
    prefix
        Prefix to add to the generated string.
    remove_defaults
        If `True`, keyword arguments that have values identical to the defaults are
        removed.
    line_length
        Maximum length of the line. If the line is longer than this, it will be split
        into multiple lines.
    copy
        If `True`, the result is copied to the clipboard.

    Examples
    --------
    >>> import numpy as np
    >>> from erlab.interactive.utils import generate_code
    >>> generate_code(
    >>>     np.linalg.norm,
    >>>     args=["|np.array([1, 2, 3])|"],
    >>>     kwargs=dict(ord=2, keepdims=False),
    >>>     module="np.linalg",
    >>>     assign="value",
    >>> )
    'value = numpy.linalg.norm(numpy.array([1, 2, 3]), ord=2)'

    """
    if remove_defaults:
        kwargs = _remove_default_kwargs(func, dict(kwargs))

    args, kwargs = _handle_xarray_dict_or_kwargs(func, args, kwargs)

    module = "" if module is None else f"{module.strip(' .')}."

    if name is None:
        name = func.__name__
    if assign is not None:
        module = f"{assign} = {module}"
    if prefix is not None:
        module = f"{prefix}{module}"

    code_str = _gen_single_function_code(
        f"{module}{name}", args, kwargs, line_length=line_length
    )

    if copy:
        return copy_to_clipboard(code_str)
    return code_str


def _handle_xarray_dict_or_kwargs(
    func: Callable, args: Collection[typing.Any], kwargs: dict[str, typing.Any]
):
    """Handle compatibility with xarray methods that accept either kwargs or a dict.

    Many xarray methods accept either keyword arguments or a dictionary as the first
    positional argument.

    Where no positional arguments are given for such functions and the keyword argument
    dictionary contains at least one key that contains spaces, a conversion of kwargs to
    the first positional argument is attempted.
    """
    if len(args) != 0 or all(k.isidentifier() for k in kwargs):
        return args, kwargs

    params = inspect.signature(func).parameters

    param_order = list(params.keys())
    if param_order[0] == "self":
        param_order = param_order[1:]

    # Name of first positional argument
    arg_name = param_order[0]
    kwarg_name = param_order[0] + "_kwargs"

    if (
        (kwarg_name not in params)
        or (params[kwarg_name].kind != inspect.Parameter.VAR_KEYWORD)
        or (params[arg_name].default is not None)
        or (params[arg_name].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ):
        return args, kwargs

    pos_arg = {}
    for k in list(kwargs.keys()):
        if k not in param_order:
            pos_arg[k] = f"|{_parse_single_arg(kwargs.pop(k))}|"
            # pos_arg[k] = _parse_single_arg(kwargs.pop(k))

    return [pos_arg], kwargs


def _remove_default_kwargs(
    func: Callable, kwargs: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """Clean up keyword arguments for a function.

    Given a function and a dictionary of keyword arguments, remove any keys that already
    have the default value in the function signature.

    Parameters
    ----------
    func
        Function to clean up keyword arguments for.
    kwargs
        Dictionary of keyword arguments.

    Returns
    -------
    dict
        Cleaned up dictionary of keyword arguments.
    """
    params = inspect.signature(func).parameters

    for k, v in dict(kwargs).items():
        if k not in params:
            continue
        if params[k].default is not inspect.Parameter.empty and v == params[k].default:
            kwargs.pop(k)
    return kwargs


def _gen_single_function_code(
    funcname: str,
    args: Collection[typing.Any],
    kwargs: Mapping[str, typing.Any],
    *,
    line_length: int = 88,
) -> str:
    """Generate the string for a Python function call.

    The first argument is the name of the function, and subsequent arguments are passed
    as positional arguments. Keyword arguments are also supported. For strings in
    arguments and keyword arguments, surrounding the string with vertical bars (``|``)
    will prevent the string from being quoted.

    Parameters
    ----------
    funcname
        Name of the function.
    args
        Mandatory arguments passed onto the function.
    kwargs
        Keyword arguments passed onto the function.
    line_length
        Maximum length of the line. If the line is longer than this, it will be split
        into multiple lines.

    Returns
    -------
    code : str
        generated code.

    """
    if len(args) == 0 and len(kwargs) == 0:
        # If no arguments are passed, return the function name
        return f"{funcname}()"

    TAB = "    "

    # Start with function call and open parenthesis
    code = f"{funcname}(\n"

    for v in args:
        # Add positional argument to code string
        code += f"{TAB}{_parse_single_arg(v)},\n"

    for k, v in kwargs.items():
        # Add keyword argument to code string
        code += f"{TAB}{k}={_parse_single_arg(v)},\n"

    # Add closing parenthesis
    code += ")"

    if len(code.replace("\n", "")) <= line_length:
        # If code fits in one line, remove newlines
        code = " ".join([s.strip() for s in code.split("\n")])
        # Remove trailing comma and space
        code = code.replace(", )", ")").replace("( ", "(")

    return code


class KeyboardEventFilter(QtCore.QObject):
    """Event filter that intercepts select all and copy shortcuts.

    For some operating systems, shortcuts are often intercepted by actions in the menu
    bar. This filter ensures that the shortcuts work as expected when the target widget
    has focus.

    This filter can be used when the target widget does receive the shortcut event with
    type `QtCore.QEvent.Type.ShortcutOverride`, but does not respond to it. If the
    target widget never receives the event, a different approach using the current focus
    widget is needed.
    """

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.ShortcutOverride
            and isinstance(obj, QtWidgets.QWidget)
            and obj.hasFocus()
        ):
            event = typing.cast("QtGui.QKeyEvent", event)
            if event.matches(QtGui.QKeySequence.StandardKey.SelectAll) or event.matches(
                QtGui.QKeySequence.StandardKey.Copy
            ):
                event.accept()
                return True
        return super().eventFilter(obj, event)


class BetterSpinBox(QtWidgets.QAbstractSpinBox):
    """An improved spinbox.

    Signals
    ----------
    valueChanged
        Emitted when the value is changed.
    textChanged
        Emitted when the text is changed.

    Parameters
    ----------
    integer
        If `True`, the spinbox will only display integer values.
    compact
        Whether to reduce the height of the spinbox.
    discrete
        If `True` the spinbox will only step to pre-determined discrete values.

        If `False`, the spinbox will just add or subtract the predetermined
        increment when increasing or decreasing the step.
    decimals
        The precision of the spinbox. See the `significant` argument for the
        meaning. When `integer` is `True`, this argument is ignored.
    significant
        If `True`, `decimals` will specify the total number of significant digits,
        before or after the decimal point, ignoring leading zeros.

        If `False`, `decimals` will specify the total number of digits after the
        decimal point, including leading zeros.

        When `integer` or `scientific` is `True`, this argument is ignored.
    scientific
        Whether to print in scientific notation.
    value
        Initial value of the spinbox.

    """

    valueChanged = QtCore.Signal(object)  #: :meta private:
    textChanged = QtCore.Signal(object)  #: :meta private:
    editingStarted = QtCore.Signal()  #: :meta private:

    def __init__(
        self,
        *args,
        integer: bool = False,
        compact: bool = True,
        discrete: bool = False,
        decimals: int = 3,
        significant: bool = False,
        scientific: bool = False,
        value: float = 0.0,
        prefix: str = "",
        **kwargs,
    ) -> None:
        self._only_int = integer
        self._is_compact = compact
        self._is_discrete = discrete
        self._is_scientific = scientific
        self._decimal_significant = significant
        self._decimals = decimals

        self._value = value
        self._lastvalue: float | None = None
        self._min = -np.inf
        self._max = np.inf
        self._step = 1 if self._only_int else 0.01
        self._prefix = prefix

        kwargs.setdefault("correctionMode", self.CorrectionMode.CorrectToPreviousValue)
        kwargs.setdefault("keyboardTracking", False)

        # PyQt6 compatibility: set options with keyword arguments
        set_dict = {}
        for k in ["singleStep", "minimum", "maximum"]:
            set_dict[k] = kwargs.pop(k, None)
        super().__init__(*args, **kwargs)
        for k, v in set_dict.items():
            if v is not None:
                getattr(self, f"set{k[0].capitalize()}{k[1:]}")(v)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.editingFinished.connect(self.editingFinishedEvent)
        self._updateHeight()
        self._updateWidth()

        if self.isReadOnly():
            line_edit = self.lineEdit()
            if line_edit is not None:
                line_edit.setReadOnly(True)
            self.setButtonSymbols(self.ButtonSymbols.NoButtons)
        self.setValue(self.value())

    @QtCore.Slot(str)
    def setPrefix(self, prefix: str) -> None:
        self._prefix = prefix

    def prefix(self) -> str:
        return self._prefix

    @QtCore.Slot(int)
    def setDecimals(self, decimals: int) -> None:
        self._decimals = decimals
        self._updateWidth()

    def decimals(self) -> int:
        return self._decimals

    def setRange(self, mn, mx) -> None:
        self.setMinimum(min(mn, mx))
        self.setMaximum(max(mn, mx))

    @QtCore.Slot(str, result=int)
    def widthFromText(self, text: str) -> int:
        return QtGui.QFontMetrics(self.font()).boundingRect(text).width()

    def widthFromValue(self, value) -> int:
        return self.widthFromText(self.textFromValue(value))

    def setMaximum(self, mx) -> None:
        if self._only_int and np.isfinite(mx):
            mx = round(mx)
        elif np.isnan(mx):
            mx = np.inf
        self._max = mx
        self._updateWidth()

    def setMinimum(self, mn) -> None:
        if self._only_int and np.isfinite(mn):
            mn = round(mn)
        elif np.isnan(mn):
            mn = -np.inf
        self._min = mn
        self._updateWidth()

    def setSingleStep(self, step) -> None:
        if self._only_int:
            step = round(step)
        self._step = abs(step)

    def singleStep(self):
        return self._step

    def maximum(self):
        return self._max

    def minimum(self):
        return self._min

    def value(self):
        if self._only_int:
            return int(self._value)
        return self._value

    def text(self) -> str:
        return self.textFromValue(self.value())

    def textFromValue(self, value) -> str:
        if (not self._only_int) or (not np.isfinite(value)):
            if self._is_scientific:
                return self.prefix() + np.format_float_scientific(
                    value,
                    precision=self.decimals(),
                    unique=False,
                    trim="k",
                    exp_digits=1,
                )
            return self.prefix() + np.format_float_positional(
                value,
                precision=self.decimals(),
                unique=False,
                fractional=not self._decimal_significant,
                trim="k",
            )
        return self.prefix() + str(int(value))

    def valueFromText(self, text: str):
        text = text[len(self.prefix()) :]
        if text == "":
            return np.nan
        if self._only_int:
            return int(text)
        return float(text)

    def stepBy(self, steps) -> None:
        self.editingStarted.emit()
        inc = self.singleStep()
        if (
            all(np.isfinite([self.maximum(), self.minimum(), self.value()]))
            and self._is_discrete
        ):
            self.setValue(
                self.minimum()
                + inc
                * max(
                    min(
                        round((self.value() + steps * inc - self.minimum()) / inc),
                        int((self.maximum() - self.minimum()) / inc),
                    ),
                    0,
                )
            )
        elif steps > 0:
            self.setValue(min(inc * steps + self.value(), self.maximum()))
        else:
            self.setValue(max(inc * steps + self.value(), self.minimum()))

    def stepEnabled(self):
        if self.isReadOnly():
            return self.StepEnabledFlag.StepNone
        if self.wrapping():
            return (
                self.StepEnabledFlag.StepDownEnabled
                | self.StepEnabledFlag.StepUpEnabled
            )
        if self.value() < self.maximum():
            if self.value() > self.minimum():
                return (
                    self.StepEnabledFlag.StepDownEnabled
                    | self.StepEnabledFlag.StepUpEnabled
                )
            return self.StepEnabledFlag.StepUpEnabled
        if self.value() > self.minimum():
            return self.StepEnabledFlag.StepDownEnabled
        return self.StepEnabledFlag.StepNone

    def setValue(self, val) -> None:
        val = np.nan if np.isnan(val) else max(self.minimum(), min(val, self.maximum()))

        if self._only_int and np.isfinite(val):
            val = round(val)

        self._lastvalue, self._value = self._value, val

        self.valueChanged.emit(self.value())
        line = self.lineEdit()
        if line is not None:
            line.setText(self.text())
        self.textChanged.emit(self.text())

    def fixup(self, _):
        # Called when the spinbox loses focus with an invalid or intermediate string
        return self.textFromValue(self._lastvalue)

    def validate(self, strn, pos):
        if strn == "-":
            ret = QtGui.QValidator.State.Intermediate
        else:
            ret = QtGui.QValidator.State.Intermediate
            try:
                val = self.valueFromText(strn)
                if val < self.maximum() and val > self.minimum():
                    ret = QtGui.QValidator.State.Acceptable
            except ValueError:
                # sys.excepthook(*sys.exc_info())
                ret = QtGui.QValidator.State.Invalid

        # note: if text is invalid, we don't change the textValid flag since the text
        # will be forced to its previous state anyway
        return (ret, strn, pos)

    def editingFinishedEvent(self) -> None:
        line = self.lineEdit()
        if line is not None:
            self.setValue(self.valueFromText(line.text()))

    def keyPressEvent(self, evt) -> None:
        line = self.lineEdit()
        if line is not None:
            if evt == QtGui.QKeySequence.StandardKey.Copy:
                if (not evt.isAutoRepeat()) and line.hasSelectedText():
                    copy_to_clipboard(line.selectedText())
            else:
                super().keyPressEvent(evt)
        else:
            super().keyPressEvent(evt)

    def focusInEvent(self, evt) -> None:
        self.editingStarted.emit()
        super().focusInEvent(evt)

    def _updateHeight(self) -> None:
        if self._is_compact:
            self.setFixedHeight(QtGui.QFontMetrics(self.font()).height() + 3)

    def _get_offset(self):
        spin = QtWidgets.QDoubleSpinBox(self)
        spin.setRange(0, 0)
        w = (
            spin.minimumSizeHint().width()
            - QtGui.QFontMetrics(spin.font())
            .boundingRect(spin.textFromValue(0.0))
            .width()
        )
        spin.setDisabled(True)
        spin.setVisible(False)
        del spin
        return w

    def _updateWidth(self) -> None:
        self.setMinimumWidth(
            max(
                self.widthFromValue(self.maximum()), self.widthFromValue(self.minimum())
            )
            + self._get_offset()
        )


class BetterAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def updateAutoSIPrefix(self) -> None:
        if self.label.isVisible():
            _range = 10 ** np.array(self.range) if self.logMode else self.range
            (scale, prefix) = pg.siScale(
                max(abs(_range[0] * self.scale), abs(_range[1] * self.scale))
            )
            if self.labelUnits == "" and prefix in [
                "k",
                "m",
            ]:  # If we are not showing units, wait until 1e6 before scaling.
                scale = 1.0
                prefix = ""
            self.autoSIPrefixScale = scale
            self.labelUnitPrefix = prefix
        else:
            self.autoSIPrefixScale = 1.0

        self._updateLabel()

    def tickStrings(self, values, scale, spacing):
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)

        places = max(0, np.ceil(-np.log10(spacing * scale)))
        strings = []
        for v in values:
            vs = v * scale
            if abs(vs) < 0.001 or abs(vs) >= 10000:
                vstr = f"{vs:g}"
            else:
                vstr = f"{vs:.{places:.0f}f}"
            strings.append(vstr.replace("-", "−"))
        return strings

    def labelString(self) -> str:
        if self.labelUnits == "":
            if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
                units = ""
            else:
                # units = re.sub(
                #     r"1E\+?(\-?)0?(\d?\d)",
                #     r"10<sup>\1\2</sup>",
                #     f"(×{1.0 / self.autoSIPrefixScale:.3G})",
                # )#.replace("-", "−")
                search = re.search(
                    r"1E\+?(\-?)0?(\d?\d)", f"{1.0 / self.autoSIPrefixScale:.3G}"
                )
                if search is not None:
                    units = "".join(search.groups())

                    for k, v in zip(
                        ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"),
                        ("⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹", "⁻"),
                        strict=True,
                    ):
                        units = units.replace(k, v)
                    units = f"10{units}"
                else:
                    units = f"{1.0 / self.autoSIPrefixScale:.3G}"
                units = f"(×{units})"

        else:
            units = f"({self.labelUnitPrefix}{self.labelUnits})"

        s = units if self.labelText == "" else f"{self.labelText} {units}"

        style = ";".join([f"{k}: {v}" for k, v in self.labelStyle.items()])

        return f"<span style='{style}'>{s}</span>"

    def setLabel(self, text=None, units=None, unitPrefix=None, **args) -> None:
        # `None` input is kept for backward compatibility!
        self.labelText = text or ""
        self.labelUnits = units or ""
        self.labelUnitPrefix = unitPrefix or ""
        if len(args) > 0:
            self.labelStyle: dict = args
        # Account empty string and `None` for units and text
        visible = bool(text or units)
        if text == units == "":
            visible = True
        self.showLabel(visible)
        self._updateLabel()


class FittingParameterWidget(QtWidgets.QWidget):
    sigParamChanged = QtCore.Signal()

    def __init__(
        self,
        name: str,
        spin_kw: dict | None = None,
        checkable: bool = True,
        fixed: bool = False,
        label: str | None = None,
        show_label: bool = True,
    ) -> None:
        super().__init__()
        if spin_kw is None:
            spin_kw = {}
        layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        self.param_name = name
        self._prefix = ""
        if label is None:
            label = self.param_name
        self.label = QtWidgets.QLabel(label)
        spin_kw.setdefault("keyboardTracking", False)
        # spin_min_width = spin_kw.pop("minimumWidth", 80)
        self.spin_value = BetterSpinBox(**spin_kw)
        self.spin_lb = BetterSpinBox(
            value=-np.inf,
            minimumWidth=60,
            toolTip="Lower Bound",
            keyboardTracking=False,
        )
        self.spin_ub = BetterSpinBox(
            value=np.inf, minimumWidth=60, toolTip="Upper Bound", keyboardTracking=False
        )
        self.spin_lb.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.spin_ub.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.check = QtWidgets.QCheckBox()
        self.check.setToolTip("Fix parameter")

        if show_label:
            layout.addWidget(self.label)
        layout.addWidget(self.spin_value)
        layout.addWidget(self.spin_lb)
        layout.addWidget(self.spin_ub)
        layout.addWidget(self.check)

        for spin in (self.spin_value, self.spin_lb, self.spin_ub):
            spin.valueChanged.connect(lambda: self.sigParamChanged.emit())
        self.spin_lb.valueChanged.connect(self._refresh_bounds)
        self.spin_ub.valueChanged.connect(self._refresh_bounds)
        self.check.stateChanged.connect(self.setFixed)

        self.setCheckable(checkable)
        self.setFixed(fixed)

    def _refresh_bounds(self) -> None:
        self.spin_value.setRange(self.spin_lb.value(), self.spin_ub.value())

    def setValue(self, value) -> None:
        self.spin_value.setValue(value)

    def checkable(self):
        return self._checkable

    def setCheckable(self, value: bool) -> None:
        self._checkable = value
        self.check.setVisible(value)

    def fixed(self):
        if self.checkable():
            return self.check.isChecked()
        return False

    def setFixed(self, value: bool | QtCore.Qt.CheckState) -> None:
        if isinstance(value, QtCore.Qt.CheckState):
            if value == QtCore.Qt.CheckState.Unchecked:
                value = False
            elif value == QtCore.Qt.CheckState.Checked:
                value = True
        else:
            self.check.setChecked(value)
        if value:
            self.spin_lb.setDisabled(True)
            self.spin_ub.setDisabled(True)
        else:
            self.spin_lb.setEnabled(True)
            self.spin_ub.setEnabled(True)

    def value(self):
        return self.spin_value.value()

    def prefix(self):
        return self._prefix

    def set_prefix(self, prefix) -> None:
        self._prefix = prefix

    def minimum(self):
        return self.spin_lb.value()

    def maximum(self):
        return self.spin_ub.value()

    @property
    def param_dict(self):
        param_info = {"value": self.value()}
        if self.checkable():
            param_info["vary"] = not self.fixed()
        if np.isfinite(self.minimum()):
            param_info["min"] = float(self.minimum())
        if np.isfinite(self.maximum()):
            param_info["max"] = float(self.maximum())
        return {self.prefix() + self.param_name: param_info}


class xImageItem(erlab.interactive.colors.BetterImageItem):
    """:class:`pyqtgraph.ImageItem` with additional functionality.

    This class provides :class:`xarray.DataArray` support and auto limits based on
    histogram analysis.

    Parameters
    ----------
    image
        Image data.
    **kwargs
        Additional arguments to :class:`pyqtgraph.ImageItem`.

    Signals
    -------
    sigToleranceChanged()

    """

    sigToleranceChanged = QtCore.Signal(float, float)  #: :meta private:

    def __init__(self, image: npt.NDArray | None = None, **kwargs) -> None:
        super().__init__(image, **kwargs)
        self.cut_tolerance = [30, 30]
        self.data_array: None | xr.DataArray = None

    def set_cut_tolerance(self, cut_tolerance) -> None:
        try:
            self.cut_tolerance = list(cut_tolerance.__iter__)
        except AttributeError:
            self.cut_tolerance = [cut_tolerance] * 2
        self.setImage(levels=self.data_cut_levels())

    def data_cut_levels(self, data=None):
        """Return appropriate levels estimated from the data."""
        if data is None:
            data = self.image
            if data is None:
                return (np.nan, np.nan)
        data = np.nan_to_num(data)
        q3, q1, pu, pl = np.percentile(
            data, [75, 25, 100 - self.cut_tolerance[0], self.cut_tolerance[1]]
        )
        ql, qu = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        self.sigToleranceChanged.emit(
            100 * (data > qu).mean(), 100 * (data < ql).mean()
        )
        mn, mx = max(min(pl, ql), data.min()), min(max(pu, qu), data.max())
        return (mn, mx)

    def setImage(self, image=None, autoLevels=None, cut_to_data=False, **kargs) -> None:
        if cut_to_data:
            kargs["levels"] = self.data_cut_levels(data=image)
        super().setImage(image=image, autoLevels=autoLevels, **kargs)
        self.data_array = None

    def setDataArray(
        self, data: xr.DataArray, update_labels: bool = True, **kargs
    ) -> None:
        rect = array_rect(data)
        if self.axisOrder == "row-major":
            img = np.ascontiguousarray(data.values)
        else:
            img = np.asfortranarray(data.values.T)

        if update_labels:
            pi = self.getPlotItem()
            if pi is not None:
                pi.setLabel("left", data.dims[0])
                pi.setLabel("bottom", data.dims[1])

        self.setImage(img, rect=rect, **kargs)
        self.data_array = data

    def getMenu(self):
        if self.menu is None:
            if not self.removable:
                self.menu = QtWidgets.QMenu()
            else:
                super().getMenu()
            if self.menu is not None:
                itoolAct = QtGui.QAction("Open in ImageTool", self.menu)
                itoolAct.triggered.connect(self.open_itool)
                self.menu.addAction(itoolAct)
                self.menu.itoolAct = itoolAct
        return self.menu

    def getPlotItem(self) -> pg.PlotItem | None:
        p: typing.Self | None = self
        while True:
            try:
                if p is not None:
                    p = p.parentItem()
            except RuntimeError:
                return None
            if p is None:
                return None
            if isinstance(p, pg.PlotItem):
                return p

    @QtCore.Slot()
    def open_itool(self) -> None:
        if self.data_array is None:
            if self.image is None:
                return
            da = xr.DataArray(np.asarray(self.image)).T
        else:
            da = self.data_array.T

        tool = erlab.interactive.itool(da, execute=False)
        if isinstance(tool, QtWidgets.QWidget):
            tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self._itool = tool
            self._itool.show()


class ParameterGroup(QtWidgets.QGroupBox):
    """Easy creation of groupboxes with multiple varying parameters.

    Can be used in many different interactive tools for dynamic data analysis.

    Parameters
    ----------
    ncols
        Number of columns in the layout.
    groupbox_kw
        Keyword arguments passed onto :class:`PySide6.QtWidgets.QGroupBox`.
    params
        See Examples.


    Signals
    -------
    sigParameterChanged(dict)

    Examples
    --------
    >>> ParameterGroup(
        **{
            "a": QtWidgets.QDoubleSpinBox(range=(0, 1), singleStep=0.01, value=0.2),
            "b": dict(qwtype="dblspin", range=(0, 2), singleStep=0.04),
            "c": QtWidgets.QSlider(range=(0, 10000))
        }
    )

    """

    VALID_QWTYPE: Mapping[str, type[QtWidgets.QWidget]] = types.MappingProxyType(
        {
            "spin": QtWidgets.QSpinBox,
            "dblspin": QtWidgets.QDoubleSpinBox,
            "btspin": BetterSpinBox,
            "slider": QtWidgets.QSlider,
            "chkbox": QtWidgets.QCheckBox,
            "pushbtn": QtWidgets.QPushButton,
            "chkpushbtn": QtWidgets.QPushButton,
            "combobox": QtWidgets.QComboBox,
            "fitparam": FittingParameterWidget,
        }
    )  # : Dictionary of valid widgets that can be added.

    sigParameterChanged: QtCore.SignalInstance = QtCore.Signal(dict)  #: :meta private:

    def __init__(
        self,
        widgets: dict[str, dict] | None = None,
        ncols: int = 1,
        groupbox_kw: dict | None = None,
        **widgets_kwargs,
    ) -> None:
        if groupbox_kw is None:
            groupbox_kw = {}
        super().__init__(**groupbox_kw)
        layout = QtWidgets.QGridLayout(self)
        self.setLayout(layout)

        self.labels = []
        self.untracked = []
        self.widgets: dict[str, QtWidgets.QWidget] = {}

        kwargs = widgets if widgets is not None else widgets_kwargs

        j = 0
        for i, (k, v) in enumerate(kwargs.items()):
            if isinstance(v, dict):
                showlabel = v.pop("showlabel", k)
                ind_eff = v.pop("colspan", 1)
                if ind_eff == "ncols":
                    ind_eff = ncols
                if v.pop("notrack", False):
                    self.untracked.append(k)
                self.widgets[k] = self.getParameterWidget(**v)
            elif isinstance(v, QtWidgets.QWidget):
                showlabel = k
                ind_eff = 1
                self.widgets[k] = v
            else:
                raise TypeError(
                    "Each value must be a QtWidgets.QWidget instance"
                    "or a dictionary of keyword arguments to getParameterWidget."
                )

            self.labels.append(QtWidgets.QLabel(str(showlabel)))
            self.labels[i].setBuddy(self.widgets[k])
            if showlabel:
                layout.addWidget(self.labels[i], j // ncols, 2 * (j % ncols))
                layout.addWidget(
                    self.widgets[k], j // ncols, 2 * (j % ncols) + 1, 1, 2 * ind_eff - 1
                )
            else:
                layout.addWidget(
                    self.widgets[k], j // ncols, 2 * (j % ncols), 1, 2 * ind_eff
                )
            j += ind_eff

        self.global_connect()

    def layout(self) -> QtWidgets.QGridLayout:
        return typing.cast("QtWidgets.QGridLayout", super().layout())

    @staticmethod
    def getParameterWidget(
        qwtype: (
            typing.Literal[
                "spin",
                "dblspin",
                "btspin",
                "slider",
                "chkbox",
                "pushbtn",
                "chkpushbtn",
                "combobox",
                "fitparam",
            ]
            | None
        ) = None,
        **kwargs,
    ):
        """
        Initialize the :class:`PySide6.QtWidgets.QWidget` corresponding to ``qwtype``.

        Parameters
        ----------
        qwtype
            Type of the widget, must a key of :obj:`ParameterGroup.VALID_QWTYPE`.

        """
        if qwtype is None:
            widget = kwargs.pop("widget")
            if not isinstance(widget, QtWidgets.QWidget):
                raise ValueError("widget is not a valid QWidget")
            return widget
        if qwtype not in ParameterGroup.VALID_QWTYPE:
            raise ValueError(
                f"qwtype must be one of {list(ParameterGroup.VALID_QWTYPE.keys())}"
            )

        widget_class = ParameterGroup.VALID_QWTYPE[qwtype]

        if qwtype == "combobox":
            items = kwargs.pop("items", None)
            currtxt = kwargs.pop("currentText", None)
            curridx = kwargs.pop("currentIndex", None)

        elif qwtype.endswith("pushbtn"):
            pressed = kwargs.pop("pressed", None)
            released = kwargs.pop("released", None)
            if qwtype == "chkpushbtn":
                kwargs["checkable"] = True
                toggled = kwargs.pop("toggled", None)
            else:
                clicked = kwargs.pop("clicked", None)
        newrange = kwargs.pop("range", None)

        valueChanged = kwargs.pop("valueChanged", None)
        textChanged = kwargs.pop("textChanged", None)

        policy = kwargs.pop("policy", None)

        alignment = kwargs.pop("alignment", None)

        if qwtype == "fitparam":
            show_param_label = kwargs.pop("show_param_label", False)
            kwargs["show_label"] = show_param_label

        fixedWidth = kwargs.pop("fixedWidth", None)
        fixedHeight = kwargs.pop("fixedHeight", None)

        value = kwargs.pop("value", None)

        widget = widget_class(**kwargs)

        if qwtype == "combobox":
            widget.addItems(items)
            if currtxt is not None:
                widget.setCurrentText(currtxt)
            if curridx is not None:
                widget.setCurrentIndex(curridx)
        elif qwtype.endswith("pushbtn"):
            if pressed is not None:
                widget.pressed.connect(pressed)
            if released is not None:
                widget.released.connect(released)
            if qwtype == "chkpushbtn":
                if toggled is not None:
                    widget.toggled.connect(toggled)
            elif clicked is not None:
                widget.clicked.connect(clicked)

        if newrange is not None:
            widget.setRange(*newrange)

        if valueChanged is not None:
            widget.valueChanged.connect(valueChanged)
        if textChanged is not None:
            widget.textChanged.connect(textChanged)

        if fixedWidth is not None:
            widget.setFixedWidth(fixedWidth)
        if fixedHeight is not None:
            widget.setFixedHeight(fixedHeight)
        if policy is not None:
            widget.setSizePolicy(*policy)
        if alignment is not None:
            widget.setAlignment(alignment)

        if value is not None:
            widget.setValue(value)

        return widget

    def set_values(self, **kwargs) -> None:
        for k, v in kwargs.items():
            widget = self.widgets[k]
            widget.blockSignals(True)
            if hasattr(widget, "setValue"):
                widget.setValue(v)
            widget.blockSignals(False)

        self.sigParameterChanged.emit(kwargs)

    def widget_value(self, widget: str | QtWidgets.QWidget):
        if isinstance(widget, str):
            widget = self.widgets[widget]
        if isinstance(
            widget,
            QtWidgets.QSpinBox
            | QtWidgets.QDoubleSpinBox
            | BetterSpinBox
            | FittingParameterWidget,
        ):
            return widget.value()
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            return widget.text()
        if isinstance(widget, QtWidgets.QAbstractSlider):
            return widget.value()
        if isinstance(widget, QtWidgets.QCheckBox):
            if widget.isTristate():
                return widget.checkState()
            return widget.isChecked()
        if isinstance(widget, QtWidgets.QAbstractButton):
            if widget.isCheckable():
                return widget.isChecked()
            return widget.isDown()
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        return None

    def widget_change_signal(self, widget):
        if isinstance(
            widget, QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox | BetterSpinBox
        ):
            return widget.valueChanged
        if isinstance(widget, FittingParameterWidget):
            return widget.sigParamChanged
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            return widget.editingFinished
        if isinstance(widget, QtWidgets.QAbstractSlider):
            return widget.valueChanged
        if isinstance(widget, QtWidgets.QCheckBox):
            return widget.stateChanged
        if isinstance(widget, QtWidgets.QAbstractButton):
            if widget.isCheckable():
                return widget.clicked
            return widget.toggled
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentTextChanged
        return None

    def global_connect(self) -> None:
        for k, v in self.widgets.items():
            if k not in self.untracked:
                self.widget_change_signal(v).connect(
                    lambda x=None, name=k: self.sigParameterChanged.emit({name: x})
                )

    def widgets_of_type(self, widgetclass):
        if isinstance(widgetclass, str):
            widgetclass = self.VALID_QWTYPE[widgetclass]
        return [w for w in self.widgets.values() if isinstance(w, widgetclass)]

    @property
    def values(self) -> dict[str, float | int | bool]:
        return {k: self.widget_value(v) for k, v in self.widgets.items()}

    # "spin": QtWidgets.QSpinBox,
    # "dblspin": QtWidgets.QDoubleSpinBox,
    # "slider": QtWidgets.QSlider,
    # "chkbox": QtWidgets.QCheckBox,
    # "pushbtn": QtWidgets.QPushButton,
    # "chkpushbtn": QtWidgets.QPushButton,
    # "combobox": QtWidgets.QComboBox,


class ROIControls(ParameterGroup):
    def __init__(self, roi: pg.ROI, spinbox_kw: dict | None = None, **kwargs) -> None:
        if spinbox_kw is None:
            spinbox_kw = {}
        self.roi = roi
        x0, y0, x1, y1 = self.roi_limits
        xm, ym, xM, yM = self.roi.maxBounds.getCoords()

        default_properties = {
            "decimals": 3,
            "singleStep": 0.002,
            "keyboardTracking": False,
        }
        for k, v in default_properties.items():
            spinbox_kw.setdefault(k, v)

        super().__init__(
            x0=dict(
                qwtype="btspin",
                value=x0,
                valueChanged=lambda x: self.modify_roi(x0=x),
                minimum=xm,
                maximum=xM,
                **spinbox_kw,
            ),
            x1=dict(
                qwtype="btspin",
                value=x1,
                valueChanged=lambda x: self.modify_roi(x1=x),
                minimum=xm,
                maximum=xM,
                **spinbox_kw,
            ),
            y0=dict(
                qwtype="btspin",
                value=y0,
                valueChanged=lambda x: self.modify_roi(y0=x),
                minimum=ym,
                maximum=yM,
                **spinbox_kw,
            ),
            y1=dict(
                qwtype="btspin",
                value=y1,
                valueChanged=lambda x: self.modify_roi(y1=x),
                minimum=ym,
                maximum=yM,
                **spinbox_kw,
            ),
            drawbtn={
                "qwtype": "chkpushbtn",
                "toggled": self.draw_mode,
                "showlabel": False,
                "text": "Draw",
            },
            **kwargs,
        )
        self.draw_button = typing.cast("QtWidgets.QPushButton", self.widgets["drawbtn"])
        self.roi_spin = [self.widgets[i] for i in ["x0", "y0", "x1", "y1"]]
        self.roi.sigRegionChanged.connect(self.update_pos)

    @property
    def roi_limits(self):
        x0, y0 = self.roi.state["pos"]
        w, h = self.roi.state["size"]
        x1, y1 = x0 + w, y0 + h
        return x0, y0, x1, y1

    @typing.no_type_check
    def update_pos(self) -> None:
        self.widgets["x0"].setMaximum(self.widgets["x1"].value())
        self.widgets["y0"].setMaximum(self.widgets["y1"].value())
        self.widgets["x1"].setMinimum(self.widgets["x0"].value())
        self.widgets["y1"].setMinimum(self.widgets["y0"].value())
        for pos, spin in zip(self.roi_limits, self.roi_spin, strict=True):
            spin.blockSignals(True)
            spin.setValue(pos)
            spin.blockSignals(False)

    def modify_roi(self, x0=None, y0=None, x1=None, y1=None, update=True) -> None:
        lim_new = (x0, y0, x1, y1)
        lim_old = self.roi_limits
        x0, y0, x1, y1 = (
            (f if f is not None else i) for i, f in zip(lim_old, lim_new, strict=True)
        )
        xm, ym, xM, yM = self.roi.maxBounds.getCoords()
        x0, y0, x1, y1 = max(x0, xm), max(y0, ym), min(x1, xM), min(y1, yM)
        self.roi.setPos((x0, y0), update=False)
        self.roi.setSize((x1 - x0, y1 - y0), update=update)

    def draw_mode(self, toggle) -> None:
        vb = self.roi.parentItem().getViewBox()

        if not toggle:
            vb.mouseDragEvent = self._drag_evt_old
            vb.setMouseMode(self._state_old)
            vb.rbScaleBox.setPen(pg.mkPen((255, 255, 100), width=1))
            vb.rbScaleBox.setBrush(pg.mkBrush(255, 255, 0, 100))

            vb.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
            return

        vb.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self._state_old: int = vb.state["mouseMode"]
        self._drag_evt_old: MouseDragEvent = vb.mouseDragEvent

        vb.setMouseMode(pg.ViewBox.RectMode)
        vb.rbScaleBox.setPen(pg.mkPen((255, 255, 255), width=1))
        vb.rbScaleBox.setBrush(pg.mkBrush(255, 255, 255, 100))

        def mouseDragEventCustom(ev, axis=None) -> None:
            ev.accept()
            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos

            mouseEnabled = np.array(vb.state["mouseEnabled"], dtype=np.float64)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1 - axis] = 0.0

            if ev.button() & QtCore.Qt.MouseButton.LeftButton:
                if vb.state["mouseMode"] == pg.ViewBox.RectMode and axis is None:
                    if ev.isFinish():
                        vb.rbScaleBox.hide()
                        ax = QtCore.QRectF(
                            pg.Point(ev.buttonDownPos(ev.button())), pg.Point(pos)
                        )
                        ax = vb.childGroup.mapRectFromParent(ax)
                        self.modify_roi(*ax.getCoords())
                    else:
                        vb.updateScaleBox(ev.buttonDownPos(), ev.pos())
            elif ev.button() & QtCore.Qt.MouseButton.MiddleButton:
                tr = vb.childGroup.transform()
                tr = pg.invertQTransform(tr)
                tr = tr.map(dif * mask) - tr.map(pg.Point(0, 0))

                x = tr.x() if mask[0] == 1 else None
                y = tr.y() if mask[1] == 1 else None

                vb._resetTarget()
                if x is not None or y is not None:
                    vb.translateBy(x=x, y=y)
                vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"])
            elif ev.button() & QtCore.Qt.MouseButton.RightButton:
                if vb.state["aspectLocked"] is not False:
                    mask[0] = 0

                dif = ev.screenPos() - ev.lastScreenPos()
                dif = np.array([dif.x(), dif.y()])
                dif[0] *= -1
                s = ((mask * 0.02) + 1) ** dif

                tr = pg.invertQTransform(vb.childGroup.transform())
                x = s[0] if mouseEnabled[0] == 1 else None
                y = s[1] if mouseEnabled[1] == 1 else None

                center = pg.Point(
                    tr.map(ev.buttonDownPos(QtCore.Qt.MouseButton.RightButton))
                )
                vb._resetTarget()
                vb.scaleBy(x=x, y=y, center=center)
                vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"])

        vb.mouseDragEvent = mouseDragEventCustom  # set to modified mouseDragEvent


class AnalysisWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        data,
        title=None,
        layout="horizontal",
        data_is_input=True,
        analysisWidget=None,
        *args,
        **kwargs,
    ) -> None:
        self.qapp = typing.cast(
            "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
        )
        if not self.qapp:
            self.qapp = QtWidgets.QApplication(sys.argv)
        self.qapp.setStyle("Fusion")

        super().__init__()

        self.data = parse_data(data)
        if title is None:
            title = self.data.name
        self.setWindowTitle(title)

        self._main = QtWidgets.QWidget(self)
        self.setCentralWidget(self._main)

        self.controlgroup = QtWidgets.QWidget()
        if layout == "vertical":
            layout = QtWidgets.QVBoxLayout(self._main)
            self.controls: QtWidgets.QBoxLayout = QtWidgets.QHBoxLayout(
                self.controlgroup
            )
        elif layout == "horizontal":
            layout = QtWidgets.QHBoxLayout(self._main)
            self.controls = QtWidgets.QVBoxLayout(self.controlgroup)
        else:
            raise ValueError("Layout must be 'vertical' or 'horizontal'.")

        if analysisWidget is None:
            self.aw = AnalysisWidgetBase(*args, **kwargs)
        elif isinstance(analysisWidget, type):
            self.aw = analysisWidget(*args, **kwargs)
        else:
            self.aw = analysisWidget

        for n in [
            "set_input",
            "add_roi",
            "axes",
            "hists",
            "images",
            # "set_pre_function",
            # "set_pre_function_args",
            # "set_main_function",
            # "set_main_function_args",
            # "refresh_output",
            # "refresh_all",
        ]:
            setattr(self, n, getattr(self.aw, n))
        layout.addWidget(self.aw)
        layout.addWidget(self.controlgroup)

        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        layout.setSpacing(0)
        self.controlgroup.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        if data_is_input:
            self.aw.set_input(data)

    def __post_init__(self, execute=None):
        with setup_qapp(execute) as execute:
            self.show()
            self.activateWindow()
            self.raise_()

    def addParameterGroup(self, *args, **kwargs):
        group = ParameterGroup(*args, **kwargs)
        self.controls.addWidget(group)
        return group

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        cb = typing.cast(
            "QtWidgets.QApplication", QtWidgets.QApplication.instance()
        ).clipboard()
        if event is not None and cb is not None and cb.text(cb.Mode.Clipboard) != "":
            pyperclip.copy(cb.text(cb.Mode.Clipboard))
        return super().closeEvent(event)


class AnalysisWidgetBase(pg.GraphicsLayoutWidget):
    """AnalysisWidgetBase.

    Parameters
    ----------
    orientation
        Sets the orientation of the plots, by default "vertical"
    num_ax
        Sets the number of axes.
    link
        Link axes, by default "both"
    cut_to_data
        Whether to remove outliers by adjusting color levels, by default "none"

    """

    def __init__(
        self,
        orientation: typing.Literal["vertical", "horizontal"] = "vertical",
        num_ax: int = 2,
        link: typing.Literal["x", "y", "both", "none"] = "both",
        cut_to_data: typing.Literal["in", "out", "both", "none"] = "none",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if orientation == "horizontal":
            self.is_vertical = False
        elif orientation == "vertical":
            self.is_vertical = True
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'.")
        self.cut_to_data = cut_to_data

        self.input: None | xr.DataArray | npt.NDArray = None

        self.initialize_layout(num_ax)

        for i in range(1, num_ax):
            if link in ("x", "both"):
                self.axes[i].setXLink(self.axes[0])
            if link in ("y", "both"):
                self.axes[i].setYLink(self.axes[0])

    def initialize_layout(self, nax: int) -> None:
        self.hists: pg.HistogramLUTItem = [pg.HistogramLUTItem() for _ in range(nax)]
        self.axes: list[pg.PlotItem] = [pg.PlotItem() for _ in range(nax)]
        self.images: list[xImageItem] = [
            xImageItem(axisOrder="row-major") for _ in range(nax)
        ]
        cmap = erlab.interactive.colors.pg_colormap_powernorm("terrain", 1.0, N=6)
        for i in range(nax):
            self.addItem(self.axes[i], *self.get_axis_pos(i))
            self.addItem(self.hists[i], *self.get_hist_pos(i))
            self.axes[i].addItem(self.images[i])
            self.hists[i].setImageItem(self.images[i])
            self.hists[i].gradient.setColorMap(cmap)
        self.roi = [None for _ in range(nax)]

    def get_axis_pos(self, ax):
        if self.is_vertical:
            return ax, 0, 1, 1
        return 0, 2 * ax, 1, 1

    def get_hist_pos(self, ax):
        if self.is_vertical:
            return ax, 1, 1, 1
        return 0, 2 * ax + 1, 1, 1

    def setStretchFactors(self, factors) -> None:
        for i, f in enumerate(factors):
            self.setStretchFactor(i, f)

    def setStretchFactor(self, i, factor) -> None:
        if self.is_vertical:
            self.ci.layout.setRowStretchFactor(i, factor)
        else:
            self.ci.layout.setColumnStretchFactor(i, factor)

    def set_input(self, data=None) -> None:
        if data is not None:
            self.input = parse_data(data)
            self.images[0].setDataArray(
                self.input, cut_to_data=self.cut_to_data in ("in", "both")
            )

    def add_roi(self, i):
        self.roi[i] = pg.ROI(
            [-0.1, -0.5],
            [0.3, 0.5],
            parent=self.images[i],
            rotatable=False,
            resizable=True,
            maxBounds=self.axes[i].getViewBox().itemBoundingRect(self.images[i]),
        )
        self.roi[i].addScaleHandle([0.5, 1], [0.5, 0])
        self.roi[i].addScaleHandle([0.5, 0], [0.5, 1])
        self.roi[i].addScaleHandle([1, 0.5], [0, 0.5])
        self.roi[i].addScaleHandle([0, 0.5], [1, 0.5])
        self.axes[i].addItem(self.roi[i])
        self.roi[i].setZValue(10)
        return self.roi[i]


class ComparisonWidget(AnalysisWidgetBase):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["num_ax"] = 2
        super().__init__(*args, **kwargs)
        self.prefunc = lambda x: x
        self.mainfunc = lambda x: x
        self.prefunc_only_values = False
        self.mainfunc_only_values = False
        self.prefunc_kwargs: dict[str, typing.Any] = {}
        self.mainfunc_kwargs: dict[str, typing.Any] = {}

    def call_prefunc(self, x):
        xval = np.asarray(x) if self.prefunc_only_values else x
        return self.prefunc(xval, **self.prefunc_kwargs)

    def set_input(self, data=None) -> None:
        if data is not None:
            self.input_ = parse_data(data)

        self.input = self.call_prefunc(self.input_)
        if self.prefunc_only_values:
            if not isinstance(self.input, np.ndarray):
                raise TypeError(
                    "Pre-function must return a numpy array when `only_values` is True."
                )
            self.images[0].setImage(
                np.ascontiguousarray(self.input),
                rect=array_rect(self.input_),
                cut_to_data=self.cut_to_data in ("in", "both"),
            )
        else:
            if not isinstance(self.input, xr.DataArray):
                raise TypeError(
                    "Pre-function must return a DataArray when `only_values` is False."
                )
            self.images[0].setDataArray(
                self.input, cut_to_data=self.cut_to_data in ("in", "both")
            )

    def set_pre_function(self, func, only_values=False, **kwargs) -> None:
        self.prefunc_only_values = only_values
        self.prefunc = func
        self.set_pre_function_args(**kwargs)
        self.set_input()

    def set_main_function(self, func, only_values=False, **kwargs) -> None:
        self.mainfunc_only_values = only_values
        self.mainfunc = func
        self.set_main_function_args(**kwargs)
        self.refresh_output()

    def set_pre_function_args(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.prefunc_kwargs[k] = v
        self.refresh_output()

    def set_main_function_args(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.mainfunc_kwargs[k] = v
        self.refresh_output()

    def refresh_output(self) -> None:
        if self.mainfunc_only_values:
            self.output = self.mainfunc(np.asarray(self.input_), **self.mainfunc_kwargs)
            self.images[1].setImage(
                np.ascontiguousarray(self.output),
                rect=array_rect(self.input),
                cut_to_data=self.cut_to_data in ("out", "both"),
            )
        else:
            self.output = self.mainfunc(self.input_, **self.mainfunc_kwargs)
            self.images[1].setDataArray(
                self.output, cut_to_data=self.cut_to_data in ("out", "both")
            )

    def refresh_all(self) -> None:
        self.set_input()
        self.refresh_output()


class DictMenuBar(QtWidgets.QMenuBar):
    def __init__(self, parent: QtWidgets.QWidget | None = None, **kwargs) -> None:
        super().__init__(parent)

        self.menu_dict: dict[str, QtWidgets.QMenu] = {}
        self.action_dict: dict[str, QtWidgets.QAction] = {}

        self.add_items(**kwargs)

    def __getattribute__(self, __name: str, /) -> typing.Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            try:
                out: typing.Any = self.menu_dict[__name]
            except KeyError:
                out = self.action_dict[__name]
            warnings.warn(
                f"Menu or Action '{__name}' called as an attribute",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            return out

    def add_items(self, **kwargs) -> None:
        self.parse_menu(self, **kwargs)

    def parse_menu(
        self, parent: QtWidgets.QMenuBar | QtWidgets.QMenu, **kwargs
    ) -> None:
        for name, opts in kwargs.items():
            menu = opts.pop("menu", None)
            actions = opts.pop("actions")

            if menu is None:
                title = opts.pop("title", None)
                icon = opts.pop("icon", None)
                if title is None:
                    title = name
                if icon is None:
                    menu = parent.addMenu(title)
                else:
                    menu = parent.addMenu(icon, title)
            else:
                menu = parent.addMenu(menu)

            self.menu_dict[name] = menu

            for actname, actopts in actions.items():
                if isinstance(actopts, QtWidgets.QAction):
                    act = actopts
                    sep_before, sep_after = False, False
                else:
                    if "actions" in actopts:
                        self.parse_menu(menu, **{actname: actopts})
                        continue
                    sep_before = actopts.pop("sep_before", False)
                    sep_after = actopts.pop("sep_after", False)
                    if "text" not in actopts:
                        actopts["text"] = actname
                    act = self.parse_action(actopts)
                if sep_before:
                    menu.addSeparator()
                menu.addAction(act)
                if (
                    act.text() is not None
                ):  # check whether it's a separator without text
                    self.action_dict[actname] = act
                if sep_after:
                    menu.addSeparator()

    def parse_action(self, actopts: dict):
        text = actopts.pop("text", None)
        tooltip = actopts.pop("tooltip", None)
        checkable = actopts.pop("checkable", None)
        shortcut = actopts.pop("shortcut", None)
        triggered = actopts.pop("triggered", None)
        toggled = actopts.pop("toggled", None)
        changed = actopts.pop("changed", None)
        separator = actopts.pop("separator", False)

        if shortcut is not None:
            shortcut = QtGui.QKeySequence(shortcut)

        parent = self.parent()
        if parent is None:
            parent = self

        action = QtGui.QAction(parent)
        if separator:
            action.setSeparator(separator)
            return action

        if text is not None:
            action.setText(text)
        if checkable is not None:
            action.setCheckable(checkable)
        if tooltip is not None:
            action.setToolTip(tooltip)
        if shortcut is not None:
            action.setShortcut(shortcut)
        if triggered is not None:
            action.triggered.connect(triggered)
        if toggled is not None:
            action.toggled.connect(toggled)
        if changed is not None:
            action.changed.connect(changed)
        return action


class ExclusiveComboGroup(QtCore.QObject):
    """A group of mutually exclusive comboboxes.

    Adapted from `this StackOverflow answer <https://stackoverflow.com/a/66093311>`_.

    This group stores only weak references to the comboboxes, so it is necessary to keep
    a reference to the comboboxes elsewhere in the code.
    """

    def __init__(self, parent=None, exclude_first: bool = False) -> None:
        super().__init__(parent)
        self._combos: list[weakref.ref[QtWidgets.QComboBox]] = []
        self._role = QtCore.Qt.ItemDataRole.UserRole + 500
        self._exclude_first = exclude_first

    def addCombo(self, combo: QtWidgets.QComboBox) -> None:
        combo.activated.connect(lambda: self._handle_activated(combo))
        self._combos.append(weakref.ref(combo))

    def _handle_activated(self, target: QtWidgets.QComboBox) -> None:
        index = target.currentIndex()
        groupid = id(target)
        for cb in self._combos:
            combo = cb()
            if combo is None or combo is target:
                continue
            previous = combo.findData(groupid, self._role)

            view: QtWidgets.QAbstractItemView | None = combo.view()

            if isinstance(view, QtWidgets.QListView):
                if previous >= 0:
                    view.setRowHidden(previous, False)
                    combo.setItemData(previous, None, self._role)
                if (index > 0) or (index == 0 and not self._exclude_first):
                    combo.setItemData(index, groupid, self._role)
                    view.setRowHidden(index, True)


class IconButton(QtWidgets.QPushButton):
    """Convenience class for creating a QPushButton with a qtawesome icon.

    This button adapts to dark mode changes by resetting the qtawesome cache when a
    color palette change is detected.

    Parameters
    ----------
    on : str, optional
        The icon to display when the button is in the "on" state. If `off` is not
        provided, this will be the only icon displayed.
    off : str, optional
        The icon to display when the button is in the "off" state. If provided, the
        button will be checkable, and the icon will change when the button is toggled.
    **kwargs
        Additional keyword arguments passed to the QPushButton constructor.

    """

    def __init__(self, on: str | None = None, off: str | None = None, **kwargs) -> None:
        self.icon_key_on = None
        self.icon_key_off = None

        if on is not None:
            self.icon_key_on = on
            kwargs["icon"] = self.get_icon(self.icon_key_on)

        if off is not None:
            if on is None and kwargs["icon"] is None:
                raise ValueError("Icon for `on` state was not supplied.")
            self.icon_key_off = off
            kwargs.setdefault("checkable", True)

        super().__init__(**kwargs)
        self.toggled.connect(self.refresh_icons)

    def setChecked(self, value: bool) -> None:
        super().setChecked(value)
        self.refresh_icons()

    def get_icon(self, icon: str) -> QtGui.QIcon:
        return qtawesome.icon(icon)

    def refresh_icons(self) -> None:
        if self.icon_key_off is not None and self.isChecked():
            self.setIcon(self.get_icon(self.icon_key_off))
            return
        if self.icon_key_on is not None:
            self.setIcon(self.get_icon(self.icon_key_on))

    def changeEvent(self, evt: QtCore.QEvent | None) -> None:  # handles dark mode
        if evt is not None and evt.type() == QtCore.QEvent.Type.PaletteChange:
            qtawesome.reset_cache()
            self.refresh_icons()
        super().changeEvent(evt)


class IconActionButton(IconButton):
    """IconButton that supports linking to a QAction.

    Parameters
    ----------
    action : QtGui.QAction
        The QAction to be associated with this button.
    on : str, optional
        The icon to display when the button is in the "on" state.
    off : str, optional
        The icon to display when the button is in the "off" state. If `action` is not
        toggleable, this icon will never be displayed.
    text_from_action : bool, optional
        If True, the button's text will be set from the QAction's text. Otherwise, the
        text will be left empty.
    **kwargs
        Additional keyword arguments passed to the IconButton constructor.

    """

    def __init__(
        self,
        action: QtGui.QAction,
        on: str | None = None,
        off: str | None = None,
        text_from_action: bool = False,
        **kwargs,
    ):
        super().__init__(on=on, off=off, **kwargs)

        self._action: QtGui.QAction | None = None
        self.text_from_action = text_from_action
        self.setAction(action)

    def setAction(self, action: QtGui.QAction) -> None:
        if self._action:
            self._action.changed.disconnect(self._update_from_action)
            self.clicked.disconnect(self._action.trigger)

        self._action = action
        if action:
            self._update_from_action()
            action.changed.connect(self._update_from_action)
            self.clicked.connect(action.trigger)

    def _update_from_action(self) -> None:
        if not self._action:
            return

        if self.text_from_action:
            self.setText(self._action.text())
        self.setEnabled(self._action.isEnabled())
        self.setCheckable(self._action.isCheckable())
        self.setChecked(self._action.isChecked())
        self.setToolTip(self._action.toolTip())
        self._action.blockSignals(True)
        self._action.setIcon(self.icon())
        self._action.blockSignals(False)


class RotatableLine(pg.InfiniteLine):
    """:class:`pyqtgraph.InfiniteLine` that rotates under drag.

    The position can be changed by providing a :class:`pyqtgraph.TargetItem` which will
    drag the line along with it.

    Using the constructor function :func:`make_crosshairs` is recommended for creating
    several lines at once.

    Parameters
    ----------
    offset
        Offset angle in degrees.
    target
        Target item to link the position of the line to.
    **kwargs
        Additional keyword arguments to pass to :class:`pyqtgraph.InfiniteLine`.

    Signals
    -------
    sigAngleChanged(float)
        Emitted when the angle of the line is changed.
    """

    sigAngleChanged = QtCore.Signal(float)  #: :meta private:
    _sigAngleChangeStarted = QtCore.Signal(float)  #: :meta private:

    def __init__(
        self, offset: float = 0.0, target: pg.TargetItem | None = None, **kwargs
    ) -> None:
        self.offset: float = offset
        kwargs.setdefault("movable", True)
        super().__init__(**kwargs)

        if target is not None:
            self.set_target(target)

    @property
    def angle_effective(self) -> float:
        """The angle of the line relative to the initial angle."""
        return np.round(self.angle - self.offset - 90.0, 2)

    def link(self, other: typing.Self) -> None:
        """Link with another :class:`RotatableLine`.

        Providing another :class:`RotatableLine` will link the angles of both lines.
        """

        def _on_angle_changed_self(angle) -> None:
            self.blockSignals(True)
            self.setAngle(angle)
            self.blockSignals(False)
            self.sigAngleChanged.emit(angle)

        def _on_angle_changed_other(angle) -> None:
            other.blockSignals(True)
            other.setAngle(angle)
            other.blockSignals(False)
            other.sigAngleChanged.emit(angle)

        self._sigAngleChangeStarted.connect(_on_angle_changed_other)
        other._sigAngleChangeStarted.connect(_on_angle_changed_self)

    def setAngle(self, angle: float) -> None:
        self.viewTransformChanged()
        angle = angle % 180
        super().setAngle(np.round(angle + self.offset, 2))
        self._sigAngleChangeStarted.emit(angle)
        self.sigAngleChanged.emit(angle)

    def set_target(self, target: pg.TargetItem) -> None:
        target.sigPositionChanged.connect(lambda: self._target_moved(target))
        self.sigPositionChanged.connect(lambda: self._sync_target(target))

    def _sync_target(self, target: pg.TargetItem) -> None:
        target.blockSignals(True)
        target.setPos(self.pos())
        target.blockSignals(False)

    def _target_moved(self, target: pg.TargetItem) -> None:
        self.setPos(target.pos())

    def mouseDragEvent(self, ev) -> None:
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()

            if not self.moving:
                return

            lastpos = self.mapToParent(ev.pos())
            dx = lastpos.x() - self.startPosition.x()
            dy = lastpos.y() - self.startPosition.y()
            self.setAngle(np.rad2deg(np.arctan2(dy, dx)) - self.offset)

            self.sigDragged.emit(self)
            if ev.isFinish():
                self.moving = False

    def _computeBoundingRect(self):
        """RotatableLine debugging."""
        _ = self.getViewBox().size()
        return super()._computeBoundingRect()


def make_crosshairs(
    n: typing.Literal[1, 2, 3] = 1,
) -> list[pg.TargetItem | RotatableLine]:
    r"""Create a :class:`pyqtgraph.TargetItem` and associated `RotatableLine`\ s.

    Parameters
    ----------
    n
        Number of lines to create. Must be 1, 2, or 3. If 1, a single line is created.
        If 2, two lines are created at 0 and 90 degrees. If 3, three lines are created
        at 0, 120, and 240 degrees.

    """
    if n == 1:
        angles: tuple[int, ...] = (0,)
    elif n == 2:
        angles = (0, 90)
    else:
        angles = (0, 120, 240)

    target = pg.TargetItem()
    lines = [RotatableLine(a, target, movable=True) for a in angles]

    for ln in lines:
        ln.set_target(target)

    for l0, l1 in itertools.combinations(lines, 2):
        l0.link(l1)

    return [*lines, target]

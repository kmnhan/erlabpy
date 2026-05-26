"""Jupyter console widget for ImageToolManager."""

from __future__ import annotations

__all__ = ["ToolNamespace", "ToolsNamespace", "_ImageToolManagerJupyterConsole"]

import contextlib
import importlib
import operator
import sys
import typing
import weakref
from dataclasses import dataclass

import numpy as np
import qtconsole.inprocess
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from IPython.core.interactiveshell import (
        ExecutionInfo,
        ExecutionResult,
        InteractiveShell,
    )

    from erlab.interactive.imagetool import ImageTool
    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )


_MPL_QT_CURSOR_PATCH_ATTR = "_erlab_original_set_cursor"


def _patch_packaged_macos_matplotlib_qt_cursor() -> None:
    """Disable Matplotlib Qt cursor updates in the packaged macOS manager."""
    if sys.platform != "darwin" or not erlab.utils.misc._IS_PACKAGED:
        return

    backend_qt = importlib.import_module("matplotlib.backends.backend_qt")
    canvas_cls = typing.cast("typing.Any", getattr(backend_qt, "FigureCanvasQT", None))
    if canvas_cls is None or hasattr(canvas_cls, _MPL_QT_CURSOR_PATCH_ATTR):
        return

    original_set_cursor = canvas_cls.set_cursor

    def _set_cursor_noop(self, cursor) -> None:
        return None

    # The standalone macOS manager can crash when Matplotlib QtAgg draw-time
    # cursor updates enter Qt Cocoa's QImage::toCGImage conversion.
    setattr(canvas_cls, _MPL_QT_CURSOR_PATCH_ATTR, original_set_cursor)
    canvas_cls.set_cursor = _set_cursor_noop


def _resolve_console_namespace(
    namespace: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    _patch_packaged_macos_matplotlib_qt_cursor()
    return {
        name: importlib.import_module(module) if isinstance(module, str) else module
        for name, module in namespace.items()
    }


def _tool_data_name(index: int) -> str:
    return f"data_{index}"


def _dedupe_script_inputs(
    inputs: Sequence[erlab.interactive.imagetool.provenance.ScriptInput],
) -> tuple[erlab.interactive.imagetool.provenance.ScriptInput, ...]:
    deduped: list[erlab.interactive.imagetool.provenance.ScriptInput] = []
    seen: set[str] = set()
    for script_input in inputs:
        if script_input.name in seen:
            continue
        seen.add(script_input.name)
        deduped.append(script_input)
    return tuple(deduped)


@dataclass(frozen=True)
class _ConsoleOperand:
    value: typing.Any
    code: str
    script_inputs: tuple[erlab.interactive.imagetool.provenance.ScriptInput, ...] = ()
    copyable: bool = True


def _literal_code(value: typing.Any) -> tuple[str, bool]:
    value = erlab.utils.misc._convert_to_native(value)
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return repr(value), True
    if isinstance(value, slice):
        start, start_copyable = _literal_code(value.start)
        stop, stop_copyable = _literal_code(value.stop)
        step, step_copyable = _literal_code(value.step)
        return (
            f"slice({start}, {stop}, {step})",
            start_copyable and stop_copyable and step_copyable,
        )
    if isinstance(value, tuple):
        parts = [_literal_code(item) for item in value]
        suffix = "," if len(parts) == 1 else ""
        return (
            f"({', '.join(part for part, _copyable in parts)}{suffix})",
            all(copyable for _part, copyable in parts),
        )
    if isinstance(value, list):
        parts = [_literal_code(item) for item in value]
        return (
            f"[{', '.join(part for part, _copyable in parts)}]",
            all(copyable for _part, copyable in parts),
        )
    if isinstance(value, dict):
        dict_parts: list[str] = []
        copyable = True
        for key, item in value.items():
            key_code, key_copyable = _literal_code(key)
            item_code, item_copyable = _literal_code(item)
            dict_parts.append(f"{key_code}: {item_code}")
            copyable = copyable and key_copyable and item_copyable
        return f"{{{', '.join(dict_parts)}}}", copyable
    return repr(value), False


def _merge_operands(
    *operands: _ConsoleOperand,
) -> tuple[tuple[erlab.interactive.imagetool.provenance.ScriptInput, ...], bool]:
    return (
        _dedupe_script_inputs(
            tuple(
                script_input
                for operand in operands
                for script_input in operand.script_inputs
            )
        ),
        all(operand.copyable for operand in operands),
    )


def _derived_operand_code(expression: str) -> str:
    if any(symbol in expression for symbol in (" + ", " - ", " * ", " / ", " // ")):
        return f"({expression})"
    if any(symbol in expression for symbol in (" % ", " ** ", " & ", " | ", " ^ ")):
        return f"({expression})"
    if any(
        symbol in expression
        for symbol in (" < ", " <= ", " > ", " >= ", " == ", " != ")
    ):
        return f"({expression})"
    return expression


def _unwrap_console_value(value: typing.Any) -> typing.Any:
    if isinstance(value, _ConsoleDataHandleMixin):
        return value.data
    if isinstance(value, tuple):
        return tuple(_unwrap_console_value(item) for item in value)
    if isinstance(value, list):
        return [_unwrap_console_value(item) for item in value]
    if isinstance(value, dict):
        return {
            _unwrap_console_value(key): _unwrap_console_value(item)
            for key, item in value.items()
        }
    return value


def _operand_from_value(value: typing.Any) -> _ConsoleOperand:
    if isinstance(value, _ConsoleDataHandleMixin):
        return value._console_operand()
    if isinstance(value, tuple):
        item_operands = tuple(_operand_from_value(item) for item in value)
        suffix = "," if len(item_operands) == 1 else ""
        inputs, copyable = _merge_operands(*item_operands)
        return _ConsoleOperand(
            tuple(operand.value for operand in item_operands),
            f"({', '.join(operand.code for operand in item_operands)}{suffix})",
            inputs,
            copyable,
        )
    if isinstance(value, list):
        item_operands = tuple(_operand_from_value(item) for item in value)
        inputs, copyable = _merge_operands(*item_operands)
        return _ConsoleOperand(
            [operand.value for operand in item_operands],
            f"[{', '.join(operand.code for operand in item_operands)}]",
            inputs,
            copyable,
        )
    if isinstance(value, dict):
        key_operands = tuple(_operand_from_value(key) for key in value)
        value_operands = tuple(_operand_from_value(item) for item in value.values())
        pair_operands = tuple(
            operand
            for pair in zip(key_operands, value_operands, strict=True)
            for operand in pair
        )
        inputs, copyable = _merge_operands(*pair_operands)
        return _ConsoleOperand(
            {
                key_operand.value: value_operand.value
                for key_operand, value_operand in zip(
                    key_operands, value_operands, strict=True
                )
            },
            "{"
            + ", ".join(
                f"{key_operand.code}: {value_operand.code}"
                for key_operand, value_operand in zip(
                    key_operands, value_operands, strict=True
                )
            )
            + "}",
            inputs,
            copyable,
        )
    code, copyable = _literal_code(value)
    return _ConsoleOperand(value, code, copyable=copyable)


def _format_call_code(
    args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]
) -> tuple[
    str,
    tuple[erlab.interactive.imagetool.provenance.ScriptInput, ...],
    tuple[typing.Any, ...],
    dict[str, typing.Any],
    bool,
]:
    arg_operands = tuple(_operand_from_value(arg) for arg in args)
    kwarg_operands = {key: _operand_from_value(value) for key, value in kwargs.items()}
    inputs, copyable = _merge_operands(*arg_operands, *tuple(kwarg_operands.values()))
    parts = [operand.code for operand in arg_operands]
    parts.extend(f"{key}={operand.code}" for key, operand in kwarg_operands.items())
    return (
        ", ".join(parts),
        inputs,
        tuple(operand.value for operand in arg_operands),
        {key: operand.value for key, operand in kwarg_operands.items()},
        copyable,
    )


class _ConsoleDataHandleMixin:
    __array_priority__ = 1000

    @property
    def data(self) -> xr.DataArray:
        raise NotImplementedError

    @property
    def _console_tools(self) -> ToolsNamespace | None:
        raise NotImplementedError

    @property
    def _console_is_source(self) -> bool:
        return False

    def _console_operand(self) -> _ConsoleOperand:
        raise NotImplementedError

    def _console_provenance_spec(
        self, *, active_name: str, label: str
    ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
        raise NotImplementedError

    def _wrap_console_result(
        self,
        value: typing.Any,
        expression: str,
        script_inputs: Sequence[erlab.interactive.imagetool.provenance.ScriptInput],
        *,
        copyable: bool,
    ) -> typing.Any:
        if not isinstance(value, xr.DataArray):
            return value
        return _DerivedDataNamespace(
            self._console_tools,
            value,
            expression,
            _dedupe_script_inputs(script_inputs),
            copyable=copyable,
        )

    def _binary_operation(
        self,
        other: typing.Any,
        symbol: str,
        operation: Callable[[typing.Any, typing.Any], typing.Any],
        *,
        reflected: bool = False,
    ) -> typing.Any:
        left_operand = (
            _operand_from_value(other) if reflected else self._console_operand()
        )
        right_operand = (
            self._console_operand() if reflected else _operand_from_value(other)
        )
        inputs, copyable = _merge_operands(left_operand, right_operand)
        result = operation(left_operand.value, right_operand.value)
        expression = f"{left_operand.code} {symbol} {right_operand.code}"
        return self._wrap_console_result(result, expression, inputs, copyable=copyable)

    def _unary_operation(
        self, symbol: str, operation: Callable[[typing.Any], typing.Any]
    ) -> typing.Any:
        operand = self._console_operand()
        result = operation(operand.value)
        return self._wrap_console_result(
            result,
            f"{symbol}{operand.code}",
            operand.script_inputs,
            copyable=operand.copyable,
        )

    def __add__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "+", operator.add)

    def __radd__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "+", operator.add, reflected=True)

    def __sub__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "-", operator.sub)

    def __rsub__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "-", operator.sub, reflected=True)

    def __mul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "*", operator.mul)

    def __rmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "*", operator.mul, reflected=True)

    def __matmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "@", operator.matmul)

    def __rmatmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "@", operator.matmul, reflected=True)

    def __truediv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "/", operator.truediv)

    def __rtruediv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "/", operator.truediv, reflected=True)

    def __floordiv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "//", operator.floordiv)

    def __rfloordiv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "//", operator.floordiv, reflected=True)

    def __mod__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "%", operator.mod)

    def __rmod__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "%", operator.mod, reflected=True)

    def __pow__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "**", operator.pow)

    def __rpow__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "**", operator.pow, reflected=True)

    def __and__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "&", operator.and_)

    def __rand__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "&", operator.and_, reflected=True)

    def __or__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "|", operator.or_)

    def __ror__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "|", operator.or_, reflected=True)

    def __xor__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "^", operator.xor)

    def __rxor__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "^", operator.xor, reflected=True)

    def __lt__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "<", operator.lt)

    def __le__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "<=", operator.le)

    def __gt__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, ">", operator.gt)

    def __ge__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, ">=", operator.ge)

    def __eq__(self, other: object) -> typing.Any:
        return self._binary_operation(other, "==", operator.eq)

    def __ne__(self, other: object) -> typing.Any:
        return self._binary_operation(other, "!=", operator.ne)

    def __neg__(self) -> typing.Any:
        return self._unary_operation("-", operator.neg)

    def __pos__(self) -> typing.Any:
        return self._unary_operation("+", operator.pos)

    def __abs__(self) -> typing.Any:
        operand = self._console_operand()
        return self._wrap_console_result(
            abs(operand.value),
            f"abs({operand.code})",
            operand.script_inputs,
            copyable=operand.copyable,
        )

    def __invert__(self) -> typing.Any:
        return self._unary_operation("~", operator.invert)

    def __array__(
        self,
        dtype: np.dtype[typing.Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        if copy is None:
            return np.asarray(self.data, dtype=dtype)
        return np.array(self.data, dtype=dtype, copy=copy)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if method != "__call__":
            return NotImplemented
        input_operands = tuple(_operand_from_value(value) for value in inputs)
        kwarg_operands = {
            key: _operand_from_value(value) for key, value in kwargs.items()
        }
        script_inputs, copyable = _merge_operands(
            *input_operands, *tuple(kwarg_operands.values())
        )
        raw_kwargs = {key: operand.value for key, operand in kwarg_operands.items()}
        result = ufunc(*(operand.value for operand in input_operands), **raw_kwargs)
        arg_code = ", ".join(
            (
                *(operand.code for operand in input_operands),
                *(f"{key}={operand.code}" for key, operand in kwarg_operands.items()),
            )
        )
        return self._wrap_console_result(
            result,
            f"np.{ufunc.__name__}({arg_code})",
            script_inputs,
            copyable=copyable,
        )

    def __getitem__(self, key: typing.Any) -> typing.Any:
        key_operand = _operand_from_value(key)
        self_operand = self._console_operand()
        result = self.data[key_operand.value]
        inputs, copyable = _merge_operands(self_operand, key_operand)
        return self._wrap_console_result(
            result,
            f"{self_operand.code}[{key_operand.code}]",
            inputs,
            copyable=copyable,
        )

    def qshow(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        tools = self._console_tools
        if tools is None:
            return self.data.qshow(*args, **kwargs)
        return tools._qshow_handle(self, *args, **kwargs)

    def __getattr__(self, attr: str) -> typing.Any:
        data_attr = getattr(self.data, attr)
        if not callable(data_attr):
            if isinstance(data_attr, xr.DataArray):
                operand = self._console_operand()
                return self._wrap_console_result(
                    data_attr,
                    f"{operand.code}.{attr}",
                    operand.script_inputs,
                    copyable=operand.copyable,
                )
            return data_attr

        def _method(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            call_code, call_inputs, raw_args, raw_kwargs, args_copyable = (
                _format_call_code(args, kwargs)
            )
            self_operand = self._console_operand()
            result = data_attr(*raw_args, **raw_kwargs)
            inputs, copyable = _merge_operands(
                self_operand,
                _ConsoleOperand(None, "", call_inputs, args_copyable),
            )
            return self._wrap_console_result(
                result,
                f"{self_operand.code}.{attr}({call_code})",
                inputs,
                copyable=copyable,
            )

        return _method


class ToolNamespace(_ConsoleDataHandleMixin):
    """A console interface that represents a single ImageTool object.

    In the manager console, this namespace can be accessed with the variable
    ``tools[idx]``, where ``idx`` is the index of a top-level ImageTool to access.
    Child ImageTools are available through ``tools[idx].children``.

    Examples
    --------
    - Access the underlying DataArray of an ImageTool:

      >>> tools[1].data

    - Setting a new DataArray:

      >>> tools[1].data = new_data

    """

    def __init__(
        self,
        wrapper: _ImageToolWrapper | _ManagedWindowNode,
        tools: ToolsNamespace | None = None,
    ) -> None:
        self._wrapper_ref = weakref.ref(wrapper)
        self._tools_ref = weakref.ref(tools) if tools is not None else None

    @property
    def _wrapper(self) -> _ImageToolWrapper | _ManagedWindowNode:
        wrapper = self._wrapper_ref()
        if wrapper:
            return wrapper
        raise LookupError("Parent was destroyed")

    @property
    def tool(self) -> ImageTool:
        """The underlying ImageTool object."""
        return typing.cast("ImageTool", self._wrapper.imagetool)

    @property
    def data(self) -> xr.DataArray:
        """The DataArray associated with the ImageTool."""
        return self.tool.slicer_area._data

    @data.setter
    def data(self, value: xr.DataArray | _ConsoleDataHandleMixin) -> None:
        provenance_spec = None
        if isinstance(value, _ConsoleDataHandleMixin):
            raw_value = value.data
            provenance_spec = value._console_provenance_spec(
                active_name=self._console_input_name,
                label=f"Replace {self._console_label} data from console",
            )
        else:
            raw_value = value
        if provenance_spec is not None:
            self._wrapper.set_detached_provenance(provenance_spec)
        self.tool.slicer_area.replace_source_data(raw_value, emit_edited=True)

    @property
    def children(self) -> _ToolChildren:
        """Child ImageTools displayed under this ImageTool in manager tree order."""
        return _ToolChildren(self)

    def _get_data_item(self, key):
        """Return a subset of the tool data."""
        return self.tool.slicer_area._data[key]

    def _set_data_item(self, key, value) -> None:
        """Safely mutate a subset of the tool data from the console."""
        self.tool.slicer_area._set_source_item(key, _unwrap_console_value(value))

    @property
    def _console_tools(self) -> ToolsNamespace | None:
        if self._tools_ref is None:
            return None
        return self._tools_ref()

    @property
    def _console_is_source(self) -> bool:
        return True

    @property
    def _console_input_name(self) -> str:
        tools = self._console_tools
        if tools is not None:
            return tools._node_data_name(self._wrapper)
        return _tool_data_name(typing.cast("_ImageToolWrapper", self._wrapper).index)

    @property
    def _console_label(self) -> str:
        tools = self._console_tools
        if tools is not None:
            return tools._node_label(self._wrapper, include_name=False)
        return f"ImageTool {typing.cast('_ImageToolWrapper', self._wrapper).index}"

    def _script_input(
        self,
    ) -> erlab.interactive.imagetool.provenance.ScriptInput:
        label = self._console_label
        if self._wrapper.name:
            label += f": {self._wrapper.name}"
        provenance_spec = (
            self._wrapper.provenance_spec.model_dump(mode="json")
            if self._wrapper.provenance_spec is not None
            else None
        )
        return erlab.interactive.imagetool.provenance.ScriptInput(
            name=self._console_input_name,
            label=label,
            node_uid=self._wrapper.uid,
            node_snapshot_token=self._wrapper.snapshot_token,
            provenance_spec=provenance_spec,
        )

    def _console_operand(self) -> _ConsoleOperand:
        return _ConsoleOperand(
            self.data,
            self._console_input_name,
            (self._script_input(),),
        )

    def _console_provenance_spec(
        self, *, active_name: str, label: str
    ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
        return self._wrapper.provenance_spec

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        key_operand = _operand_from_value(key)
        value_operand = _operand_from_value(value)
        self_operand = self._console_operand()
        script_inputs, copyable = _merge_operands(
            self_operand, key_operand, value_operand
        )
        provenance_spec = erlab.interactive.imagetool.provenance.script(
            erlab.interactive.imagetool.provenance.ScriptCodeOperation(
                label=f"Set {self._console_label} data item from console",
                code=(
                    f"{self_operand.code}[{key_operand.code}] = {value_operand.code}"
                ),
                copyable=copyable,
            ),
            start_label="Run ImageTool manager console code",
            active_name=self._console_input_name,
            script_inputs=script_inputs,
        )
        self._set_data_item(key_operand.value, value_operand.value)
        self._wrapper.set_detached_provenance(provenance_spec)

    def __getattr__(self, attr):  # implicitly wrap methods from ImageToolWrapper
        if hasattr(self._wrapper, attr):
            m = getattr(self._wrapper, attr)
            if callable(m):
                return m
        return super().__getattr__(attr)

    def __repr__(self) -> str:
        time_repr = self._wrapper._created_time.isoformat(sep=" ", timespec="seconds")
        label = self._console_label
        if self._wrapper.name:
            label += f": {self._wrapper.name}"
        out = f"{label}\n"
        out += f"  Added: {time_repr}\n"
        out += f"  Linked: {self.tool.slicer_area.is_linked}\n"
        return out


class _ToolChildren:
    def __init__(self, parent: ToolNamespace) -> None:
        self._parent = parent

    def _nodes(self) -> list[_ManagedWindowNode]:
        tools = self._parent._console_tools
        if tools is None:
            return []
        return tools._child_imagetool_nodes(self._parent._wrapper)

    def __getitem__(self, index: int | slice) -> ToolNamespace | list[ToolNamespace]:
        tools = self._parent._console_tools
        if tools is None:
            raise LookupError("Parent was destroyed")
        nodes = self._nodes()
        if isinstance(index, slice):
            return [ToolNamespace(node, tools) for node in nodes[index]]
        return ToolNamespace(nodes[index], tools)

    def __iter__(self):
        tools = self._parent._console_tools
        if tools is None:
            return iter(())
        return (ToolNamespace(node, tools) for node in self._nodes())

    def __len__(self) -> int:
        return len(self._nodes())

    def __repr__(self) -> str:
        tools = self._parent._console_tools
        if tools is None:
            return "No child ImageTools"
        nodes = self._nodes()
        if not nodes:
            return f"No child ImageTools for {self._parent._console_label}"
        lines = [f"Child ImageTools for {self._parent._console_label}:"]
        for index, node in enumerate(nodes):
            lines.append(f"[{index}] {tools._node_label(node, include_name=True)}")
        return "\n".join(lines)


class _DerivedDataNamespace(_ConsoleDataHandleMixin):
    def __init__(
        self,
        tools: ToolsNamespace | None,
        data: xr.DataArray,
        expression: str,
        script_inputs: Sequence[erlab.interactive.imagetool.provenance.ScriptInput],
        *,
        copyable: bool,
    ) -> None:
        self._tools_ref = weakref.ref(tools) if tools is not None else None
        self._data = data
        self._expression = expression
        self._script_inputs = _dedupe_script_inputs(script_inputs)
        self._copyable = copyable
        self._console_name: str | None = None

    @property
    def data(self) -> xr.DataArray:
        return self._data

    @property
    def _console_tools(self) -> ToolsNamespace | None:
        if self._tools_ref is None:
            return None
        return self._tools_ref()

    def _set_console_name(self, name: str) -> None:
        self._console_name = name

    def _console_operand(self) -> _ConsoleOperand:
        if self._console_name is None:
            return _ConsoleOperand(
                self.data,
                _derived_operand_code(self._expression),
                self._script_inputs,
                self._copyable,
            )
        provenance_spec = self._console_provenance_spec(
            active_name=self._console_name,
            label=f"Assign console variable {self._console_name!r}",
        )
        provenance_payload = (
            provenance_spec.model_dump(mode="json")
            if provenance_spec is not None
            else None
        )
        return _ConsoleOperand(
            self.data,
            self._console_name,
            (
                erlab.interactive.imagetool.provenance.ScriptInput(
                    name=self._console_name,
                    label=f"console variable {self._console_name!r}",
                    provenance_spec=provenance_payload,
                ),
            ),
            provenance_spec is not None,
        )

    def _console_provenance_spec(
        self, *, active_name: str, label: str
    ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
        if not self._script_inputs:
            return None
        return erlab.interactive.imagetool.provenance.script(
            erlab.interactive.imagetool.provenance.ScriptCodeOperation(
                label=label,
                code=f"{active_name} = {self._expression}",
                copyable=self._copyable,
            ),
            start_label="Run ImageTool manager console code",
            active_name=active_name,
            script_inputs=self._script_inputs,
        )

    def __repr__(self) -> str:
        return repr(self.data)


class ToolsNamespace:
    """A console interface that represents the ImageToolManager and its tools.

    In the manager console, this namespace can be accessed with the variable `tools`.

    Examples
    --------
    - Print the list of tools:

      >>> tools

    - Access :class:`ToolNamespace` by index:

      >>> tools[1]

    """

    def __init__(self, manager: ImageToolManager) -> None:
        self._manager_ref = weakref.ref(manager)
        self._shell_ref: weakref.ReferenceType[InteractiveShell] | None = None
        self._namespace_snapshot: dict[str, int] = {}

    @property
    def _manager(self) -> ImageToolManager:
        """Access the ImageToolManager instance."""
        manager = self._manager_ref()
        if manager:
            return manager
        raise LookupError("Parent was destroyed")

    @property
    def selected_data(self) -> list[xr.DataArray]:
        """Get a list of DataArrays from the selected windows."""
        return [
            self._manager.get_imagetool(target).slicer_area._data
            for target in self._manager._selected_imagetool_targets()
        ]

    @property
    def selected(self) -> list[ToolNamespace]:
        """Get provenance-aware handles for the selected windows."""
        return [
            ToolNamespace(self._manager._node_for_target(target), self)
            for target in self._manager._selected_imagetool_targets()
        ]

    def __getitem__(self, index: int) -> ToolNamespace | None:
        """Access a specific ImageTool object by its index."""
        if index not in self._manager._imagetool_wrappers:
            print(f"Tool {index} not found")
            return None

        return ToolNamespace(self._manager._imagetool_wrappers[index], self)

    def _node_path(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> list[int] | None:
        path: list[int] = []
        current = node
        while current.parent_uid is not None:
            image_parent = self._image_parent_node(current)
            if image_parent is None:
                return None
            child_nodes = self._child_imagetool_nodes(image_parent)
            child_uids = [child.uid for child in child_nodes]
            if current.uid not in child_uids:
                return None
            child_index = child_uids.index(current.uid)
            path.append(child_index)
            current = image_parent
        for root_index, wrapper in self._manager._imagetool_wrappers.items():
            if wrapper.uid == current.uid:
                return [root_index, *reversed(path)]
        return None

    def _child_imagetool_nodes(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> list[_ManagedWindowNode]:
        output: list[_ManagedWindowNode] = []

        def collect(parent: _ImageToolWrapper | _ManagedWindowNode) -> None:
            for child_uid in parent._childtool_indices:
                child = self._manager._all_nodes.get(child_uid)
                if child is None:
                    continue
                if child.is_imagetool:
                    output.append(child)
                    continue
                collect(child)

        collect(node)
        return output

    def _image_parent_node(
        self, node: _ManagedWindowNode
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        parent_uid = node.parent_uid
        while parent_uid is not None:
            parent = self._manager._all_nodes.get(parent_uid)
            if parent is None:
                return None
            if parent.is_imagetool:
                return parent
            parent_uid = parent.parent_uid
        return None

    def _node_data_name(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        path = self._node_path(node)
        if path is not None:
            return "data_" + "_".join(str(index) for index in path)
        uid_name = "".join(char if char.isalnum() else "_" for char in node.uid)
        return f"data_{uid_name[:8]}"

    def _node_label(
        self, node: _ImageToolWrapper | _ManagedWindowNode, *, include_name: bool
    ) -> str:
        path = self._node_path(node)
        display_index = ".".join(str(index) for index in path) if path else node.uid[:8]
        label = f"ImageTool {display_index}"
        if include_name and node.name:
            label += f": {node.name}"
        return label

    def bind_shell(self, shell: InteractiveShell) -> None:
        self._shell_ref = weakref.ref(shell)
        shell.events.register("pre_run_cell", self._pre_run_cell)
        shell.events.register("post_run_cell", self._post_run_cell)

    def unbind_shell(self) -> None:
        if self._shell_ref is None:
            return
        shell = self._shell_ref()
        if shell is not None:
            with contextlib.suppress(KeyError, ValueError):
                shell.events.unregister("pre_run_cell", self._pre_run_cell)
            with contextlib.suppress(KeyError, ValueError):
                shell.events.unregister("post_run_cell", self._post_run_cell)
        self._shell_ref = None

    def _pre_run_cell(self, info: ExecutionInfo | None = None) -> None:
        if self._shell_ref is None:
            return
        shell = self._shell_ref()
        if shell is None:
            return
        self._namespace_snapshot = {
            name: id(value) for name, value in shell.user_ns.items()
        }

    def _post_run_cell(self, result: ExecutionResult | None = None) -> None:
        if result is not None and (
            result.error_before_exec is not None or result.error_in_exec is not None
        ):
            return
        if self._shell_ref is not None and (shell := self._shell_ref()) is not None:
            for name, value in shell.user_ns.items():
                if (
                    name.startswith("_")
                    or self._namespace_snapshot.get(name) == id(value)
                    or not isinstance(value, _DerivedDataNamespace)
                ):
                    continue
                value._set_console_name(name)
        if result is None or not isinstance(result.result, _DerivedDataNamespace):
            return
        self._show_handle(
            result.result,
            active_name=result.result._console_name or "derived",
            label="Evaluate console expression",
        )

    def _manager_argument_targets_this_manager(self, value: typing.Any) -> bool:
        if value is False:
            return False
        if value is None or value is True:
            return True
        try:
            target = operator.index(value)
        except TypeError:
            return False
        return target == getattr(self._manager, "manager_index", None)

    def _show_dataarray_with_provenance(
        self,
        data: xr.DataArray,
        provenance_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec,
        **kwargs,
    ) -> bool:
        display_kwargs = dict(kwargs)
        display_kwargs.pop("execute", None)
        tool = erlab.interactive.itool(
            data, manager=False, execute=False, **display_kwargs
        )
        if not isinstance(tool, erlab.interactive.imagetool.ImageTool):
            return False
        self._manager.add_imagetool(
            tool,
            show=True,
            activate=True,
            provenance_spec=provenance_spec,
        )
        return True

    def _show_handle(
        self,
        handle: _ConsoleDataHandleMixin,
        *,
        active_name: str,
        label: str,
        **kwargs: typing.Any,
    ) -> bool:
        provenance_spec = handle._console_provenance_spec(
            active_name=active_name,
            label=label,
        )
        if provenance_spec is None:
            return False
        return self._show_dataarray_with_provenance(
            handle.data, provenance_spec, **kwargs
        )

    def itool(
        self, data: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        manager = kwargs.get("manager")
        replace = kwargs.get("replace")
        if (
            isinstance(data, _ConsoleDataHandleMixin)
            and not args
            and replace is None
            and self._manager_argument_targets_this_manager(manager)
        ):
            display_kwargs = dict(kwargs)
            display_kwargs.pop("manager", None)
            display_kwargs.pop("replace", None)
            if self._show_handle(
                data,
                active_name=(
                    data._console_name
                    if isinstance(data, _DerivedDataNamespace)
                    and data._console_name is not None
                    else "derived"
                ),
                label="Evaluate console expression",
                **display_kwargs,
            ):
                return None
        return erlab.interactive.itool(_unwrap_console_value(data), *args, **kwargs)

    def _qshow_handle(
        self,
        data: _ConsoleDataHandleMixin,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        manager = kwargs.get("manager")
        if not args and self._manager_argument_targets_this_manager(manager):
            display_kwargs = dict(kwargs)
            display_kwargs.pop("manager", None)
            if self._show_handle(
                data,
                active_name=(
                    data._console_name
                    if isinstance(data, _DerivedDataNamespace)
                    and data._console_name is not None
                    else "derived"
                ),
                label="Evaluate console expression",
                **display_kwargs,
            ):
                return None
        return data.data.qshow(*args, **kwargs)

    def __repr__(self) -> str:
        output = []
        for index, wrapper in self._manager._imagetool_wrappers.items():
            output.append(f"{index}: {wrapper.name}")
        if not output:
            return "No tools"
        return "\n".join(output)


class _JupyterConsoleWidget(qtconsole.inprocess.QtInProcessRichJupyterWidget):
    """A Jupyter console widget for ImageToolManager.

    This widget is derived from qtconsole with some modifications such as:

    - Support for dark mode

    - Custom banner text

    - Lazy kernel initialization, including lazily evaluated namespace injection

    - Automated storing of data from ImageTools with the ``%store`` magic command


    Parameters
    ----------
    parent
        The parent widget for the console.
    namespace
        A dictionary of objects to inject into the console namespace. The keys are the
        names of the objects in the console, and the values are the objects themselves.
        If the value is a string, it is imported as a module upon kernel initialization,
        improving startup time for the ImageToolManager application.

    """

    def __init__(
        self, parent=None, namespace: dict[str, typing.Any] | None = None
    ) -> None:
        super().__init__(parent)
        self.kernel_manager = qtconsole.inprocess.QtInProcessKernelManager()
        self._namespace = namespace
        self._kernel_banner_default: str = ""
        self._kernel_initializing = False
        self._erlab_loader_name: str | None = None
        self._erlab_data_dir: str | None = None
        self._erlab_io_hooks_registered = False
        self._tools_namespace: ToolsNamespace | None = None

    def _restore_erlab_io_state(self, *args, **kwargs) -> None:
        erlab.io.set_loader(self._erlab_loader_name)
        erlab.io.set_data_dir(self._erlab_data_dir)

    def _persist_erlab_io_state(self, *args, **kwargs) -> None:
        loader = erlab.io.loaders.current_loader
        self._erlab_loader_name = None if loader is None else loader.name

        data_dir = erlab.io.loaders.current_data_dir
        self._erlab_data_dir = None if data_dir is None else str(data_dir)

    def _register_erlab_io_hooks(self) -> None:
        if self._erlab_io_hooks_registered or not self.kernel_manager.kernel:
            return

        shell = self.kernel_manager.kernel.shell
        shell.events.register("pre_run_cell", self._restore_erlab_io_state)
        shell.events.register("post_run_cell", self._persist_erlab_io_state)
        self._erlab_io_hooks_registered = True
        self._persist_erlab_io_state()

    def initialize_kernel(self) -> None:
        if self.kernel_manager.kernel or self._kernel_initializing:
            return
        self._kernel_initializing = True
        try:
            self.kernel_manager.start_kernel()
            self.kernel_client = self.kernel_manager.client()
            self.kernel_client.start_channels()

            super().execute(r"%load_ext storemagic", hidden=True, interactive=False)
            super().execute(
                r"%load_ext erlab.interactive", hidden=True, interactive=False
            )

            if self._namespace is not None:
                resolved_namespace = _resolve_console_namespace(self._namespace)
                self.kernel_manager.kernel.shell.push(resolved_namespace)
                tools_namespace = resolved_namespace.get("tools")
                if isinstance(tools_namespace, ToolsNamespace):
                    self._tools_namespace = tools_namespace
                    tools_namespace.bind_shell(self.kernel_manager.kernel.shell)
                super().execute(
                    r"xr.set_options(keep_attrs=True)",
                    hidden=True,
                    interactive=False,
                )
            self._register_erlab_io_hooks()
        finally:
            self._kernel_initializing = False

    def execute(
        self, source: str | None = None, hidden: bool = False, interactive: bool = False
    ) -> None:
        if not self.kernel_manager.kernel and not self._kernel_initializing:
            self.initialize_kernel()
        super().execute(source, hidden=hidden, interactive=interactive)

    def store_data_as(self, tool_index: int, name: str) -> None:
        """Store the data in an ImageTool with IPython to reuse in other scripts."""
        self.initialize_kernel()
        store_commands = (
            f"{name} = tools[{tool_index}].data",
            f"get_ipython().run_line_magic('store', '{name}')",
            f"del {name}",
        )
        self.execute("\n".join(store_commands), hidden=True)

    @QtCore.Slot()
    def shutdown_kernel(self) -> None:
        if self.kernel_manager.kernel:
            if self._tools_namespace is not None:
                self._tools_namespace.unbind_shell()
                self._tools_namespace = None
            if self._erlab_io_hooks_registered:
                shell = self.kernel_manager.kernel.shell
                with contextlib.suppress(KeyError, ValueError):
                    shell.events.unregister(
                        "pre_run_cell", self._restore_erlab_io_state
                    )
                with contextlib.suppress(KeyError, ValueError):
                    shell.events.unregister(
                        "post_run_cell", self._persist_erlab_io_state
                    )
                self._erlab_io_hooks_registered = False
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()

    def _banner_default(self) -> str:
        banner = super()._banner_default()
        return banner.strip() + f" | ERLabPy {erlab.__version__}\n"

    @property
    def kernel_banner(self) -> str:
        def _command_ansi(title: str, command_list: list[str]):
            out = f"\033[1m* {title}\033[0m"
            for command in command_list:
                out += f"\n  {command}"
            return out

        info_str = (
            _command_ansi(
                "Access raw data",
                [
                    "tools[<index>].data",
                    "tools[<index>].children[0].data",
                    "tools.selected_data",
                ],
            )
            + "\n"
            + _command_ansi(
                "Track provenance",
                [
                    "tools[0] - tools[1]",
                    "tools[0].children[0] - tools[1]",
                    "tools.selected[0]",
                ],
            )
            + "\n"
            + _command_ansi("Change data", ["tools[<index>].data = <value>"])
            + "\n"
            + _command_ansi(
                "Control window visibility",
                ["tools[<index>].show(), .close(), .dispose()"],
            )
            + "\n"
        )

        return f"{self._kernel_banner_default}{info_str}"

    @kernel_banner.setter
    def kernel_banner(self, value: str) -> None:
        self._kernel_banner_default = value

    def _update_colors(self) -> None:
        """Detect dark mode and update the console colors accordingly."""
        if self.kernel_manager.kernel:
            colors = "linux" if erlab.interactive.colors.is_dark_mode() else "lightbg"
            self.set_default_style(colors)
            self._syntax_style_changed()
            self._style_sheet_changed()
            self._execute(
                f"""
from IPython.core.ultratb import VerboseTB
if getattr(VerboseTB, 'tb_highlight_style', None) is not None:
    VerboseTB.tb_highlight_style = '{self.syntax_style}'
elif getattr(VerboseTB, '_tb_highlight_style', None) is not None:
    VerboseTB._tb_highlight_style = '{self.syntax_style}'
else:
    get_ipython().run_line_magic('colors', '{colors}')
del VerboseTB
""",
                True,
            )  # Adapted from qtconsole.mainwindow.MainWindow.set_syntax_style

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(300, 186)


class _ImageToolManagerJupyterConsole(QtWidgets.QDockWidget):
    """A dock widget containing the Jupyter console."""

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__("Console", manager, flags=QtCore.Qt.WindowType.Window)
        tools_namespace = ToolsNamespace(manager)

        self._console_widget = _JupyterConsoleWidget(
            parent=self,
            namespace={
                "np": np,
                "xr": xr,
                "erlab": erlab,
                "eri": erlab.interactive,
                "itool": tools_namespace.itool,
                "tools": tools_namespace,
                "era": "erlab.analysis",
                "eplt": "erlab.plotting",
                "plt": "matplotlib.pyplot",
            },
        )
        qapp = QtWidgets.QApplication.instance()

        if qapp:
            # Shutdown kernel when application quits
            qapp.aboutToQuit.connect(self._console_widget.shutdown_kernel)

        self.setWidget(self._console_widget)
        manager.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self)
        self.setFloating(False)
        self.hide()

        # Start kernel when console is shown
        self._console_widget.installEventFilter(self)

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if (
            hasattr(self, "_console_widget")
            and obj == self._console_widget
            and event is not None
            and event.type() == QtCore.QEvent.Type.Show
        ):
            self._console_widget.initialize_kernel()
            self._console_widget._update_colors()
        return super().eventFilter(obj, event)

    def changeEvent(self, evt: QtCore.QEvent | None) -> None:
        if evt is not None and evt.type() == QtCore.QEvent.Type.PaletteChange:
            self._console_widget._update_colors()

        super().changeEvent(evt)

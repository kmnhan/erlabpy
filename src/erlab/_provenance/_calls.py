"""Public Python calls used by both recorded execution and notebook export."""

from __future__ import annotations

import keyword
import operator
import sys
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pydantic
import xarray as xr

from erlab._provenance._model import ToolProvenanceOperation, _assignment_code
from erlab._provenance._settings import ExecutionSettings

if typing.TYPE_CHECKING:
    from collections.abc import Collection, Sequence


@dataclass(frozen=True, slots=True)
class InputRef:
    """An operand supplied by a named input slot."""

    slot: str


def _argument(value: typing.Any, inputs: Mapping[str, xr.DataArray]) -> typing.Any:
    if isinstance(value, InputRef):
        return inputs[value.slot]
    if isinstance(value, list):
        return [_argument(item, inputs) for item in value]
    if isinstance(value, tuple):
        return tuple(_argument(item, inputs) for item in value)
    if isinstance(value, Mapping):
        return {
            _argument(key, inputs): _argument(item, inputs)
            for key, item in value.items()
        }
    return value


def _argument_code(value: typing.Any, names: Mapping[str, str]) -> str:
    if isinstance(value, InputRef):
        return names[value.slot]
    if isinstance(value, list):
        return "[" + ", ".join(_argument_code(item, names) for item in value) + "]"
    if isinstance(value, tuple):
        items = ", ".join(_argument_code(item, names) for item in value)
        return f"({items},)" if len(value) == 1 else f"({items})"
    if isinstance(value, Mapping):
        items = (
            f"{_argument_code(key, names)}: {_argument_code(item, names)}"
            for key, item in value.items()
        )
        return "{" + ", ".join(items) + "}"
    if isinstance(value, slice):
        items = (value.start, value.stop, value.step)
        return "slice(" + ", ".join(_argument_code(item, names) for item in items) + ")"
    if isinstance(value, np.ndarray | np.generic):
        dtype = value.dtype
        dtype_spec = dtype.descr if dtype.fields else dtype.str
        # Preserve the storage unit of timestamps and durations.
        if dtype.kind in "mM":
            items = value.astype(np.int64).tolist()
        elif dtype.char in {"g", "G"}:
            # tolist() can return another longdouble, even on platforms where it
            # uses eight bytes. Decimal text preserves the native precision.
            items = value.astype(str).tolist()
            dtype_spec = dtype.byteorder + dtype.char
        else:
            items = value.tolist()
        literal = f"np.array({_argument_code(items, names)}, dtype={dtype_spec!r})"
        if isinstance(value, np.generic):
            return literal + "[()]"
        if value.size == 0 and value.ndim > 1:
            return f"{literal}.reshape({value.shape!r})"
        return literal
    if isinstance(value, float) and not np.isfinite(value):
        return f"float({str(value)!r})"
    if isinstance(value, complex):
        real = _argument_code(value.real, names)
        imag = _argument_code(value.imag, names)
        return f"complex({real}, {imag})"
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return repr(value)
    raise TypeError(f"Cannot emit a Python literal for {type(value).__name__}")


def _public_name(function: Callable[..., typing.Any]) -> str:
    name = function.__name__
    if getattr(np, name, None) is function:
        return f"np.{name}"
    if getattr(xr, name, None) is function:
        return f"xr.{name}"
    # Installed code supplies the callable. Check the public package before
    # emitting a name for a function re-exported from an implementation module.
    module = function.__module__.split("._", 1)[0]
    if getattr(sys.modules.get(module), name, None) is not function:
        raise ValueError(f"{function!r} has no public export at {module}.{name}")
    if module.startswith("erlab.analysis."):
        module = module.replace("erlab.analysis.", "era.", 1)
    return f"{module}.{name}"


@dataclass(frozen=True, slots=True)
class Call:
    """One public function call, or a method on the primary array."""

    target: Callable[..., typing.Any] | str
    args: Sequence[typing.Any] = ()
    kwargs: Mapping[str, typing.Any] = field(default_factory=dict)

    @classmethod
    def function(
        cls,
        function: Callable[..., typing.Any],
        *,
        args: Sequence[typing.Any] = (),
        kwargs: Mapping[str, typing.Any] | None = None,
    ) -> Call:
        return cls(function, args, {} if kwargs is None else kwargs)

    @classmethod
    def method(
        cls,
        name: str,
        *,
        args: Sequence[typing.Any] = (),
        kwargs: Mapping[str, typing.Any] | None = None,
    ) -> Call:
        return cls(name, args, {} if kwargs is None else kwargs)

    def evaluate(self, inputs: Mapping[str, xr.DataArray]) -> typing.Any:
        target = self.target
        if isinstance(target, str):
            target = operator.attrgetter(target)(inputs["data"])
        return target(
            *(_argument(arg, inputs) for arg in self.args),
            **{key: _argument(value, inputs) for key, value in self.kwargs.items()},
        )

    def expression(self, names: Mapping[str, str]) -> str:
        target = (
            f"{names['data']}.{self.target}"
            if isinstance(self.target, str)
            else _public_name(self.target)
        )
        args = [_argument_code(arg, names) for arg in self.args]
        for key, value in self.kwargs.items():
            if key.isidentifier() and not keyword.iskeyword(key):
                args.append(f"{key}={_argument_code(value, names)}")
            else:
                args.append(f"**{_argument_code({key: value}, names)}")
        return f"{target}({', '.join(args)})"


@dataclass(frozen=True, slots=True)
class SetAttribute:
    """Assign a public metadata property on an isolated array wrapper."""

    accessor: str
    attribute: str
    value: typing.Any

    def evaluate(self, data: xr.DataArray) -> xr.DataArray:
        result = data.copy(deep=False)
        setattr(getattr(result, self.accessor), self.attribute, self.value)
        return result

    def statement(self, input_name: str, output_name: str) -> str:
        value = _argument_code(self.value, {})
        assignment = f"{output_name}.{self.accessor}.{self.attribute} = {value}"
        if input_name == output_name:
            return assignment
        return f"{output_name} = {input_name}.copy(deep=False)\n{assignment}"


class CallOperation(ToolProvenanceOperation):
    """A built-in transform defined by one public, read-only Python call."""

    def invocation(self) -> Call:
        raise NotImplementedError

    def evaluate(self, operands: Mapping[str, xr.DataArray]) -> xr.DataArray:
        result = self.invocation().evaluate(operands)
        if not isinstance(result, xr.DataArray):
            raise TypeError("An operation must produce a DataArray")
        return result

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return self.evaluate({"data": data})

    def apply_with_inputs(
        self,
        data: xr.DataArray,
        inputs: Mapping[str, xr.DataArray],
        *,
        authorization: object | None = None,
    ) -> xr.DataArray:
        return self.evaluate({**inputs, "data": data})

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        return self.invocation().expression({"data": input_name})

    def expression_code_with_inputs(
        self,
        input_name: str,
        inputs: Mapping[str, str],
        *,
        source_name: str | None = None,
    ) -> str:
        return self.invocation().expression({**inputs, "data": input_name})


class ArrayMethodOperation(CallOperation):
    """A method with positional literals on the primary DataArray."""

    method: typing.ClassVar[str]

    def arguments(self) -> tuple[typing.Any, ...]:
        raise NotImplementedError

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return getattr(data, self.method)(*self.arguments())

    def evaluate(self, operands: Mapping[str, xr.DataArray]) -> xr.DataArray:
        return self.apply(operands["data"])

    def invocation(self) -> Call:
        return Call.method(self.method, args=self.arguments())


class WorkerCallOperation(CallOperation):
    """A public call that installs its recorded options in the worker."""

    # Older records used the options in effect at replay time.
    execution_options: dict[str, bool | str] | None = pydantic.Field(
        default=None, exclude_if=lambda value: value is None
    )

    @pydantic.field_validator("execution_options")
    @classmethod
    def _valid_execution_options(
        cls, options: dict[str, bool | str] | None
    ) -> dict[str, bool | str] | None:
        if options is None:
            return None
        return ExecutionSettings.from_options(options).as_dict()

    def effective_settings(self, defaults: ExecutionSettings) -> ExecutionSettings:
        if self.execution_options is None:
            return defaults
        return ExecutionSettings.from_options(
            defaults.as_dict() | self.execution_options
        )


class TupleOutputOperation(CallOperation):
    """Select one returned array and retain the public tuple unpacking in notebooks."""

    outputs: typing.ClassVar[tuple[str, ...]]

    @property
    def selected_output(self) -> str:
        raise NotImplementedError

    def evaluate(self, operands: Mapping[str, xr.DataArray]) -> xr.DataArray:
        results = self.invocation().evaluate(operands)
        if not isinstance(results, tuple) or len(results) != len(self.outputs):
            raise TypeError("The call did not return the declared tuple of outputs")
        selected = results[self.outputs.index(self.selected_output)]
        if not isinstance(selected, xr.DataArray):
            raise TypeError("An operation must produce a DataArray")
        return selected

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        raise NotImplementedError

    def expression_code_with_inputs(
        self,
        input_name: str,
        inputs: Mapping[str, str],
        *,
        source_name: str | None = None,
    ) -> str:
        raise NotImplementedError

    def _output_statement(
        self,
        names: Mapping[str, str],
        output_name: str,
        reserved_names: Collection[str],
    ) -> str:
        targets = []
        used = {output_name, *names.values(), *reserved_names}
        for base in self.outputs:
            if base == self.selected_output:
                targets.append(output_name)
                continue
            target = base
            suffix = 2
            while target in used:
                target = f"{base}_{suffix}"
                suffix += 1
            targets.append(target)
            used.add(target)
        return _assignment_code(", ".join(targets), self.invocation().expression(names))

    def statement_code(
        self, input_name: str, *, output_name: str, source_name: str | None = None
    ) -> str:
        return self._output_statement({"data": input_name}, output_name, ())

    def _statement_replay_code_with_inputs(
        self,
        input_name: str,
        inputs: Mapping[str, str],
        *,
        output_name: str,
        source_name: str | None = None,
        reserved_names: Collection[str] = (),
    ) -> str:
        return self._output_statement(
            {**inputs, "data": input_name}, output_name, reserved_names
        )


class MetadataSetterOperation(ToolProvenanceOperation):
    """An accessor setter that changes metadata on its own array wrapper."""

    statement_mutates_input: typing.ClassVar[bool] = True

    def assignment(self) -> SetAttribute:
        raise NotImplementedError

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return self.assignment().evaluate(data)

    def statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
    ) -> str:
        return self.assignment().statement(input_name, output_name)

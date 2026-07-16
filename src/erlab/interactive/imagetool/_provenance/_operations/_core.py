"""Script-backed and model-fitting provenance operations."""

from __future__ import annotations

import typing

import numpy as np
import pydantic
import xarray as xr

import erlab
from erlab.interactive.imagetool._provenance._code import _provenance_value_code
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    NullableProvenanceHashable,
    ProvenanceHashable,
    ProvenanceMapping,
    ToolProvenanceOperation,
)

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Mapping


class ScriptCodeOperation(ToolProvenanceOperation):
    op: typing.Literal["script_code"] = "script_code"
    label: str
    code: str | None
    copyable: bool = True
    visible: bool = True

    live_applicable: typing.ClassVar[bool] = False

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        raise TypeError(
            "script_code operations do not support live updates from ImageTool data"
        )

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(self.label, self.code, self.copyable)


_MODEL_FIT_MODEL_NAMES = (
    "FermiEdgeModel",
    "MultiPeakModel",
    "PolynomialModel",
    "StepEdgeModel",
    "SymmetrizedGapModel",
    "TLLModel",
)


class _ModelFitParameterSpec(pydantic.BaseModel):
    """Serializable scalar or one-dimensional parameter initialization."""

    value: float | tuple[float, ...] | None = None
    minimum: float | tuple[float, ...] | None = None
    maximum: float | tuple[float, ...] | None = None
    vary: bool = True
    expr: str | None = None

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def _validate_parameter(self) -> typing.Self:
        if self.expr is not None:
            if not self.expr.strip():
                raise ValueError("model-fit parameter expressions must not be empty")
            if any(
                value is not None for value in (self.value, self.minimum, self.maximum)
            ):
                raise ValueError("expression parameters cannot define values or bounds")
            if not self.vary:
                raise ValueError("expression parameters cannot define vary=False")
            return self
        if self.value is None:
            raise ValueError("model-fit parameters must define a value or expression")

        sequence_lengths: set[int] = set()
        for field_name in ("value", "minimum", "maximum"):
            field_value = getattr(self, field_name)
            if isinstance(field_value, tuple):
                if not field_value:
                    raise ValueError(
                        f"model-fit parameter {field_name} arrays must not be empty"
                    )
                sequence_lengths.add(len(field_value))
            if field_value is not None and any(
                np.isnan(item)
                for item in (
                    field_value if isinstance(field_value, tuple) else (field_value,)
                )
            ):
                raise ValueError(
                    f"model-fit parameter {field_name} must not contain NaN"
                )
            if (
                field_name == "value"
                and field_value is not None
                and any(
                    not np.isfinite(item)
                    for item in (
                        field_value
                        if isinstance(field_value, tuple)
                        else (field_value,)
                    )
                )
            ):
                raise ValueError("model-fit parameter values must be finite")
        if len(sequence_lengths) > 1:
            raise ValueError("model-fit parameter arrays must have equal lengths")
        return self

    @property
    def has_array_value(self) -> bool:
        return any(
            isinstance(value, tuple)
            for value in (self.value, self.minimum, self.maximum)
        )


def _model_fit_value_code(
    value: float | tuple[float, ...],
    *,
    input_name: str,
    broadcast_dim: Hashable | None,
) -> str:
    if not isinstance(value, tuple):
        return _provenance_value_code(value)
    if broadcast_dim is None:  # pragma: no cover - operation validation guard.
        raise ValueError("array-valued parameters require a broadcast dimension")
    broadcast_dim_code = _provenance_value_code(broadcast_dim)
    values_code = _provenance_value_code(list(value))
    return "\n".join(
        (
            "xr.DataArray(",
            f"    {values_code},",
            f"    coords={{{broadcast_dim_code}: "
            f"{input_name}.get_index({broadcast_dim_code})}},",
            f"    dims=({broadcast_dim_code},),",
            ")",
        )
    )


def _model_fit_parameter_entry_code(
    parameter: _ModelFitParameterSpec,
    *,
    input_name: str,
    broadcast_dim: Hashable | None,
) -> list[str]:
    if parameter.expr is not None:
        return [f"{{'expr': {parameter.expr!r}}}"]
    if parameter.value is None:  # pragma: no cover - model validation guard.
        raise ValueError("model-fit parameter has no value")

    value_code = _model_fit_value_code(
        parameter.value,
        input_name=input_name,
        broadcast_dim=broadcast_dim,
    )
    if parameter.minimum is None and parameter.maximum is None and parameter.vary:
        return value_code.splitlines()

    fields = [("value", value_code)]
    if parameter.minimum is not None:
        fields.append(
            (
                "min",
                _model_fit_value_code(
                    parameter.minimum,
                    input_name=input_name,
                    broadcast_dim=broadcast_dim,
                ),
            )
        )
    if parameter.maximum is not None:
        fields.append(
            (
                "max",
                _model_fit_value_code(
                    parameter.maximum,
                    input_name=input_name,
                    broadcast_dim=broadcast_dim,
                ),
            )
        )
    if not parameter.vary:
        fields.append(("vary", "False"))
    lines = ["{"]
    for name, value in fields:
        value_lines = value.splitlines()
        if len(value_lines) == 1:
            lines.append(f"    {name!r}: {value},")
            continue
        lines.append(f"    {name!r}: {value_lines[0]}")
        lines.extend(f"    {line}" for line in value_lines[1:-1])
        lines.append(f"    {value_lines[-1]},")
    lines.append("}")
    return lines


def _model_fit_parameters_code(
    parameters: Mapping[str, _ModelFitParameterSpec],
    *,
    input_name: str,
    broadcast_dim: Hashable | None,
) -> str:
    """Return readable xarray-lmfit parameter mapping code."""
    lines = ["{"]
    for name, parameter in parameters.items():
        entry_lines = _model_fit_parameter_entry_code(
            parameter,
            input_name=input_name,
            broadcast_dim=broadcast_dim,
        )
        if len(entry_lines) == 1:
            lines.append(f"    {name!r}: {entry_lines[0]},")
            continue
        lines.append(f"    {name!r}: {entry_lines[0]}")
        lines.extend(f"    {line}" for line in entry_lines[1:-1])
        lines.append(f"    {entry_lines[-1]},")
    lines.append("}")
    return "\n".join(lines)


def _model_fit_runtime_value(
    value: float | tuple[float, ...],
    *,
    data: xr.DataArray,
    broadcast_dim: Hashable | None,
) -> float | xr.DataArray:
    if not isinstance(value, tuple):
        return value
    if broadcast_dim is None:  # pragma: no cover - operation validation guard.
        raise ValueError("array-valued parameters require a broadcast dimension")
    if broadcast_dim not in data.dims:
        raise ValueError(
            f"Model-fit broadcast dimension {broadcast_dim!r} was not found in data"
        )
    if len(value) != data.sizes[broadcast_dim]:
        raise ValueError(
            f"Model-fit parameter array length {len(value)} does not match "
            f"dimension {broadcast_dim!r} size {data.sizes[broadcast_dim]}"
        )
    return xr.DataArray(
        list(value),
        coords={broadcast_dim: data.get_index(broadcast_dim)},
        dims=(broadcast_dim,),
    )


class ModelFitOperation(ToolProvenanceOperation):
    """Fit a supported model and return one parameter value or standard error."""

    supported_models: typing.ClassVar[frozenset[str]] = frozenset(
        _MODEL_FIT_MODEL_NAMES
    )

    op: typing.Literal["model_fit"] = "model_fit"
    fit_dim: ProvenanceHashable
    model: str
    model_kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)
    parameters: dict[str, _ModelFitParameterSpec]
    method: str
    parameter: str
    output: typing.Literal["value", "stderr"] = "value"
    broadcast_dim: NullableProvenanceHashable = None
    normalize: bool = False

    @pydantic.model_validator(mode="after")
    def _validate_model_fit(self) -> typing.Self:
        if self.model not in self.supported_models:
            raise ValueError(f"Unsupported model-fit model {self.model!r}")
        if isinstance(self.fit_dim, str) and not self.fit_dim:
            raise ValueError("model-fit dimension must not be empty")
        if self.broadcast_dim == self.fit_dim:
            raise ValueError("model-fit and broadcast dimensions must differ")
        if not self.method.strip():
            raise ValueError("model-fit method must not be empty")
        if not self.parameter:
            raise ValueError("model-fit output parameter must not be empty")
        if not self.parameters:
            raise ValueError("model-fit parameters must not be empty")
        if any(not name for name in self.parameters):
            raise ValueError("model-fit parameter names must not be empty")
        if any(not isinstance(key, str) for key in self.model_kwargs):
            raise TypeError("model-fit constructor kwargs must use string keys")
        if (
            any(parameter.has_array_value for parameter in self.parameters.values())
            and self.broadcast_dim is None
        ):
            raise ValueError(
                "array-valued model-fit parameters require a broadcast dimension"
            )
        return self

    def _model(self):
        model_type = getattr(erlab.analysis.fit.models, self.model)
        return model_type(**dict(self.model_kwargs))

    def _runtime_parameters(self, data: xr.DataArray) -> dict[str, typing.Any]:
        parameters: dict[str, typing.Any] = {}
        for name, parameter in self.parameters.items():
            if parameter.expr is not None:
                parameters[name] = {"expr": parameter.expr}
                continue
            if parameter.value is None:  # pragma: no cover - validation guard.
                raise ValueError(f"Model-fit parameter {name!r} has no value")
            value = _model_fit_runtime_value(
                parameter.value,
                data=data,
                broadcast_dim=self.broadcast_dim,
            )
            if (
                parameter.minimum is None
                and parameter.maximum is None
                and parameter.vary
            ):
                parameters[name] = value
                continue
            entry: dict[str, typing.Any] = {"value": value}
            if parameter.minimum is not None:
                entry["min"] = _model_fit_runtime_value(
                    parameter.minimum,
                    data=data,
                    broadcast_dim=self.broadcast_dim,
                )
            if parameter.maximum is not None:
                entry["max"] = _model_fit_runtime_value(
                    parameter.maximum,
                    data=data,
                    broadcast_dim=self.broadcast_dim,
                )
            if not parameter.vary:
                entry["vary"] = False
            parameters[name] = entry
        return parameters

    @property
    def output_name(self) -> str:
        suffix = "stderr" if self.output == "stderr" else "values"
        return f"{self.parameter}_{suffix}"

    def preferred_replay_output_name(self) -> str:
        return "parameter_stderr" if self.output == "stderr" else "parameter_values"

    def preferred_replay_input_name(self) -> str:
        return "fit_data"

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        if self.fit_dim not in data.dims:
            raise ValueError(
                f"Model-fit dimension {self.fit_dim!r} was not found in data"
            )
        mean_dim = self.fit_dim if isinstance(self.fit_dim, str) else (self.fit_dim,)
        fit_data = data / data.mean(mean_dim) if self.normalize else data
        fit_result = fit_data.xlm.modelfit(
            self.fit_dim,
            model=self._model(),
            params=self._runtime_parameters(data),
            method=self.method,
        )
        result_variable = (
            "modelfit_stderr" if self.output == "stderr" else "modelfit_coefficients"
        )
        output = fit_result[result_variable].sel(param=self.parameter, drop=True)
        if self.output == "stderr":
            output = output.fillna(0.0)
        return output.rename(self.output_name)

    def derivation_label(self) -> str:
        output = "standard errors" if self.output == "stderr" else "values"
        return f"Fit {self.model} and extract {self.parameter!r} parameter {output}"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        model_kwargs_values = typing.cast(
            "dict[typing.Hashable, typing.Any]", dict(self.model_kwargs)
        )
        model_kwargs = erlab.interactive.utils.format_call_kwargs(model_kwargs_values)
        model_code = f"era.fit.models.{self.model}({model_kwargs})"
        parameters_code = _model_fit_parameters_code(
            self.parameters,
            input_name=input_name,
            broadcast_dim=self.broadcast_dim,
        )
        fit_input = (
            f"({input_name} / {input_name}.mean("
            f"{_provenance_value_code(self.fit_dim)}))"
            if self.normalize
            else input_name
        )
        lines = [
            f"{fit_input}.xlm.modelfit(",
            f"    {_provenance_value_code(self.fit_dim)},",
        ]
        model_line = f"    model={model_code},"
        if len(model_line) <= 88:
            lines.append(model_line)
        else:
            lines.append(f"    model=era.fit.models.{self.model}(")
            lines.extend(
                "        "
                + erlab.interactive.utils.format_call_kwargs({key: value})
                + ","
                for key, value in model_kwargs_values.items()
            )
            lines.append("    ),")
        parameter_lines = parameters_code.splitlines()
        lines.append(f"    params={parameter_lines[0]}")
        lines.extend(f"    {line}" for line in parameter_lines[1:-1])
        lines.append(f"    {parameter_lines[-1]},")
        lines.extend((f"    method={self.method!r},", ")"))
        result_variable = (
            "modelfit_stderr" if self.output == "stderr" else "modelfit_coefficients"
        )
        lines[-1] += f".{result_variable}.sel("
        lines.extend((f"    param={self.parameter!r},", "    drop=True,", ")"))
        if self.output == "stderr":
            lines[-1] += ".fillna(0.0)"
        lines[-1] += f".rename({self.output_name!r})"
        return "\n".join(lines)

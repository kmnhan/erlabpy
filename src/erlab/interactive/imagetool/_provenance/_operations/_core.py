"""Script-backed and model-fitting provenance operations."""

from __future__ import annotations

import typing

import numpy as np
import pydantic
import xarray as xr

import erlab
from erlab.interactive.imagetool._provenance._code import (
    _format_selection_expr,
    _provenance_value_code,
)
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    NullableProvenanceHashable,
    ProvenanceHashable,
    ProvenanceMapping,
    ToolProvenanceOperation,
)

if typing.TYPE_CHECKING:
    from collections.abc import Collection, Hashable, Mapping


class ScriptCodeOperation(ToolProvenanceOperation):
    op: typing.Literal["script_code"] = "script_code"
    label: str
    code: str | None
    copyable: bool = True
    visible: bool = True
    # This is a code-generation hint. Serialized values do not establish origin or
    # trust.
    uses_implicit_framework_imports: bool = pydantic.Field(
        default=False,
        exclude_if=lambda value: not value,
    )

    live_applicable: typing.ClassVar[bool] = False

    def apply(self, data: xr.DataArray) -> xr.DataArray:
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
    output: typing.Literal["value", "stderr", "value_valid_stderr"] = "value"
    broadcast_dim: NullableProvenanceHashable = None
    normalize: bool = False
    weighting: typing.Literal["none", "uncertainty"] = "none"
    scale_covar: bool = True
    uncertainty_sel: ProvenanceMapping = pydantic.Field(default_factory=dict)
    uncertainty_isel: ProvenanceMapping = pydantic.Field(default_factory=dict)

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
        if self.weighting == "none" and (self.uncertainty_sel or self.uncertainty_isel):
            raise ValueError(
                "Unweighted model fits cannot define uncertainty selections"
            )
        return self

    def input_slots(self) -> tuple[str, ...]:
        return ("uncertainty",) if self.weighting == "uncertainty" else ()

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

    def _fit_result(
        self,
        data: xr.DataArray,
        *,
        uncertainty: xr.DataArray | None,
        authorization: object | None = None,
    ) -> xr.Dataset:
        if self.fit_dim not in data.dims:
            raise ValueError(
                f"Model-fit dimension {self.fit_dim!r} was not found in data"
            )
        mean_dim = self.fit_dim if isinstance(self.fit_dim, str) else (self.fit_dim,)
        mean = data.mean(mean_dim)
        fit_data = data / mean if self.normalize else data
        weights: xr.DataArray | None = None
        if uncertainty is not None:
            if self.uncertainty_sel:
                uncertainty = uncertainty.sel(self.uncertainty_sel)
            if self.uncertainty_isel:
                uncertainty = uncertainty.isel(self.uncertainty_isel)
            uncertainty = uncertainty.broadcast_like(data)
            if self.normalize:
                uncertainty = uncertainty / abs(mean)
            weights = 1 / uncertainty
        from erlab.interactive._code_trust import execution_capability_allows
        from erlab.interactive.imagetool._provenance._trust import (
            provenance_operation_code_trust_entries,
        )

        entries = provenance_operation_code_trust_entries(
            self,
            location_prefix="operation",
        )
        if not execution_capability_allows(authorization, entries):
            raise PermissionError("Model-fit parameter expressions are not authorized")
        return fit_data.xlm.modelfit(
            self.fit_dim,
            model=self._model(),
            params=self._runtime_parameters(data),
            method=self.method,
            weights=weights,
            scale_covar=self.scale_covar,
        )

    def _output_from_fit(self, fit_result: xr.Dataset) -> xr.DataArray:
        values = fit_result.modelfit_coefficients.sel(
            param=self.parameter,
            drop=True,
        )
        stderr = fit_result.modelfit_stderr.sel(
            param=self.parameter,
            drop=True,
        )
        if self.output == "stderr":
            output = stderr.where(np.isfinite(stderr) & (stderr > 0))
        elif self.output == "value_valid_stderr":
            output = values.where(
                np.isfinite(values) & np.isfinite(stderr) & (stderr > 0)
            )
        else:
            output = values
        return output.rename(self.output_name)

    def apply(
        self,
        data: xr.DataArray,
        *,
        authorization: object | None = None,
    ) -> xr.DataArray:
        if self.weighting != "none":
            raise ValueError("Weighted model fits require a bound uncertainty input")
        return self._output_from_fit(
            self._fit_result(data, uncertainty=None, authorization=authorization)
        )

    def apply_with_inputs(
        self,
        data: xr.DataArray,
        inputs: Mapping[str, xr.DataArray],
        *,
        authorization: object | None = None,
    ) -> xr.DataArray:
        return self._output_from_fit(
            self._fit_result(
                data,
                uncertainty=inputs["uncertainty"],
                authorization=authorization,
            )
        )

    def derivation_label(self) -> str:
        if self.output == "stderr":
            output = "standard errors"
        elif self.output == "value_valid_stderr":
            output = "values with valid standard errors"
        else:
            output = "values"
        return f"Fit {self.model} and extract {self.parameter!r} parameter {output}"

    def _uncertainty_expression_code(
        self,
        input_name: str,
        uncertainty_name: str,
    ) -> str:
        expression = uncertainty_name
        if self.uncertainty_sel:
            expression = _format_selection_expr(
                expression,
                "sel",
                self.uncertainty_sel,
            )
        if self.uncertainty_isel:
            expression = _format_selection_expr(
                expression,
                "isel",
                self.uncertainty_isel,
            )
        expression += f".broadcast_like({input_name})"
        if self.normalize:
            fit_dim = _provenance_value_code(self.fit_dim)
            expression = f"({expression} / abs({input_name}.mean({fit_dim})))"
        return expression

    def _fit_expression_code(
        self,
        input_name: str,
        *,
        uncertainty_name: str | None,
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
        lines.append(f"    method={self.method!r},")
        if uncertainty_name is not None:
            uncertainty_expression = self._uncertainty_expression_code(
                input_name,
                uncertainty_name,
            )
            lines.append(f"    weights=1 / ({uncertainty_expression}),")
        if not self.scale_covar or uncertainty_name is not None:
            lines.append(f"    scale_covar={self.scale_covar!r},")
        lines.append(")")
        return "\n".join(lines)

    def _parameter_expression_code(
        self,
        input_name: str,
        *,
        uncertainty_name: str | None,
    ) -> str:
        if self.output == "value_valid_stderr":
            raise NotImplementedError
        lines = self._fit_expression_code(
            input_name,
            uncertainty_name=uncertainty_name,
        ).splitlines()
        result_variable = (
            "modelfit_stderr" if self.output == "stderr" else "modelfit_coefficients"
        )
        lines[-1] += f".{result_variable}.sel("
        lines.extend((f"    param={self.parameter!r},", "    drop=True,", ")"))
        if self.output == "stderr":
            lines[-1] += ".where("
            lines.extend(
                (
                    "    lambda error: error.notnull()",
                    "    & (error > 0)",
                    '    & (error < float("inf")),',
                    ")",
                )
            )
        lines[-1] += f".rename({self.output_name!r})"
        return "\n".join(lines)

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        del source_name
        if self.weighting != "none":
            raise NotImplementedError
        return self._parameter_expression_code(input_name, uncertainty_name=None)

    def expression_code_with_inputs(
        self,
        input_name: str,
        inputs: Mapping[str, str],
        *,
        source_name: str | None = None,
    ) -> str:
        del source_name
        uncertainty_name = (
            inputs["uncertainty"] if self.weighting == "uncertainty" else None
        )
        return self._parameter_expression_code(
            input_name,
            uncertainty_name=uncertainty_name,
        )

    def _statement_replay_code_with_inputs(
        self,
        input_name: str,
        inputs: Mapping[str, str],
        *,
        output_name: str,
        source_name: str | None = None,
        reserved_names: Collection[str] = (),
    ) -> str:
        del source_name
        unavailable = {input_name, output_name, *inputs.values(), *reserved_names}

        def available_name(base: str) -> str:
            name = base
            suffix = 2
            while name in unavailable:
                name = f"{base}_{suffix}"
                suffix += 1
            unavailable.add(name)
            return name

        fit_name = available_name("parameter_fit")
        values_name = available_name("fit_parameter_values")
        stderr_name = available_name("fit_parameter_stderr")
        uncertainty_name = (
            inputs["uncertainty"] if self.weighting == "uncertainty" else None
        )
        fit_expression = self._fit_expression_code(
            input_name,
            uncertainty_name=uncertainty_name,
        )
        return "\n".join(
            (
                f"{fit_name} = {fit_expression}",
                f"{values_name} = {fit_name}.modelfit_coefficients.sel(",
                f"    param={self.parameter!r},",
                "    drop=True,",
                ")",
                f"{stderr_name} = {fit_name}.modelfit_stderr.sel(",
                f"    param={self.parameter!r},",
                "    drop=True,",
                ")",
                f"{output_name} = {values_name}.where(",
                f"    {values_name}.notnull()",
                f'    & (abs({values_name}) < float("inf"))',
                f"    & {stderr_name}.notnull()",
                f"    & ({stderr_name} > 0)",
                f'    & ({stderr_name} < float("inf"))',
                f").rename({self.output_name!r})",
            )
        )

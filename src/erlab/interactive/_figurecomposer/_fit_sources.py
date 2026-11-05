"""Immutable fit-result snapshots used by Figure Composer."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import typing

import numpy as np
import xarray as xr

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class FigureFitParameterSourceSnapshot:
    """Saved parameter values and standard errors without an executable model."""

    dataset: xr.Dataset
    _provenance_json: str | None = None

    @classmethod
    def from_dataset(
        cls,
        result_ds: xr.Dataset,
        *,
        provenance_spec: Mapping[str, typing.Any] | None = None,
    ) -> FigureFitParameterSourceSnapshot:
        data = result_ds[["modelfit_coefficients", "modelfit_stderr"]].copy(deep=True)
        data.attrs = {}
        errors = data.modelfit_stderr
        data["modelfit_stderr"] = errors.where(np.isfinite(errors) & (errors > 0))
        if provenance_spec is not None:
            from erlab.interactive.imagetool._provenance._model import (
                ReplayStep,
                parse_tool_provenance_spec,
            )
            from erlab.interactive.imagetool._provenance._operations import (
                ScriptCodeOperation,
            )

            spec = parse_tool_provenance_spec(dict(provenance_spec))
            if spec is None or spec.active_name is None:
                provenance_spec = None
            else:
                name = spec.active_name
                operation = ScriptCodeOperation(
                    label="Extract fitted parameter values and standard errors",
                    code=(
                        f"{name} = {name}[\n"
                        '    ["modelfit_coefficients", "modelfit_stderr"]\n'
                        "]\n"
                        f'{name}["modelfit_stderr"] = {name}.modelfit_stderr.where(\n'
                        f"    ({name}.modelfit_stderr > 0)\n"
                        f'    & (abs({name}.modelfit_stderr) < float("inf"))\n'
                        ")\n"
                        f"{name}.attrs = {{}}"
                    ),
                )
                spec = spec.model_copy(
                    update={"steps": (*spec.steps, ReplayStep(operation=operation))}
                )
                provenance_spec = json.loads(
                    erlab.interactive.utils._model_dump_json(spec)
                )
        return cls(
            data,
            None if provenance_spec is None else json.dumps(dict(provenance_spec)),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.dataset.param.values)

    @property
    def dimension(self) -> str | None:
        return next(
            (
                str(dim)
                for dim in self.dataset.modelfit_coefficients.dims
                if dim != "param"
            ),
            None,
        )

    @property
    def native_dataset(self) -> xr.Dataset:
        return self.dataset.copy(deep=True)

    @property
    def provenance_spec(self) -> dict[str, typing.Any] | None:
        return (
            None if self._provenance_json is None else json.loads(self._provenance_json)
        )


@dataclasses.dataclass(frozen=True)
class FigureFitSourceSnapshot:
    """A detached fit result with safe access to its saved numeric arrays."""

    blob: bytes
    _provenance_json: str | None = None
    supports_resampling: bool = False
    supports_confidence: bool = False
    independent_vars: tuple[str, ...] = ()
    model_name: str = ""
    component_names: tuple[str, ...] = ()

    @classmethod
    def from_dataset(
        cls,
        result_ds: xr.Dataset,
        *,
        provenance_spec: Mapping[str, typing.Any] | None = None,
    ) -> FigureFitSourceSnapshot:
        """Capture a complete fit result without retaining its mutable objects."""
        blob = erlab.interactive.utils._serialize_fit_dataset_blob(result_ds).tobytes()
        result_name = next(
            name for name in result_ds if str(name).endswith("modelfit_results")
        )
        result = result_ds[result_name].compute().item()
        independent_vars = tuple(result.model.independent_vars)
        return cls(
            blob,
            None if provenance_spec is None else json.dumps(dict(provenance_spec)),
            len(independent_vars) == 1,
            bool(result.errorbars)
            and result.covar is not None
            and bool(np.isfinite(result.covar).all())
            and result.nfree > 0
            and all(
                parameter.stderr is not None and bool(np.isfinite(parameter.stderr))
                for parameter in result.params.values()
            ),
            independent_vars,
            str(result.model.name),
            tuple(
                dict.fromkeys(
                    str(component.prefix or component._name)
                    for component in result.model.components
                )
            ),
        )

    @property
    def native_dataset(self) -> xr.Dataset:
        """Read saved arrays without loading an lmfit model or Python callable."""
        with xr.open_dataset(io.BytesIO(self.blob), engine="h5netcdf") as stored:
            native = stored.drop_vars(
                [name for name in stored if str(name).endswith("modelfit_results")]
            ).load()
        from erlab.interactive.imagetool import _serialization

        name = erlab.interactive.utils._fit_dataset_private_coord_data_name(native)
        if name is not None:
            native = _serialization.restore_private_coords(native, name)
        return native

    @property
    def provenance_spec(self) -> dict[str, typing.Any] | None:
        return (
            None if self._provenance_json is None else json.loads(self._provenance_json)
        )

    def result_dataset(self) -> xr.Dataset:
        """Decode the result after the caller validates its execution capability."""
        return erlab.interactive.utils._deserialize_fit_dataset_blob(
            np.frombuffer(self.blob, dtype=np.uint8)
        )

    def to_payload(self) -> dict[str, typing.Any]:
        """Return an opaque JSON-compatible persistence payload."""
        return {
            "blob": base64.b64encode(self.blob).decode("ascii"),
            "provenance": self._provenance_json,
            "supports_resampling": self.supports_resampling,
            "supports_confidence": self.supports_confidence,
            "independent_vars": self.independent_vars,
            "model_name": self.model_name,
            "component_names": self.component_names,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, typing.Any]) -> FigureFitSourceSnapshot:
        """Restore bytes without executing the stored result."""
        return cls(
            base64.b64decode(payload["blob"], validate=True),
            payload.get("provenance"),
            bool(payload.get("supports_resampling", False)),
            bool(payload.get("supports_confidence", False)),
            tuple(payload.get("independent_vars", ())),
            str(payload.get("model_name", "")),
            tuple(payload.get("component_names", ())),
        )

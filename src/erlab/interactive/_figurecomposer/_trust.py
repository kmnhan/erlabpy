"""Figure Composer adapter for the general executable-code trust service."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from erlab.interactive._code_trust._core import CodeTrustEntry


from erlab.interactive._code_trust import create_entry
from erlab.interactive._figurecomposer._model._gridspec import _gridspec_all_axes_ids
from erlab.interactive._figurecomposer._model._sources import _valid_source_variable
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationKind,
    FigureRecipeState,
)
from erlab.interactive._figurecomposer._operations._method._catalog import _method_spec
from erlab.interactive._figurecomposer._operations._method._state import (
    _method_has_transform_control,
)


def _figure_axes_namespace(recipe: FigureRecipeState) -> dict[str, object]:
    if recipe.setup.layout_mode == "gridspec":
        return {
            "layout_mode": "gridspec",
            "axes_ids": list(_gridspec_all_axes_ids(recipe.setup)),
        }
    return {
        "layout_mode": "subplots",
        "shape": [recipe.setup.nrows, recipe.setup.ncols],
    }


def _figure_operation_code_trust_entries(
    recipe: FigureRecipeState,
    operation_index: int,
    *,
    location_prefix: str,
    include_inactive: bool,
    axes_namespace: dict[str, object] | None = None,
):
    operation = recipe.operations[operation_index]
    if not include_inactive and not operation.enabled:
        return ()
    is_custom_code = (
        operation.kind == FigureOperationKind.CUSTOM and operation.code.strip()
    )
    is_custom_transform = (
        operation.kind == FigureOperationKind.METHOD
        and operation.method_transform_expression.strip()
        and _method_has_transform_control(_method_spec(operation))
        and (include_inactive or operation.method_transform == "custom")
    )
    if not is_custom_code and not is_custom_transform:
        return ()
    if axes_namespace is None:
        axes_namespace = _figure_axes_namespace(recipe)
    location = f"{location_prefix}/operations/{operation_index}"
    common_context = {
        "axes": operation.axes.model_dump(mode="json"),
        "enabled": operation.enabled,
        "kind": operation.kind.value,
        "sources": list(operation.sources),
    }
    if is_custom_code:
        return (
            create_entry(
                "erlab.figure-composer.custom-code",
                location,
                operation.code,
                {
                    **common_context,
                    "namespace": {
                        "axes": axes_namespace,
                        "source_variables": [
                            _valid_source_variable(source.name)
                            for source in recipe.sources
                        ],
                    },
                },
            ),
        )
    return (
        create_entry(
            "erlab.figure-composer.custom-transform",
            f"{location}/transform",
            operation.method_transform_expression,
            {
                **common_context,
                "axes_namespace": axes_namespace,
                "method_family": operation.method_family.value,
                "method_name": operation.method_name,
                "transform": operation.method_transform,
                "transform_x": operation.method_transform_x,
                "transform_y": operation.method_transform_y,
            },
        ),
    )


def figure_operation_execution_entries(
    recipe: FigureRecipeState,
    operation_index: int,
    *,
    location_prefix: str,
    source_payloads: Mapping[str, object] | None = None,
):
    """Return the entries that one render operation is about to execute."""
    entries = _figure_operation_code_trust_entries(
        recipe,
        operation_index,
        location_prefix=location_prefix,
        include_inactive=False,
    )
    operation = recipe.operations[operation_index]
    if operation.enabled and source_payloads:
        from erlab.interactive._figurecomposer._operations._registry import (
            source_requests,
        )

        entries += figure_source_entries(
            {
                request.alias: source_payloads[request.alias]
                for request in source_requests(operation, source_payloads)
                if request.evaluate and request.alias in source_payloads
            },
            location_prefix=location_prefix,
        )
    return entries


def figure_code_trust_entries(recipe: FigureRecipeState, *, location_prefix: str):
    """Return the Python entries stored in one Figure Composer recipe."""
    entries = []
    axes_namespace = _figure_axes_namespace(recipe)
    for index in range(len(recipe.operations)):
        entries.extend(
            _figure_operation_code_trust_entries(
                recipe,
                index,
                location_prefix=location_prefix,
                include_inactive=True,
                axes_namespace=axes_namespace,
            )
        )
    return tuple(entries)


def figure_source_entries(
    source_payloads: Mapping[str, object], *, location_prefix: str
) -> tuple[CodeTrustEntry, ...]:
    """Inspect registered source payloads without materializing their values."""
    from erlab.interactive._figurecomposer._sources import (
        OpaqueSourcePayload,
        source_spec_for_payload,
    )

    entries: list[CodeTrustEntry] = []
    for name, payload in source_payloads.items():
        if isinstance(payload, OpaqueSourcePayload):
            continue
        inspect = source_spec_for_payload(payload).trust_entries
        if inspect is not None:
            entries.extend(inspect(payload, name, location_prefix))
    return tuple(entries)

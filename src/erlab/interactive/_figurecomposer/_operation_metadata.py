"""Qt-widget-independent metadata derived from Figure Composer operations."""

from __future__ import annotations

import enum
import typing

from erlab.interactive._figurecomposer._custom_code import (
    _custom_code_names,
    _renamed_source_loads,
)
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._state import (
    FigureMethodFamily,
    FigureOperationKind,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from erlab.interactive._figurecomposer._state import FigureOperationState


class MethodTargetDomain(enum.StrEnum):
    """Object domain targeted by one curated method operation."""

    AXES = "axes"
    FIGURE = "figure"
    NONE = "none"


_BUILTIN_METHOD_TARGET_DOMAINS = {
    **{
        (FigureMethodFamily.AXES, name): MethodTargetDomain.AXES
        for name in (
            "text",
            "plot",
            "errorbar",
            "legend",
            "axvline",
            "axhline",
            "axvspan",
            "axhspan",
            "set_xticks",
            "set_yticks",
            "set_xlim",
            "set_ylim",
            "set_title",
            "set_xlabel",
            "set_ylabel",
            "set_xscale",
            "set_yscale",
            "margins",
            "set_aspect",
            "grid",
            "tick_params",
            "set_axis_off",
            "set_axis_on",
            "invert_xaxis",
            "invert_yaxis",
        )
    },
    **{
        (FigureMethodFamily.FIGURE, name): MethodTargetDomain.FIGURE
        for name in (
            "supxlabel",
            "supylabel",
            "suptitle",
            "legend",
            "subplots_adjust",
            "set_layout_engine",
        )
    },
    **{
        (FigureMethodFamily.ERLAB, name): MethodTargetDomain.AXES
        for name in (
            "clean_labels",
            "fancy_labels",
            "integer_ticks",
            "label_subplots",
            "label_subplot_properties",
            "nice_colorbar",
            "proportional_colorbar",
            "unify_clim",
            "set_titles",
            "set_xlabels",
            "set_ylabels",
            "fermiline",
            "mark_points",
            "sizebar",
            "scale_units",
        )
    },
}
_method_target_domains = dict(_BUILTIN_METHOD_TARGET_DOMAINS)


def register_method_target_domain(
    family: FigureMethodFamily,
    name: str,
    target_domain: MethodTargetDomain,
    *,
    replace: bool = False,
) -> None:
    """Register target metadata for a dynamically declared method spec."""
    key = (family, name)
    builtin_target = _BUILTIN_METHOD_TARGET_DOMAINS.get(key)
    if builtin_target is not None and builtin_target != target_domain:
        raise ValueError(
            f"Target domain for built-in {family.value} method {name!r} must be "
            f"{builtin_target.value!r}, not {target_domain.value!r}"
        )
    existing = _method_target_domains.get(key)
    if existing is not None and existing != target_domain and not replace:
        raise ValueError(
            f"Target domain for {family.value} method {name!r} is already "
            f"{existing.value!r}"
        )
    _method_target_domains[key] = target_domain


def unregister_method_target_domain(
    family: FigureMethodFamily,
    name: str,
    target_domain: MethodTargetDomain,
) -> None:
    """Remove target metadata owned by one method catalog entry."""
    key = (family, name)
    existing = _method_target_domains.get(key)
    if existing is None:
        return
    if existing != target_domain:
        raise ValueError(
            f"Cannot unregister {family.value} method {name!r} with target "
            f"{target_domain.value!r}; registered target is {existing.value!r}"
        )
    del _method_target_domains[key]


def validate_builtin_method_target_domains(
    target_domains: dict[tuple[FigureMethodFamily, str], MethodTargetDomain],
) -> None:
    """Reject drift between the curated method catalog and Qt-free metadata."""
    if target_domains == _BUILTIN_METHOD_TARGET_DOMAINS:
        return
    missing = tuple(sorted(set(_BUILTIN_METHOD_TARGET_DOMAINS) - set(target_domains)))
    unexpected = tuple(
        sorted(set(target_domains) - set(_BUILTIN_METHOD_TARGET_DOMAINS))
    )
    mismatched = tuple(
        sorted(
            key
            for key in set(target_domains) & set(_BUILTIN_METHOD_TARGET_DOMAINS)
            if target_domains[key] != _BUILTIN_METHOD_TARGET_DOMAINS[key]
        )
    )
    raise ValueError(
        "Curated method target metadata is out of sync: "
        f"missing={missing!r}, unexpected={unexpected!r}, mismatched={mismatched!r}"
    )


def method_target_domain(family: FigureMethodFamily, name: str) -> MethodTargetDomain:
    """Return target metadata without importing Qt operation editors."""
    try:
        return _method_target_domains[(family, name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported {family.value} method: {name!r}") from exc


def _unique_source_names(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


def is_axes_plot_data_method(family: FigureMethodFamily, name: str) -> bool:
    """Return whether a method can plot values selected from recipe sources."""
    return family == FigureMethodFamily.AXES and name in {"plot", "errorbar"}


def is_axes_errorbar_data_method(family: FigureMethodFamily, name: str) -> bool:
    """Return whether a source-backed method accepts error values."""
    return family == FigureMethodFamily.AXES and name == "errorbar"


def operation_uses_axes(operation: FigureOperationState) -> bool:
    """Return whether an operation targets one or more Matplotlib axes.

    This is deliberately independent of the Qt operation registry so document
    transformations can preserve target semantics without importing editors.
    """
    if operation.kind == FigureOperationKind.METHOD:
        return (
            method_target_domain(operation.method_family, operation.method_name)
            == MethodTargetDomain.AXES
        )
    return operation.kind not in {
        FigureOperationKind.SET_PALETTE,
        FigureOperationKind.CUSTOM,
    }


def declared_operation_source_names(
    operation: FigureOperationState,
) -> tuple[str, ...]:
    """Return data sources explicitly selected by an operation's controls."""
    if operation.kind in {
        FigureOperationKind.PLOT_ARRAY,
        FigureOperationKind.PLOT_SLICES,
    }:
        return _unique_source_names(operation.sources)
    if operation.kind == FigureOperationKind.LINE:
        if len(operation.map_selections) > 1:
            return _unique_source_names(
                selection.source for selection in operation.map_selections
            )
        return (operation.line_source,) if operation.line_source is not None else ()
    if operation.kind == FigureOperationKind.PHOTON_ENERGY_OVERLAY:
        return (
            (operation.hv_overlay_source,)
            if operation.hv_overlay_source is not None
            else ()
        )
    if (
        operation.kind == FigureOperationKind.METHOD
        and is_axes_plot_data_method(operation.method_family, operation.method_name)
        and operation.method_plot_data_mode == "from_data"
    ):
        values = [operation.method_plot_x, operation.method_plot_y]
        if is_axes_errorbar_data_method(operation.method_family, operation.method_name):
            values.extend((operation.method_plot_xerr, operation.method_plot_yerr))
        return _unique_source_names(
            value.source for value in values if value is not None
        )
    return ()


def recipe_operation_source_names(
    operation: FigureOperationState,
    available_source_names: Iterable[str],
) -> tuple[str, ...]:
    """Return every recipe source read by an operation during execution."""
    names = list(declared_operation_source_names(operation))
    if operation.kind == FigureOperationKind.CUSTOM:
        loaded_names = _custom_code_names(operation.code)
        names.extend(name for name in available_source_names if name in loaded_names)
    return _unique_source_names(names)


def rename_operation_sources(
    operation: FigureOperationState, rename_map: Mapping[str, str]
) -> FigureOperationState:
    """Return an operation with every source reference renamed atomically."""
    updates: dict[str, typing.Any] = {}
    if operation.kind == FigureOperationKind.CUSTOM:
        try:
            code = _renamed_source_loads(operation.code, dict(rename_map))
        except ValueError as exc:
            raise FigureComposerInputError(
                "Could not rename source references in Python step "
                f"{operation.label!r}: {exc}."
            ) from exc
        if code != operation.code:
            updates["code"] = code
    if operation.sources:
        sources = tuple(
            rename_map.get(source_name, source_name)
            for source_name in operation.sources
        )
        if sources != operation.sources:
            updates["sources"] = sources
    if operation.map_selections:
        map_selections = tuple(
            selection.model_copy(update={"source": source})
            if (source := rename_map.get(selection.source, selection.source))
            != selection.source
            else selection
            for selection in operation.map_selections
        )
        if map_selections != operation.map_selections:
            updates["map_selections"] = map_selections
    if operation.line_source is not None:
        line_source = rename_map.get(operation.line_source, operation.line_source)
        if line_source != operation.line_source:
            updates["line_source"] = line_source
    for field in (
        "method_plot_x",
        "method_plot_y",
        "method_plot_xerr",
        "method_plot_yerr",
    ):
        state = getattr(operation, field)
        if state is not None:
            source = rename_map.get(state.source, state.source)
            if source != state.source:
                updates[field] = state.model_copy(update={"source": source})
    if operation.hv_overlay_source is not None:
        hv_overlay_source = rename_map.get(
            operation.hv_overlay_source, operation.hv_overlay_source
        )
        if hv_overlay_source != operation.hv_overlay_source:
            updates["hv_overlay_source"] = hv_overlay_source
    if not updates:
        return operation
    return operation.model_copy(update=updates)

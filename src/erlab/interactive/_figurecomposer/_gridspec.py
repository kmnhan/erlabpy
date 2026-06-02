"""GridSpec layout helpers for Figure Composer."""

from __future__ import annotations

import keyword
import re
import typing

from erlab.interactive._figurecomposer._state import (
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureSubplotsState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


def _gridspec_all_axes(
    setup: FigureSubplotsState,
) -> tuple[FigureGridSpecAxesState, ...]:
    return tuple(_iter_axes(setup.gridspec.root))


def _gridspec_all_axes_ids(setup: FigureSubplotsState) -> tuple[str, ...]:
    return tuple(axis.axes_id for axis in _gridspec_all_axes(setup))


def _gridspec_valid_axes_ids(
    setup: FigureSubplotsState, axes_ids: Iterable[str]
) -> tuple[str, ...]:
    axes_by_id = _gridspec_valid_axes_by_id(setup)
    return tuple(
        dict.fromkeys(axis_id for axis_id in axes_ids if axis_id in axes_by_id)
    )


def _gridspec_invalid_axes_ids(
    setup: FigureSubplotsState, axes_ids: Iterable[str]
) -> tuple[str, ...]:
    axes_by_id = _gridspec_valid_axes_by_id(setup)
    return tuple(
        dict.fromkeys(axis_id for axis_id in axes_ids if axis_id not in axes_by_id)
    )


def _gridspec_axes_by_id(
    setup: FigureSubplotsState,
) -> dict[str, FigureGridSpecAxesState]:
    return {axis.axes_id: axis for axis in _gridspec_all_axes(setup)}


def _gridspec_valid_axes_by_id(
    setup: FigureSubplotsState,
) -> dict[str, FigureGridSpecAxesState]:
    return {
        axis.axes_id: axis
        for _grid, axis in _iter_valid_axes_with_grid(setup.gridspec.root)
    }


def _gridspec_axis_display_name(setup: FigureSubplotsState, axes_id: str) -> str:
    axes_by_id = _gridspec_axes_by_id(setup)
    axis = axes_by_id.get(axes_id)
    if axis is None:
        return axes_id
    return axis.label.strip() or _gridspec_axis_code_names(setup).get(axes_id, axes_id)


def _gridspec_axis_display_names(
    setup: FigureSubplotsState, axes_ids: Iterable[str]
) -> tuple[str, ...]:
    return tuple(_gridspec_axis_display_name(setup, axes_id) for axes_id in axes_ids)


def _gridspec_axis_code_names(setup: FigureSubplotsState) -> dict[str, str]:
    used: set[str] = set()
    names: dict[str, str] = {}
    for index, axis in enumerate(_gridspec_all_axes(setup)):
        preferred = _sanitize_axes_name(axis.label) if axis.label else f"ax{index}"
        name = preferred or f"ax{index}"
        if name in used:
            base = name
            suffix = 1
            while f"{base}_{suffix}" in used:
                suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        names[axis.axes_id] = name
    return names


def _gridspec_root_from_subplots(setup: FigureSubplotsState) -> FigureGridSpecGridState:
    axes = [
        FigureGridSpecAxesState(
            span=FigureGridSpecSpanState(
                row_start=row,
                row_stop=row + 1,
                col_start=col,
                col_stop=col + 1,
            )
        )
        for row in range(setup.nrows)
        for col in range(setup.ncols)
    ]
    return FigureGridSpecGridState(
        grid_id="root",
        label="Root",
        nrows=setup.nrows,
        ncols=setup.ncols,
        width_ratios=setup.width_ratios,
        height_ratios=setup.height_ratios,
        axes=tuple(axes),
    )


def _gridspec_setup_from_subplots(setup: FigureSubplotsState) -> FigureSubplotsState:
    root = _gridspec_root_from_subplots(setup)
    return setup.model_copy(
        update={
            "layout_mode": "gridspec",
            "gridspec": FigureGridSpecLayoutState(root=root),
        }
    )


def _subplots_setup_from_gridspec(setup: FigureSubplotsState) -> FigureSubplotsState:
    root = setup.gridspec.root
    return setup.model_copy(
        update={
            "layout_mode": "subplots",
            "nrows": root.nrows,
            "ncols": root.ncols,
            "width_ratios": root.width_ratios,
            "height_ratios": root.height_ratios,
        }
    )


def _gridspec_grid_by_id(
    setup: FigureSubplotsState, grid_id: str
) -> FigureGridSpecGridState | None:
    return _find_grid(setup.gridspec.root, grid_id)


def _gridspec_grid_path(
    setup: FigureSubplotsState, grid_id: str
) -> tuple[FigureGridSpecGridState, ...]:
    path = _find_grid_path(setup.gridspec.root, grid_id)
    if path:
        return path
    return (setup.gridspec.root,)


def _gridspec_replace_grid(
    setup: FigureSubplotsState,
    grid_id: str,
    updater: Callable[[FigureGridSpecGridState], FigureGridSpecGridState],
) -> FigureSubplotsState:
    root = _replace_grid(setup.gridspec.root, grid_id, updater)
    return setup.model_copy(
        update={"gridspec": setup.gridspec.model_copy(update={"root": root})}
    )


def _gridspec_region_overlaps(
    grid: FigureGridSpecGridState,
    span: FigureGridSpecSpanState,
    *,
    ignore_axes_id: str | None = None,
    ignore_grid_id: str | None = None,
) -> bool:
    for axis in grid.axes:
        if axis.axes_id != ignore_axes_id and _spans_overlap(axis.span, span):
            return True
    for child in grid.child_grids:
        if (
            child.grid_id != ignore_grid_id
            and child.span is not None
            and _spans_overlap(child.span, span)
        ):
            return True
    return False


def _gridspec_region_valid(
    grid: FigureGridSpecGridState, span: FigureGridSpecSpanState
) -> bool:
    return (
        0 <= span.row_start < span.row_stop <= grid.nrows
        and 0 <= span.col_start < span.col_stop <= grid.ncols
    )


def _gridspec_has_invalid_regions(grid: FigureGridSpecGridState) -> bool:
    for axis in grid.axes:
        if not _gridspec_region_valid(grid, axis.span):
            return True
    for child in grid.child_grids:
        if child.span is None or not _gridspec_region_valid(grid, child.span):
            return True
        if _gridspec_has_invalid_regions(child):
            return True
    return False


def _iter_valid_axes_with_grid(
    grid: FigureGridSpecGridState,
) -> Iterator[tuple[FigureGridSpecGridState, FigureGridSpecAxesState]]:
    for axis in grid.axes:
        if _gridspec_region_valid(grid, axis.span):
            yield grid, axis
    for child in grid.child_grids:
        if child.span is not None and _gridspec_region_valid(grid, child.span):
            yield from _iter_valid_axes_with_grid(child)


def _gridspec_remove_region(
    setup: FigureSubplotsState, grid_id: str, region_id: str
) -> FigureSubplotsState:
    def updater(grid: FigureGridSpecGridState) -> FigureGridSpecGridState:
        return grid.model_copy(
            update={
                "axes": tuple(axis for axis in grid.axes if axis.axes_id != region_id),
                "child_grids": tuple(
                    child for child in grid.child_grids if child.grid_id != region_id
                ),
            }
        )

    return _gridspec_replace_grid(setup, grid_id, updater)


def _gridspec_update_axis_label(
    setup: FigureSubplotsState, axes_id: str, label: str
) -> FigureSubplotsState:
    def update_grid(grid: FigureGridSpecGridState) -> FigureGridSpecGridState:
        axes = tuple(
            axis.model_copy(update={"label": label})
            if axis.axes_id == axes_id
            else axis
            for axis in grid.axes
        )
        children = tuple(update_grid(child) for child in grid.child_grids)
        return grid.model_copy(update={"axes": axes, "child_grids": children})

    root = update_grid(setup.gridspec.root)
    return setup.model_copy(
        update={"gridspec": setup.gridspec.model_copy(update={"root": root})}
    )


def _gridspec_region_label(
    setup: FigureSubplotsState, grid: FigureGridSpecGridState, region_id: str
) -> str:
    code_names = _gridspec_axis_code_names(setup)
    for axis in grid.axes:
        if axis.axes_id == region_id:
            return axis.label.strip() or code_names.get(axis.axes_id, axis.axes_id)
    for child in grid.child_grids:
        if child.grid_id == region_id:
            return child.label.strip() or "Grid"
    return region_id


def _gridspec_span_code(
    span: FigureGridSpecSpanState, grid: FigureGridSpecGridState
) -> str:
    return (
        "["
        f"{_slice_code(span.row_start, span.row_stop, grid.nrows)}, "
        f"{_slice_code(span.col_start, span.col_stop, grid.ncols)}"
        "]"
    )


def _gridspec_axis_code_tuple(
    setup: FigureSubplotsState, axes_ids: Iterable[str]
) -> tuple[str, ...]:
    code_names = _gridspec_axis_code_names(setup)
    return tuple(code_names[axes_id] for axes_id in axes_ids if axes_id in code_names)


def _iter_axes(grid: FigureGridSpecGridState) -> Iterator[FigureGridSpecAxesState]:
    yield from grid.axes
    for child in grid.child_grids:
        yield from _iter_axes(child)


def _find_grid(
    grid: FigureGridSpecGridState, grid_id: str
) -> FigureGridSpecGridState | None:
    if grid.grid_id == grid_id:
        return grid
    for child in grid.child_grids:
        found = _find_grid(child, grid_id)
        if found is not None:
            return found
    return None


def _find_grid_path(
    grid: FigureGridSpecGridState, grid_id: str
) -> tuple[FigureGridSpecGridState, ...]:
    if grid.grid_id == grid_id:
        return (grid,)
    for child in grid.child_grids:
        path = _find_grid_path(child, grid_id)
        if path:
            return (grid, *path)
    return ()


def _replace_grid(
    grid: FigureGridSpecGridState,
    grid_id: str,
    updater: Callable[[FigureGridSpecGridState], FigureGridSpecGridState],
) -> FigureGridSpecGridState:
    if grid.grid_id == grid_id:
        return updater(grid)
    children = tuple(
        _replace_grid(child, grid_id, updater) for child in grid.child_grids
    )
    return grid.model_copy(update={"child_grids": children})


def _spans_overlap(
    first: FigureGridSpecSpanState, second: FigureGridSpecSpanState
) -> bool:
    return not (
        first.row_stop <= second.row_start
        or second.row_stop <= first.row_start
        or first.col_stop <= second.col_start
        or second.col_stop <= first.col_start
    )


def _sanitize_axes_name(label: str) -> str:
    cleaned = re.sub(r"\W+", "_", label.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        return ""
    if cleaned[0].isdigit():
        cleaned = f"ax_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def _slice_code(start: int, stop: int, size: int) -> str:
    if stop == start + 1:
        return str(start)
    if start == 0 and stop == size:
        return ":"
    if start == 0:
        return f":{stop}"
    return f"{start}:{stop}"

"""Virtualize and materialize workspace payloads that are not open as windows."""

from __future__ import annotations

import base64
import binascii
import collections
import collections.abc
import contextlib
import copy
import html
import json
import logging
import typing
import warnings

import numpy as np
from qtpy import QtGui

import erlab
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.slicer
import erlab.interactive.imagetool.viewer_linking
from erlab.interactive.imagetool import _serialization
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME, ImageTool
from erlab.interactive.imagetool._provenance._model import (
    ScriptInputDataRole,
    ToolProvenanceSpec,
    parse_tool_provenance_operation,
    parse_tool_provenance_spec,
    require_live_source_spec,
)
from erlab.interactive.imagetool._provenance._operations import (
    ImageToolSelectionSourceBinding,
)
from erlab.interactive.imagetool.manager._widgets import _curve_preview_data
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    import os
    import pathlib
    from collections.abc import Hashable, Mapping, Sequence

    import h5py
    import xarray as xr

    from erlab.interactive.imagetool.manager._workspace._controller import (
        _WorkspaceController,
    )
    from erlab.interactive.imagetool.manager._workspace._loading import _WorkspaceLoader
    from erlab.interactive.imagetool.viewer import ImageSlicerArea
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")


logger = logging.getLogger(__name__)
_PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES = 128 * 1024 * 1024
_PENDING_WORKSPACE_PREVIEW_DISPLAY_AXES = (0, 1)


class _PendingWorkspaceLinkTarget:
    def __init__(
        self, array_slicer: erlab.interactive.imagetool.slicer.ArraySlicer
    ) -> None:
        self.array_slicer = array_slicer

    @property
    def data(self) -> xr.DataArray:
        return self.array_slicer._obj


class _PendingWorkspacePayloads:
    """Own lazy payload metadata, previews, materialization, and link state."""

    def __init__(
        self, loader: _WorkspaceLoader, controller: _WorkspaceController
    ) -> None:
        self._loader = loader
        self._manager = loader._manager
        self._controller = controller

    @staticmethod
    def _pending_workspace_data_with_saved_dim_order(
        data: xr.DataArray, attrs: Mapping[typing.Any, typing.Any]
    ) -> xr.DataArray:
        raw_state = attrs.get("itool_state")
        if isinstance(raw_state, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                raw_state = raw_state.decode()
        if not isinstance(raw_state, str):
            return data
        try:
            state = json.loads(raw_state)
        except Exception:
            logger.debug("Ignoring invalid pending ImageTool state", exc_info=True)
            return data
        if not isinstance(state, collections.abc.Mapping):
            return data
        slice_state = state.get("slice")
        if not isinstance(slice_state, collections.abc.Mapping):
            return data
        raw_dims = slice_state.get("dims")
        if not isinstance(raw_dims, (list, tuple)):
            return data

        saved_dims = tuple(str(dim) for dim in raw_dims)
        # Match ArraySlicer dimension promotion before applying its saved order.
        normalized = erlab.utils.array._make_dims_uniform(data)
        dims_by_text = {str(dim): dim for dim in normalized.dims}
        if (
            len(saved_dims) != normalized.ndim
            or len(dims_by_text) != normalized.ndim
            or set(saved_dims) != set(dims_by_text)
        ):
            logger.debug(
                "Ignoring incompatible saved pending ImageTool dimension order %s "
                "for data dims %s",
                saved_dims,
                normalized.dims,
            )
            return data
        ordered_dims = tuple(dims_by_text[dim] for dim in saved_dims)
        if ordered_dims == normalized.dims:
            return normalized
        try:
            return normalized.transpose(*ordered_dims, transpose_coords=True)
        except ValueError:
            logger.debug(
                "Could not apply saved pending ImageTool dimension order %s",
                saved_dims,
                exc_info=True,
            )
            return data

    @staticmethod
    def _apply_pending_workspace_filter(
        data: xr.DataArray, filter_payload: object
    ) -> xr.DataArray:
        if filter_payload is None:
            return data
        if not isinstance(filter_payload, dict):
            raise TypeError("Invalid pending filter operation")
        operation = parse_tool_provenance_operation(filter_payload)
        source_shape = tuple(data.sizes[dim] for dim in data.dims)
        filtered = operation.apply(data, parent_data=data)
        if filtered.ndim != data.ndim or set(filtered.dims) != set(data.dims):
            raise ValueError("Pending filter changed data dimensions")
        expected_shape = tuple(data.sizes[dim] for dim in filtered.dims)
        if filtered.shape != expected_shape:
            raise ValueError("Pending filter changed data shape")
        data = filtered.transpose(*data.dims, transpose_coords=True)
        if data.shape != source_shape:
            raise ValueError("Pending filter result is misaligned")
        return data

    def _pending_workspace_lazy_source_data(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        data_role: ScriptInputDataRole = "displayed",
        owner_node: _ImageToolWrapper | _ManagedWindowNode | None = None,
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] | None = None,
    ) -> xr.DataArray:
        opened = self._loader._workspace_imagetool_reference_dataset(
            node,
            owner_node=owner_node,
            reference_datasets=reference_datasets,
        )
        ds = workspace_format._restore_workspace_dataset_attrs(opened.copy(deep=False))
        ds = _serialization.restore_private_coords(ds, _ITOOL_DATA_NAME)
        name = None if node.name == "" else node.name
        data = self._loader._workspace_imagetool_payload_data(ds).rename(name)
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            attrs = ds.attrs
        data = self._pending_workspace_data_with_saved_dim_order(data, attrs)
        data = erlab.utils.array._restore_nonuniform_dims(data)
        if data_role == "displayed":
            raw_state = attrs.get("itool_state")
            if isinstance(raw_state, bytes):
                with contextlib.suppress(UnicodeDecodeError):
                    raw_state = raw_state.decode()
            if isinstance(raw_state, str):
                try:
                    state = json.loads(raw_state)
                except Exception:
                    logger.debug(
                        "Ignoring invalid pending ImageTool state", exc_info=True
                    )
                else:
                    if isinstance(state, collections.abc.Mapping):
                        data = self._apply_pending_workspace_filter(
                            data, state.get("filter_operation")
                        )
        return node._finalize_script_input_data(data)

    @staticmethod
    def _pending_workspace_data_with_loaded_coords(data: xr.DataArray) -> xr.DataArray:
        loaded_coords = {
            key: coord.copy(deep=False).load() for key, coord in data.coords.items()
        }
        return data.copy(deep=False).assign_coords(loaded_coords)

    def _pending_workspace_imagetool_metadata_data(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> xr.DataArray:
        """Return pending ImageTool metadata without materializing its data window."""
        data = self._pending_workspace_lazy_source_data(
            node, data_role="source", owner_node=node
        )
        scalar_coords = {
            key: coord.copy(deep=False).load()
            for key, coord in data.coords.items()
            if isinstance(key, str) and coord.ndim == 0
        }
        return data.copy(deep=False).assign_coords(scalar_coords)

    def _pending_workspace_imagetool_info_text(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        pending = node.pending_workspace_memory_payload
        if pending is None:
            return None
        workspace_path, payload_path = pending
        try:
            ds = self._loader._read_workspace_imagetool_payload_dataset(
                workspace_path, payload_path, load_data=False
            )
            try:
                ds = _serialization.restore_private_coords(ds, _ITOOL_DATA_NAME)
                if _ITOOL_DATA_NAME not in ds:
                    return None
                name = None if node.name == "" else node.name
                data = ds[_ITOOL_DATA_NAME].rename(name)
                attrs = node.pending_workspace_payload_attrs
                if attrs is None:
                    attrs = ds.attrs
                data = self._pending_workspace_data_with_saved_dim_order(data, attrs)
                data = erlab.utils.array._restore_nonuniform_dims(data)
                additional_info = [f"Added {node.added_time_display}"]
                try:
                    metadata_data = self._pending_workspace_data_with_loaded_coords(
                        data
                    )
                    text = erlab.utils.formatting.format_darr_html(
                        metadata_data,
                        show_size=True,
                        additional_info=additional_info,
                    )
                except Exception:
                    logger.debug(
                        "Failed to load coordinates for pending workspace metadata",
                        exc_info=True,
                    )
                    text = erlab.utils.formatting.format_darr_html(
                        data,
                        show_size=True,
                        load_values=False,
                        additional_info=additional_info,
                    )
                return erlab.interactive.utils._apply_qt_accent_color(text)
            finally:
                ds.close()
        except Exception:
            logger.debug(
                "Failed to read metadata for pending workspace payload %s from %s",
                payload_path,
                workspace_path,
                exc_info=True,
            )
            return None

    def _pending_workspace_tool_info_text(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        if node.pending_workspace_tool_payload is None:
            return None
        attrs = node.pending_workspace_payload_attrs or {}
        tool_name = self._loader._workspace_tool_display_name_from_attrs(attrs)
        lines = [f"<p><b>{html.escape(tool_name)}</b></p>"]
        data_name = workspace_format._decode_workspace_attr_text(
            attrs.get("tool_data_name")
        )
        if data_name is not None and data_name != "<none-value>":
            lines.append(f"<p>Data: <code>{html.escape(data_name)}</code></p>")
        source_state = workspace_format._decode_workspace_attr_text(
            attrs.get("tool_source_state")
        )
        if source_state is not None:
            lines.append(f"<p>Source: {html.escape(source_state)}</p>")
        lines.append(f"<p>Added {html.escape(node.added_time_display)}</p>")
        return erlab.interactive.utils._apply_qt_accent_color("".join(lines))

    def _pending_workspace_info_text(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        match node.pending_workspace_payload_kind:
            case "imagetool":
                return self._pending_workspace_imagetool_info_text(node)
            case "tool":
                return self._pending_workspace_tool_info_text(node)
            case _:
                return None

    @staticmethod
    def _pending_workspace_tool_preview_image(
        node: _ImageToolWrapper | _ManagedWindowNode,
    ) -> tuple[float, QtGui.QPixmap] | None:
        if node.pending_workspace_tool_payload is None:
            return None
        attrs = node.pending_workspace_payload_attrs or {}
        encoded = attrs.get("figure_composer_preview_cache_png")
        if isinstance(encoded, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                encoded = encoded.decode()
        if not isinstance(encoded, str) or not encoded:
            return None
        try:
            png_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (ValueError, binascii.Error):
            return None
        pixmap = QtGui.QPixmap()
        if not pixmap.loadFromData(png_bytes, "PNG") or pixmap.isNull():
            return None
        width = pixmap.width()
        if width <= 0:
            return None
        return pixmap.height() / width, pixmap

    @staticmethod
    def _pending_preview_axis_state(
        shape: tuple[int, ...],
        state: Mapping[str, typing.Any],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[bool, ...]] | None:
        slice_state = state.get("slice")
        if not isinstance(slice_state, collections.abc.Mapping):
            return None
        state_bins = slice_state.get("bins")
        state_indices = slice_state.get("indices")
        if not isinstance(state_bins, list) or not state_bins:
            return None
        if not isinstance(state_indices, list):
            state_indices = []

        cursor = int(state.get("current_cursor", 0))
        cursor = min(max(cursor, 0), len(state_bins) - 1)
        bins_for_cursor = state_bins[cursor]
        indices_for_cursor = (
            state_indices[cursor] if cursor < len(state_indices) else []
        )
        if not isinstance(bins_for_cursor, list):
            return None
        if not isinstance(indices_for_cursor, list):
            indices_for_cursor = []

        center_indices = erlab.interactive.imagetool.slicer._center_indices_for_shape(
            shape
        )
        bins: list[int] = []
        indices: list[int] = []
        for axis, size in enumerate(shape):
            try:
                axis_bin = int(bins_for_cursor[axis])
            except (IndexError, TypeError, ValueError):
                axis_bin = 1
            if axis_bin < 1:
                axis_bin = 1
            try:
                axis_index = int(indices_for_cursor[axis])
            except (IndexError, TypeError, ValueError):
                axis_index = center_indices[axis]
            indices.append(
                erlab.interactive.imagetool.slicer._normalized_axis_index(
                    axis_index, size
                )
            )
            bins.append(axis_bin)
        return tuple(indices), tuple(bins), tuple(axis_bin != 1 for axis_bin in bins)

    @staticmethod
    def _pending_preview_dataset_dims(
        dataset: h5py.Dataset, state_dims: tuple[str, ...]
    ) -> tuple[tuple[str, ...], int | None] | None:
        stack_axis: int | None = None
        expected_dims = state_dims
        if len(state_dims) != dataset.ndim:
            if (
                dataset.ndim == 1
                and len(state_dims) == 2
                and state_dims.count("stack_dim") == 1
            ):
                stack_axis = state_dims.index("stack_dim")
                expected_dims = tuple(dim for dim in state_dims if dim != "stack_dim")
            else:
                return None

        dataset_dims: list[str] = []
        used_dims: set[str] = set()
        for axis, dim in enumerate(dataset.dims):
            dim_keys = tuple(str(key) for key in dim)
            candidates: list[str] = []
            for expected_dim in expected_dims:
                if expected_dim in used_dims:
                    continue
                if expected_dim in dim_keys or (
                    expected_dim.endswith("_idx")
                    and expected_dim.removesuffix("_idx") in dim_keys
                ):
                    candidates.append(expected_dim)
            candidates = list(dict.fromkeys(candidates))
            positional_dim = expected_dims[axis]
            if positional_dim in candidates:
                dataset_dim = positional_dim
            elif len(candidates) == 1:
                dataset_dim = candidates[0]
            else:
                return None
            dataset_dims.append(dataset_dim)
            used_dims.add(dataset_dim)
        stored_dims = tuple(dataset_dims)
        if set(expected_dims) != set(stored_dims):
            return None
        return stored_dims, stack_axis

    def _pending_workspace_imagetool_preview_curve(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pending = node.pending_workspace_memory_payload
        if pending is None:
            return None
        attrs = node.pending_workspace_payload_attrs
        workspace_path, payload_path = pending
        try:
            with (
                workspace_arrays._workspace_file_lock(workspace_path),
                h5py.File(workspace_path, "r") as h5_file,
            ):
                group = h5_file.get(payload_path.strip("/"))
                if not isinstance(group, h5py.Group):
                    return None
                dataset = group.get(_ITOOL_DATA_NAME)
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
                    return None
                if dataset.dtype.kind not in "biuf":
                    return None

                group_attrs = (
                    workspace_arrays._h5py_attrs_to_dict(group.attrs)
                    if attrs is None
                    else attrs
                )
                state = group_attrs.get("itool_state")
                if isinstance(state, bytes):
                    state = state.decode()
                if isinstance(state, str):
                    state = json.loads(state)
                if not isinstance(state, collections.abc.Mapping):
                    return None
                slice_state = state.get("slice")
                if not isinstance(slice_state, collections.abc.Mapping):
                    return None
                raw_state_dims = slice_state.get("dims")
                if not isinstance(raw_state_dims, (list, tuple)):
                    return None
                state_dims = tuple(str(dim) for dim in raw_state_dims)
                resolved_dims = self._pending_preview_dataset_dims(dataset, state_dims)
                if resolved_dims is None:
                    return None
                stored_dims, promoted_stack_axis = resolved_dims
                if promoted_stack_axis is None:
                    return None

                data_nbytes = int(dataset.size) * int(np.dtype(dataset.dtype).itemsize)
                if data_nbytes > _PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES:
                    return None
                y_values = np.asarray(dataset[()]).reshape(-1)

                x_values: np.ndarray | None = None
                coord_dataset = None
                stored_dim = stored_dims[0]
                coord_names = (stored_dim,)
                if stored_dim.endswith("_idx"):
                    coord_names = (stored_dim, stored_dim.removesuffix("_idx"))
                for coord_name in coord_names:
                    with contextlib.suppress(KeyError, TypeError, RuntimeError):
                        maybe_coord = dataset.dims[0][coord_name]
                        if isinstance(maybe_coord, h5py.Dataset):
                            coord_dataset = maybe_coord
                            break
                if coord_dataset is None:
                    for coord_name in coord_names:
                        maybe_coord = group.get(coord_name)
                        if isinstance(maybe_coord, h5py.Dataset):
                            coord_dataset = maybe_coord
                            break
                if (
                    coord_dataset is not None
                    and coord_dataset.shape == dataset.shape
                    and coord_dataset.dtype.kind in "biuf"
                ):
                    coord_nbytes = int(coord_dataset.size) * int(
                        np.dtype(coord_dataset.dtype).itemsize
                    )
                    if (
                        data_nbytes + coord_nbytes
                        <= _PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES
                    ):
                        x_values = np.asarray(coord_dataset[()]).reshape(-1)
                if x_values is None:
                    x_values = np.arange(y_values.size, dtype=np.float64)
                return _curve_preview_data(x_values, y_values)
        except Exception:
            logger.debug(
                "Failed to render curve preview for pending workspace payload "
                "%s from %s",
                payload_path,
                workspace_path,
                exc_info=True,
            )
            return None

    def _pending_workspace_imagetool_preview_image(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[float, QtGui.QPixmap] | None:
        pending = node.pending_workspace_memory_payload
        if pending is None:
            return None
        attrs = node.pending_workspace_payload_attrs
        workspace_path, payload_path = pending
        try:
            with (
                workspace_arrays._workspace_file_lock(workspace_path),
                h5py.File(workspace_path, "r") as h5_file,
            ):
                group = h5_file.get(payload_path.strip("/"))
                if not isinstance(group, h5py.Group):
                    return None
                dataset = group.get(_ITOOL_DATA_NAME)
                if not isinstance(dataset, h5py.Dataset):
                    return None
                if dataset.dtype.kind not in "biuf":
                    return None

                group_attrs = (
                    workspace_arrays._h5py_attrs_to_dict(group.attrs)
                    if attrs is None
                    else attrs
                )
                state = group_attrs.get("itool_state")
                if isinstance(state, bytes):
                    state = state.decode()
                if isinstance(state, str):
                    state = json.loads(state)
                if not isinstance(state, collections.abc.Mapping):
                    return None

                slice_state = state.get("slice")
                if not isinstance(slice_state, collections.abc.Mapping):
                    return None
                raw_state_dims = slice_state.get("dims")
                if not isinstance(raw_state_dims, (list, tuple)):
                    return None
                state_dims = tuple(str(dim) for dim in raw_state_dims)
                resolved_dims = self._pending_preview_dataset_dims(dataset, state_dims)
                if resolved_dims is None:
                    return None
                stored_dims, promoted_stack_axis = resolved_dims
                if promoted_stack_axis is not None:
                    return None

                stored_shape = tuple(int(size) for size in dataset.shape)
                if len(stored_shape) < 2:
                    return None
                slicer_shape = tuple(
                    stored_shape[stored_dims.index(dim)] for dim in state_dims
                )
                axis_state = self._pending_preview_axis_state(slicer_shape, state)
                if axis_state is None:
                    return None
                indices, bins, binned = axis_state
                hidden_axes = (
                    erlab.interactive.imagetool.slicer._hidden_axes_for_display(
                        len(slicer_shape),
                        _PENDING_WORKSPACE_PREVIEW_DISPLAY_AXES,
                    )
                )
                slicer_selection, reduction_axes, _any_binned, _all_binned = (
                    erlab.interactive.imagetool.slicer._reduced_axes_selection(
                        slicer_shape,
                        hidden_axes,
                        indices,
                        bins,
                        binned,
                    )
                )
                final_ndim = sum(
                    isinstance(item, slice) for item in slicer_selection
                ) - len(reduction_axes)
                if final_ndim != 2:
                    return None

                stored_selection: list[slice | int] = [slice(None)] * dataset.ndim
                for slicer_axis, item in enumerate(slicer_selection):
                    stored_axis = stored_dims.index(state_dims[slicer_axis])
                    stored_selection[stored_axis] = item

                read_nbytes = int(np.dtype(dataset.dtype).itemsize)
                for size, item in zip(stored_shape, stored_selection, strict=True):
                    if isinstance(item, slice):
                        start = 0 if item.start is None else item.start
                        stop = size if item.stop is None else item.stop
                        read_nbytes *= max(0, stop - start)
                if read_nbytes > _PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES:
                    return None

                data = np.asarray(dataset[tuple(stored_selection)])

            stored_remaining_dims = tuple(
                dim
                for dim, item in zip(stored_dims, stored_selection, strict=True)
                if isinstance(item, slice)
            )
            slicer_remaining_dims = tuple(
                dim
                for dim, item in zip(state_dims, slicer_selection, strict=True)
                if isinstance(item, slice)
            )
            if stored_remaining_dims != slicer_remaining_dims:
                axis_order = tuple(
                    stored_remaining_dims.index(dim) for dim in slicer_remaining_dims
                )
                data = data.transpose(axis_order)
            if reduction_axes:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.filterwarnings(
                        "ignore", r"Mean of empty slice", RuntimeWarning
                    )
                    data = np.nanmean(data, axis=reduction_axes)
            if data.ndim != 2 or data.size == 0:
                return None
            if 0 in _PENDING_WORKSPACE_PREVIEW_DISPLAY_AXES:
                data = data.T
            data = erlab.interactive.imagetool.slicer._display_safe_values(data)

            pixmap = self._workspace_pending_preview_pixmap(data, state)
            if pixmap is None or pixmap.isNull():
                return None
            pixmap = pixmap.transformed(QtGui.QTransform().scale(1.0, -1.0))
            if pixmap.width() <= 0:
                return None
            return pixmap.height() / pixmap.width(), pixmap
        except Exception:
            logger.debug(
                "Failed to render preview for pending workspace payload %s from %s",
                payload_path,
                workspace_path,
                exc_info=True,
            )
            return None

    @staticmethod
    def _workspace_pending_preview_pixmap(
        data: np.ndarray, state: Mapping[str, typing.Any]
    ) -> QtGui.QPixmap | None:
        color = state.get("color", {})
        if not isinstance(color, collections.abc.Mapping):
            color = {}
        image_item = erlab.interactive.colors.BetterImageItem()
        image_item.set_colormap(
            color.get("cmap", "viridis"),
            float(color.get("gamma", 1.0)),
            bool(color.get("reverse", False)),
            high_contrast=bool(color.get("high_contrast", False)),
            zero_centered=bool(color.get("zero_centered", False)),
            update=False,
        )
        levels_locked = bool(color.get("levels_locked", False))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"All-NaN (slice|axis) encountered", RuntimeWarning
            )
            image_item.setImage(data, autoLevels=not levels_locked)
        levels = color.get("levels")
        if levels_locked and isinstance(levels, (list, tuple)) and len(levels) == 2:
            image_item.setLevels((float(levels[0]), float(levels[1])), update=True)
        return image_item.getPixmap()

    @staticmethod
    def _workspace_imagetool_name_from_attrs(
        attrs: Mapping[typing.Any, typing.Any],
    ) -> str:
        name = attrs.get("itool_name", "")
        if isinstance(name, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                name = name.decode()
        return "" if name is None else str(name)

    def _workspace_tool_source_metadata(
        self, attrs: Mapping[str, typing.Any]
    ) -> tuple[
        ToolProvenanceSpec | None,
        ImageToolSelectionSourceBinding | None,
        bool,
        _ManagedWindowNode._source_state_type,
    ]:
        source_spec = None
        raw_source_spec = attrs.get(erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR)
        if raw_source_spec is not None:
            try:
                source_spec = require_live_source_spec(
                    parse_tool_provenance_spec(
                        typing.cast(
                            "Mapping[str, typing.Any]",
                            json.loads(raw_source_spec),
                        )
                    )
                )
            except Exception:
                logger.warning(
                    "Ignoring invalid saved tool source provenance",
                    exc_info=True,
                )
        source_binding = None
        if (
            source_spec is None
            and erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR in attrs
        ):
            try:
                source_binding = ImageToolSelectionSourceBinding.model_validate(
                    typing.cast(
                        "Mapping[str, typing.Any]",
                        json.loads(
                            attrs[erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR]
                        ),
                    )
                )
            except Exception:
                logger.warning(
                    "Ignoring invalid saved tool source binding",
                    exc_info=True,
                )
        return (
            source_spec,
            source_binding,
            bool(attrs.get(erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR)),
            self._loader._workspace_tool_source_state_from_attrs(attrs),
        )

    def _register_pending_workspace_imagetool(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        node_path: str | None,
        kwargs: dict[str, typing.Any],
        pending_workspace_memory_payload: tuple[str | os.PathLike[str], str],
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> int | str:
        name = self._workspace_imagetool_name_from_attrs(ds.attrs)
        if parent_target is not None:
            parent_node = self._manager._node_for_target(parent_target)
            node = _ManagedWindowNode(
                self._manager,
                self._manager._next_node_uid(kwargs.get("uid")),
                parent_node.uid,
                None,
                window_kind="imagetool",
                name=name,
                provenance_spec=kwargs.get("provenance_spec"),
                source_spec=kwargs.get("source_spec"),
                source_binding=kwargs.get("source_binding"),
                source_auto_update=bool(kwargs.get("source_auto_update", False)),
                source_state=typing.cast(
                    "_ManagedWindowNode._source_state_type",
                    kwargs.get("source_state", "fresh"),
                ),
                output_id=kwargs.get("output_id"),
                snapshot_token=kwargs.get("snapshot_token"),
                source_snapshot_token=kwargs.get("source_snapshot_token"),
                created_time=kwargs.get("created_time"),
                note=kwargs.get("note"),
            )
            node.set_pending_workspace_memory_payload(
                *pending_workspace_memory_payload,
                payload_attrs=ds.attrs,
            )
            self._manager._register_child_node(node)
            if node.output_id is not None and parent_node.tool_window is not None:
                parent_node.tool_window._register_output_imagetool_target(
                    node.output_id, node.uid
                )
            self._manager.tree_view.childtool_added(node.uid, parent_target)
            self._controller._mark_node_added(node.uid)
            self._loader._record_workspace_loaded_node_target(
                ds, node.uid, loaded_targets_by_uid
            )
            return node.uid

        root_kwargs = self._loader._root_workspace_imagetool_kwargs(ds, kwargs)
        preferred_index: int | None = None
        if node_path is not None and "/" not in node_path:
            with contextlib.suppress(ValueError):
                preferred_index = int(node_path)
            if preferred_index is not None and preferred_index < 0:
                preferred_index = None
        if (
            preferred_index is None
            or preferred_index in self._manager._tool_graph.root_wrappers
        ):
            preferred_index = int(self._manager.next_idx)

        wrapper = _ImageToolWrapper(
            self._manager,
            preferred_index,
            self._manager._next_node_uid(root_kwargs.get("uid")),
            None,
            watched_var=root_kwargs.get("watched_var"),
            watched_workspace_link_id=root_kwargs.get("watched_workspace_link_id"),
            watched_source_label=root_kwargs.get("watched_source_label"),
            watched_source_uid=root_kwargs.get("watched_source_uid"),
            watched_connected=bool(root_kwargs.get("watched_connected", True)),
            source_input_ndim=typing.cast(
                "int | None", root_kwargs.get("source_input_ndim")
            ),
            provenance_spec=root_kwargs.get("provenance_spec"),
            source_spec=root_kwargs.get("source_spec"),
            source_binding=root_kwargs.get("source_binding"),
            source_auto_update=bool(root_kwargs.get("source_auto_update", False)),
            source_state=typing.cast(
                "_ManagedWindowNode._source_state_type",
                root_kwargs.get("source_state", "fresh"),
            ),
            snapshot_token=root_kwargs.get("snapshot_token"),
            source_snapshot_token=root_kwargs.get("source_snapshot_token"),
            created_time=root_kwargs.get("created_time"),
            note=root_kwargs.get("note"),
            name=name,
        )
        wrapper.set_pending_workspace_memory_payload(
            *pending_workspace_memory_payload,
            payload_attrs=ds.attrs,
        )
        self._manager._register_root_wrapper(wrapper)
        self._manager.tree_view.imagetool_added(preferred_index)
        self._controller._mark_node_added(wrapper.uid)
        self._loader._record_workspace_loaded_node_target(
            ds, preferred_index, loaded_targets_by_uid
        )
        return preferred_index

    def _materialize_pending_workspace_payload(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> bool:
        if node.pending_workspace_payload_kind == "tool":
            return self._materialize_pending_workspace_tool_payload(node)

        pending = node.pending_workspace_memory_payload
        if pending is None:
            return True

        workspace_path, payload_path = pending
        origin = self._controller._active_managed_window() or self._manager
        try:
            name = node.name
            internal_workspace_operation = (
                self._manager._workspace_state.loading_depth > 0
                or self._manager._workspace_state.saving_depth > 0
                or self._manager._workspace_state.closing_document
            )
            wait_context = (
                contextlib.nullcontext()
                if internal_workspace_operation
                else erlab.interactive.utils.wait_dialog(
                    origin, "Loading ImageTool data..."
                )
            )
            with wait_context:
                ds = self._loader._read_workspace_imagetool_payload_dataset(
                    workspace_path, payload_path, load_data=True
                )
                name = None if name == "" else name
                ds = ds.copy(deep=False)
                pending_attrs = node.pending_workspace_payload_attrs
                if pending_attrs is not None:
                    ds.attrs.update(pending_attrs)
                ds.attrs["itool_name"] = "" if name is None else name
                ds = self._loader._dataset_without_missing_workspace_colormap(
                    ds, payload_path
                )
                with self._controller._workspace_load_context():
                    if node.imagetool is None:
                        tool = ImageTool.from_dataset(
                            ds,
                            _in_manager=True,
                            _defer_state_refresh=True,
                            _defer_secondary_plots=True,
                            options_model=(self._manager.effective_interactive_options),
                        )
                        node.window = tool
                        if node.parent_uid is not None:
                            self._manager._parent_node(node).add_child_reference(
                                node.uid, tool
                            )
                    else:
                        state = copy.deepcopy(node.slicer_area.state)
                        data = ds[_ITOOL_DATA_NAME].rename(name)
                        node.slicer_area.set_data(data, auto_compute=False)
                        node.slicer_area.state = state
                    node.clear_pending_workspace_payload()
                    node.update_title()
                    self._loader._sync_materialized_workspace_link_group(node)
                self._manager.tree_view.refresh(node.uid)
                self._manager._update_info(uid=node.uid)
                return True
        except Exception:
            logger.exception(
                "Error while loading pending workspace payload %s from %s",
                payload_path,
                workspace_path,
                extra={"suppress_ui_alert": True},
            )
            self._manager._show_operation_error(
                "Could Not Load ImageTool Data",
                "The saved data for this ImageTool could not be read from the "
                "workspace file. The file may have moved, been deleted, or changed "
                "on disk.",
            )
            return False

    def _materialize_pending_workspace_tool_payload(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> bool:
        pending = node.pending_workspace_tool_payload
        if pending is None:
            return True

        workspace_path, payload_path = pending
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] = {}
        try:
            ds = self._loader._read_workspace_tool_payload_dataset(
                workspace_path, payload_path
            )
            pending_attrs = node.pending_workspace_payload_attrs
            if pending_attrs is not None:
                ds = ds.copy(deep=False)
                ds.attrs.update(pending_attrs)

            source_parent_data, tool_data_reference_resolver = (
                self._loader._workspace_tool_restore_references(
                    ds,
                    parent_target=node.parent_uid,
                    owner_node=node,
                    reference_datasets=reference_datasets,
                    resolver_error_types=(Exception,),
                    log_resolver_errors=True,
                )
            )

            with self._controller._workspace_load_context():
                tool: erlab.interactive.utils.ToolWindow = (
                    erlab.interactive.utils.ToolWindow.from_dataset(
                        ds,
                        _source_parent_data=source_parent_data,
                        _tool_data_reference_resolver=tool_data_reference_resolver,
                        _defer_restore_work=True,
                    )
                )
                if node.parent_uid is None:
                    self._loader._require_workspace_root_tool_is_figure(tool)
                    node.window = tool
                    if not tool._tool_display_name:
                        tool._tool_display_name = node.name
                    self._manager._configure_figure_tool(node, tool)
                    self._manager._figure_collection.sync(select_uid=None)
                else:
                    parent = self._manager._node_for_target(node.parent_uid)
                    node.window = tool
                    if not tool._tool_display_name:
                        tool._tool_display_name = parent.name
                    parent_uid = parent.uid

                    def _source_parent_fetcher() -> xr.DataArray:
                        return self._loader._workspace_tool_reference_source_data(
                            parent_uid, owner_node=node
                        )

                    def _input_provenance_parent_fetcher() -> ToolProvenanceSpec | None:
                        return self._manager._node_for_target(
                            parent_uid
                        ).displayed_provenance_spec

                    tool.set_source_parent_fetcher(_source_parent_fetcher)
                    tool.set_input_provenance_parent_fetcher(
                        _input_provenance_parent_fetcher
                    )
                    parent.add_child_reference(node.uid, tool)
                node._set_workspace_tool_data_references(
                    type(tool)._saved_tool_data_references(ds)
                )
                node._adopt_workspace_reference_datasets(reference_datasets)
                reference_datasets = {}
                node.clear_pending_workspace_payload()
        except Exception:
            self._loader._close_workspace_reference_datasets(reference_datasets)
            logger.exception(
                "Error while loading pending workspace ToolWindow payload %s from %s",
                payload_path,
                workspace_path,
                extra={"suppress_ui_alert": True},
            )
            self._manager._show_operation_error(
                "Could Not Open Saved Tool",
                "This saved tool could not be opened from the workspace file. "
                "The tool data may be unavailable or incompatible with this version.",
            )
            return False
        else:
            self._manager.tree_view.refresh(node.uid)
            self._manager._update_info(uid=node.uid)
            return True

    def _has_pending_workspace_linked_slicers(
        self, source: ImageSlicerArea, *, color: bool
    ) -> bool:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return False
        node = self._manager.node_from_slicer_area(source)
        if (
            node is None
            or not node.is_imagetool
            or node.workspace_link_key is None
            or (color and not node.workspace_link_colors)
        ):
            return False
        return any(
            linked_node.uid != node.uid
            and linked_node.is_imagetool
            and linked_node.workspace_link_key == node.workspace_link_key
            and linked_node.pending_workspace_memory_payload is not None
            for linked_node in self._manager._tool_graph.nodes.values()
        )

    def _sync_pending_workspace_linked_slicers(
        self,
        source: ImageSlicerArea,
        funcname: str,
        arguments: dict[str, typing.Any],
        source_dims: tuple[Hashable, ...],
        indices: bool,
        steps: bool,
        color: bool,
        transaction_id: str | None,
        keep_pending: bool,
    ) -> None:
        if not self._has_pending_workspace_linked_slicers(source, color=color):
            return
        node = self._manager.node_from_slicer_area(source)
        if node is None or node.workspace_link_key is None:
            return

        for linked_node in self._manager._tool_graph.nodes.values():
            if (
                linked_node.uid == node.uid
                or not linked_node.is_imagetool
                or linked_node.workspace_link_key != node.workspace_link_key
                or linked_node.pending_workspace_memory_payload is None
            ):
                continue
            if self._apply_pending_workspace_link_operation(
                source,
                linked_node,
                funcname,
                arguments,
                source_dims,
                indices,
                steps,
                transaction_id,
                keep_pending,
            ):
                self._controller._mark_workspace_dirty(uid=linked_node.uid, state=True)

    def _sync_pending_workspace_linked_manual_limits(
        self,
        source: ImageSlicerArea,
        manual_limits: Mapping[str, Sequence[typing.Any]],
    ) -> None:
        if not self._has_pending_workspace_linked_slicers(source, color=False):
            return
        node = self._manager.node_from_slicer_area(source)
        if node is None or node.workspace_link_key is None:
            return

        normalized_limits = self._normalized_pending_workspace_manual_limits(
            manual_limits
        )
        for linked_node in self._manager._tool_graph.nodes.values():
            if (
                linked_node.uid == node.uid
                or not linked_node.is_imagetool
                or linked_node.workspace_link_key != node.workspace_link_key
                or linked_node.pending_workspace_memory_payload is None
            ):
                continue
            if self._update_pending_workspace_manual_limits(
                linked_node, normalized_limits
            ):
                self._controller._mark_workspace_dirty(uid=linked_node.uid, state=True)

    @staticmethod
    def _normalized_pending_workspace_manual_limits(
        manual_limits: Mapping[str, Sequence[typing.Any]],
    ) -> dict[str, list[float]]:
        normalized: dict[str, list[float]] = {}
        for dim, limits in manual_limits.items():
            try:
                normalized[str(dim)] = [float(limits[0]), float(limits[1])]
            except (IndexError, TypeError, ValueError):
                continue
        return normalized

    def _update_pending_workspace_manual_limits(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        manual_limits: Mapping[str, list[float]],
    ) -> bool:
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            return False
        raw_state = attrs.get("itool_state")
        if isinstance(raw_state, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                raw_state = raw_state.decode()
        if not isinstance(raw_state, str):
            return False
        try:
            state = typing.cast("dict[str, typing.Any]", json.loads(raw_state))
        except Exception:
            logger.debug("Ignoring invalid pending ImageTool state", exc_info=True)
            return False
        slice_state = state.get("slice")
        valid_dims: tuple[str, ...] = ()
        if isinstance(slice_state, collections.abc.Mapping):
            raw_dims = slice_state.get("dims")
            if isinstance(raw_dims, (list, tuple)):
                valid_dims = tuple(str(dim) for dim in raw_dims)
        if not valid_dims:
            pending = node.pending_workspace_memory_payload
            if pending is None:
                return False
            workspace_path, payload_path = pending
            try:
                ds = self._loader._read_workspace_imagetool_payload_dataset(
                    workspace_path, payload_path, load_data=False
                )
                try:
                    ds = _serialization.restore_private_coords(ds, _ITOOL_DATA_NAME)
                    if _ITOOL_DATA_NAME not in ds:
                        return False
                    valid_dims = tuple(str(dim) for dim in ds[_ITOOL_DATA_NAME].dims)
                finally:
                    ds.close()
            except Exception:
                logger.debug(
                    "Could not update manual limits for pending workspace payload "
                    "%s from %s",
                    payload_path,
                    workspace_path,
                    exc_info=True,
                )
                return False

        valid_dim_set = set(valid_dims)
        updated_limits = {
            dim: limits for dim, limits in manual_limits.items() if dim in valid_dim_set
        }
        if state.get("manual_limits", {}) == updated_limits:
            return False
        state["manual_limits"] = updated_limits
        updated_attrs = dict(attrs)
        updated_attrs["itool_state"] = json.dumps(state)
        node.update_pending_workspace_payload_attrs(updated_attrs)
        return True

    def _apply_pending_workspace_link_operation(
        self,
        source: ImageSlicerArea,
        node: _ImageToolWrapper | _ManagedWindowNode,
        funcname: str,
        arguments: dict[str, typing.Any],
        source_dims: tuple[Hashable, ...],
        indices: bool,
        steps: bool,
        transaction_id: str | None,
        keep_pending: bool,
    ) -> bool:
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            return False
        raw_state = attrs.get("itool_state")
        if isinstance(raw_state, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                raw_state = raw_state.decode()
        if not isinstance(raw_state, str):
            return False
        try:
            state = typing.cast("dict[str, typing.Any]", json.loads(raw_state))
        except Exception:
            logger.debug("Ignoring invalid pending ImageTool state", exc_info=True)
            return False

        pending = node.pending_workspace_memory_payload
        if pending is None:
            return False
        workspace_path, payload_path = pending
        try:
            cache = node._pending_workspace_link_slicer_cache
            if (
                cache is not None
                and cache[0] == pending
                and erlab.interactive.utils.qt_is_valid(cache[2])
            ):
                _, cached_state, array_slicer = cache
                if cached_state != raw_state:
                    array_slicer.state = copy.deepcopy(state["slice"])
                    node._pending_workspace_link_slicer_cache = (
                        pending,
                        raw_state,
                        array_slicer,
                    )
            else:
                node._clear_pending_workspace_link_slicer_cache()
                ds = self._loader._read_workspace_imagetool_payload_dataset(
                    workspace_path, payload_path, load_data=False
                )
                try:
                    ds = _serialization.restore_private_coords(ds, _ITOOL_DATA_NAME)
                    if _ITOOL_DATA_NAME not in ds:
                        return False
                    target_data = ds[_ITOOL_DATA_NAME]
                    target_data = self._pending_workspace_data_with_saved_dim_order(
                        target_data, attrs
                    )
                    array_slicer = erlab.interactive.imagetool.slicer.ArraySlicer(
                        target_data,
                        node,
                        display_value_abs_limit=(
                            source.array_slicer.display_value_abs_limit
                        ),
                    )
                    node._pending_workspace_link_slicer_cache = (
                        pending,
                        raw_state,
                        array_slicer,
                    )
                    array_slicer.state = copy.deepcopy(state["slice"])
                finally:
                    ds.close()

            target = _PendingWorkspaceLinkTarget(array_slicer)
            converter = erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy()
            converted_args = converter.convert_args(
                source,
                typing.cast("typing.Any", target),
                dict(arguments),
                source_dims,
                indices,
                steps,
                transaction_id,
                keep_pending,
            )
            if converted_args is None:
                return False
            for key in (
                "__slicer_skip_sync",
                "__slicer_transaction_id",
                "__slicer_keep_pending",
            ):
                converted_args.pop(key, None)
            original_state_json = json.dumps(state)
            if not self._update_pending_link_state_for_operation(
                state, array_slicer, source, funcname, converted_args
            ):
                return False
            state_json = json.dumps(state)
            if state_json == original_state_json:
                return False
            updated_attrs = dict(attrs)
            updated_attrs["itool_state"] = state_json
            node.update_pending_workspace_payload_attrs(updated_attrs)
            node._pending_workspace_link_slicer_cache = (
                pending,
                state_json,
                array_slicer,
            )
        except Exception:
            node._clear_pending_workspace_link_slicer_cache()
            logger.debug(
                "Could not apply linked operation %s to pending workspace payload %s "
                "from %s",
                funcname,
                payload_path,
                workspace_path,
                exc_info=True,
            )
            return False
        else:
            return True

    def _update_pending_link_state_for_operation(
        self,
        state: dict[str, typing.Any],
        array_slicer: erlab.interactive.imagetool.slicer.ArraySlicer,
        source: ImageSlicerArea,
        funcname: str,
        arguments: dict[str, typing.Any],
    ) -> bool:
        if funcname in {"refresh", "refresh_current"}:
            return False
        if funcname == "view_all":
            state["manual_limits"] = {}
            return True
        if funcname == "center_all_cursors":
            for cursor in range(array_slicer.n_cursors):
                array_slicer.center_cursor(cursor, update=False)
            state["slice"] = array_slicer.state
            return True
        if funcname == "center_cursor":
            cursor = int(state.get("current_cursor", 0)) % array_slicer.n_cursors
            array_slicer.center_cursor(cursor, update=False)
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_current_cursor":
            state["current_cursor"] = int(arguments["cursor"]) % array_slicer.n_cursors
            return True
        if funcname == "set_axis_inverted":
            dim = str(arguments["dim"])
            if dim not in {str(dim_name) for dim_name in array_slicer._obj.dims}:
                return False
            axis_inversions = dict(state.get("axis_inversions", {}))
            if bool(arguments["inverted"]):
                axis_inversions[dim] = True
            else:
                axis_inversions.pop(dim, None)
            state["axis_inversions"] = axis_inversions
            return True
        if funcname == "swap_axes":
            array_slicer.swap_axes(int(arguments["ax1"]), int(arguments["ax2"]))
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_index":
            cursor = self._pending_link_cursor_argument(state, arguments)
            array_slicer.set_index(
                cursor,
                int(arguments["axis"]),
                int(arguments["value"]),
                update=False,
            )
            state["slice"] = array_slicer.state
            return True
        if funcname == "step_index":
            cursor = self._pending_link_cursor_argument(state, arguments)
            array_slicer.step_index(
                cursor,
                int(arguments["axis"]),
                int(arguments["value"]),
                update=False,
            )
            state["slice"] = array_slicer.state
            return True
        if funcname == "step_index_all":
            for cursor in range(array_slicer.n_cursors):
                array_slicer.step_index(
                    cursor,
                    int(arguments["axis"]),
                    int(arguments["value"]),
                    update=False,
                )
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_value":
            cursor = self._pending_link_cursor_argument(state, arguments)
            array_slicer.set_value(
                cursor,
                int(arguments["axis"]),
                float(arguments["value"]),
                update=False,
                uniform=bool(arguments.get("uniform", False)),
            )
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_bin":
            cursor = self._pending_link_cursor_argument(state, arguments)
            bins: list[int | None] = [None] * array_slicer._obj.ndim
            bins[int(arguments["axis"])] = int(arguments["value"])
            array_slicer.set_bins(cursor, bins, update=False)
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_bin_all":
            bins = [None] * array_slicer._obj.ndim
            bins[int(arguments["axis"])] = int(arguments["value"])
            for cursor in range(array_slicer.n_cursors):
                array_slicer.set_bins(cursor, bins, update=False)
            state["slice"] = array_slicer.state
            return True
        if funcname == "add_cursor":
            current_cursor = (
                int(state.get("current_cursor", 0)) % array_slicer.n_cursors
            )
            array_slicer.add_cursor(current_cursor, update=False)
            cursor_colors = list(state.get("cursor_colors", ()))
            color = arguments.get("color")
            if color is None:
                cursor_colors.append(
                    self._pending_link_default_cursor_color(
                        source, array_slicer.n_cursors - 1, cursor_colors
                    )
                )
            else:
                cursor_colors.append(QtGui.QColor(color).name())
            state["cursor_colors"] = cursor_colors
            state["current_cursor"] = array_slicer.n_cursors - 1
            state["slice"] = array_slicer.state
            return True
        if funcname == "remove_cursor":
            if array_slicer.n_cursors == 1:
                return False
            index = int(arguments["index"]) % array_slicer.n_cursors
            array_slicer.remove_cursor(index, update=False)
            cursor_colors = list(state.get("cursor_colors", ()))
            if index < len(cursor_colors):
                cursor_colors.pop(index)
            current_cursor = int(state.get("current_cursor", 0))
            if current_cursor == index:
                if index == 0:
                    current_cursor = 1
                current_cursor -= 1
            elif current_cursor > index:
                current_cursor -= 1
            state["cursor_colors"] = cursor_colors
            state["current_cursor"] = max(0, current_cursor)
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_colormap":
            self._update_pending_link_colormap_state(state, arguments)
            return True
        if funcname == "toggle_snap":
            slice_state = dict(state["slice"])
            value = arguments.get("value")
            snap_to_data = (
                not bool(slice_state.get("snap_to_data", False))
                if value is None
                else bool(value)
            )
            array_slicer.snap_to_data = snap_to_data
            slice_state["snap_to_data"] = snap_to_data
            state["slice"] = slice_state
            return True
        return False

    @staticmethod
    def _pending_link_cursor_argument(
        state: Mapping[str, typing.Any], arguments: Mapping[str, typing.Any]
    ) -> int:
        cursor = arguments.get("cursor")
        if cursor is None:
            cursor = state.get("current_cursor", 0)
        return int(cursor)

    @staticmethod
    def _pending_link_default_cursor_color(
        source: ImageSlicerArea, index: int, cursor_colors: list[str]
    ) -> str:
        colors = [QtGui.QColor(color).name() for color in source.COLORS]
        color = colors[index % len(colors)]
        while color in cursor_colors:
            color = colors[index % len(colors)]
            if len(cursor_colors) + 1 > len(colors):
                break
            index += 1
        return color

    @staticmethod
    def _update_pending_link_colormap_state(
        state: dict[str, typing.Any], arguments: Mapping[str, typing.Any]
    ) -> None:
        color_state = dict(state.get("color", {}))
        for key in ("cmap", "gamma", "reverse", "high_contrast", "zero_centered"):
            value = arguments.get(key)
            if value is not None:
                color_state[key] = value
        levels_locked = arguments.get("levels_locked")
        if levels_locked is not None:
            color_state["levels_locked"] = bool(levels_locked)
        levels = arguments.get("levels")
        if levels is not None:
            color_state["levels"] = [float(levels[0]), float(levels[1])]
        state["color"] = color_state

    def _register_pending_workspace_tool(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        pending_workspace_tool_payload: tuple[str | os.PathLike[str], str],
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> int | str:
        uid = self._manager._next_node_uid(
            self._loader._workspace_saved_uid_from_dataset(ds)
        )
        attrs = dict(ds.attrs)
        name = self._loader._workspace_tool_display_name_from_attrs(attrs)
        if parent_target is None:
            node = _ManagedWindowNode(
                self._manager,
                uid,
                None,
                None,
                window_kind="tool",
                name=name,
                snapshot_token=attrs.get("manager_node_snapshot_token"),
                source_snapshot_token=attrs.get("manager_node_source_snapshot_token"),
                created_time=attrs.get("manager_node_added_at"),
                note=attrs.get("manager_node_note"),
            )
            self._manager._register_figure_node(node)
            self._manager._figure_collection.sync(select_uid=None)
        else:
            parent_node = self._manager._node_for_target(parent_target)
            node = _ManagedWindowNode(
                self._manager,
                uid,
                parent_node.uid,
                None,
                window_kind="tool",
                name=name,
                snapshot_token=attrs.get("manager_node_snapshot_token"),
                source_snapshot_token=attrs.get("manager_node_source_snapshot_token"),
                created_time=attrs.get("manager_node_added_at"),
                note=attrs.get("manager_node_note"),
            )
            self._manager._register_child_node(node)
            self._manager.tree_view.childtool_added(node.uid, parent_target)
        source_spec, source_binding, auto_update, source_state = (
            self._workspace_tool_source_metadata(attrs)
        )
        node.set_restored_source_binding_metadata(
            source_spec,
            source_binding,
            auto_update=auto_update,
            state=source_state,
        )
        node.set_pending_workspace_payload(
            "tool",
            *pending_workspace_tool_payload,
            payload_attrs=attrs,
        )
        self._controller._mark_node_added(node.uid)
        self._loader._record_workspace_loaded_node_target(
            ds, node.uid, loaded_targets_by_uid
        )
        return node.uid

    @staticmethod
    def _validate_pending_workspace_tool_dataset(
        ds: xr.Dataset, *, parent_target: int | str | None
    ) -> None:
        tool_cls = erlab.interactive.utils.ToolWindow._saved_tool_class_from_dataset(ds)
        if not tool_cls.can_save_and_load():
            raise TypeError("Saved tool class cannot be restored")
        tool_state = ds.attrs.get("tool_state")
        if not isinstance(tool_state, str):
            raise TypeError("Saved tool dataset is missing a valid tool state")
        tool_cls.StateModel.model_validate_json(tool_state)
        tool_cls._saved_tool_data_references(ds)
        if parent_target is None and tool_cls.manager_collection != "figures":
            raise ValueError("Workspace tool node has no parent")

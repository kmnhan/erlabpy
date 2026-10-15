"""Live tool outputs that can be captured as detached figure sources."""

from __future__ import annotations

import copy
import typing
from dataclasses import dataclass, field

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from erlab.interactive._fit1d import Fit1DTool
    from erlab.interactive.imagetool._provenance._model import ScriptInput
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )


@dataclass(frozen=True)
class FigureSourceCapture:
    """One captured output, with no dependency on its live window."""

    payload: object
    name: str
    label: str = ""
    provenance_spec: dict[str, typing.Any] | None = None
    selection: dict[str, typing.Any] = field(default_factory=dict)
    revision_token: str | None = None
    data_role: typing.Literal["displayed", "source"] = "displayed"
    use_node_data: bool = False


class FigureSourceProvider(typing.Protocol):
    """Advertise and capture one stable output from an open tool."""

    provider_id: str
    label: str
    payload_kind: str

    def is_ready(self, selection: Mapping[str, typing.Any] | None = None) -> bool: ...

    def capture(
        self, selection: Mapping[str, typing.Any] | None = None
    ) -> FigureSourceCapture: ...


@dataclass(frozen=True)
class FigureSourceTarget:
    """Select one output when a Manager node offers several providers."""

    node_uid: str
    provider_id: str


class ImageToolSourceProvider:
    """Adapt ImageTool data without requiring it to inherit ``ToolWindow``."""

    provider_id = "array"
    label = "ImageTool"
    payload_kind = "array"

    def __init__(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        script_input_getter: Callable[..., ScriptInput] | None = None,
    ) -> None:
        self._node = node
        self._script_input_getter = script_input_getter

    def is_ready(self, selection: Mapping[str, typing.Any] | None = None) -> bool:
        return self._node.is_imagetool

    def capture(
        self, selection: Mapping[str, typing.Any] | None = None
    ) -> FigureSourceCapture:
        from erlab.interactive._figurecomposer._model._sources import (
            _source_alias_candidate,
        )
        from erlab.interactive._figurecomposer._sources import ArraySourcePayload

        data_role = (selection or {}).get("data_role", "displayed")
        if data_role not in {"displayed", "source"}:
            raise ValueError("Unknown ImageTool source data role")
        data = self._node.data_for_role(data_role)
        script_input = (
            None
            if self._script_input_getter is None
            else self._script_input_getter(self._node, data_role=data_role)
        )
        return FigureSourceCapture(
            payload=ArraySourcePayload(data=data.copy(deep=True)),
            name=_source_alias_candidate(data)
            or ("data" if script_input is None else script_input.name),
            label=self._node.display_text
            if script_input is None
            else script_input.label,
            provenance_spec=None
            if script_input is None
            else copy.deepcopy(script_input.provenance_spec),
            selection={"data_role": data_role},
            revision_token=None
            if script_input is None
            else script_input.node_snapshot_token,
            data_role=data_role,
            use_node_data=True,
        )


class FitSourceProvider:
    """Capture the completed fit selected in Fit1D or Fit2D."""

    provider_id = "fit"
    label = "Fit"
    payload_kind = "fit"

    def __init__(self, tool: Fit1DTool) -> None:
        self._tool = tool

    def is_ready(self, selection: Mapping[str, typing.Any] | None = None) -> bool:
        return self._tool.has_figure_fit_source(
            None if not selection else dict(selection)
        )

    def capture(
        self, selection: Mapping[str, typing.Any] | None = None
    ) -> FigureSourceCapture:
        snapshot = self._tool.figure_fit_source_snapshot(
            None if not selection else dict(selection)
        )
        frozen_selection = (
            self._tool.figure_fit_selection() if selection is None else selection
        )
        return FigureSourceCapture(
            payload=snapshot,
            name=self._tool.figure_fit_source_name(),
            label=self._tool.windowTitle(),
            provenance_spec=copy.deepcopy(snapshot.provenance_spec),
            selection=copy.deepcopy(dict(frozen_selection or {})),
            revision_token=str(self._tool.provenance_revision),
        )

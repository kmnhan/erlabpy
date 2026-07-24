"""Semantic persistence for non-ImageTool plot appearance."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import typing
import weakref

import numpy as np
import pydantic
import pyqtgraph as pg
from qtpy import QtCore

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from erlab.interactive.colors import BetterColorBarItem

logger = logging.getLogger(__name__)

TOOL_VIEW_STATE_ATTR = "tool_view_state"
_TOOL_VIEW_STATE_VERSION = 1
_MAX_SAVED_COLORMAP_STOPS = 256


class _PlotStateModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore", frozen=True)


class NamedColormapState(_PlotStateModel):
    """A colormap that can be reconstructed from its registered name."""

    kind: typing.Literal["named"] = "named"
    name: str = pydantic.Field(min_length=1)
    reverse: bool = False


class GradientStopState(_PlotStateModel):
    """One normalized stop in a saved custom gradient."""

    position: float = pydantic.Field(ge=0.0, le=1.0)
    color: tuple[int, int, int, int]

    @pydantic.field_validator("position")
    @classmethod
    def _position_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("gradient stop positions must be finite")
        return value

    @pydantic.field_validator("color")
    @classmethod
    def _color_channels_must_be_bytes(
        cls, value: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("gradient color channels must be between 0 and 255")
        return value


class GradientColormapState(_PlotStateModel):
    """An explicit pyqtgraph gradient."""

    kind: typing.Literal["gradient"] = "gradient"
    mode: typing.Literal["rgb", "hsv"] = "rgb"
    stops: tuple[GradientStopState, ...] = pydantic.Field(min_length=2)


ColormapState = typing.Annotated[
    NamedColormapState | GradientColormapState,
    pydantic.Field(discriminator="kind"),
]


class PowerNormState(_PlotStateModel):
    """ERLab's power-normalization parameters."""

    kind: typing.Literal["power"] = "power"
    gamma: float = pydantic.Field(gt=0.0)
    high_contrast: bool = False
    zero_centered: bool = False

    @pydantic.field_validator("gamma")
    @classmethod
    def _gamma_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("gamma must be finite")
        return value


class PlotAppearanceState(_PlotStateModel):
    """Backend-neutral color appearance for one semantic plot."""

    colormap: ColormapState
    norm: PowerNormState | None = None
    levels: tuple[float, float] | None = None

    @pydantic.field_validator("levels")
    @classmethod
    def _validate_levels(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        lower, upper = value
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("color levels must be finite and nondecreasing")
        return (lower, upper)


class ToolViewState(_PlotStateModel):
    """Saved appearance for the registered plots in one ToolWindow."""

    version: typing.Literal[1] = 1
    plots: dict[str, PlotAppearanceState] = pydantic.Field(default_factory=dict)


def _parse_tool_view_state(payload: object) -> ToolViewState:
    if isinstance(payload, bytes):
        payload = payload.decode()
    if not isinstance(payload, str):
        raise TypeError("saved tool view state must be JSON text")
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise TypeError("saved tool view state must be a JSON object")
    if raw.get("version", _TOOL_VIEW_STATE_VERSION) != _TOOL_VIEW_STATE_VERSION:
        raise ValueError("unsupported saved tool view state version")
    raw_plots = raw.get("plots", {})
    if not isinstance(raw_plots, dict):
        raise TypeError("saved tool view-state plots must be a JSON object")

    plots: dict[str, PlotAppearanceState] = {}
    for plot_id, state in raw_plots.items():
        if not isinstance(plot_id, str) or not plot_id:
            logger.warning("Ignoring saved plot appearance with an invalid plot ID")
            continue
        try:
            plots[plot_id] = PlotAppearanceState.model_validate(state)
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid saved plot appearance for %s",
                plot_id,
                exc_info=True,
            )
    return ToolViewState(plots=plots)


def _sample_colormap(cmap: pg.ColorMap) -> GradientColormapState:
    positions, colors = cmap.getStops(pg.ColorMap.BYTE)
    positions = np.asarray(positions, dtype=float)
    colors = np.asarray(colors, dtype=np.uint8)
    if positions.size > _MAX_SAVED_COLORMAP_STOPS:
        indices = np.linspace(
            0, positions.size - 1, _MAX_SAVED_COLORMAP_STOPS, dtype=int
        )
        positions = positions[indices]
        colors = colors[indices]
    return GradientColormapState(
        stops=tuple(
            GradientStopState(
                position=float(position),
                color=typing.cast(
                    "tuple[int, int, int, int]",
                    tuple(int(channel) for channel in color),
                ),
            )
            for position, color in zip(positions, colors, strict=True)
        )
    )


def _gradient_state(
    state: Mapping[str, typing.Any],
) -> GradientColormapState:
    mode = state.get("mode", "rgb")
    if mode not in {"rgb", "hsv"}:
        mode = "rgb"
    raw_stops = state.get("ticks", ())
    stops = tuple(
        GradientStopState(
            position=float(position),
            color=typing.cast(
                "tuple[int, int, int, int]",
                tuple(int(channel) for channel in color),
            ),
        )
        for position, color in sorted(raw_stops, key=lambda stop: float(stop[0]))
    )
    return GradientColormapState(
        mode=typing.cast('typing.Literal["rgb", "hsv"]', mode),
        stops=stops,
    )


def _gradient_restore_payload(state: GradientColormapState) -> dict[str, object]:
    return {
        "mode": state.mode,
        "ticks": [
            (stop.position, stop.color)
            for stop in sorted(state.stops, key=lambda stop: stop.position)
        ],
    }


class _PlotAppearanceAdapter:
    def __init__(
        self,
        target: object,
        *,
        changed: Callable[[str], None],
        image_changed: Callable[[], None],
    ) -> None:
        self._target_ref = weakref.ref(target)
        self._changed_callback = changed
        self._image_changed_callback = image_changed
        self._connections: list[tuple[object, object, Callable[..., None]]] = []
        self._applying = False
        self.manual_levels = False
        self.desired_state: PlotAppearanceState | None = None

    @property
    def target(self) -> object | None:
        return self._target_ref()

    def _connect(
        self,
        owner: object,
        signal: object,
        slot: Callable[..., None],
    ) -> None:
        signal.connect(slot)  # type: ignore[attr-defined]
        self._connections.append((owner, signal, slot))

    def disconnect(self) -> None:
        from erlab.interactive.utils import qt_is_valid

        for owner, signal, slot in self._connections:
            if not qt_is_valid(owner):
                continue
            with contextlib.suppress(RuntimeError, TypeError):
                signal.disconnect(slot)  # type: ignore[attr-defined]
        self._connections.clear()

    def _changed(self, component: str) -> None:
        if not self._applying:
            self._changed_callback(component)

    def _image_changed(self) -> None:
        if not self._applying:
            self._image_changed_callback()

    def capture(self) -> PlotAppearanceState | None:
        raise NotImplementedError

    def apply(self, state: PlotAppearanceState) -> None:
        raise NotImplementedError

    def set_desired_state(self, state: PlotAppearanceState) -> None:
        self.desired_state = state
        self.manual_levels = state.levels is not None
        self.apply(state)

    def reapply(self) -> None:
        if self.desired_state is not None:
            self.apply(self.desired_state)


class _BetterColorBarAdapter(_PlotAppearanceAdapter):
    @property
    def colorbar(self) -> BetterColorBarItem | None:
        return typing.cast("BetterColorBarItem | None", self.target)

    def __init__(
        self,
        target: BetterColorBarItem,
        *,
        changed: Callable[[str], None],
        image_changed: Callable[[], None],
    ) -> None:
        super().__init__(target, changed=changed, image_changed=image_changed)
        self._expanded_limits = False
        self._connect(
            target,
            target.sigColorChanged,
            lambda: self._changed("color"),
        )
        self._connect(
            target,
            target.sigLevelsChangeFinished,
            lambda: self._changed("levels"),
        )
        for image_ref in tuple(target.images):
            image = image_ref()
            if image is not None:
                self._connect(image, image.sigImageChanged, self._image_changed)

    def capture(self) -> PlotAppearanceState | None:
        colorbar = self.colorbar
        if colorbar is None:
            return None
        image_ref = colorbar.primary_image
        image = None if image_ref is None else image_ref()
        if image is None:
            return None
        cmap = image.getColorMap()
        if cmap is None:
            return None
        properties = colorbar.colormap_properties
        if properties is None:
            colormap: ColormapState = _sample_colormap(cmap)
            norm = None
        else:
            colormap = NamedColormapState(
                name=properties["cmap"],
                reverse=properties["reverse"],
            )
            norm = PowerNormState(
                gamma=properties["gamma"],
                high_contrast=properties["high_contrast"],
                zero_centered=properties["zero_centered"],
            )
        levels = None
        if self.manual_levels:
            levels = typing.cast(
                "tuple[float, float]",
                tuple(float(value) for value in colorbar.spanRegion()),
            )
        return PlotAppearanceState(
            colormap=colormap,
            norm=norm,
            levels=levels,
        )

    def apply(self, state: PlotAppearanceState) -> None:
        colorbar = self.colorbar
        if colorbar is None:
            return
        self._applying = True
        try:
            if state.levels is not None:
                image_ref = colorbar.primary_image
                image = None if image_ref is None else image_ref()
                if image is not None:
                    data_limits = tuple(
                        float(value) for value in image.quickMinMax(targetSize=2**16)
                    )
                    if all(math.isfinite(value) for value in data_limits):
                        domain = (
                            min(data_limits[0], state.levels[0]),
                            max(data_limits[1], state.levels[1]),
                        )
                    else:
                        domain = state.levels
                        if domain[0] == domain[1]:
                            lower = math.nextafter(domain[0], -math.inf)
                            upper = math.nextafter(domain[1], math.inf)
                            domain = (
                                lower if math.isfinite(lower) else domain[0],
                                upper if math.isfinite(upper) else domain[1],
                            )
                    if domain != data_limits:
                        colorbar.setLimits(domain)
                        self._expanded_limits = True
                    elif self._expanded_limits:
                        colorbar.setLimits(None)
                        self._expanded_limits = False
            if isinstance(state.colormap, NamedColormapState):
                norm = state.norm or PowerNormState(gamma=1.0)
                try:
                    colorbar.set_colormap(
                        state.colormap.name,
                        gamma=norm.gamma,
                        reverse=state.colormap.reverse,
                        high_contrast=norm.high_contrast,
                        zero_centered=norm.zero_centered,
                    )
                except RuntimeError:
                    logger.warning(
                        "Saved colormap %r is unavailable; applying its normalization "
                        "to the current colormap",
                        state.colormap.name,
                    )
                    fallback = colorbar.colormap_properties
                    if fallback is not None and fallback["cmap"] != state.colormap.name:
                        fallback_name = fallback["cmap"]
                        colorbar.set_colormap(
                            fallback_name,
                            gamma=norm.gamma,
                            reverse=state.colormap.reverse,
                            high_contrast=norm.high_contrast,
                            zero_centered=norm.zero_centered,
                        )
                        self.desired_state = state.model_copy(
                            update={
                                "colormap": NamedColormapState(
                                    name=fallback_name,
                                    reverse=state.colormap.reverse,
                                )
                            }
                        )
                    else:
                        image_ref = colorbar.primary_image
                        image = None if image_ref is None else image_ref()
                        fallback_cmap = None if image is None else image.getColorMap()
                        if image is not None and fallback_cmap is not None:
                            colorbar.set_colormap(
                                fallback_cmap,
                                gamma=norm.gamma,
                                reverse=state.colormap.reverse,
                                high_contrast=norm.high_contrast,
                                zero_centered=norm.zero_centered,
                            )
                            applied_cmap = image.getColorMap()
                            if applied_cmap is not None:
                                self.desired_state = state.model_copy(
                                    update={
                                        "colormap": _sample_colormap(applied_cmap),
                                        "norm": None,
                                    }
                                )
            else:
                cmap = pg.ColorMap(
                    [stop.position for stop in state.colormap.stops],
                    [stop.color for stop in state.colormap.stops],
                    name="saved-gradient",
                )
                if state.norm is None:
                    colorbar.set_pg_colormap(cmap)
                else:
                    colorbar.set_colormap(
                        cmap,
                        gamma=state.norm.gamma,
                        high_contrast=state.norm.high_contrast,
                        zero_centered=state.norm.zero_centered,
                    )
            if state.levels is not None:
                colorbar.setAutoLevels(False)
                colorbar.setSpanRegion(state.levels)
                for image_ref in tuple(colorbar.images):
                    image = image_ref()
                    if image is not None:
                        image.setLevels(state.levels)
        finally:
            self._applying = False


class _HistogramLUTAdapter(_PlotAppearanceAdapter):
    @property
    def histogram(self) -> pg.HistogramLUTItem | None:
        return typing.cast("pg.HistogramLUTItem | None", self.target)

    def __init__(
        self,
        target: pg.HistogramLUTItem,
        *,
        changed: Callable[[str], None],
        image_changed: Callable[[], None],
    ) -> None:
        super().__init__(target, changed=changed, image_changed=image_changed)
        self._level_change_started = False
        self._connect(
            target.gradient,
            target.gradient.sigGradientChangeFinished,
            lambda *_args: self._changed("color"),
        )
        self._connect(
            target.region,
            target.region.sigRegionChanged,
            self._level_region_changed,
        )
        for line in target.region.lines:
            self._connect(line, line.sigDragged, self._level_change_start)
        self._connect(
            target,
            target.sigLevelChangeFinished,
            self._level_change_finished,
        )
        image = target.imageItem()
        if image is not None:
            self._connect(image, image.sigImageChanged, self._image_changed)

    def _level_change_start(self, *_args) -> None:
        if not self._applying:
            self._level_change_started = True

    def _level_region_changed(self, *_args) -> None:
        histogram = self.histogram
        if histogram is not None and histogram.region.moving:
            self._level_change_start()

    def _level_change_finished(self, *_args) -> None:
        if self._level_change_started:
            self._level_change_started = False
            self._changed("levels")

    def capture(self) -> PlotAppearanceState | None:
        histogram = self.histogram
        if histogram is None:
            return None
        levels = None
        if self.manual_levels and histogram.levelMode == "mono":
            levels = typing.cast(
                "tuple[float, float]",
                tuple(float(value) for value in histogram.getLevels()),
            )
        return PlotAppearanceState(
            colormap=_gradient_state(histogram.gradient.saveState()),
            levels=levels,
        )

    def apply(self, state: PlotAppearanceState) -> None:
        histogram = self.histogram
        if histogram is None:
            return
        self._applying = True
        try:
            if isinstance(state.colormap, GradientColormapState):
                histogram.gradient.restoreState(
                    _gradient_restore_payload(state.colormap)
                )
            else:
                try:
                    cmap = pg.colormap.get(
                        state.colormap.name,
                        source="matplotlib",
                        skipCache=True,
                    )
                except (FileNotFoundError, KeyError, ValueError):
                    logger.warning(
                        "Saved colormap %r is unavailable; keeping the current "
                        "colormap",
                        state.colormap.name,
                    )
                else:
                    if state.colormap.reverse:
                        cmap.reverse()
                    histogram.gradient.setColorMap(cmap)
            if state.levels is not None and histogram.levelMode == "mono":
                histogram.setLevels(*state.levels)
        finally:
            self._applying = False


class ToolPlotStateRegistry(QtCore.QObject):
    """Track and persist registered plot appearance for one ToolWindow."""

    def __init__(
        self,
        parent: QtCore.QObject,
        *,
        state_changed: Callable[[], None],
        info_changed: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._state_changed = state_changed
        self._info_changed = info_changed
        self._adapters: dict[str, _PlotAppearanceAdapter] = {}
        self._restored_states: dict[str, PlotAppearanceState] = {}
        self._pending_changes: set[str] = set()
        self._pending_reapply: set[str] = set()
        self._flush_timer = QtCore.QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_pending)
        parent.destroyed.connect(self._disconnect_all)

    def _adapter(
        self,
        plot_id: str,
        target: object,
    ) -> _PlotAppearanceAdapter:
        from erlab.interactive.colors import BetterColorBarItem

        registry_ref = weakref.ref(self)

        def changed(component: str) -> None:
            registry = registry_ref()
            if registry is not None:
                registry._queue_change(plot_id, component)

        def image_changed() -> None:
            registry = registry_ref()
            if registry is not None:
                registry._queue_reapply(plot_id)

        if isinstance(target, BetterColorBarItem):
            return _BetterColorBarAdapter(
                target,
                changed=changed,
                image_changed=image_changed,
            )
        if isinstance(target, pg.HistogramLUTItem):
            return _HistogramLUTAdapter(
                target,
                changed=changed,
                image_changed=image_changed,
            )
        raise TypeError(
            "plot appearance targets must be BetterColorBarItem or HistogramLUTItem"
        )

    def register(self, plot_id: str, target: object) -> None:
        if not plot_id:
            raise ValueError("plot appearance IDs must be non-empty strings")
        if plot_id in self._pending_changes:
            self._flush_pending()
        previous = self._adapters.pop(plot_id, None)
        desired_state = None
        if previous is not None:
            desired_state = previous.desired_state
            previous.disconnect()
        adapter = self._adapter(plot_id, target)
        self._adapters[plot_id] = adapter
        restored_state = self._restored_states.pop(plot_id, None)
        if restored_state is not None:
            adapter.set_desired_state(restored_state)
        elif desired_state is not None:
            adapter.set_desired_state(desired_state)

    def _queue_change(self, plot_id: str, component: str) -> None:
        adapter = self._adapters.get(plot_id)
        if adapter is None:
            return
        if component == "levels":
            adapter.manual_levels = True
        state = adapter.capture()
        if state is None:
            return
        adapter.desired_state = state
        self._pending_changes.add(plot_id)
        self._flush_timer.start(0)

    def _queue_reapply(self, plot_id: str) -> None:
        self._pending_reapply.add(plot_id)
        self._flush_timer.start(0)

    def _disconnect_all(self, *_args) -> None:
        self._flush_timer.stop()
        for adapter in self._adapters.values():
            adapter.disconnect()
        self._adapters.clear()
        self._pending_changes.clear()
        self._pending_reapply.clear()

    def _flush_pending(self) -> None:
        changed = False
        for plot_id in tuple(self._pending_changes):
            adapter = self._adapters.get(plot_id)
            if adapter is None:
                continue
            if adapter.desired_state is not None:
                changed = True
        self._pending_changes.clear()

        for plot_id in tuple(self._pending_reapply):
            adapter = self._adapters.get(plot_id)
            if adapter is not None:
                adapter.reapply()
        self._pending_reapply.clear()

        if changed:
            self._state_changed()
            self._info_changed()

    def state(self) -> ToolViewState:
        self._flush_pending()
        plots = dict(self._restored_states)
        for plot_id, adapter in self._adapters.items():
            state = adapter.capture()
            if state is not None:
                plots[plot_id] = state
        return ToolViewState(plots=plots)

    def state_json(self) -> str | None:
        state = self.state()
        if not state.plots:
            return None
        return state.model_dump_json()

    def restore_json(self, payload: object | None) -> None:
        if payload is None:
            return
        try:
            state = _parse_tool_view_state(payload)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid saved tool view state", exc_info=True)
            return
        self._restored_states = dict(state.plots)
        for plot_id, adapter in self._adapters.items():
            restored = self._restored_states.pop(plot_id, None)
            if restored is not None:
                adapter.set_desired_state(restored)

    def reapply(self) -> None:
        for adapter in self._adapters.values():
            adapter.reapply()

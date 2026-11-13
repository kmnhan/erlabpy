"""Publish ImageTool calculations and project their data revisions into the Manager."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import pathlib
import typing
import uuid
import weakref

import numpy as np
from qtpy import QtCore

import erlab
from erlab._provenance import (
    Application,
    EarlierStep,
    FileLoad,
    FileSource,
    Follow,
    Pin,
    ProvenanceGraph,
    Recipe,
    UpdatePlan,
)
from erlab._provenance import _graph as provenance_graph
from erlab._provenance._graph import DisplayedEdit, _SupersededEvaluation
from erlab._provenance._records import FileLoadPresentation, lazy_input_guards
from erlab.interactive._code_trust import (
    execution_capability_allows,
    issue_local_edit_capability,
)
from erlab.interactive.imagetool._load_source import (
    _load_source_details,
    _resolve_load_func,
)
from erlab.interactive.imagetool._provenance._execution import (
    _load_file_source_data,
    execute_replay_graph,
    replay_script_provenance,
)
from erlab.interactive.imagetool._provenance._graph import (
    ReplayGraph,
    ReplayGraphError,
    SelectedProvenanceScope,
    _add_file_load_node,
    _compile_replay_steps,
    _selected_notebook_spec,
    selected_steps_spec,
)
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    FileLoadSource,
    FileReplayCall,
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    _direct_replay_source_name,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    _SourceViewOperation,
    compose_display_provenance,
    file_load,
    recipe,
    script,
    strip_partial_operation_groups,
)
from erlab.interactive.imagetool._provenance._operations import (
    AverageOperation,
    ExtensionRoutineOperation,
    IselOperation,
    QSelAggregationOperation,
    QSelOperation,
    RenameOperation,
    SelOperation,
    SqueezeOperation,
    TransposeOperation,
)
from erlab.interactive.imagetool._provenance._operations._coordinates import (
    ImageToolInputOperation,
)
from erlab.interactive.imagetool._provenance._program import (
    CalculatedInput,
    ConsoleInput,
    PreparedFileCall,
    RecordedProgram,
)
from erlab.interactive.imagetool._provenance._recorded_code import (
    _notebook_name,
    full_code,
    recorded_file_source,
)
from erlab.interactive.imagetool._provenance._trust import (
    provenance_code_trust_entries,
    provenance_operation_requires_code_trust,
)
from erlab.interactive.imagetool.manager._widgets import (
    _TrustedProvenanceReplayCancelled,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import xarray as xr

    from erlab.interactive._code_trust._core import CodeTrustEntry
    from erlab.interactive.imagetool._load_source import _LoadFunc
    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool._provenance._model import ToolProvenanceOperation
    from erlab.interactive.imagetool._provenance._operations import (
        ImageToolSelectionSourceBinding,
    )
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._revisions import _ProvenanceRevisionStore
    from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode

_Command = typing.Literal["refresh", "reload"]


class _ReloadCancelled(Exception):
    pass


class _ResultWindowUnavailable(RuntimeError):
    """Window construction did not return an ImageTool for the calculated result."""


class _UnboundInput(ValueError):
    """An authored calculation has no resolvable graph input."""


@dataclasses.dataclass(frozen=True)
class _InputRevision:
    binding: Follow | Pin | EarlierStep
    revision: str


@dataclasses.dataclass(frozen=True)
class _WindowInput:
    data: xr.DataArray
    revision: _InputRevision
    require_current: Callable[[], None]


@dataclasses.dataclass(frozen=True)
class _OperationRequest:
    """Settings and ordered operands captured before an operation executes."""

    operations: tuple[ToolProvenanceOperation, ...]
    inputs: tuple[_InputRevision, ...]
    input_names: tuple[str, ...] = ()
    output_name: str | None = None
    source_kind: typing.Literal["full_data", "public_data", "selection"] | None = None


@dataclasses.dataclass(frozen=True)
class _CalculatedResult:
    capture: provenance_graph.CapturedRevision
    output_name: str | None
    window_uid: str | None = None

    @property
    def data(self) -> xr.DataArray:
        return self.capture.calculated_data


@dataclasses.dataclass(frozen=True)
class _ImageToolInput:
    path: pathlib.Path | None
    loader: _LoadFunc | None
    preparation: tuple[ToolProvenanceOperation, ...]
    raw: xr.DataArray | None = None

    @classmethod
    def from_window(cls, tool: ImageTool) -> _ImageToolInput:
        """Read the load details of an already constructed window."""
        area = tool.slicer_area
        return cls(area._file_path, area._load_func, area._load_preparation_operations)


@dataclasses.dataclass(frozen=True)
class _RawReplacement:
    source_ndim: int
    source_dtype: np.dtype
    publish: Callable[[], str] | None
    raw_revision: str | None
    watched_name: str | None


_SourceWrite = _RawReplacement | _ImageToolInput | typing.Literal["file"]


@dataclasses.dataclass
class _WindowWrite:
    signals: int = 0
    newer_source: _SourceWrite | None = None

    @property
    def superseded(self) -> bool:
        return self.signals > 1 or self.newer_source is not None


@dataclasses.dataclass(frozen=True)
class _ImportedHistory:
    provenance: ToolProvenanceSpec | None
    source: ToolProvenanceSpec | None
    binding: ImageToolSelectionSourceBinding | None


class _SnapshotSignals(QtCore.QObject):
    finished = QtCore.Signal()


class _InputSnapshotTask(QtCore.QRunnable):
    """Copy an in-memory input while its owner continues processing UI events."""

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.data: xr.DataArray | None = data
        self.signals = _SnapshotSignals()
        self.result: xr.DataArray | None = None
        self.error: BaseException | None = None
        self.waiting = True

    def run(self) -> None:
        try:
            self.result = provenance_graph.checkpoint_data(
                typing.cast("xr.DataArray", self.data)
            )
        except BaseException as exc:
            self.error = exc
        finally:
            self.data = None
            if not self.waiting:
                self.result = None
                self.error = None
            self.signals.finished.emit()


# Keep workers alive if application shutdown exits the local event loop before
# their copies finish. During normal operation the queued completion releases them.
_INPUT_SNAPSHOTS: set[_InputSnapshotTask] = set()


# Only settings with these plain value types can participate in code export's
# readback. Backend and chunk-manager options can dispatch to third-party hooks.
_NETCDF_VERIFICATION_OPTIONS = {
    "engine": str,
    "group": str,
    "chunks": type(None),
    "decode_cf": bool,
    "decode_coords": bool,
    "decode_times": bool,
    "decode_timedelta": bool,
    "mask_and_scale": bool,
    "concat_characters": bool,
    "cache": bool,
    "create_default_indexes": bool,
}


class _ManagerDataGraph:
    def __init__(self, manager: ImageToolManager) -> None:
        self.manager = manager
        self._snapshot_active = False
        self.graph = ProvenanceGraph(
            read_file=self._read_file,
            apply_operation=self._apply_operation,
            validate_update=self._validate_update,
            placement=self._placement,
            execution_scope=self._execution_scope,
            install=self._install,
            installed_step=self._installed_step,
            published=self._published,
            schedule_auto=self._schedule_auto,
            snapshot_input=self._snapshot_input,
        )
        self._changing: set[str] = set()
        self._source_writes: dict[str, _WindowWrite] = {}
        self._staged_windows: set[str] = set()
        self._renaming: set[str] = set()
        self._legacy_seed_revisions: dict[str, str] = {}
        self._branched: set[str] = set()
        self._exports: dict[tuple[str, bool], tuple[str, int, ToolProvenanceSpec]] = {}
        self._uses_extensions = False
        self._console_program: RecordedProgram | None = None
        self._approved_entries: dict[bytes, CodeTrustEntry] | None = None
        self._checkpoint_sources: set[str] = set()
        self._checkpoint_reasons: dict[str, tuple[str, str]] = {}
        self._adopting: set[str] = set()
        # Older documents and externally supplied ImageTools can carry a
        # replay workflow with no graph IDs for its intermediate values.
        self._imported: set[str] = set()
        self._watched_sources: dict[str, tuple[str, ToolProvenanceSpec]] = {}
        # A display filter or axis order is a view of this row's source revision.
        self._views: dict[str, tuple[str, int]] = {}
        self._view_state: dict[
            str, tuple[weakref.ReferenceType[xr.DataArray], tuple, int]
        ] = {}

    def _snapshot_input(self, data: xr.DataArray) -> xr.DataArray:
        # Small copies cost less than posting a worker. The large-input path is
        # used only during evaluation, before any graph or window commit starts.
        if data.nbytes < 8 * 1024**2:
            return provenance_graph.checkpoint_data(data)
        if self._snapshot_active:
            raise _SupersededEvaluation(
                "An input snapshot is in progress. Retry when it finishes."
            )
        task = _InputSnapshotTask(data)
        loop = QtCore.QEventLoop()
        task.signals.finished.connect(
            loop.quit, QtCore.Qt.ConnectionType.QueuedConnection
        )
        task_ref = weakref.ref(task)

        def release_task() -> None:
            if worker := task_ref():
                if not worker.waiting:
                    worker.result = None
                    worker.error = None
                _INPUT_SNAPSHOTS.discard(worker)

        task.signals.finished.connect(
            release_task,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        _INPUT_SNAPSHOTS.add(task)
        self._snapshot_active = True
        try:
            try:
                typing.cast(
                    "QtCore.QThreadPool", QtCore.QThreadPool.globalInstance()
                ).start(task)
            except Exception:
                task.data = None
                _INPUT_SNAPSHOTS.discard(task)
                raise
            loop.exec()
            if task.error is not None:
                raise task.error
            if task.result is None:
                raise _SupersededEvaluation("Input capture was interrupted")
            if (
                not erlab.interactive.utils.qt_is_valid(self.manager)
                or self.manager._workspace_state.closing_document
            ):
                raise _SupersededEvaluation("The workspace closed during input capture")
            return task.result
        finally:
            self._snapshot_active = False
            task.waiting = False
            task.result = None
            task.error = None

    def manages(self, uid: str) -> bool:
        return self.reloads_result(uid) and (
            self.graph.has_recipe(uid) or self._checkpoint_reason(uid) is not None
        )

    def _checkpoint_reason(self, uid: str) -> str | None:
        saved = self._checkpoint_reasons.get(uid)
        if (
            saved is not None
            and uid in self.graph
            and self.graph.current_revision(uid) == saved[0]
        ):
            return saved[1]
        return None

    def _placement(self, uid: str) -> tuple[str | None, int] | None:
        tree = self.manager._tool_graph
        node = tree.nodes.get(uid)
        if node is None:
            return None
        if node.parent_uid is None:
            index = getattr(node, "index", None)
            if index in tree.displayed_indices:
                return None, tree.displayed_indices.index(index)
        else:
            siblings = tree.nodes[node.parent_uid]._childtool_indices
            if uid in siblings:
                return node.parent_uid, siblings.index(uid)
        return None

    @contextlib.contextmanager
    def staging_window(self, uid: str) -> Iterator[None]:
        """Keep public reads out of registration and replacement callbacks."""
        nested = uid in self._staged_windows
        self._staged_windows.add(uid)
        try:
            yield
        finally:
            if not nested:
                self._staged_windows.discard(uid)

    def require_published(self, uid: str) -> None:
        if uid in self._staged_windows:
            raise _SupersededEvaluation(
                f"{self.label(uid)} is being updated. Retry when the update finishes."
            )

    def console_data(self, node: _ManagedWindowNode) -> xr.DataArray:
        """Return an independent snapshot with the recorded lazy file guards."""
        if not node.materialize_pending_workspace_payload():
            raise ValueError("Could not read the saved ImageTool data")
        self._ensure_source(node)
        return node.slicer_area.displayed_data.copy(deep=True)

    def read_input(
        self,
        target: int | str,
        *,
        data_role: typing.Literal["source", "displayed"] = "displayed",
    ) -> _WindowInput:
        node = self.manager._node_for_target(target)
        if not node.materialize_pending_workspace_payload() or not self._ensure_source(
            node
        ):
            raise _SupersededEvaluation("The selected input is unavailable")
        check = self.observe_input(target, data_role=data_role)
        binding = Follow(node.uid, source=data_role == "source")
        revision = _InputRevision(binding, self.graph.followed_revision(binding))
        data = self.graph.calculation_data(revision.revision)
        check()
        return _WindowInput(data, revision, check)

    def observe_input(
        self,
        target: int | str,
        *,
        data_role: typing.Literal["source", "displayed"] = "displayed",
    ) -> Callable[[], None]:
        """Check that a result still describes the ImageTool used to calculate it."""
        node = self.manager._node_for_target(target)
        self.require_published(node.uid)
        label = self.label(node.uid)
        area = node.slicer_area
        source = node.source_snapshot_token
        layout = area._public_source_dims_for_current_layout()
        displayed = node.snapshot_token
        accepted_filter = area.accepted_filter_revision

        def require_current() -> None:
            self.require_published(node.uid)
            if (
                self.manager._tool_graph.nodes.get(node.uid) is not node
                or node.imagetool is None
                or node.slicer_area is not area
                or node.source_snapshot_token != source
                or area._public_source_dims_for_current_layout() != layout
                or (
                    data_role == "displayed"
                    and (
                        node.snapshot_token != displayed
                        or area.accepted_filter_revision != accepted_filter
                    )
                )
            ):
                raise _SupersededEvaluation(
                    f"{label} changed during the calculation. "
                    "Run the transform again using its current data."
                )

        return require_current

    @staticmethod
    def verify_new_window(tool: ImageTool, *input_checks: Callable[[], None]) -> None:
        """Discard an unpublished window if an input changed while it was created."""
        try:
            for check in input_checks:
                check()
        except _SupersededEvaluation:
            tool.close()
            tool.deleteLater()
            raise

    @contextlib.contextmanager
    def changing(self, uid: str) -> Iterator[None]:
        if uid in self._changing:
            yield
            return
        self._changing.add(uid)
        try:
            yield
        finally:
            self._changing.remove(uid)

    @contextlib.contextmanager
    def replacing(self, node: _ManagedWindowNode) -> Iterator[None]:
        """Keep an external replacement rollbackable until its recipe is published."""
        with self._stage_window(node), self._write_window(node):
            yield

    def _require_single_write(self, uid: str) -> None:
        write = self._source_writes.get(uid)
        if write is not None and write.superseded:
            raise _SupersededEvaluation("ImageTool data changed during installation")

    @contextlib.contextmanager
    def _write_window(self, node: _ManagedWindowNode) -> Iterator[None]:
        uid = node.uid
        if uid in self._changing:
            raise _SupersededEvaluation(
                "This ImageTool is being updated. Retry after the update."
            )
        self._source_writes[uid] = _WindowWrite()
        try:
            with self.changing(uid):
                yield
                self._require_single_write(uid)
        finally:
            # The wrapper suspends publication while it installs data. A second
            # source write from a synchronous slot is a new input. Publish it after
            # releasing the wrapper, so rollback preserves that newer revision.
            writes = self._source_writes.pop(uid)
            if (
                writes.superseded
                and self.manager._tool_graph.nodes.get(uid) is node
                and node.imagetool is not None
            ):
                with self.changing(uid):
                    self.prepare_source_update(node, writes.newer_source)()
                node.record_source_input(writes.newer_source)
                if not self.graph.has_recipe(uid):
                    node.set_detached_provenance(None, replay_source_data=None)
                    self._checkpoint_reasons[uid] = (
                        self.graph.current_revision(uid),
                        (
                            "A newer source write replaced this calculation. "
                            "Repeat the calculation from its inputs to restore updates."
                        ),
                    )
                node._advance_snapshot_token(defer_refresh=True)
                self.manager._mark_node_data_dirty(uid)
                self.manager._lineage_controller._propagate_source_change_from_uid(uid)

    def _schedule_auto(self, run: Callable[[], None]) -> None:
        def update() -> None:
            if erlab.interactive.utils.qt_is_valid(self.manager):
                run()
                self._refresh_inspector()

        self.manager._queue_idle_work(("data-graph-auto",), update)

    def _refresh_inspector(self) -> None:
        if (
            erlab.interactive.utils.qt_is_valid(self.manager)
            and not self.manager._workspace_state.closing_document
        ):
            viewport = self.manager.tree_view.viewport()
            if viewport is not None:
                viewport.update()
            self.manager._update_info()
            self.manager._update_actions()

    @contextlib.contextmanager
    def _replay_publication(
        self,
    ) -> Iterator[tuple[Callable[[], None], Callable[[], None]]]:
        # All extension executions in one graph request share the final capability
        # check and publish their verified script bytes only after the graph commits.
        previous = self._approved_entries
        approved: dict[bytes, CodeTrustEntry] = {}
        self._approved_entries = approved
        try:
            with self.manager._extensions.execution.capture_replay_sources() as capture:

                def require_current() -> None:
                    capture.require_current_for_publication()
                    workspace = self.manager._workspace_controller
                    if (
                        approved
                        and workspace.issue_code_execution_capability(
                            tuple(approved.values()), focus_on_block=True
                        )
                        is None
                    ):
                        raise _TrustedProvenanceReplayCancelled(
                            "Workspace approval changed during this calculation. "
                            "Review its stored code before updating the result."
                        )

                yield require_current, capture.publish
        finally:
            self._approved_entries = previous

    @contextlib.contextmanager
    def _execution_scope(self) -> Iterator[Callable[[], None]]:
        with self._replay_publication() as (require_current, publish):
            yield require_current
            publish()

    def _authorize(
        self, entries: tuple[CodeTrustEntry, ...], *, reason: str
    ) -> object | None:
        capability = self.manager._lineage_controller._authorize_provenance_execution(
            entries, reason=reason, raise_on_block=False
        )
        if capability is None:
            raise _TrustedProvenanceReplayCancelled(
                "Workspace trust is blocking this calculation. "
                "Review its stored code before updating the result."
            )
        if self._approved_entries is not None:
            for item in entries:
                if execution_capability_allows(capability, (item,)):
                    self._approved_entries[item.execution_identity()] = item
        return capability

    def _file_edge(self, spec: ToolProvenanceSpec, call: FileReplayCall) -> FileLoad:
        source = spec.file_load_source
        if source is None:
            raise ValueError("The recorded load requires a file origin")
        presentation = (
            FileLoadPresentation(
                spec.start_label or "Load file",
                spec.active_name or "derived",
                spec.seed_code,
                source.loader_label,
                source.loader_text,
                source.kwargs_text,
                source.load_code,
            )
            if source.replay_call is not None
            and source.replay_call.cast_float64 == call.cast_float64
            else None
        )
        return FileLoad(self.graph.file_source(source.path), call, presentation)

    def _read_file(self, source: FileSource, call: FileReplayCall) -> xr.DataArray:
        file = FileLoadSource(
            path=source.path,
            replay_call=call,
            loader_label="Loader",
            loader_text=call.target,
            kwargs_text="",
        )
        if isinstance(call, PreparedFileCall):
            replay = ReplayGraph()
            raw = _add_file_load_node(
                replay, file, setup_code=None, load_code=None, active_name="data"
            )
            replay.output_key = _compile_replay_steps(
                replay, raw, call.steps, display=False
            )
            return execute_replay_graph(
                replay,
                extension_executor=self.manager._extensions.execution.run_operation,
                extension_loader_executor=self.manager._extensions.replay_loader,
                authorize=functools.partial(
                    self._authorize, reason="reload this prepared file"
                ),
            )
        return _load_file_source_data(
            file,
            extension_loader_executor=self.manager._extensions.replay_loader,
        )

    def _apply_operation(
        self, application: Application, values: Mapping[str, xr.DataArray]
    ) -> xr.DataArray:
        operation = application.operation
        if isinstance(operation, RecordedProgram):
            inputs = {
                name: values[key]
                for name, key in operation.input_names(application.inputs).items()
            }
            for item in operation.spec.replay_inputs:
                if selection := item.parsed_source_spec():
                    entries = provenance_code_trust_entries(
                        selection, location_prefix="workspace/provenance"
                    )
                    authorization = self._authorize(
                        entries, reason="apply the recorded input selection"
                    )
                    inputs[item.name] = (
                        self.manager._lineage_controller._apply_provenance(
                            selection,
                            inputs[item.name],
                            reason="apply the recorded input selection",
                            authorization=authorization,
                        )
                    )
            local = operation is self._console_program
            return replay_script_provenance(
                operation.spec,
                inputs,
                extension_executor=self.manager._extensions.execution.run_operation,
                extension_loader_executor=self.manager._extensions.replay_loader,
                authorize=self._authorize_console_call
                if local
                else functools.partial(
                    self._authorize, reason="run this recorded console calculation"
                ),
                owned_inputs=local,
            )
        if getattr(operation, "op", None) == "extension_routine":
            return self.manager._extensions.execution.run_operation(
                operation, values[application.inputs[0][1]]
            )
        if provenance_operation_requires_code_trust(operation):
            replay = ReplayGraph()
            for key, data in values.items():
                replay.add_node(key, "live_input", payload={"data": data})
            (_, primary), *others = application.inputs
            replay.output_key = replay.add_node(
                "result",
                "operation",
                parents=(primary, primary, *(key for _, key in others)),
                payload={"operation": operation, "input_bindings": tuple(others)},
            )
            return execute_replay_graph(
                replay,
                authorize=functools.partial(
                    self._authorize, reason="repeat this operation"
                ),
            )
        return application.apply(values)

    def _validate_update(self, uid: str, data: xr.DataArray) -> None:
        self.manager.get_imagetool(uid)
        erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(data)

    def legacy_replay_seed(self, uid: str) -> str | None:
        """Reuse a recorded origin while installing its compatibility provenance."""
        revision = self._legacy_seed_revisions.get(uid)
        if revision is None:
            return None
        origins = [
            record
            for record in self.graph.recorded_lineage(revision)
            if not isinstance(record.producer, Application)
        ]
        if len(origins) != 1:
            return None
        origin = origins[0]
        store = self.manager._provenance_revisions
        if store.has_data(origin.revision_id):
            return origin.revision_id
        reader = self.graph.checkpoint_reader(origin.revision_id)
        if reader is None:
            return None
        return store.add(
            revision_id=None if origin.revision_id in store else origin.revision_id,
            source_owner_uid=origin.owner_id,
            data_reader=reader,
        )

    @contextlib.contextmanager
    def _stage_window(self, node: _ManagedWindowNode) -> Iterator[None]:
        """Restore this window if any window in the same request rejects its data."""
        uid = node.uid
        if uid in self._changing:
            raise _SupersededEvaluation(
                "This ImageTool is being updated. Retry after the update."
            )
        area = node.slicer_area
        data, state = area._data, area.state
        restore_history = area._capture_history_for_rollback()
        path, loader, preparation = (
            area._file_path,
            area._load_func,
            area._load_preparation_operations,
        )
        input_dtype, degree_coords = area._source_input_dtype, area._degree_coords
        provenance = node._provenance_owner_snapshot()
        filter_revision = area.accepted_filter_revision
        layout = area._public_source_dims_for_current_layout()
        tokens = node._source_snapshot_token, node._snapshot_token
        source = (
            node._source_binding,
            node._output_id,
            node._source_auto_update,
            node._source_state,
            node._replay_source_data,
        )
        view = self._views.get(uid)
        revision = self.graph.current_revision(uid) if uid in self.graph else None
        with self.staging_window(uid):
            try:
                yield
            except Exception:
                # A callback may have closed this row or installed a newer revision
                # while another window in the batch was updating. Preserve that work.
                if (
                    self.manager._tool_graph.nodes.get(uid) is not node
                    or node.imagetool is not provenance.imagetool
                    or (self.graph.current_revision(uid) if uid in self.graph else None)
                    != revision
                ):
                    raise
                with (
                    self._write_window(node),
                    node._suspend_descendant_propagation(),
                    node._suspend_snapshot_updates(),
                ):
                    if area._data is not data:
                        area._set_data(
                            data,
                            source_replaced=True,
                            file_path=path,
                            load_func=loader,
                            preparation_operations=preparation,
                            auto_compute=False,
                        )
                        self._require_single_write(uid)
                        area.state = state
                        area._source_input_dtype = input_dtype
                        area._degree_coords = degree_coords
                    elif (
                        area.accepted_filter_revision != filter_revision
                        or area._public_source_dims_for_current_layout() != layout
                    ):
                        area.state = state
                    (
                        node._source_binding,
                        node._output_id,
                        node._source_auto_update,
                        node._source_state,
                        node._replay_source_data,
                    ) = source
                    node._source_snapshot_token, node._snapshot_token = tokens
                    node.restore_provenance_owners(provenance)
                    if view is not None:
                        self._views[uid] = view
                    else:
                        self._views.pop(uid, None)
                    self._view_state[uid] = (
                        weakref.ref(area._data),
                        area._public_source_dims_for_current_layout(),
                        area.accepted_filter_revision,
                    )
                    restore_history()
                raise

    def _refresh_imported_spec(
        self, node: _ManagedWindowNode
    ) -> ToolProvenanceSpec | None:
        """Refresh the interchange payload of an imported workflow in its own format."""
        spec = node.provenance_spec
        if node.source_spec is not None and node.parent_uid is not None:
            parent = self.manager._parent_node(node)
            spec = compose_display_provenance(
                parent.displayed_provenance_spec,
                node.source_spec,
                parent_data=parent.current_source_data(),
            )
        elif spec is not None and spec.replay_inputs:
            lineage = self.manager._lineage_controller
            spec = spec.with_replay_inputs(
                tuple(
                    lineage._script_input_for_node(
                        self.manager._node_for_target(item.node_uid),
                        name=item.name,
                        source_spec=item.parsed_source_spec(),
                        data_role=item.data_role,
                    )
                    if item.follow_source
                    and item.node_uid in self.manager._tool_graph.nodes
                    else item
                    for item in spec.replay_inputs
                )
            )
        return spec

    @contextlib.contextmanager
    def _install(
        self, entries: Mapping[str, str], values: Mapping[str, xr.DataArray]
    ) -> Iterator[Mapping[str, xr.DataArray]]:
        materialized: dict[str, xr.DataArray] = {}
        with contextlib.ExitStack() as batch:
            for uid, revision in entries.items():
                node = self.manager._node_for_target(uid)
                batch.enter_context(self._stage_window(node))
                base = revision
                count = 0
                recorded_filter = None
                while (
                    isinstance((step := self.graph.recipe_for(base)), Recipe)
                    and step.view_role is not None
                ):
                    applied = self.graph.revision(base).producer
                    if not isinstance(applied, Application):
                        raise TypeError("Expected a recorded display operation")
                    if step.view_role == "filter":
                        recorded_filter = (applied.operation, values[base])
                    base = applied.inputs[0][1]
                    count += 1
                if count:
                    self._views[uid] = base, count
                else:
                    self._views.pop(uid, None)
                if uid in self._renaming:
                    # Rename changes only the name. Keep the array backing, layout,
                    # filter, and separate source and display tokens intact.
                    with self._write_window(node):
                        node._rename_imagetool_data(
                            "" if values[base].name is None else str(values[base].name),
                            record_provenance=False,
                        )
                        if uid in self._imported and node._source_spec is not None:
                            node._source_spec = node._source_spec.append_final_rename(
                                typing.cast("str | None", values[base].name)
                            )
                            node.set_displayed_provenance(
                                self.export_spec(base, target_uid=uid),
                                advance_snapshot=False,
                            )
                    area = node.slicer_area
                    self._view_state[uid] = (
                        weakref.ref(area._data),
                        area._public_source_dims_for_current_layout(),
                        area.accepted_filter_revision,
                    )
                    continue
                # The compatibility checkpoint store keeps the displayed and
                # previous values for v3 workspace saves and standalone transfer.
                spec = (
                    self._refresh_imported_spec(node)
                    if uid in self._imported
                    else self.export_spec(base, target_uid=uid)
                )
                file_origin = self.graph.revision(base).producer
                file_path = (
                    spec.file_load_source.path
                    if spec is not None and spec.file_load_source is not None
                    else file_origin.source.path
                    if isinstance(file_origin, FileLoad)
                    else None
                )
                data = materialized.get(base)
                if data is None:
                    data = values[base] if base in values else self.graph.data(base)
                if erlab.interactive.imagetool.viewer._should_auto_compute(data):
                    if recorded_filter is None:
                        data = data.compute()
                    else:
                        # Materialize both outputs of the same Dask graph. Shared
                        # source chunks are read once and the filter is not rerun.
                        import dask

                        op, filtered = recorded_filter
                        data, filtered = dask.compute(data, filtered)
                        recorded_filter = op, filtered
                    materialized[base] = data
                if self.owns_result(uid):
                    # Capture the previous version while the wrapper still sees
                    # the graph's published revision.
                    node.revision_id_for_role("source", retain_expression=True)
                with self._write_window(node):
                    self._legacy_seed_revisions[uid] = (
                        self.graph.current_revision(uid) if uid in self.graph else base
                    )
                    try:

                        def installed(uid: str = uid, base: str = base) -> None:
                            self._require_single_write(uid)
                            self._legacy_seed_revisions[uid] = base

                        node._replace_imagetool_data(
                            # Detach eager values from later window edits. For lazy
                            # data this copies the expression, without reading chunks.
                            data.copy(deep=True),
                            spec,
                            propagate_descendants=False,
                            after_replace=installed,
                            recorded_filter=recorded_filter,
                            primary_checkpoint_required=False,
                            detach_source_binding=uid in self._branched,
                            preserve_file=isinstance(file_origin, FileLoad)
                            and node.slicer_area._file_path is not None
                            and file_path is not None
                            and pathlib.Path(file_path) == node.slicer_area._file_path,
                        )
                    finally:
                        self._legacy_seed_revisions.pop(uid, None)
                area = node.slicer_area
                if (
                    revision != base
                    and values[revision].chunks is not None
                    and area.displayed_data.chunks is None
                ):
                    materialized[revision] = area.displayed_data.copy(deep=True)
                self._view_state[uid] = (
                    weakref.ref(area._data),
                    area._public_source_dims_for_current_layout(),
                    area.accepted_filter_revision,
                )

            yield materialized

    def _installed_step(self, uid: str, key: str, data: xr.DataArray) -> Recipe | None:
        area = self.manager._node_for_target(uid).slicer_area
        prior = EarlierStep(key) if self.graph.recipe_for(key) is not None else Pin(key)
        if (
            data.dtype not in (np.float32, np.float64)
            and area._data.dtype == np.float64
        ):
            return Recipe.bind(
                ImageToolInputOperation(cast_float64=True).capture_execution_options(),
                prior,
            )
        dims = area._public_source_dims_for_current_layout()
        if dims != data.dims:
            base, depth = self._views.get(uid, (key, 0))
            self._views[uid] = base, depth + 1
            return Recipe(
                TransposeOperation(dims=dims),
                (("data", prior),),
                view_role="layout",
            )
        return None

    def _published(self, entries: Sequence[str]) -> None:
        for uid in entries:
            if uid not in self._changing and uid in self.manager._tool_graph.nodes:
                # A selected ToolWindow can consume this result in the same Reload
                # request, before the queued window notification runs.
                node = self.manager._node_for_target(uid)
                node.update_title()
                node._flush_snapshot_token_refresh(uid)
                self.manager._lineage_controller._propagate_source_change_from_uid(uid)
        if self.manager.tree_view.selectedIndexes():
            self.manager._queue_idle_work(
                ("data-graph-inspector",), self._refresh_inspector
            )

    @staticmethod
    def _extension_capabilities(
        producers: Sequence[Recipe | FileLoad],
    ) -> Iterator[tuple[str, str, typing.Literal["routine", "loader"], str]]:
        for producer in producers:
            if isinstance(producer, FileLoad):
                call = producer._captured
                if call.kind == "extension_loader":
                    yield (
                        call.target,
                        typing.cast("str", call.source_hash),
                        "loader",
                        typing.cast("str", call.capability_id),
                    )
                operations = (
                    [step.operation for step in call.steps]
                    if isinstance(call, PreparedFileCall)
                    else []
                )
            else:
                operations = [producer._captured.operation]
            for operation in operations:
                if isinstance(operation, RecordedProgram):
                    pending = [step.operation for step in operation.spec.steps]
                    for item in operation.spec.replay_inputs:
                        if spec := item.parsed_source_spec():
                            pending.extend(spec.operations)
                else:
                    pending = [operation]
                for op in pending:
                    if isinstance(op, ExtensionRoutineOperation):
                        yield (op.script_name, op.source_hash, "routine", op.routine_id)

    def _note_producers(self, producers: Sequence[Recipe | FileLoad]) -> None:
        if not self._uses_extensions:
            self._uses_extensions = any(self._extension_capabilities(producers))

    def prepare_window(
        self,
        uid: str,
        tool: ImageTool,
        origin: _ImageToolInput | _CalculatedResult,
        *,
        parent_uid: str | None,
        auto: bool,
        watched_name: str | None = None,
    ) -> Callable[[], str] | None:
        if uid in self._changing:
            # A graph reload already owns the revision for this destination.
            return None
        if isinstance(origin, _ImageToolInput):
            return self.prepare_input(
                uid, tool, origin, parent_uid=parent_uid, watched_name=watched_name
            )
        return self.prepare_calculation(
            uid,
            origin,
            parent_uid=parent_uid,
            tool=tool,
            auto=auto,
        )

    def prepare_input(
        self,
        uid: str,
        tool: ImageTool,
        origin: _ImageToolInput,
        *,
        parent_uid: str | None,
        watched_name: str | None = None,
    ) -> Callable[[], str]:
        """Record the input accepted by the window and its file replay information."""
        area = tool.slicer_area
        path, preparation, raw = origin.path, origin.preparation, origin.raw
        if raw is not None:
            from erlab.interactive.imagetool.manager._workspace._arrays import (
                lease_workspace_input,
            )

            raw = lease_workspace_input(raw)
        resolved = (
            _resolve_load_func(
                origin.loader,
                capture_session=True,
                source_input_dtype=None if preparation else area._source_input_dtype,
            )
            if path is not None
            else None
        )
        steps: list[Recipe | FileLoad] = []
        checkpoints = {}
        binding: Pin | Follow = Follow(uid)
        extra = list(preparation)
        cast = area._source_input_dtype not in (None, np.float32, np.float64)
        if resolved is not None and resolved.cast_float64:
            cast = False
            if raw is not None and area._degree_coords:
                if not raw.variable._in_memory and raw.chunks is None:
                    raw = raw.chunk("auto")
                raw = raw.astype("float64")
        if cast or area._degree_coords:
            extra.append(
                ImageToolInputOperation(
                    cast_float64=cast, degree_coords=area._degree_coords
                )
            )
        if path is not None and resolved is not None:
            call = resolved.replay_call()
            details = _load_source_details(path, origin.loader, resolved)
            presentation = FileLoadPresentation(
                f"Load data from file {path.name!r}",
                "derived",
                resolved.load_code(path, assign="derived"),
                details.loader_label,
                details.loader_text,
                details.kwargs_text,
                None if call.kind == "extension_loader" else details.load_code,
            )
            if raw is None and extra:
                # A client can send only the prepared file value. Its checkpoint
                # includes these steps. Apply them again only on a new file read.
                call = PreparedFileCall(
                    **call.model_dump(),
                    steps=tuple(ReplayStep(operation=op) for op in extra),
                )
                if resolved.session_loader is not None:
                    call._session_loader = resolved.session_loader
                extra.clear()
            steps.append(
                FileLoad(self.graph.file_source(str(path)), call, presentation)
            )
        elif raw is not None and extra:
            key = uuid.uuid4().hex
            checkpoints[key] = uid, raw
            binding = Pin(key)
        else:
            extra.clear()
        for index, operation in enumerate(extra):
            steps.append(
                Recipe.bind(
                    operation.capture_execution_options(),
                    binding if index == 0 else Follow(uid),
                )
            )
        self._note_producers(steps)
        if extra:
            publish = self.graph.prepare_evaluation(
                uid,
                steps,
                parent_id=parent_uid,
                file_data=raw,
                checkpoints=checkpoints,
                prepare_view=self._install_evaluated_source,
            )
        else:
            publish = self.graph.prepare_source(
                uid,
                area._data,
                file=typing.cast("FileLoad", steps[0]) if steps else None,
                parent_id=parent_uid,
                prepare_view=self._prepare_view,
            )
        return self._commit_publication(
            uid,
            publish,
            imported=False,
            watched_name=watched_name if resolved is None else None,
            raw_revision=binding.revision_id if isinstance(binding, Pin) else None,
        )

    def _ensure_source(self, node: _ManagedWindowNode) -> bool:
        """Read a registered revision or materialize an explicitly imported history."""
        self.require_published(node.uid)
        if node.uid in self.graph:
            return True
        if (
            node.imagetool is None
            or self.manager._workspace_state.loading_depth
            or node.uid in self._adopting
            or node.uid not in self._imported
        ):
            return False
        self._adopting.add(node.uid)
        try:
            self._adopt_source(node)
            self._imported.add(node.uid)
        finally:
            self._adopting.remove(node.uid)
        return True

    def _adopt_source(self, node: _ManagedWindowNode) -> None:
        spec = node.provenance_spec
        relative = node.source_spec
        area = node.slicer_area
        # Older workspaces saved only the displayed dtype. Replay ImageTool's
        # conversion for these bindings without evaluating their inputs on restore.
        input_dtype = area._source_input_dtype
        cast_result = input_dtype is not None and input_dtype not in (
            np.float32,
            np.float64,
        )
        normalize_dtype = input_dtype is None and area._data.dtype == np.float64
        conversion = (
            ImageToolInputOperation(
                cast_float64=cast_result, normalize_dtype=normalize_dtype
            )
            if cast_result or normalize_dtype
            else None
        )
        if relative is not None and conversion is not None:
            relative = relative.append_operations(conversion)
        if (
            node.output_id is None
            and relative is not None
            and relative.is_live_source
            and node.parent_uid is not None
            and self._adopt_parent_binding(node, relative)
        ):
            return
        steps: tuple[FileLoad, ...] = ()
        if spec is not None and node.output_id is None and relative is None:
            source = spec.file_load_source
            if (
                spec.kind == "file"
                and source is not None
                and source.replay_call is not None
                and all(
                    getattr(step.operation, "op", None) != "script_code"
                    for step in spec.steps
                )
            ):
                call = source.replay_call
                preparation = spec.steps
                file_conversion = (
                    conversion if preparation or not call.cast_float64 else None
                )
                has_display_conversion = bool(preparation) and isinstance(
                    preparation[-1].operation, ImageToolInputOperation
                )
                if not has_display_conversion and (
                    file_conversion is not None or area._degree_coords
                ):
                    preparation = (
                        *preparation,
                        ReplayStep(
                            operation=(
                                file_conversion or ImageToolInputOperation()
                            ).model_copy(update={"degree_coords": area._degree_coords})
                        ),
                    )
                if preparation:
                    # Only the prepared result survived. Keep it as the file
                    # checkpoint, with the original load and steps for the next read.
                    call = PreparedFileCall(**call.model_dump(), steps=preparation)
                steps = (self._file_edge(spec, call),)
            elif spec.replay_inputs:
                if conversion is not None:
                    spec = spec.append_operations(conversion)
                publish = self.prepare_imported_workflow(
                    node.uid,
                    node.slicer_area._data,
                    spec,
                    parent_uid=node.parent_uid,
                    allow_program=True,
                )
                if publish is not None:
                    publish()
                    return
        self._note_producers(steps)
        opaque = (
            not steps
            and spec is not None
            and (
                not getattr(node, "watched", False) or node._provenance_spec is not None
            )
        )
        with self.changing(node.uid):
            self.graph.prepare_import(
                node.uid,
                node.slicer_area._data,
                steps,
                live=True,
                prepare_view=self._prepare_view,
                imported_history=spec.model_dump_json()
                if opaque and spec is not None
                else None,
            )()
            if opaque:
                self._checkpoint_sources.add(node.uid)

    def _adopt_parent_binding(
        self, node: _ManagedWindowNode, relative: ToolProvenanceSpec
    ) -> bool:
        parent = self.manager._parent_node(node)
        if node.source_state == "fresh":
            publish = self.prepare_imported_transform(
                node.uid,
                parent,
                relative,
                node.slicer_area._data,
                parent_uid=node.parent_uid,
                auto=node.source_auto_update,
            )
            if publish is None:
                return False
            publish()
            return True
        if not self._ensure_source(parent):
            return False
        # The workspace retained the stale result and its live binding. The
        # parent's current revision did not produce the array on screen.
        next_spec = script(
            start_label="Refresh from source ImageTool",
            seed_code="derived = data",
            active_name="derived",
            script_inputs=(
                ScriptInput(name="data", source_spec=relative.model_dump(mode="json")),
            ),
        )
        pending = Recipe.bind(RecordedProgram(spec=next_spec), Follow(parent.uid))
        self._note_producers((pending,))
        with self.changing(node.uid):
            self.graph.prepare_import(
                node.uid,
                node.slicer_area._data,
                (),
                auto=node.source_auto_update,
                parent_id=node.parent_uid,
                next_recipe=pending,
                live=True,
                prepare_view=self._prepare_view,
            )()
        return True

    def retire_watched_input(self, node: _ManagedWindowNode) -> None:
        """Stop using a notebook name as soon as its next value arrives."""
        self.forget_watched_input(node.uid)
        if node.uid in self._imported:
            # Materialize the saved inputs once. The imported window's portable
            # recipe can still refer to the previous notebook value by name.
            self.displayed_edit_revision(node)
            self._imported.discard(node.uid)
            self._exports.clear()

    def forget_watched_input(self, uid: str) -> None:
        if self._watched_sources.pop(uid, None) is not None:
            self._exports.clear()

    def _bind_watched_input(self, uid: str, name: str, revision: str) -> None:
        if not erlab.utils.misc._is_valid_identifier(name):
            return
        store = self.manager._provenance_revisions
        if not store.has_data(revision):
            with self.graph.retain(revision):
                reader = self.graph.checkpoint_reader(revision)
            if reader is None:
                raise ValueError("A watched source requires a checkpoint")
            if revision in store:
                store.attach_checkpoint_reader(revision, reader)
            else:
                store.add(
                    revision_id=revision,
                    data_reader=reader,
                    source_owner_uid=uid,
                )
        self._watched_sources[uid] = (
            revision,
            script(
                start_label=f"Start from watched variable {name!r}",
                seed_code=f"derived = {name}",
                active_name="derived",
                script_inputs=(
                    ScriptInput(
                        name=name,
                        label="Saved",
                        revision_id=revision,
                        follow_source=False,
                    ),
                ),
            ),
        )
        self._exports.clear()

    def _watched_seed(
        self, uid: str, revision: str
    ) -> tuple[str, ToolProvenanceSpec] | None:
        seed = self._watched_sources.get(uid)
        source = self.manager._tool_graph.nodes.get(uid)
        if (
            seed is None
            or source is None
            or not getattr(source, "watched", False)
            or source.imagetool is None
            or seed[0] != revision
        ):
            return None
        name = getattr(source, "_watched_varname", None)
        return (name, seed[1]) if name else None

    def watched_name(self, revision: str | None) -> str | None:
        if revision is None or not self.graph.has_revision(revision):
            return None
        seed = self._watched_seed(self.graph.owner(revision), revision)
        return seed[0] if seed else None

    def watched_persistence_spec(self, uid: str) -> ToolProvenanceSpec | None:
        revision = self.source_revision(uid, displayed=False)
        if self.watched_name(revision) is not None:
            # An unchanged watched array is itself the workspace's main payload.
            return None
        # Include a retired origin even if it has no steps. Otherwise the legacy
        # workspace reader would associate it with the newer notebook variable.
        return self.export_spec(revision, target_uid=uid)

    def _prepare_view(
        self, uid: str, prior: Pin | EarlierStep, source: xr.DataArray
    ) -> list[Recipe]:
        area = self.manager._node_for_target(uid).slicer_area
        area._guard_lazy_data(lazy_input_guards(source))
        dims = area._public_source_dims_for_current_layout()
        steps = self._view_steps(
            uid, prior, source.dims, dims, area._accepted_filter_provenance_operation
        )
        self._require_single_write(uid)
        self._view_state[uid] = (
            weakref.ref(area._data),
            dims,
            area.accepted_filter_revision,
        )
        if steps:
            self._views[uid] = prior.revision_id, len(steps)
        else:
            self._views.pop(uid, None)
        return steps

    def _install_evaluated_source(
        self, uid: str, prior: Pin | EarlierStep, source: xr.DataArray
    ) -> list[Recipe]:
        area = self.manager._node_for_target(uid).slicer_area
        # Input preparation can change values and coordinates. Use the revision
        # the graph evaluated, preserving the slicer's navigation and file handle.
        area._data = (
            source.compute()
            if source.chunks is not None and area._data.chunks is None
            else source
        )
        area._own_managed_source()
        return self._prepare_view(uid, prior, source)

    @staticmethod
    def _view_steps(
        uid: str,
        prior: Pin | EarlierStep,
        source_dims: tuple,
        dims: tuple,
        operation: ToolProvenanceOperation | None,
    ) -> list[Recipe]:
        steps = []
        if operation is not None:
            steps.append(Recipe(operation, (("data", prior),), view_role="filter"))
        if dims != source_dims:
            steps.append(
                Recipe(
                    TransposeOperation(dims=dims),
                    (("data", Follow(uid) if steps else prior),),
                    view_role="layout",
                )
            )
        return steps

    def handles_filter(self, node: _ManagedWindowNode) -> bool:
        return (
            node.uid not in self._changing
            and not node._suspend_snapshot_token_updates
            and self._ensure_source(node)
        )

    def prepare_filter(
        self,
        node: _ManagedWindowNode,
        operation: ToolProvenanceOperation | None,
        *,
        update: bool,
        emit_edited: bool,
    ) -> tuple[Callable[[], None], Callable[[], None]]:
        uid = node.uid
        if not self._ensure_source(node):
            raise ValueError("The display filter requires a registered source")
        check = self.observe_input(uid)
        depth = self._views[uid][1] if uid in self._views else 0
        prior = self.graph.view_input(uid, depth)
        area = node.slicer_area
        # A filter always reads the source in its original dimension order.
        # The unfiltered public view may already be transposed.
        operation = (
            operation.capture_execution_options().model_copy(deep=True)
            if operation is not None
            else None
        )
        layout = area._public_source_dims_for_current_layout()
        source = area._data
        if operation is not None and provenance_operation_requires_code_trust(
            operation
        ):
            raise TypeError(
                "Operations that can execute stored Python cannot be used "
                "as display filters"
            )
        graph_source = self.graph.calculation_data(prior.revision_id)
        steps = self._view_steps(uid, prior, graph_source.dims, layout, operation)
        filtered = None
        publish = None
        if steps:
            publish = self.graph.prepare_evaluation(
                uid,
                steps,
                parent_id=node.parent_uid,
                auto=self.graph.auto_enabled(uid),
            )
            if operation is not None:
                filtered = area._normalize_filter_result_for_source_dims(
                    graph_source, publish.value, tuple(area.data.dims)
                )

        def install() -> None:
            check()
            with self.replacing(node):
                area._install_filter(
                    operation, filtered, update=update, preview=False, emit_edited=False
                )
                self._require_single_write(uid)
                if (
                    area._data is not source
                    or area._public_source_dims_for_current_layout() != layout
                    or area._accepted_filter_provenance_operation != operation
                ):
                    raise _SupersededEvaluation(
                        "ImageTool changed while installing its display filter"
                    )
                if publish is not None:
                    self._views[uid] = prior.revision_id, len(steps)
                    publish()
                elif uid in self._views:
                    self._views.pop(uid)
                    self.graph.restore_view(uid, prior)
                self._view_state[uid] = (
                    weakref.ref(area._data),
                    layout,
                    area.accepted_filter_revision,
                )
            if emit_edited:
                area.sigSourceDataReplaced.emit(area._tool_source_parent_data())
                area.sigDataEdited.emit()

        return check, install

    def _record_view(self, node: _ManagedWindowNode) -> None:
        area = node.slicer_area
        uid = node.uid
        state = self._view_state.get(uid)
        if (
            state
            and state[0]() is area._data
            and state[1:]
            == (
                area._public_source_dims_for_current_layout(),
                area.accepted_filter_revision,
            )
        ):
            return
        depth = self._views[uid][1] if uid in self._views else 0
        base = self.graph.view_input(uid, depth)
        dims = area._public_source_dims_for_current_layout()
        prior = base
        filter_depth = 0
        if (
            state
            and state[0]() is area._data
            and state[2] == area.accepted_filter_revision
            and area._accepted_filter_provenance_operation is not None
            and depth
        ):
            # Only the axis order changed. Reuse the accepted filter revision.
            prior = self.graph.view_input(uid, depth - 1)
            filter_depth = 1
        source = self.graph.calculation_data(prior.revision_id)
        steps = self._view_steps(
            uid,
            prior,
            source.dims,
            dims,
            None if filter_depth else area._accepted_filter_provenance_operation,
        )
        if steps:
            self.graph.prepare_evaluation(uid, steps)()
        elif depth:
            self.graph.restore_view(uid, prior)
        depth = filter_depth + len(steps)
        if depth:
            self._views[uid] = base.revision_id, depth
        else:
            self._views.pop(uid, None)
        self._view_state[uid] = (
            weakref.ref(area._data),
            dims,
            area.accepted_filter_revision,
        )

    def view_changed(self, node: _ManagedWindowNode) -> None:
        state = self._view_state.get(node.uid)
        if (
            node.uid in self.graph
            and node.uid not in self._changing
            and not node._suspend_snapshot_token_updates
            and node.imagetool is not None
            and (state is None or state[0]() is node.slicer_area._data)
        ):
            # The slicer clears its old filter before installing a new source array.
            # Its source-data signal publishes that change after the layout is ready.
            with self.changing(node.uid):
                self._record_view(node)

    def source_changed(self, node: _ManagedWindowNode, *, signal: bool = False) -> None:
        uid = node.uid
        if signal and uid in self._source_writes:
            self._source_writes[uid].signals += 1
        if not node._suspend_snapshot_token_updates and uid not in self._changing:
            self.forget_watched_input(uid)
        if (
            uid in self.graph
            and not node._suspend_snapshot_token_updates
            and uid not in self._changing
            and node.imagetool is not None
        ):
            with self.changing(uid):
                self.prepare_source_update(node)()

    def prepare_source_update(
        self,
        node: _ManagedWindowNode,
        source_write: _SourceWrite | None = None,
    ) -> Callable[[], str]:
        """Record the input supplied by an external write or a source reload."""
        uid = node.uid
        area = node.slicer_area
        if isinstance(source_write, _RawReplacement):
            replacement = source_write
            if replacement.publish is not None:
                return self._commit_publication(
                    uid,
                    replacement.publish,
                    imported=False,
                    watched_name=replacement.watched_name,
                    raw_revision=replacement.raw_revision,
                )
            if replacement.watched_name is not None:
                publish = self.graph.prepare_source(
                    uid,
                    area._data,
                    parent_id=node.parent_uid,
                    prepare_view=self._prepare_view,
                )
                return self._commit_publication(
                    uid, publish, imported=False, watched_name=replacement.watched_name
                )
            # A plain update keeps the saved executor of an explicitly imported
            # workflow. Live windows keep only the graph's next-update recipe.
        if (
            isinstance(source_write, _ImageToolInput)
            or source_write == "file"
            or uid not in self.graph
        ):
            tool = self.manager.get_imagetool(uid)
            return self.prepare_input(
                uid,
                tool,
                source_write
                if isinstance(source_write, _ImageToolInput)
                else _ImageToolInput.from_window(tool),
                parent_uid=node.parent_uid,
            )
        current_recipe = self.graph.entry(uid).recipe
        steps: tuple[FileLoad, ...] = ()
        spec = node.provenance_spec if uid in self._imported else None
        if node.slicer_area._source_reload_in_progress and spec is not None:
            source = spec.file_load_source
            if source is not None and source.replay_call is not None and not spec.steps:
                steps = (
                    FileLoad(self.graph.file_source(source.path), source.replay_call),
                )
        if node.slicer_area._source_reload_in_progress and isinstance(
            current_recipe, FileLoad
        ):
            steps = (current_recipe,)
        publish = self.graph.prepare_source(
            uid,
            node.slicer_area._data,
            file=steps[0] if steps else None,
            next_recipe=self.graph.replacement_recipe(uid) if not steps else None,
            prepare_view=self._prepare_view,
            imported_history=spec.model_dump_json()
            if spec is not None and not steps
            else None,
        )
        return self._commit_publication(uid, publish, imported=uid in self._imported)

    def capture_input(self, node: _ManagedWindowNode, item: ScriptInput) -> None:
        if item.revision_id is not None and self._ensure_source(node):
            self.graph.capture_input(
                Follow(node.uid, source=item.data_role == "source"),
                item.revision_id,
                item._capture if isinstance(item, ConsoleInput) else item,
            )

    def calculate(
        self,
        operations: Sequence[ToolProvenanceOperation],
        inputs: Sequence[_WindowInput],
        *,
        input_names: tuple[str, ...] = (),
        output_name: str | None = None,
        source_kind: typing.Literal["full_data", "public_data", "selection"]
        | None = None,
        keep_name: bool = False,
        replace_uid: str | None = None,
    ) -> _CalculatedResult:
        """Evaluate the captured operations and retain their result before placement."""
        request = _OperationRequest(
            tuple(operations),
            tuple(item.revision for item in inputs),
            input_names,
            output_name,
            source_kind,
        )
        result = self.evaluate_request(request, replace_uid=replace_uid)
        if keep_name:
            source_name = self.graph.calculation_data(request.inputs[0].revision).name
            if result.data.name != source_name:
                revision = result.capture.revision_id
                result = self.evaluate_request(
                    _OperationRequest(
                        (RenameOperation(name=typing.cast("str | None", source_name)),),
                        (_InputRevision(EarlierStep(revision), revision),),
                        output_name=output_name,
                    ),
                    replace_uid=result.window_uid,
                )
        for item in inputs:
            item.require_current()
        return result

    def transform(
        self,
        source: _WindowInput,
        spec: ToolProvenanceSpec,
        *,
        keep_name: bool = False,
        replace_uid: str | None = None,
    ) -> _CalculatedResult:
        kind = typing.cast(
            'typing.Literal["full_data", "public_data", "selection"]', spec.kind
        )
        return self.calculate(
            spec.operations or (_SourceViewOperation(source_kind=kind),),
            (source,),
            source_kind=kind,
            keep_name=keep_name,
            replace_uid=replace_uid,
        )

    def evaluate_request(
        self,
        request: _OperationRequest,
        *,
        replace_uid: str | None = None,
    ) -> _CalculatedResult:
        """Evaluate the ordered recipe using the graph values of its exact operands."""
        uid = replace_uid or self.manager._next_node_uid()
        first, *rest = (op.capture_execution_options() for op in request.operations)
        slots = ("data", *first.input_slots())
        inputs = tuple(zip(slots, request.inputs, strict=True))
        steps = [
            Recipe(
                first,
                tuple((slot, item.binding) for slot, item in inputs),
                input_names=request.input_names,
                output_name=request.output_name,
                source_kind=request.source_kind,
            ),
            *(Recipe.bind(op, Follow(uid)) for op in rest),
        ]
        self._note_producers(steps)
        holder = provenance_graph.CapturedRevision()
        self.graph.prepare_evaluation(
            uid,
            steps,
            calculation_holder=holder,
            window_calculation=True,
            recorded_inputs={(0, slot): item.revision for slot, item in inputs},
        )()
        return _CalculatedResult(holder, request.output_name, uid)

    def accept_worker_result(
        self, result: provenance_graph.WorkerCalculation
    ) -> _CalculatedResult:
        self._note_producers((result.recipe,))
        holder = self.graph.retain_completed(result)
        return _CalculatedResult(holder, None, result.owner)

    @staticmethod
    def window_preparation(tool: ImageTool) -> tuple[ToolProvenanceOperation, ...]:
        area = tool.slicer_area
        operations = area._load_preparation_operations
        cast = (
            area._source_input_dtype is not None
            and area._source_input_dtype not in (np.float32, np.float64)
        )
        if cast or area._degree_coords:
            operations = (
                *operations,
                ImageToolInputOperation(
                    cast_float64=cast, degree_coords=area._degree_coords
                ),
            )
        return operations

    def prepare_imported_transform(
        self,
        uid: str,
        source: _ManagedWindowNode,
        spec: ToolProvenanceSpec,
        data: xr.DataArray,
        *,
        parent_uid: str | None,
        auto: bool = False,
        input_dtype: np.dtype | None = None,
    ) -> Callable[[], str] | None:
        if (
            self.manager._workspace_state.loading_depth
            or not spec.is_live_source
            or not self._ensure_source(source)
        ):
            return None
        operations = spec.operations or (
            _SourceViewOperation(
                source_kind=typing.cast(
                    "typing.Literal['full_data', 'public_data', 'selection']", spec.kind
                )
            ),
        )
        steps = [
            Recipe(
                op,
                (("data", Follow(source.uid if i == 0 else uid)),),
                source_kind=typing.cast(
                    'typing.Literal["full_data", "public_data", "selection"]', spec.kind
                ),
            )
            for i, op in enumerate(operations)
        ]
        return self._prepare_import(
            uid, data, steps, parent_uid=parent_uid, auto=auto, input_dtype=input_dtype
        )

    def prepare_replacement(
        self,
        node: _ManagedWindowNode,
        data: xr.DataArray,
        spec: ToolProvenanceSpec | None,
        *,
        staged: _ProvenanceRevisionStore | None = None,
    ) -> Callable[[], str] | None:
        if node.uid in self._changing or self.manager._workspace_state.loading_depth:
            return None
        if (
            spec is not None
            and spec.kind == "file"
            and spec.file_load_source is not None
        ):
            call = spec.file_load_source.replay_call
            if call is not None:
                if spec.steps:
                    call = PreparedFileCall(**call.model_dump(), steps=spec.steps)
                return self._prepare_import(
                    node.uid,
                    data,
                    (self._file_edge(spec, call),),
                    parent_uid=node.parent_uid,
                    imported=node.uid in self._imported,
                )
        if spec is None:
            return self._prepare_import(node.uid, data, (), parent_uid=node.parent_uid)
        return self.prepare_imported_workflow(
            node.uid,
            data,
            spec,
            parent_uid=node.parent_uid,
            allow_program=True,
            staged=staged,
            imported=node.uid in self._imported,
        )

    def prepare_calculation(
        self,
        uid: str,
        result: _CalculatedResult,
        *,
        parent_uid: str | None,
        tool: ImageTool,
        item_update: bool = False,
        auto: bool = False,
    ) -> Callable[[], str]:
        data = tool.slicer_area._data
        output_name = result.output_name
        binding = self.graph.calculation_input(result.capture.revision_id, target=uid)
        if item_update:
            # Item writes restore source dimension order and keep the layout.
            operations: tuple[ToolProvenanceOperation, ...] = (
                (TransposeOperation(dims=data.dims),)
                if result.data.dims != data.dims
                else ()
            )
        else:
            operations = self.window_preparation(tool)
        if operations:
            steps = [
                Recipe(
                    op.capture_execution_options(),
                    (("data", binding if i == 0 else Follow(uid)),),
                    output_name=output_name,
                )
                for i, op in enumerate(operations)
            ]
            publish = self.graph.prepare_evaluation(
                uid,
                steps,
                parent_id=parent_uid,
                auto=auto,
                prepare_view=self._install_evaluated_source,
            )
            return self._commit_publication(uid, publish, imported=False)
        publish = self.graph.prepare_calculation(
            uid,
            result.capture,
            parent_id=parent_uid,
            output_name=output_name,
            auto=auto,
            prepare_view=self._prepare_view,
        )
        return self._commit_publication(uid, publish, imported=False)

    def prepare_imported_workflow(
        self,
        uid: str,
        data: xr.DataArray,
        spec: ToolProvenanceSpec | None,
        *,
        parent_uid: str | None,
        allow_program: bool = False,
        imported: bool = False,
        staged: _ProvenanceRevisionStore | None = None,
        auto: bool = False,
        input_dtype: np.dtype | None = None,
    ) -> Callable[[], str] | None:
        """Resolve the inputs recorded in a saved workflow."""
        if uid in self._changing or self.manager._workspace_state.loading_depth:
            return None

        if spec is None or spec.file_load_source is not None:
            return None

        def checkpoint(reason: str) -> Callable[[], str] | None:
            # Preserve the existing executor of an imported recipe. For a new
            # console/paste result, display exactly why the result cannot update.
            if not allow_program or imported or uid in self._adopting:
                return None
            values = (
                data
                if data.dtype in (np.float32, np.float64)
                else data.astype(np.float64)
            )
            publish = self.graph.prepare_import(
                uid, values, (), parent_id=parent_uid, prepare_view=self._prepare_view
            )

            def commit() -> str:
                with self.changing(uid):
                    self._require_single_write(uid)
                    key = publish()
                    self.graph.set_auto(uid, enabled=False)
                    self._checkpoint_reasons[uid] = key, reason
                    return key

            return commit

        checkpoints: dict[str, tuple[str, xr.DataArray]] = {}
        intermediates: dict[str, tuple[Recipe, Mapping[str, str]]] = {}
        file_origins: dict[str, FileLoad] = {}
        checkpoint_histories: dict[str, str] = {}
        try:
            steps, recorded = self._bind_workflow(
                uid,
                spec,
                consume_target=False,
                allow_program=allow_program,
                checkpoints=checkpoints,
                intermediates=intermediates,
                file_origins=file_origins,
                checkpoint_histories=checkpoint_histories,
                staged=staged,
            )
        except _UnboundInput as exc:
            return checkpoint(str(exc))
        return self._prepare_import(
            uid,
            data,
            steps,
            imported=imported,
            parent_uid=parent_uid,
            auto=auto,
            checkpoints=checkpoints,
            recorded_inputs=recorded,
            intermediates=intermediates,
            file_origins=file_origins,
            checkpoint_histories=checkpoint_histories,
            input_dtype=input_dtype,
        )

    def _bind_workflow(
        self,
        uid: str,
        spec: ToolProvenanceSpec,
        *,
        consume_target: bool,
        allow_program: bool,
        checkpoints: dict[str, tuple[str, xr.DataArray]],
        intermediates: dict[str, tuple[Recipe, Mapping[str, str]]],
        file_origins: dict[str, FileLoad],
        checkpoint_histories: dict[str, str],
        staged: _ProvenanceRevisionStore | None,
    ) -> tuple[list[Recipe], dict[tuple[int, str], str]]:
        if not spec.replay_inputs:
            raise _UnboundInput(
                "This calculation has no recorded input. "
                "Repeat it with a named ImageTool input to enable updates."
            )
        primary, use_program = self._workflow_primary(spec)
        local = _selected_notebook_spec(spec)
        if use_program and (
            not allow_program
            or any(
                getattr(step.operation, "op", None) == "script_code"
                and not getattr(step.operation, "copyable", False)
                for step in local.steps
            )
        ):
            raise _UnboundInput(
                "The calculation did not record replayable code. "
                "Repeat it with a supported operation to enable updates."
            )
        bindings: dict[str, Follow | Pin | EarlierStep] = {}
        historical: dict[str, str] = {}
        local_inputs = []

        def stage_workflow(
            nested: ToolProvenanceSpec, result_key: str | None = None
        ) -> EarlierStep:
            nested_steps, used = self._bind_workflow(
                uid,
                nested,
                allow_program=True,
                consume_target=consume_target,
                checkpoints=checkpoints,
                intermediates=intermediates,
                staged=staged,
                file_origins=file_origins,
                checkpoint_histories=checkpoint_histories,
            )
            previous: EarlierStep | None = None
            for index, nested_step in enumerate(nested_steps):
                if previous is not None:
                    nested_step = nested_step._with_inputs(
                        (
                            ("data", previous),
                            *nested_step.inputs[1:],
                        )
                    )
                revision = (
                    result_key
                    if result_key is not None and index == len(nested_steps) - 1
                    else uuid.uuid4().hex
                )
                intermediates[revision] = (
                    nested_step,
                    {slot: key for (n, slot), key in used.items() if n == index},
                )
                previous = EarlierStep(revision)
            if previous is None:
                raise _UnboundInput("The console input contains no recorded steps.")
            return previous

        for item, argument in zip(spec.replay_inputs, local.replay_inputs, strict=True):
            key = item.resolved_revision_id or item.revision_id
            store = (
                staged
                if staged is not None and key in staged
                else self.manager._provenance_revisions
            )
            selection = item.source_spec
            source_uid = item.node_uid
            if source_uid is None and item.revision_id is None:
                nested = item.parsed_provenance_spec()
                if nested is not None and nested.replay_inputs:
                    bindings[item.name] = stage_workflow(nested)
                    local_inputs.append(argument)
                    continue
            source = self.manager._tool_graph.nodes.get(source_uid or "")
            if not item.follow_source or source is None:
                key = item.resolved_revision_id or item.revision_id
                captured = self.graph.captured_input(key)
                if key is None or (captured is None and not store.has_data(key)):
                    raise _UnboundInput(
                        f"The recorded input {item.name!r} is unavailable. "
                        "Supply the input and repeat the calculation to enable updates."
                    )
                record = store.get(key)
                owner = record.source_owner_uid if record is not None else None
                if (
                    consume_target
                    and owner == uid
                    and key
                    == self.manager._node_for_target(uid).revision_id_for_role(
                        item.data_role
                    )
                ):
                    source_uid = uid
                else:
                    bindings[item.name] = Pin(captured or key)
                    if (
                        captured is None
                        and not self.graph.has_revision(key)
                        and key not in checkpoints
                    ):
                        checkpoints[key] = (owner or item.name, store.resolve_data(key))
                        origin = item.parsed_provenance_spec()
                        if (
                            origin is not None
                            and origin.kind == "file"
                            and origin.file_load_source is not None
                        ):
                            source_file = origin.file_load_source
                            file_call = source_file.replay_call
                            if file_call is not None:
                                if origin.steps:
                                    file_call = PreparedFileCall(
                                        **file_call.model_dump(), steps=origin.steps
                                    )
                                file_origins[key] = self._file_edge(origin, file_call)
                                if item.follow_source:
                                    bindings[item.name] = EarlierStep(key)
                        elif (
                            origin is not None
                            and origin.replay_inputs
                            and all(
                                arg.revision_id != key for arg in origin.replay_inputs
                            )
                        ):
                            stage_workflow(origin, key)
                        elif origin is not None:
                            checkpoint_histories[key] = (
                                origin.to_replay_spec().model_dump_json()
                            )
                    local_inputs.append(
                        argument.model_copy(
                            update={
                                "source_spec": None
                                if key == item.resolved_revision_id
                                else selection
                            }
                        )
                    )
                    continue
            node = (
                self.manager._tool_graph.nodes.get(source_uid)
                if source_uid is not None
                else None
            )
            if node is None or not node.is_imagetool or not self._ensure_source(node):
                raise _UnboundInput(
                    f"Input {item.name!r} has no available ImageTool revision. "
                    "Reopen the input and repeat the calculation to enable updates."
                )
            if (
                item.node_snapshot_token is not None
                and item.node_snapshot_token
                != node.snapshot_token_for_role(item.data_role)
            ):
                key = item.revision_id
                if not selection:
                    key = item.resolved_revision_id or key
                captured = self.graph.captured_input(key)
                if key is None or (captured is None and not store.has_data(key)):
                    raise _UnboundInput(
                        f"Input {item.name!r} changed before publication. "
                        "Its earlier revision is unavailable. Repeat the calculation "
                        "with its current data to enable updates."
                    )
                historical[item.name] = captured or key
                if captured is None and not self.graph.has_revision(key):
                    checkpoints[key] = (node.uid, store.resolve_data(key))
            local_inputs.append(argument.model_copy(update={"source_spec": selection}))
            bindings[item.name] = Follow(node.uid, source=item.data_role == "source")
        for index, item in enumerate(local_inputs):
            selection = item.parsed_source_spec()
            if selection is None:
                continue
            for operation in selection.operations:
                revision = uuid.uuid4().hex
                earlier = historical.pop(item.name, None)
                intermediates[revision] = (
                    Recipe.bind(operation, bindings[item.name]),
                    {"data": earlier} if earlier is not None else {},
                )
                bindings[item.name] = EarlierStep(revision)
            local_inputs[index] = item.model_copy(update={"source_spec": None})
        local = local.with_replay_inputs(local_inputs)
        return self._workflow_steps(
            uid, local, primary, use_program, bindings, historical
        )

    @staticmethod
    def _workflow_primary(spec: ToolProvenanceSpec) -> tuple[str | None, bool]:
        primary = spec.replay_primary_input
        if primary is None and len(spec.replay_inputs) == 1:
            name = spec.replay_inputs[0].name
            if spec.seed_code == f"{spec.active_name} = {name}":
                primary = name
        use_program = (
            primary is None
            or not spec.steps
            or any(
                step.context_names
                or step.legacy_context
                or getattr(step.operation, "op", None) == "script_code"
                for step in spec.steps
            )
        )
        if primary is not None and spec.seed_code not in (
            None,
            f"{spec.active_name} = {primary}",
        ):
            use_program = True
        return primary, use_program

    @staticmethod
    def _workflow_steps(
        uid: str,
        local: ToolProvenanceSpec,
        primary: str | None,
        use_program: bool,
        bindings: Mapping[str, Follow | Pin | EarlierStep],
        historical: Mapping[str, str],
    ) -> tuple[list[Recipe], dict[tuple[int, str], str]]:
        recorded: dict[tuple[int, str], str] = {}
        if use_program:
            program = RecordedProgram(spec=local)
            first, *others = local.replay_inputs
            steps = [
                Recipe.bind(
                    program,
                    bindings[first.name],
                    **{
                        slot: bindings[item.name]
                        for slot, item in zip(
                            program.input_slots(), others, strict=True
                        )
                    },
                )
            ]
            for slot, item in zip(
                ("data", *program.input_slots()), local.replay_inputs, strict=True
            ):
                if item.name in historical:
                    recorded[0, slot] = historical[item.name]
        else:
            primary = typing.cast("str", primary)
            steps = []
            for index, step in enumerate(local.steps):
                names = (
                    primary,
                    *(
                        step.input_bindings[slot]
                        for slot in step.operation.input_slots()
                    ),
                )
                step_bindings = (
                    ("data", bindings[primary] if index == 0 else Follow(uid)),
                    *(
                        (slot, bindings[name])
                        for slot, name in zip(
                            step.operation.input_slots(), names[1:], strict=True
                        )
                    ),
                )
                steps.append(
                    Recipe(
                        step.operation,
                        step_bindings,
                        input_names=names,
                        output_name=local.active_name,
                        source_kind="public_data"
                        if step.input_policy == "restored"
                        else "full_data"
                        if step.input_policy == "current"
                        else None,
                    )
                )
            if primary in historical:
                recorded[0, "data"] = historical[primary]
            for index, step in enumerate(local.steps):
                for slot, name in step.input_bindings.items():
                    if name in historical:
                        recorded[index, slot] = historical[name]
        return steps, recorded

    def _authorize_console_call(
        self, entries: tuple[CodeTrustEntry, ...]
    ) -> object | None:
        _, capability = issue_local_edit_capability(
            self.manager._workspace_state.code_trust, entries, edited_entries=entries
        )
        return capability

    def calculate_console(
        self,
        operation: ToolProvenanceOperation,
        operands: Sequence[ScriptInput],
        holder: provenance_graph.CapturedRevision,
    ) -> None:
        """Execute and retain one application of the exact captured console operands."""
        if self.manager._workspace_state.loading_depth:
            raise RuntimeError("Cannot capture a calculation during a workspace change")
        names = tuple(item.name for item in operands)
        program = isinstance(operation, RecordedProgram)
        if isinstance(operation, RecordedProgram):
            # Store Python names with the code. The graph owns their values.
            local = operation.spec.with_replay_inputs(
                tuple(
                    ScriptInput(name=item.name, label=item.label) for item in operands
                )
            ).as_bound_script()
            operation = RecordedProgram(spec=local)
        bindings: list[tuple[str, Follow | Pin | EarlierStep]] = []
        input_edges = []
        for slot, operand in zip(
            ("data", *operation.input_slots()), operands, strict=True
        ):
            if not isinstance(operand, ConsoleInput):
                raise TypeError(
                    "A live calculation requires a captured console operand"
                )
            key = operand._capture.revision_id
            if not self.graph.has_revision(key):
                raise ValueError("Console operands must belong to the same Manager")
            source = operand._capture.source
            if isinstance(operand, CalculatedInput):
                binding: Follow | Pin | EarlierStep = self.graph.calculation_input(key)
            elif source is not None and source.entry_id in self.graph:
                binding = source
            else:
                binding = Pin(key)
            bindings.append((slot, binding))
            input_edges.append((slot, key))
        edges = tuple(input_edges)

        step = Recipe(
            operation,
            tuple(bindings),
            input_names=() if program else names,
            output_name=None if program else "derived",
        )
        self._note_producers((step,))
        previous = self._console_program
        self._console_program = (
            typing.cast("RecordedProgram", step._captured.operation)
            if program
            else None
        )
        try:
            self.graph.prepare_evaluation(
                uuid.uuid4().hex,
                (step,),
                calculation_holder=holder,
                recorded_inputs={(0, slot): key for slot, key in edges},
            )()
        finally:
            self._console_program = previous

    def _prepare_import(
        self,
        uid: str,
        data: xr.DataArray,
        steps: Sequence[Recipe | FileLoad],
        *,
        parent_uid: str | None,
        auto: bool = False,
        imported: bool = False,
        checkpoints: Mapping[str, tuple[str, xr.DataArray]] | None = None,
        recorded_inputs: Mapping[tuple[int, str], str] | None = None,
        intermediates: Mapping[str, tuple[Recipe, Mapping[str, str]]] | None = None,
        file_origins: Mapping[str, FileLoad] | None = None,
        checkpoint_histories: Mapping[str, str] | None = None,
        input_dtype: np.dtype | None = None,
    ) -> Callable[[], str]:
        file_data = data if steps and isinstance(steps[0], FileLoad) else None
        if (input_dtype if input_dtype is not None else data.dtype) not in (
            np.float32,
            np.float64,
        ):
            if steps:
                steps = (
                    *steps,
                    Recipe.bind(
                        ImageToolInputOperation(cast_float64=True), Follow(uid)
                    ),
                )
            if data.dtype != np.float64:
                data = data.astype(np.float64)
        self._note_producers(steps)
        publish = self.graph.prepare_import(
            uid,
            data,
            steps,
            parent_id=parent_uid,
            auto=auto,
            file_data=file_data,
            checkpoints=checkpoints,
            recorded_inputs=recorded_inputs,
            intermediates=intermediates,
            file_origins=file_origins,
            checkpoint_histories=checkpoint_histories,
            prepare_view=self._prepare_view,
        )

        return self._commit_publication(uid, publish, imported=imported)

    def _commit_publication(
        self,
        uid: str,
        publish: Callable[[], str],
        *,
        imported: bool,
        watched_name: str | None = None,
        raw_revision: str | None = None,
    ) -> Callable[[], str]:
        def commit() -> str:
            with self.changing(uid):
                self._require_single_write(uid)
                revision = publish()
                if watched_name is not None:
                    self._bind_watched_input(
                        uid,
                        watched_name,
                        raw_revision or self.source_revision(uid, displayed=False),
                    )
                if imported or uid in self._adopting:
                    self._imported.add(uid)
                else:
                    self._imported.discard(uid)
                return revision

        return commit

    def remove(self, uid: str) -> None:
        self._checkpoint_reasons.pop(uid, None)
        self._branched.discard(uid)
        self._exports.pop((uid, True), None)
        self._exports.pop((uid, False), None)
        self._imported.discard(uid)
        self._checkpoint_sources.discard(uid)
        self._watched_sources.pop(uid, None)
        self._views.pop(uid, None)
        self._view_state.pop(uid, None)
        if uid in self.graph:
            self.graph.remove_entry(uid)
            self.manager._queue_idle_work(
                ("data-graph-inspector",), self._refresh_inspector
            )

    def label(self, uid: str) -> str:
        node = self.manager._tool_graph.nodes.get(uid)
        if node is None:
            return f"Closed ImageTool {uid}"
        return self.manager._details_panel._script_input_current_node_label(node)

    def _revision_owner_label(self, revision: str) -> str:
        if self.graph.is_calculation(revision):
            return "Console calculation"
        return self.label(self.graph.owner(revision))

    def revision_label(self, revision: str) -> str:
        return f"{self._revision_owner_label(revision)} [{revision[:8]}]"

    def unavailable(self, uid: str) -> str | None:
        if reason := self._checkpoint_reason(uid):
            return reason
        missing = {
            source
            for source in self.graph.follows(uid)
            if source not in self.manager._tool_graph.nodes
            and not self.graph.is_closed(source)
        }
        if missing:
            return (
                "Missing input: "
                + ", ".join(self.label(source) for source in sorted(missing))
                + ". Reopen the input or select a new source."
            )
        return None

    def _extension_issue(self, plan: UpdatePlan) -> str | None:
        for name, version, kind, capability in dict.fromkeys(
            self._extension_capabilities((*plan.file_loads, *plan.operations))
        ):
            status = self.manager._extensions.capability_status(
                name, version, kind, capability
            )
            if status != "ready":
                return (
                    f"Extension {kind} {capability!r} from {name!r} is unavailable "
                    f"({status}). Open Manage Extensions and enable the recorded "
                    f"version ({version[:12]})."
                )
        return None

    def command_issue(self, uid: str, command: _Command) -> str | None:
        if uid in self._staged_windows:
            return "This ImageTool is being updated. Retry after the update."
        if command == "reload" and not self.manages(uid):
            return "This ImageTool holds a memory input with no recorded reload source."
        reason = self.unavailable(uid)
        if reason is None:
            plan = self.graph.plan_update(
                uid, command=command, include_operations=self._uses_extensions
            )
            for load in plan.file_loads:
                if not pathlib.Path(load.source.path).is_file():
                    return (
                        f"The recorded source file is unavailable: {load.source.path}"
                    )
            if plan.input_issue is not None:
                resolution = (
                    "Reload cannot replace this fixed input. Open the current file "
                    "in a new ImageTool and repeat the calculation."
                    if command == "reload"
                    else "Use Reload to read the current files. For a fixed input, "
                    "open the current file in a new ImageTool and repeat "
                    "the calculation."
                )
                return f"{plan.input_issue} {resolution}"
            if self._uses_extensions:
                reason = self._extension_issue(plan)
        return reason

    def preview(self, uid: str, command: _Command) -> str:
        if reason := self.unavailable(uid):
            return reason
        try:
            plan = self.graph.plan_update(
                uid, command=command, include_operations=self._uses_extensions
            )
        except KeyError as exc:
            return f"Missing input {exc}. Restore the source before updating."
        if self._uses_extensions and (issue := self._extension_issue(plan)):
            return issue
        parts = []
        if command == "refresh" and self.graph.auto_enabled(uid):
            waiting = self.graph.auto_blockers(uid)
            if waiting:
                parts.append(
                    "Auto is waiting for "
                    + ", ".join(self.label(key) for key in sorted(waiting))
                    + ". Update those sources, or choose Update Now."
                )
        if plan.current_inputs:
            parts.append(
                "Use "
                + ", ".join(self.revision_label(rev) for _, rev in plan.current_inputs)
            )
            for entry, _ in plan.current_inputs:
                if entry in self._checkpoint_sources:
                    parts.append(
                        f"Use the displayed checkpoint of {self.label(entry)}. "
                        "To replay its earlier history, reload that source first."
                    )
        if plan.pinned_inputs:
            parts.append(
                "Keep "
                + ", ".join(self.revision_label(rev) for rev in plan.pinned_inputs)
            )
        if plan.file_loads:
            parts.append(
                "Read " + ", ".join(load.source.path for load in plan.file_loads)
            )
        if plan.earlier_steps:
            parts.append(
                "Rebuild "
                + ", ".join(
                    self.revision_label(step.revision_id) for step in plan.earlier_steps
                )
            )
        if plan.entries:
            parts.append(
                "Update " + ", ".join(self.label(entry) for entry in plan.entries)
            )
        if plan.affected_followers:
            parts.append(
                "Affected results: "
                + ", ".join(
                    self.label(entry) for entry in sorted(plan.affected_followers)
                )
            )
        return "\n".join(parts)

    @contextlib.contextmanager
    def _reload_destinations(
        self, targets: Sequence[str], values: Mapping[str, xr.DataArray]
    ) -> Iterator[dict[str, str]]:
        redirects = {}
        new_targets = []
        lineage = self.manager._lineage_controller
        for uid in targets:
            if uid not in values:
                continue
            node = self.manager._node_for_target(uid)
            current = node.current_source_data()
            if erlab.interactive.imagetool.slicer.check_cursors_compatible(
                current, values[uid]
            ):
                continue
            choice = lineage._prompt_incompatible_reload_commit(
                lineage._reload_incompatibility_details(current, values[uid])
            )
            if choice == "cancel":
                raise _ReloadCancelled
            if choice == "new":
                new_targets.append(uid)
        with contextlib.ExitStack() as rollback:
            for uid in new_targets:
                tool = erlab.interactive.itool(
                    values[uid], manager=False, execute=False
                )
                if not isinstance(tool, erlab.interactive.imagetool.ImageTool):
                    raise TypeError("Could not open the reloaded result")
                destination = self.manager._next_node_uid()
                # The reload transaction publishes the computed revision after
                # all destination windows have been allocated.
                with self.changing(destination):
                    index = self.manager.add_imagetool(
                        tool, uid=destination, activate=True
                    )
                rollback.callback(
                    self.manager._remove_imagetool, index, preflight=False
                )
                redirects[uid] = self.manager._node_for_target(index).uid
            yield redirects
            rollback.pop_all()

    def reload_selected(self, uids: Sequence[str]) -> bool:
        if self._staged_windows.intersection(uids):
            return False
        try:
            self.graph.reload_entries(
                uids, destinations=functools.partial(self._reload_destinations, uids)
            )
        except _ReloadCancelled:
            return False
        except Exception as exc:
            erlab.interactive.utils.MessageDialog.critical(
                self.manager,
                "Reload Failed",
                "The selected results could not be updated.",
                informative_text=str(exc),
            )
            return False
        return True

    def run(self, uid: str, command: _Command) -> bool:
        reason = self.unavailable(uid)
        if reason is None:
            try:
                if command == "reload":
                    return self.manages(uid) and self.reload_selected((uid,))
                self.graph.refresh(uid)
            except Exception as exc:
                reason = str(exc)
            else:
                return True
        erlab.interactive.utils.MessageDialog.critical(
            self.manager,
            f"{command.title()} Failed",
            f"{self.label(uid)} could not be updated.",
            informative_text=reason,
        )
        return False

    def edit_operation(
        self,
        uid: str,
        displayed: str,
        revisions: Sequence[str],
        changes: Sequence[ToolProvenanceOperation],
        *,
        input_changes: Mapping[str, Follow] | None = None,
    ) -> None:
        for binding in (input_changes or {}).values():
            source = self.manager._node_for_target(binding.entry_id)
            if not self._ensure_source(source):
                raise ValueError("The chosen input is unavailable")
        recipe = self.graph.edited_operation_recipe(
            uid,
            displayed_revision=displayed,
            edited_revision=revisions[-1],
            group_prefix=revisions[:-1],
            operations=changes,
            input_changes=input_changes,
        )
        if recipe is not None:
            self.apply_edits({uid: (displayed, recipe)})

    def apply_edits(self, updates: Mapping[str, tuple[str, Recipe | FileLoad]]) -> None:
        new_branches = updates.keys() - self._branched
        self._branched.update(new_branches)
        try:
            self.graph.apply_operation_edits(updates)
        except Exception:
            self._branched.difference_update(new_branches)
            raise

    def restore_results(
        self,
        targets: Mapping[str, tuple[str, str]],
        *,
        program_steps: Mapping[str, int | None] | None = None,
    ) -> None:
        prefixes = {}
        for uid, index in (program_steps or {}).items():
            if index is None:
                continue
            revision = targets[uid][1]
            step = self.graph.recipe_for(revision)
            if not isinstance(step, Recipe) or not isinstance(
                step.operation, RecordedProgram
            ):
                raise TypeError(
                    "The selected step does not belong to a recorded program"
                )
            spec = step.operation.spec
            if index == len(spec.steps) - 1:
                continue
            prefix = spec._prefix_through_ref(
                _ProvenanceStepRef("operation", operation_index=index)
            )
            prefixes[uid] = Recipe(
                RecordedProgram(spec=prefix.with_replay_inputs(spec.replay_inputs)),
                step.inputs,
                input_names=step.input_names,
                output_name=prefix.active_name,
            )
        new_branches = targets.keys() - self._branched
        self._branched.update(new_branches)
        try:
            self.graph.restore_revisions(targets, prefixes=prefixes)
        except Exception:
            self._branched.difference_update(new_branches)
            raise

    def duplicate_results(self, uid_map: Mapping[str, str]) -> None:
        owners = {old: new for old, new in uid_map.items() if old in self.graph}
        origin_histories = {}
        for old in owners:
            watched = self._watched_sources.get(old)
            if watched is None or not self.graph.has_revision(watched[0]):
                continue
            key, seed = watched
            if self._watched_seed(old, key) is not None:
                origin_histories[key] = seed.model_dump_json()
        keys = self.graph.duplicate_entries(owners, origin_histories=origin_histories)
        for old, new in owners.items():
            if old in self._imported:
                self._imported.add(new)
            else:
                self._imported.discard(new)
            if old in self._views:
                source, count = self._views[old]
                self._views[new] = keys.get(source, source), count
            if old in self._checkpoint_reasons:
                key, reason = self._checkpoint_reasons[old]
                if key in keys:
                    self._checkpoint_reasons[new] = keys[key], reason
            area = self.manager._node_for_target(new).slicer_area
            self._view_state[new] = (
                weakref.ref(area._data),
                area._public_source_dims_for_current_layout(),
                area.accepted_filter_revision,
            )

    def imported_workflow(self, uid: str) -> None:
        """Retain the imported history for workflows that lack graph records."""
        if uid not in self.graph:
            self._imported.add(uid)

    def owns_result(self, uid: str) -> bool:
        """Return whether readers can query the published graph result."""
        return (
            uid not in self._changing
            and uid not in self._adopting
            and uid not in self._imported
            and uid in self.graph
        )

    def reloads_result(self, uid: str) -> bool:
        """Use the graph for live entries and imports with a graph recipe."""
        if uid in self._staged_windows or (
            uid in self.graph
            and uid not in self._imported
            and uid not in self._adopting
        ):
            return True
        # An imported program can pin workspace checkpoints that only its
        # saved replay plan knows how to restore.
        return (
            uid in self._imported
            and self.graph.has_recipe(uid)
            and (
                isinstance(self.graph.entry(uid).recipe, FileLoad)
                or bool(self.graph.follows(uid))
            )
        )

    def window_staged(self, uid: str) -> bool:
        return uid in self._staged_windows

    def installing(self, uid: str) -> bool:
        return uid in self._source_writes

    @contextlib.contextmanager
    def superseding_source(
        self,
        uid: str,
        source: _SourceWrite,
    ) -> Iterator[None]:
        """Publish a callback's write when the outer transaction unwinds."""
        write = self._source_writes[uid]
        previous = write.newer_source
        write.newer_source = source
        try:
            yield
        except Exception:
            if write.newer_source is source:
                write.newer_source = previous
            raise

    def local_operations(self, revision: str) -> tuple[tuple[str, Application], ...]:
        """Return the calculation on the primary path up to its first bound input."""
        return self.graph.local_operations(revision)

    def displayed_edit_revision(self, node: _ManagedWindowNode) -> str:
        if not node.materialize_pending_workspace_payload() or not self._ensure_source(
            node
        ):
            raise ValueError("The selected ImageTool data is unavailable")
        return self.graph.current_revision(node.uid)

    @contextlib.contextmanager
    def _authoring_results(self, uids: Sequence[str]) -> Iterator[None]:
        """Use graph exports for the new results, including edits to imports."""
        imported = self._imported.intersection(uids)
        self._imported.difference_update(imported)
        try:
            yield
        except Exception:
            self._imported.update(imported)
            raise

    def rename_source(self, node: _ManagedWindowNode, name: str | None) -> None:
        displayed = self.displayed_edit_revision(node)
        source = self.source_revision(node.uid, displayed=False)
        self.require_published(node.uid)
        self._renaming.add(node.uid)
        try:
            edits = {
                node.uid: DisplayedEdit(
                    displayed, source, (RenameOperation(name=name),)
                )
            }
            if (
                node.uid in self._imported
                and node.parent_uid is None
                and node._source_spec is not None
            ):
                # A v3 root can carry a live source specification without a
                # graph parent. Preserve its saved update binding on rename.
                self.graph.edit_displayed(
                    edits, identity=_SourceViewOperation(source_kind="full_data")
                )
            else:
                self.edit_displayed(edits)
        finally:
            self._renaming.discard(node.uid)

    def edit_displayed(self, edits: Mapping[str, DisplayedEdit]) -> None:
        with self._authoring_results(tuple(edits)):
            self.graph.edit_displayed(
                edits, identity=_SourceViewOperation(source_kind="full_data")
            )

    def paste_workflow(
        self,
        uid: str,
        displayed: str,
        spec: ToolProvenanceSpec,
        staged: _ProvenanceRevisionStore,
        require_current: Callable[[], None],
    ) -> None:
        checkpoints: dict[str, tuple[str, xr.DataArray]] = {}
        intermediates: dict[str, tuple[Recipe, Mapping[str, str]]] = {}
        file_origins: dict[str, FileLoad] = {}
        histories: dict[str, str] = {}
        steps, recorded = self._bind_workflow(
            uid,
            spec,
            consume_target=True,
            allow_program=True,
            checkpoints=checkpoints,
            intermediates=intermediates,
            file_origins=file_origins,
            checkpoint_histories=histories,
            staged=staged,
        )
        self._note_producers(steps)
        if self.graph.current_revision(uid) != displayed:
            raise _SupersededEvaluation(
                "The destination changed while resolving the clipboard inputs"
            )
        with self._authoring_results((uid,)):
            self.graph.prepare_import(
                uid,
                None,
                steps,
                recorded_inputs=recorded,
                execution_scope=lambda: contextlib.nullcontext(require_current),
                intermediates=intermediates,
                checkpoints=checkpoints,
                file_origins=file_origins,
                checkpoint_histories=histories,
            )()

    def relative_spec(
        self, node: _ManagedWindowNode, *, displayed: bool
    ) -> ToolProvenanceSpec | None:
        relative = self.parent_operations(node, displayed=displayed)
        if relative is None:
            return None
        kind, operations = relative
        return ToolProvenanceSpec(kind=kind).append_operations(*operations)

    def parent_operations(
        self, node: _ManagedWindowNode, *, displayed: bool
    ) -> (
        tuple[
            typing.Literal["full_data", "public_data", "selection"],
            list[ToolProvenanceOperation],
        ]
        | None
    ):
        if node.parent_uid is None:
            return None
        revision = self.source_revision(node.uid, displayed=displayed)
        step = self.graph.recipe_for(revision)
        if not displayed and revision == self.graph.current_revision(node.uid):
            step = self.graph.entry(node.uid).recipe
        operations = []
        while isinstance(step, Recipe) and len(step.inputs) == 1:
            operations.append(step.operation)
            binding = step.inputs[0][1]
            if isinstance(binding, Follow) and binding.entry_id == node.parent_uid:
                break
            if not isinstance(binding, EarlierStep):
                return None
            step = (
                binding.pending.recipe
                if binding.pending
                else self.graph.recipe_for(binding.revision_id)
            )
        else:
            return None
        operations.reverse()
        # The compatibility child loader applies ImageTool's input dtype
        # conversion from the saved window state after the relative transform.
        if (
            operations
            and isinstance(operations[-1], ImageToolInputOperation)
            and not operations[-1].degree_coords
        ):
            operations.pop()
        kind: typing.Literal["full_data", "public_data", "selection"] = (
            step.source_kind or "full_data"
        )
        if operations and isinstance(first_op := operations[0], _SourceViewOperation):
            kind = first_op.source_kind
            operations.pop(0)
        if any(not op.live_applicable for op in operations):
            return None
        return kind, operations

    def recorded_files(self, uid: str) -> tuple[FileLoad, ...]:
        files: dict[str, FileLoad] = {}
        for record in self.graph.recorded_lineage(self.graph.current_revision(uid)):
            if isinstance(record.producer, FileLoad):
                files.setdefault(record.producer.source.path, record.producer)
        return tuple(files.values())

    def result_spec(self, uid: str, *, displayed: bool) -> ToolProvenanceSpec | None:
        revision = self.source_revision(uid, displayed=displayed)
        record = self.graph.revision(revision)
        if record.producer is None and record.imported_history is None:
            watched = self._watched_seed(uid, revision)
            return watched[1].model_copy(deep=True) if watched else None
        key = uid, displayed
        generation = self.manager._tool_graph.structure_generation
        previous = self._exports.get(key)
        if previous is None or previous[:2] != (revision, generation):
            spec = self.export_spec(revision, target_uid=uid)
            name = self.graph.entry(uid).output_name
            if name is not None and spec.active_name != name:
                spec = RecordedProgram(spec=spec).with_output_name(name).spec
            previous = revision, generation, spec
            self._exports[key] = previous
        return previous[2].model_copy(deep=True)

    def source_revision(self, uid: str, *, displayed: bool) -> str:
        if not displayed and uid in self._views:
            return self._views[uid][0]
        return self.graph.current_revision(uid)

    def export_spec(
        self, revision: str, *, target_uid: str | None = None
    ) -> ToolProvenanceSpec:
        """Project the recorded inputs of a revision into the interchange format."""
        specs: dict[str, ToolProvenanceSpec] = {}
        store = self.manager._provenance_revisions

        def register_checkpoint(key: str) -> bool:
            if not store.has_data(key):
                reader = self.graph.checkpoint_reader(key)
                if reader is None:
                    return False
                if key in store:
                    store.attach_checkpoint_reader(key, reader)
                else:
                    store.add(
                        revision_id=key,
                        source_owner_uid=self.graph.owner(key),
                        data_reader=reader,
                    )
            return True

        for record in self.graph.recorded_lineage(revision):
            key = record.revision_id
            applied = record.producer
            if isinstance(applied, FileLoad):
                call = applied.call
                presentation = applied.presentation
                file = recorded_file_source(applied)
                seed = (
                    presentation.seed_code
                    if presentation is not None
                    else recorded_file_source(
                        applied, assign="derived", generate_code=True
                    ).load_code
                )
                if seed is not None or call.kind == "extension_loader":
                    filename = pathlib.Path(applied.source.path).name
                    specs[key] = file_load(
                        start_label=presentation.label
                        if presentation is not None
                        else f"Load data from file {filename!r}",
                        active_name=presentation.output_name
                        if presentation is not None
                        else "derived",
                        seed_code=seed,
                        file_load_source=file,
                        steps=call.steps if isinstance(call, PreparedFileCall) else (),
                    )
                    continue
            if applied is None or isinstance(applied, FileLoad):
                if applied is None and (
                    watched := self._watched_seed(record.owner_id, key)
                ):
                    specs[key] = watched[1].model_copy(deep=True)
                    continue
                if record.imported_history is not None:
                    specs[key] = ToolProvenanceSpec.model_validate_json(
                        record.imported_history
                    ).to_replay_spec()
                    continue
                register_checkpoint(key)
                specs[key] = script(
                    start_label="Recorded source",
                    active_name="data",
                    root_input=ScriptInput(
                        name="data",
                        revision_id=key,
                        follow_source=False,
                        label=self.revision_label(key),
                    ),
                )
                continue
            step = typing.cast("Recipe", self.graph.recipe_for(key))
            op = applied.operation
            if isinstance(op, RecordedProgram):
                names = tuple(item.name for item in op.spec.replay_inputs)
            else:
                names = step.input_names or tuple(slot for slot, _ in applied.inputs)
            inputs = []
            for name, (_, source), (_, binding) in zip(
                names, applied.inputs, step._captured.inputs, strict=True
            ):
                owner = self.manager._tool_graph.nodes.get(
                    binding.entry_id if isinstance(binding, Follow) else ""
                )
                # A window can display a retained calculation that consumed its
                # previous value. Its portable recipe must bind that old revision.
                if owner is not None and owner.uid == target_uid:
                    owner = None
                role: typing.Literal["source", "displayed"] = (
                    "source"
                    if isinstance(binding, Follow) and binding.source
                    else "displayed"
                )
                token = None
                if owner is not None:
                    current = self.source_revision(
                        owner.uid, displayed=role == "displayed"
                    )
                    token = (
                        owner.snapshot_token_for_role(role)
                        if current == source
                        else source
                    )
                # A console calculation has a retained value before any window
                # exists. Its nested v3 recipe must still update from live inputs.
                virtual = isinstance(
                    binding, EarlierStep
                ) and self.graph.is_calculation(source)
                checkpoint = (
                    source if not virtual and register_checkpoint(source) else None
                )
                item = ScriptInput(
                    name=name,
                    label=self._revision_owner_label(source),
                    node_uid=owner.uid if owner is not None else None,
                    node_snapshot_token=token,
                    data_role=role,
                    revision_id=checkpoint,
                    resolved_revision_id=checkpoint,
                    follow_source=owner is not None,
                    provenance_spec=(
                        specs[source].model_dump(mode="json")
                        if (
                            self.graph.revision(source).producer is not None
                            or self.graph.revision(source).imported_history is not None
                            or self._watched_seed(self.graph.owner(source), source)
                            is not None
                        )
                        else None
                    ),
                )
                inputs.append(item)
            if isinstance(op, RecordedProgram):
                specs[key] = op.spec.with_replay_inputs(
                    tuple(
                        item.model_copy(update={"source_spec": original.source_spec})
                        for item, original in zip(
                            inputs, op.spec.replay_inputs, strict=True
                        )
                    )
                )
            elif len(inputs) == 1 and (
                isinstance(step._captured.inputs[0][1], EarlierStep)
                or self._watched_seed(record.owner_id, applied.inputs[0][1]) is not None
            ):
                prefix = specs[applied.inputs[0][1]].append_operations(op)
                if (
                    step.output_name is not None
                    and step.output_name != prefix.active_name
                ):
                    if (
                        prefix.replay_primary_input
                        and prefix.seed_code is None
                        and all(
                            getattr(member.operation, "op", None) != "script_code"
                            for member in prefix.steps
                        )
                    ):
                        prefix = prefix.model_copy(
                            update={"active_name": step.output_name}
                        )
                    else:
                        prefix = recipe(
                            ReplayStep(operation=op),
                            start_label=op.derivation_label(),
                            active_name=step.output_name,
                            primary_input=inputs[0].name,
                            script_inputs=tuple(inputs),
                        )
                marker = op.group
                if (
                    marker is not None
                    and marker.console_prefix
                    and len(prefix.steps) >= marker.size
                ):
                    members = tuple(
                        member.model_copy(
                            update={
                                "operation": marker.member_operation(
                                    member.operation, index
                                )
                            }
                        )
                        for index, member in enumerate(prefix.steps[-marker.size :])
                    )
                    prefix = prefix.model_copy(
                        update={"steps": (*prefix.steps[: -marker.size], *members)}
                    )
                specs[key] = prefix
            else:
                specs[key] = recipe(
                    ReplayStep(
                        operation=op,
                        input_bindings={
                            slot: name
                            for (slot, _), name in zip(
                                applied.inputs[1:], names[1:], strict=True
                            )
                        },
                    ),
                    start_label=op.derivation_label(),
                    active_name=step.output_name or "derived",
                    script_inputs=tuple({item.name: item for item in inputs}.values()),
                    primary_input=names[0],
                )
        return specs[revision]

    def selected_scopes(
        self, uid: str, revisions: Mapping[str, set[int | None]]
    ) -> list[SelectedProvenanceScope]:
        """Export selected applications with their exact connections to one another."""
        selected = [
            record
            for record in self.graph.recorded_lineage(self.graph.current_revision(uid))
            if record.revision_id in revisions
        ]
        if len(selected) != len(revisions):
            return []
        operations: list[ToolProvenanceOperation] = []
        for record in selected:
            if isinstance(record.producer, Application):
                operations.append(record.producer.operation)
            elif record.imported_history is not None:
                operations.append(
                    RecordedProgram(
                        spec=ToolProvenanceSpec.model_validate_json(
                            record.imported_history
                        )
                    )
                )
            else:
                return []
        selected_indexes = {record.revision_id: i for i, record in enumerate(selected)}
        for record, operation in zip(selected, operations, strict=True):
            marker = operation.group
            if (
                marker is None
                or not marker.console_prefix
                or not isinstance(record.producer, Application)
            ):
                continue
            members = self._group_ending_at(record.revision_id, record.producer)
            if members and all(key in selected_indexes for key in members):
                for index, key in enumerate(members):
                    pos = selected_indexes[key]
                    operations[pos] = marker.member_operation(operations[pos], index)
        result: list[SelectedProvenanceScope] = []
        indexes: dict[str, int] = {}
        for record, operation in zip(
            selected, strip_partial_operation_groups(tuple(operations)), strict=True
        ):
            applied = record.producer
            step = self.graph.recipe_for(record.revision_id)
            slots = applied.inputs if isinstance(applied, Application) else ()
            names = (
                step.input_names
                if isinstance(step, Recipe) and step.input_names
                else tuple(slot for slot, _ in slots)
            )
            spec = self.export_spec(record.revision_id)
            input_name = names[0] if names else "data"
            output_name = spec.active_name or "derived"
            input_alias = None
            if isinstance(operation, RecordedProgram):
                program = operation.spec
                positions = sorted(
                    i for i in revisions[record.revision_id] if i is not None
                )
                if not positions or positions[-1] >= len(program.steps):
                    return []
                first = _ProvenanceStepRef("operation", operation_index=positions[0])
                last = _ProvenanceStepRef("operation", operation_index=positions[-1])
                before = program._prefix_before_ref(first)
                source_name = (
                    _direct_replay_source_name(before) if not before.steps else None
                )
                input_name = source_name or before.active_name or "data"
                input_alias = before.active_name if source_name else None
                output_name = (
                    program._prefix_through_ref(last).active_name or output_name
                )
                steps = tuple(program.steps[i] for i in positions)
                steps = tuple(
                    step.model_copy(update={"operation": op})
                    for step, op in zip(
                        steps,
                        strip_partial_operation_groups(
                            tuple(step.operation for step in steps)
                        ),
                        strict=True,
                    )
                )
                names = tuple(item.name for item in program.replay_inputs)
            else:
                steps = (
                    ReplayStep(
                        operation=operation,
                        input_bindings=dict(
                            zip(operation.input_slots(), names[1:], strict=True)
                        ),
                    ),
                )
            projected = selected_steps_spec(
                steps,
                input_name=input_name,
                output_name=output_name,
                input_alias=input_alias,
                script_inputs=spec.replay_inputs,
            )
            linked = {}
            if isinstance(applied, Application):
                linked = {
                    name: indexes[key]
                    for name, (_, key) in zip(names, applied.inputs, strict=True)
                    if key in indexes
                }
            if (
                not isinstance(operation, RecordedProgram)
                or len(operation.spec.steps) - 1 in revisions[record.revision_id]
            ):
                indexes[record.revision_id] = len(result)
            result.append(SelectedProvenanceScope(projected, linked))
        return result

    def _checkpoint_input_name(self, spec: ToolProvenanceSpec) -> str | None:
        """Recover an explicit notebook source from an unchanged saved recipe."""
        pending = [spec]
        names = set()
        while pending:
            current = pending.pop()
            if (
                current.kind != "script"
                or current.file_load_source is not None
                or any(
                    getattr(step.operation, "op", None)
                    in {"extension_routine", "script_code"}
                    for step in current.steps
                )
            ):
                return None
            if not current.replay_inputs:
                name = _direct_replay_source_name(current)
                if (
                    name is None
                    or current.seed_code != f"{current.active_name} = {name}"
                ):
                    return None
                names.add(name)
                continue
            if current.seed_code is not None:
                target, _, value = current.seed_code.strip().partition("=")
                if not target.strip().isidentifier() or value.strip() not in {
                    item.name for item in current.replay_inputs
                }:
                    return None
            for item in current.replay_inputs:
                source = self.manager._tool_graph.nodes.get(item.node_uid or "")
                nested = item.parsed_provenance_spec()
                if (
                    source is None
                    or nested is None
                    or item.node_snapshot_token
                    != source.snapshot_token_for_role(item.data_role)
                ):
                    return None
                pending.append(nested)
        return names.pop() if len(names) == 1 else None

    def has_recorded_code(self, node: _ManagedWindowNode) -> bool:
        """Resolve a saved calculation's bound inputs when code is requested."""
        if self.owns_result(node.uid) or self.manages(node.uid):
            return True
        if node.uid not in self._imported or not node.is_imagetool:
            return False
        spec = node.passive_displayed_provenance_spec
        if spec is None or not spec.replay_inputs:
            if node.imagetool is None:
                loader = self.manager._workspace_controller.loading
                pending = loader.pending._pending_workspace_state(node)
                filtered = (
                    pending is not None
                    and pending[1].get("filter_operation") is not None
                )
            else:
                filtered = (
                    node.slicer_area._accepted_filter_provenance_operation is not None
                )
            if not filtered:
                # An unfiltered file recipe supplies its code from metadata.
                return False
        if not node.materialize_pending_workspace_payload() or not self._ensure_source(
            node
        ):
            return False
        revision = self.graph.current_revision(node.uid)
        return self.graph.revision(revision).producer is not None

    def copy_code(
        self,
        node: _ManagedWindowNode,
        ask_name: Callable[[str, str, frozenset[str]], str | None],
    ) -> str | None:
        """Generate code for the selected result even while its live sources change."""
        self.require_published(node.uid)
        revision = self.graph.current_revision(node.uid)
        with self.graph.retain(revision):
            names = {}
            watched = {}
            used_names: set[str] = set()
            verified: set[str] = set()
            checkpoints = self.graph.recorded_code_inputs(revision)
            for record in self.graph.recorded_lineage(revision, stop_at=checkpoints):
                if (
                    isinstance(record.producer, Application)
                    and record.revision_id not in checkpoints
                ):
                    continue
                available = self.graph.checkpoint_problem(record.revision_id) is None
                if (
                    available
                    and isinstance(record.producer, FileLoad)
                    and self._file_reproduces(record.producer, record.revision_id)
                ):
                    verified.add(record.revision_id)
                    continue
                source = self.manager._tool_graph.nodes.get(record.owner_id)
                captured = (
                    ToolProvenanceSpec.model_validate_json(record.imported_history)
                    if record.imported_history is not None
                    else None
                )
                if (
                    available
                    and record.revision_id not in checkpoints
                    and captured is not None
                    and record.owner_id in self.graph
                    and self.graph.current_revision(record.owner_id)
                    == record.revision_id
                    and (input_name := self._checkpoint_input_name(captured))
                    is not None
                    and _notebook_name(input_name) not in used_names
                ):
                    watched[record.revision_id] = input_name, captured
                    used_names.add(_notebook_name(input_name))
                    continue
                seed = self._watched_seed(record.owner_id, record.revision_id)
                if (
                    available
                    and record.revision_id not in checkpoints
                    and seed
                    and _notebook_name(seed[0]) not in used_names
                ):
                    # This notebook variable supplies the exact raw revision.
                    names[record.revision_id] = seed[0]
                    used_names.add(_notebook_name(seed[0]))
                    continue
                try:
                    base = _notebook_name(source.name) if source is not None else None
                except (ValueError, ReplayGraphError):
                    base = None
                if base is None or base in {"derived", "result", "data"}:
                    base = erlab.utils.misc._normalize_identifier(
                        f"source_{record.owner_id}"
                    )
                suggestion = base
                suffix = 2
                while suggestion in used_names:
                    suggestion = f"{base}_{suffix}"
                    suffix += 1
                name = ask_name(record.revision_id, suggestion, frozenset(used_names))
                if name is None:
                    return None
                names[record.revision_id] = name
                used_names.add(_notebook_name(name))
            output = None
            for key, applied in reversed(self.local_operations(revision)):
                step = self.graph.recipe_for(key)
                if isinstance(step, Recipe):
                    output = (
                        applied.operation.spec.active_name
                        if isinstance(applied.operation, RecordedProgram)
                        else step.output_name
                    )
                    if output:
                        break
            return full_code(
                self.graph,
                revision,
                output_name=self.graph.entry(node.uid).output_name
                or output
                or "derived",
                source_names=names,
                watched_sources=watched,
                verify_file=lambda record: record.revision_id in verified,
            )

    def _file_reproduces(self, load: FileLoad, revision: str) -> bool:
        # Only an explicit code export checks a file against its checkpoint. Never
        # invoke a user loader while copying code or inspecting the provenance panel.
        call = load._captured
        engine = call.kwargs.get("engine")
        if (
            call.kind != "callable"
            or call.target
            not in {
                "xarray.load_dataarray",
                "xarray.load_dataset",
                "xarray.open_dataarray",
                "xarray.open_dataset",
            }
            # Auto detection and registered backends can run third-party hooks.
            or type(engine) is not str
            or engine not in {"h5netcdf", "netcdf4", "scipy"}
            or any(
                type(value) is not _NETCDF_VERIFICATION_OPTIONS.get(key)
                for key, value in call.kwargs.items()
            )
            or not self.graph.checkpoint_is_materialized(revision)
            or not pathlib.Path(load.source.path).is_file()
        ):
            return False
        # A prepared checkpoint may include a fit or extension. Exporting Python
        # must never execute those steps to find out whether the file still matches.
        if isinstance(call, PreparedFileCall) and any(
            type(step.operation)
            not in {
                IselOperation,
                SelOperation,
                QSelOperation,
                QSelAggregationOperation,
                AverageOperation,
                TransposeOperation,
                RenameOperation,
                SqueezeOperation,
                _SourceViewOperation,
                ImageToolInputOperation,
            }
            for step in call.steps
        ):
            return False
        try:
            current = self._read_file(load.source, call)
            try:
                saved = self.graph.data(revision)
                # xarray can consider equal values identical across dtypes. Integer
                # widths and coordinate dtypes can affect a subsequent calculation.
                return (
                    current.dtype == saved.dtype
                    and current.identical(saved)
                    and all(
                        current.coords[key].dtype == coord.dtype
                        for key, coord in saved.coords.items()
                    )
                )
            finally:
                current.close()
        except (ImportError, OSError, KeyError, ValueError, TypeError):
            return False

    def open_recorded_input(self, revision: str) -> None:
        with (
            self.graph.retain(revision),
            self._replay_publication() as (require_current, publish),
        ):
            data = self.graph.data(revision)
            require_current()
            result = self.calculate(
                (_SourceViewOperation(source_kind="full_data"),),
                (
                    _WindowInput(
                        data, _InputRevision(Pin(revision), revision), require_current
                    ),
                ),
                output_name="derived",
            )
            index = self.manager.add_imagetool(
                result,
                activate=True,
                _on_commit=require_current,
                _input_checks=(require_current,),
            )
            try:
                require_current()
            except Exception:
                self.manager._remove_imagetool(index, preflight=False)
                raise
            publish()

    def _group_ending_at(self, revision: str, applied: Application) -> tuple[str, ...]:
        marker = applied.operation.group
        if marker is None or marker.index != marker.size - 1:
            return ()
        # Only the primary input edge can connect consecutive group members.
        members = [revision]
        for index in reversed(range(marker.index)):
            if len(applied.inputs) != 1:
                return ()
            revision = applied.inputs[0][1]
            producer = self.graph.revision(revision).producer
            if not isinstance(producer, Application):
                return ()
            member = marker.member_operation(producer.operation, index)
            if not marker._matches_member(member.group, index):
                return ()
            members.append(revision)
            applied = producer
        return tuple(reversed(members))

    def provenance_rows(
        self,
        uid: str,
        imported_rows: tuple[_ProvenanceDisplayRow, ...] | None = None,
    ) -> tuple[_ProvenanceDisplayRow, ...]:
        entry = self.graph.entry(uid)
        applied = self.graph.revision(entry.revision_id).producer

        def row(label: str) -> _ProvenanceDisplayRow:
            return _ProvenanceDisplayRow(DerivationEntry(label, None))

        def application_rows(
            revision: str,
            applied: Application,
            group: tuple[str, ...],
            ref: _ProvenanceStepRef | None = None,
            *,
            path: tuple[int, ...] = (),
            show_inputs: bool = False,
        ) -> tuple[_ProvenanceDisplayRow, ...]:
            children = (
                tuple(
                    input_row(slot, source, group, (*path, index))
                    for index, (slot, source) in enumerate(applied.inputs)
                )
                if show_inputs
                else ()
            )
            operation = applied.operation
            if (
                imported_rows is None
                and isinstance(operation, RecordedProgram)
                and operation.spec.steps
            ):
                return tuple(
                    _ProvenanceDisplayRow(
                        step.operation.derivation_entry(),
                        graph_revision=revision,
                        script_input_path=path,
                        on_primary_chain=not any(path),
                        program_index=index,
                        edit_ref=_ProvenanceStepRef("operation", operation_index=index),
                        replay_ref=_ProvenanceStepRef(
                            "operation", operation_index=index
                        ),
                        children=children if index == 0 else (),
                    )
                    for index, step in enumerate(operation.spec.steps)
                )
            return (
                _ProvenanceDisplayRow(
                    operation.derivation_entry(),
                    graph_revision=revision,
                    graph_group=group,
                    script_input_path=path,
                    on_primary_chain=not any(path),
                    edit_ref=ref,
                    replay_ref=ref,
                    children=children,
                ),
            )

        def history(
            node: str, group: tuple[str, ...], path: tuple[int, ...]
        ) -> tuple[_ProvenanceDisplayRow, ...]:
            record = self.graph.revision(node)
            if isinstance(record.producer, FileLoad):
                call = record.producer.call
                preparation = (
                    tuple(
                        _ProvenanceDisplayRow(step.operation.derivation_entry())
                        for step in call.steps
                    )
                    if isinstance(call, PreparedFileCall)
                    else ()
                )
                label = (
                    record.producer.presentation.label
                    if record.producer.presentation is not None
                    else f"Load {record.producer.source.path}"
                )
                ref = _ProvenanceStepRef("file_load")
                return (
                    _ProvenanceDisplayRow(
                        DerivationEntry(label, None),
                        graph_revision=node,
                        script_input_path=path,
                        on_primary_chain=not any(path),
                        edit_ref=ref,
                        replay_ref=ref,
                    ),
                    *preparation,
                )
            if record.producer is None:
                if record.imported_history is not None and imported_rows is None:
                    saved_program = ToolProvenanceSpec.model_validate_json(
                        record.imported_history
                    )
                    if saved_program.steps:
                        return tuple(
                            _ProvenanceDisplayRow(
                                step.operation.derivation_entry(),
                                graph_revision=node,
                                program_index=index,
                                replay_ref=_ProvenanceStepRef(
                                    "operation", operation_index=index
                                ),
                                script_input_path=path,
                                on_primary_chain=False,
                            )
                            for index, step in enumerate(saved_program.steps)
                        )
                saved = self._checkpoint_reasons.get(record.owner_id)
                if saved is not None and saved[0] == node:
                    return (row(saved[1]),)
                if record.owner_id in self._checkpoint_sources:
                    return (row("Checkpoint of the source ImageTool's displayed data"),)
                if watched := self._watched_seed(record.owner_id, node):
                    return (row(f"Watched variable {watched[0]}"),)
                return (row("Notebook or in-memory data"),)
            if node not in group:
                group = self._group_ending_at(node, record.producer)
            return application_rows(
                node,
                record.producer,
                group,
                _ProvenanceStepRef("operation", operation_index=0)
                if imported_rows is None
                else None,
                path=path,
                show_inputs=True,
            )

        def input_row(
            slot: str, node: str, group: tuple[str, ...], path: tuple[int, ...]
        ) -> _ProvenanceDisplayRow:
            return _ProvenanceDisplayRow(
                DerivationEntry(f"{slot} — {self.revision_label(node)}", None),
                on_primary_chain=not any(path) and imported_rows is None,
                script_input_path=path,
                revision_id=node,
                _children_factory=functools.partial(history, node, group, path),
            )

        # Fold the earlier steps of this calculation onto its row. An input
        # from a different ImageTool remains accessible under Inputs used.
        rows: list[_ProvenanceDisplayRow] = []
        if imported_rows is not None:
            rows.extend(imported_rows)
        else:
            chain = self.local_operations(entry.revision_id)
            group: tuple[str, ...] = ()
            for index in reversed(range(len(chain))):
                current, producer = chain[index]
                if current not in group:
                    group = self._group_ending_at(current, producer)
                if isinstance(producer.operation, _SourceViewOperation):
                    rows.append(row(producer.operation.derivation_label()))
                    continue
                ref = _ProvenanceStepRef("operation", operation_index=index)
                rows.extend(reversed(application_rows(current, producer, group, ref)))
            rows.reverse()
            if chain:
                first, applied_step = chain[0]
                first_recipe = self.graph.recipe_for(first)
                if (
                    isinstance(first_recipe, Recipe)
                    and first_recipe.source_kind is not None
                    and not isinstance(applied_step.operation, _SourceViewOperation)
                ):
                    label = (
                        "Use data from Saved"
                        if isinstance(first_recipe.inputs[0][1], Pin)
                        else "Start from selected parent ImageTool data"
                        if first_recipe.source_kind == "selection"
                        else "Start from current parent ImageTool data"
                    )
                    rows.insert(0, row(label))
            if not chain:
                rows.extend(history(entry.revision_id, (), ()))
        if reason := self._checkpoint_reason(uid):
            rows.append(row(f"Recorded result checkpoint. {reason}"))
        if applied is None and entry.recipe is not None:
            rows.append(
                row(
                    "Recorded result checkpoint — "
                    + self.revision_label(entry.revision_id)
                )
            )
        if isinstance(applied, Application):
            group = self._group_ending_at(entry.revision_id, applied)
            rows.insert(
                0,
                _ProvenanceDisplayRow(
                    DerivationEntry("Inputs used", None),
                    is_input_group=True,
                    children=tuple(
                        input_row(slot, rev, group, (index,))
                        for index, (slot, rev) in enumerate(applied.inputs)
                    ),
                ),
            )
        return tuple(rows)

    def next_input_labels(self, uid: str) -> tuple[str, ...]:
        entry = self.graph.entry(uid)
        labels = []
        for slot, source in self.graph.next_inputs(uid):
            if isinstance(source, Follow):
                label = self.label(source.entry_id)
            elif isinstance(source, EarlierStep):
                label = "Rebuild " + self.revision_label(source.revision_id)
            elif isinstance(entry.recipe, FileLoad):
                label = entry.recipe.source.path
            else:
                label = self.revision_label(source)
            labels.append(f"{slot} — {label}")
        return tuple(labels)

"""Connect managed operation submissions to data revisions and ImageTool windows."""

from __future__ import annotations

import functools
import pathlib
import typing
import weakref
from concurrent.futures import CancelledError
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace

import erlab
from erlab._provenance import (
    Application,
    EvaluatedEntry,
    FileLoad,
    FileSource,
    FileVersion,
    Follow,
    ProvenanceGraph,
    Recipe,
)
from erlab._provenance._checkpoint import protected_data
from erlab._provenance._files import (
    _SUPPORTED_CALLABLE_LOADERS,
    _capture_loader_file,
    _deserialize_loader_kwargs,
    _load_file_source_data,
    _load_provenance_from_file_details,
)
from erlab._provenance._model import (
    FileLoadSource,
    FileReplayCall,
    ReplayStep,
    ScriptInput,
    recipe,
)
from erlab._provenance._operations import RenameOperation, TransposeOperation
from erlab.interactive.imagetool.manager._revisions import _ProvenanceRevisionStore

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import xarray as xr

    from erlab._provenance._files import _LoadFunc
    from erlab._provenance._model import (
        ScriptInputDataRole,
        ToolProvenanceOperation,
        ToolProvenanceSpec,
    )
    from erlab.interactive.imagetool import ImageTool
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode


def _file_loader(call: FileReplayCall) -> _LoadFunc:
    """Only a built-in single-file loader can supply a verified native origin."""
    loader = _SUPPORTED_CALLABLE_LOADERS.get(call.target.replace(":", "."))
    if call.kind != "callable" or loader is None:
        raise ValueError(f"Cannot reload file with loader {call.target!r}")
    return loader, _deserialize_loader_kwargs(call.kwargs), call.selection


def _read_file(
    source: FileSource, call: FileReplayCall
) -> xr.DataArray | tuple[xr.DataArray, FileVersion]:
    loader, kwargs, _selection = _file_loader(call)
    backing = _capture_loader_file(source.path, loader, kwargs)
    record = FileLoadSource(
        path=source.path,
        replay_call=call,
        loader_text=call.target,
        loader_label="Loader",
        kwargs_text="",
    )
    data = _load_file_source_data(record, backing=backing)
    return data if backing is None else (data, backing)


@dataclass(frozen=True)
class _Placement:
    parent: str | None
    make_window: Callable[[xr.DataArray], ImageTool]
    output_name: str


class _ManagedOperations:
    def __init__(
        self,
        manager: ImageToolManager,
        revisions: _ProvenanceRevisionStore | None = None,
    ) -> None:
        self.manager = manager
        self.revisions = (
            revisions if revisions is not None else _ProvenanceRevisionStore()
        )
        self._deferred_auto: Callable[[], None] | None = None
        self.graph = ProvenanceGraph(
            read_file=_read_file,
            install=self._install,
            schedule_auto=self._schedule_auto,
        )
        self._tokens: dict[str, str] = {}
        self._placements: dict[str, _Placement] = {}
        self._installing: set[str] = set()

    def _schedule_auto(self, callback: Callable[[], None]) -> None:
        def run() -> None:
            if self.manager._operations is self:
                callback()
            else:
                self._deferred_auto = callback

        erlab.interactive.utils.single_shot(self.manager, 0, run)

    @contextmanager
    def replace_workspace(self, revisions: _ProvenanceRevisionStore) -> Iterator[None]:
        """Activate a separate graph and checkpoint store for the incoming document."""
        incoming = _ManagedOperations(self.manager, revisions)
        self.manager._operations = incoming
        try:
            yield
        except BaseException:
            self.manager._operations = self
            incoming.discard_workspace()
            if self._deferred_auto is not None:
                callback, self._deferred_auto = self._deferred_auto, None
                self._schedule_auto(callback)
            raise
        else:
            self.discard_workspace()

    def discard_workspace(self) -> None:
        # Releasing rows also releases their graph-owned checkpoint projections.
        self._deferred_auto = None
        for uid in self.graph.entry_ids:
            self.graph.remove_entry(uid)
        self._tokens.clear()
        self.revisions.discard_unreachable(())

    def observe(
        self, node: _ManagedWindowNode, *, displayed_data: xr.DataArray | None = None
    ) -> None:
        """Bind a legacy displayed value once to its captured revision."""
        uid = node.uid
        if uid in self._installing or self._tokens.get(uid) == node.snapshot_token:
            return
        revision = node.revision_id_for_role(
            "displayed", graph_owned=True, displayed_data=displayed_data
        )
        store = self.revisions
        shown = store._captured_data(revision)
        area = node.slicer_area if node.imagetool is not None else None
        operation = None if area is None else area._accepted_filter_provenance_operation
        display = None
        source_key, source_data = revision, shown
        if operation is not None:
            source_key = node.revision_id_for_role("source", graph_owned=True)
            source_data = store._captured_data(source_key)
            display = (revision, operation, shown)
        if (
            area is not None
            and (origin := area._file_input) is not None
            and origin.call is not None
        ):
            source = area._data
            steps: list[ToolProvenanceOperation] = []
            if source.name != origin.name:
                steps.append(
                    RenameOperation(name=typing.cast("str | None", source.name))
                )
            dims = area._public_source_dims_for_current_layout()
            if dims != origin.dims:
                steps.append(TransposeOperation(dims=dims))
            raw = (
                source.rename(origin.name).transpose(*origin.dims)
                if steps
                else source_data
            )
            self.graph.adopt_file_view(
                uid,
                FileLoad(self.graph.file_source(origin.backing.path), origin.call),
                origin.backing,
                raw=raw,
                steps=steps,
                source=(source_key, source_data),
                display=display,
            )
        else:
            self.graph.adopt_checkpoint(uid, source_key, source_data, display=display)
        if display is not None:
            self._retain_host_revision(source_key, uid)
        elif area is not None:
            # A temporary preview does not change the committed source or display.
            node.bind_source_revision(revision)
        self._retain_host_revision(revision, uid)
        self._tokens[uid] = node.snapshot_token

    def _retain_host_revision(self, revision: str, uid: str) -> None:
        """Keep the host projection until the graph and its current row release it."""
        self.graph.on_release(
            revision,
            functools.partial(
                self._release_input, self.revisions, weakref.ref(self.manager), uid
            ),
        )

    @staticmethod
    def _release_input(
        store: _ProvenanceRevisionStore,
        manager_ref: weakref.ReferenceType[ImageToolManager],
        uid: str,
        revision: str,
    ) -> None:
        manager = manager_ref()
        node = (
            None
            if manager is None or manager._provenance_revisions is not store
            else manager._tool_graph.nodes.get(uid)
        )
        if node is not None and revision in node.active_revision_ids():
            return
        store.release_graph_result(revision)

    def displayed_spec(self, node: _ManagedWindowNode) -> ToolProvenanceSpec | None:
        """Return the recorded projection while this window displays that revision."""
        uid = node.uid
        if (
            self._tokens.get(uid) != node.snapshot_token
            or uid in self._installing
            or not self.graph.has_recipe(uid)
        ):
            return None
        record = self.revisions.get(self.graph.current_revision(uid))
        return record.provenance_spec if record is not None else None

    def output_name(self, uid: str) -> str:
        """Get the recorded producer name, including when a display filter is active."""
        record = self.revisions.get(self.graph.source_revision(uid))
        if record is not None and record.provenance_spec is not None:
            return record.provenance_spec.active_name or "derived"
        return "derived"

    def installing(self, uid: str) -> bool:
        return uid in self._installing

    @contextmanager
    def authored_edit(
        self,
        node: _ManagedWindowNode,
        operation: ToolProvenanceOperation | None,
        *,
        input_role: ScriptInputDataRole,
    ) -> Iterator[None]:
        """Publish a recorded edit once the existing Manager command has succeeded."""
        if operation is None or (
            node.uid not in self.graph and node.slicer_area._file_input is None
        ):
            yield
            return
        self.observe(node)
        self._installing.add(node.uid)
        try:
            yield
            key = node.revision_id_for_role("source", graph_owned=True)
            store = self.revisions
            display = None
            accepted = node.slicer_area._accepted_filter_provenance_operation
            if accepted is not None:
                shown = node.revision_id_for_role("displayed", graph_owned=True)
                display = (shown, accepted, store._captured_data(shown))
            self.graph.adopt_application(
                node.uid,
                key,
                store._captured_data(key),
                operation,
                input_role=input_role,
                display=display,
            )
            if display is not None:
                self._retain_host_revision(display[0], node.uid)
            node.bind_displayed_revision(self.graph.current_revision(node.uid))
            self._retain_host_revision(key, node.uid)
            self._tokens[node.uid] = node.snapshot_token
        finally:
            self._installing.remove(node.uid)

    def changed(
        self, node: _ManagedWindowNode, *, displayed_data: xr.DataArray | None = None
    ) -> None:
        uid = node.uid
        if uid not in self.graph or uid in self._installing:
            return
        source = self.graph.source_revision(uid)
        # Only a display publication on the same source can keep its recipe.
        if (
            displayed_data is not None
            and node.imagetool is not None
            and node.current_revision_id == source
        ):
            operation = node.slicer_area._accepted_filter_provenance_operation
            if operation is not None or self.graph.current_revision(uid) != source:
                store = self.revisions
                if operation is None:
                    key = self.graph.set_display_operation(uid, None)
                else:
                    key = node.revision_id_for_role(
                        "displayed", graph_owned=True, displayed_data=displayed_data
                    )
                    self.graph.set_display_operation(
                        uid, operation, revision_id=key, data=store._captured_data(key)
                    )
                    self._retain_host_revision(key, uid)
                node.bind_displayed_revision(key)
                self._tokens[uid] = node.snapshot_token
                return
        self.observe(node, displayed_data=displayed_data)

    def _release_unused_inputs(self, uids: set[str]) -> None:
        for uid in uids:
            if (
                uid in self.graph
                and not self.graph.has_followers(uid)
                and not self.graph.has_recipe(uid)
                and not self.graph.has_recorded_derivation(uid)
            ):
                self.graph.remove_entry(uid)
                self._tokens.pop(uid, None)

    def removed(self, uid: str) -> None:
        self._tokens.pop(uid, None)
        if uid in self.graph:
            sources = self.graph.followed_entries(uid)
            self.graph.remove_entry(uid)
            self._release_unused_inputs(sources)

    def submit(
        self,
        operation: ToolProvenanceOperation,
        inputs: Mapping[str, str],
        *,
        placement: typing.Literal["nest", "detach", "replace"],
        make_window: Callable[[xr.DataArray], ImageTool],
        output_name: str,
    ) -> str:
        """Resolve the explicitly selected entries and publish one calculation."""
        primary = inputs["data"]
        for uid in dict.fromkeys(inputs.values()):
            self.observe(self.manager._node_for_target(uid))
        bound = Recipe(
            operation,
            tuple(
                (slot, Follow(inputs[slot]))
                for slot in ("data", *operation.input_slots())
            ),
        )
        uid = primary if placement == "replace" else self.manager._next_node_uid()
        parent = primary if placement == "nest" else None
        self._placements[uid] = _Placement(parent, make_window, output_name)
        try:
            if placement == "replace":
                self.graph.replace(uid, bound)
            else:
                self.graph.derive(uid, bound, parent_id=parent)
        finally:
            del self._placements[uid]
            self._release_unused_inputs(set(inputs.values()))
        return uid

    def _legacy_projection(
        self, update: EvaluatedEntry, output_name: str
    ) -> ToolProvenanceSpec:
        """Describe the evaluated edges for existing workspace and clipboard readers."""
        producer = update.revision.producer
        if not isinstance(producer, Application) or not isinstance(
            update.recipe, Recipe
        ):
            raise TypeError("The Manager expected an evaluated operation")
        bindings = dict(update.recipe.inputs)
        inputs = []
        for name, revision in producer.inputs:
            binding = bindings[name]
            uid = binding.entry_id if isinstance(binding, Follow) else None
            node = self.manager._tool_graph.nodes.get(uid) if uid is not None else None
            captured = self.revisions.get(revision)
            source = None if captured is None else captured.provenance_spec
            inputs.append(
                ScriptInput(
                    name=name,
                    label=name if node is None else node.display_text,
                    node_uid=uid if node is not None else None,
                    node_snapshot_token=node.snapshot_token
                    if node is not None
                    else None,
                    follow_source=node is not None,
                    revision_id=revision,
                    resolved_revision_id=revision,
                    data_role="displayed",
                    provenance_spec=None
                    if source is None
                    else source.to_replay_spec().model_dump(mode="json"),
                )
            )
        return recipe(
            ReplayStep(
                operation=producer.operation,
                input_bindings={slot: slot for slot, _ in producer.inputs[1:]},
            ),
            start_label=producer.operation.derivation_label(),
            active_name=output_name,
            script_inputs=inputs,
            primary_input="data",
        )

    @contextmanager
    def _install(self, updates: tuple[EvaluatedEntry, ...]) -> Iterator[None]:
        """Restore every affected view if installation or the graph commit fails."""
        with ExitStack() as rollback:
            for update in updates:
                uid = update.entry_id
                node = self.manager._tool_graph.nodes.get(uid)
                rollback.callback(
                    self.revisions.discard_unpublished,
                    update.revision.revision_id,
                )
                if update.display is not None:
                    rollback.callback(
                        self.revisions.discard_unpublished,
                        update.display.revision.revision_id,
                    )
                if node is not None:
                    if node.imagetool is None:
                        self.manager.get_imagetool(uid)
                    rollback.callback(node._undo_data_installation())
                else:
                    rollback.callback(self._remove_uncommitted_window, uid)
            with ExitStack() as notifications:
                for update in updates:
                    self._installing.add(update.entry_id)
                    notifications.callback(self._installing.discard, update.entry_id)
                try:
                    self._install_rows(updates)
                    yield
                except BaseException:
                    rollback.close()
                    for update in updates:
                        uid = update.entry_id
                        node = self.manager._tool_graph.nodes.get(uid)
                        if node is None:
                            self._tokens.pop(uid, None)
                        else:
                            self._tokens[uid] = node.snapshot_token
                    raise
                rollback.pop_all()
        for update in updates:
            self.graph.on_release(
                update.revision.revision_id,
                self.revisions.release_graph_result,
            )
            if update.display is not None:
                self.graph.on_release(
                    update.display.revision.revision_id,
                    self.revisions.release_graph_result,
                )
            uid = update.entry_id
            self.manager._lineage_controller._propagate_source_change_from_uid(uid)
            self.manager._schedule_selected_details_refresh(changed_uid=uid)

    def _remove_uncommitted_window(self, uid: str) -> None:
        node = self.manager._tool_graph.nodes.get(uid)
        if node is not None:
            from erlab.interactive.imagetool.manager._wrapper import _ImageToolWrapper

            if isinstance(node, _ImageToolWrapper):
                self.manager.remove_imagetool(node.index)
            else:
                self.manager._remove_uid_target(uid)

    def _install_file(
        self, node: _ManagedWindowNode, update: EvaluatedEntry, load: FileLoad
    ) -> ToolProvenanceSpec:
        call = load.call
        path = pathlib.Path(load.source.path)
        loader = _file_loader(call)
        spec = _load_provenance_from_file_details(
            path, loader, cast_float64=call.cast_float64
        )
        if spec is None:
            raise ValueError(f"Cannot describe the recorded load from {path!s}")
        with node._suspend_descendant_propagation(), node._suspend_snapshot_updates():
            node.slicer_area.replace_source_data(
                protected_data(update.data),
                file_path=path,
                load_func=loader,
                _file_backing=update.revision.backing,
            )
            origin = node.slicer_area._file_input
            if origin is not None:
                node.slicer_area._file_input = replace(origin, call=call)
            node.set_displayed_provenance(spec, advance_snapshot=False)
        node._advance_snapshot_token(defer_refresh=True)
        self.manager.tree_view.refresh(node.uid)
        return spec

    def _install_rows(self, updates: tuple[EvaluatedEntry, ...]) -> None:
        for update in updates:
            uid = update.entry_id
            destination = self._placements.get(uid)
            node = self.manager._tool_graph.nodes.get(uid)
            name = (
                destination.output_name
                if destination is not None
                else self.output_name(uid)
            )
            if isinstance(update.revision.producer, FileLoad):
                if node is None:
                    raise RuntimeError("The file window is no longer available.")
                spec = self._install_file(node, update, update.revision.producer)
            else:
                spec = self._legacy_projection(update, name)
                data = protected_data(update.data)
                if node is None:
                    if destination is None:
                        raise RuntimeError(
                            "The destination window is no longer available."
                        )
                    tool = destination.make_window(data)
                    if destination.parent is None:
                        self.manager.add_imagetool(
                            tool, uid=uid, activate=True, provenance_spec=spec
                        )
                    else:
                        self.manager.add_imagetool_child(
                            tool,
                            destination.parent,
                            uid=uid,
                            activate=True,
                            provenance_spec=spec,
                        )
                    node = self.manager._node_for_target(uid)
                else:
                    node.replace_with_detached_data(
                        data,
                        spec,
                        propagate_descendants=False,
                        preserve_filter=False,
                        replay_source_data=None,
                        restored_revision_id=update.revision.revision_id,
                        materialize=False,
                    )
                    self.manager.tree_view.refresh(uid)
            self.revisions.adopt_result(
                update.revision.revision_id, update.data, spec, uid
            )
            node.restore_current_revision_id(update.revision.revision_id)
            if update.display is not None:
                displayed = update.display
                application = typing.cast("Application", displayed.revision.producer)
                area = node.slicer_area
                with (
                    node._suspend_descendant_propagation(),
                    node._suspend_snapshot_updates(),
                ):
                    area._apply_filter_result(
                        displayed.data,
                        area._filter_func_from_operation(application.operation),
                        operation=application.operation,
                        accept=True,
                    )
                node._advance_displayed_snapshot_token(defer_refresh=True)
                self.revisions.adopt_result(
                    displayed.revision.revision_id,
                    displayed.data,
                    typing.cast(
                        "ToolProvenanceSpec", area.displayed_provenance_spec(spec)
                    ),
                    uid,
                )
                node.bind_displayed_revision(displayed.revision.revision_id)
            self._tokens[uid] = node.snapshot_token

    def reload_problem(self, uid: str) -> str | None:
        for load in self.graph.plan_update(uid, command="reload").file_loads:
            if not pathlib.Path(load.source.path).is_file():
                return f"Restore the missing file {load.source.path!r}, then reload."
        return None

    def _reload(self, uid: str) -> None:
        node = self.manager._node_for_target(uid)
        graph = self.graph
        if isinstance(graph.revision(graph.source_revision(uid)).producer, FileLoad):
            graph.reload(uid)
            return

        def make_window(data: xr.DataArray) -> ImageTool:
            tool = erlab.interactive.itool(data, manager=False, execute=False)
            if not isinstance(tool, erlab.interactive.imagetool.ImageTool):
                raise TypeError("Expected an ImageTool for the reloaded data.")
            return tool

        with ExitStack() as placements:

            def choose_destination(rebuilt: xr.DataArray) -> str | None:
                current = node.current_source_data()
                if erlab.interactive.imagetool.slicer.check_cursors_compatible(
                    current, rebuilt
                ):
                    return uid
                lineage = self.manager._lineage_controller
                details = lineage._reload_incompatibility_details(current, rebuilt)
                match lineage._prompt_incompatible_reload_commit(details):
                    case "replace":
                        return uid
                    case "new":
                        created = self.manager._next_node_uid()
                        self._placements[created] = _Placement(
                            None, make_window, self.output_name(uid)
                        )
                        placements.callback(self._placements.pop, created)
                        return created
                    case _:
                        return None

            graph.reload(uid, choose_destination=choose_destination)

    def update(self, uid: str, *, reload: bool = False) -> bool:
        try:
            if reload:
                problem = self.reload_problem(uid)
                if problem is not None:
                    erlab.interactive.utils._show_reload_unavailable_dialog(
                        self.manager, problem
                    )
                    return False
                self._reload(uid)
            else:
                self.graph.refresh(uid)
        except CancelledError:
            return False
        except Exception as exc:
            erlab.interactive.utils.MessageDialog.critical(
                self.manager,
                "Could not update the ImageTool",
                f"{exc}. Check the recorded inputs, then try again.",
            )
            return False
        self.manager._schedule_selected_details_refresh(changed_uid=uid)
        return True

"""Bind deferred extension tasks to their recorded xarray options."""

from __future__ import annotations

import functools
import itertools
import pickle
import typing
import uuid
from dataclasses import dataclass

import cloudpickle
from dask.core import get, get_dependencies
from dask.highlevelgraph import HighLevelGraph, Layer, MaterializedLayer

from erlab.extensions._routine import _run_isolated

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    import dask.array as da
    import xarray as xr

    from erlab._provenance._settings import ExecutionSettings


def _task_value(key, payload: bytes, dependencies, *values):
    cache = dict(zip(dependencies, values, strict=True))
    (value,) = get({key: pickle.loads(payload)}, [key], cache=cache)
    return value


# A private interpreter retains at most the graphs of its most recent result.
# The portable layer owns its serialized copy for workers that start later.
_cached_result: str | None = None
_cached_graphs: dict[str, HighLevelGraph] = {}


def _begin_result(group: str) -> None:
    global _cached_result
    _cached_result = group
    _cached_graphs.clear()


def _inspect_graph(group, name, inspect, args, payload=None):
    graph = _cached_graphs.get(name) if group == _cached_result else None
    if graph is None:
        if payload is None:
            return None
        if group != _cached_result:
            _begin_result(group)
        graph = _cached_graphs[name] = pickle.loads(payload)
    return inspect(graph, *args)


@dataclass(frozen=True)
class _RoutineGraph:
    settings: ExecutionSettings
    prefix: str
    name: str
    payload: bytes

    def inspect(self, function, *args):
        query = functools.partial(
            _run_isolated,
            self.settings,
            _inspect_graph,
            self.prefix,
            self.name,
            function,
            args,
        )
        result = query()
        return query(self.payload) if result is None else result


def _selected_tasks(graph, layer_name, keys):
    # Cull the original layers before expanding their tasks or running their
    # key-dependent scheduling annotations.
    if layer_name is None:
        tasks = (
            graph if keys is None else graph.cull({key for key in keys if key in graph})
        )
        annotations = {}
    else:
        tasks = graph.layers[layer_name]
        if keys is not None:
            tasks, _ = tasks.cull(
                keys.intersection(tasks.get_output_keys()),
                graph.get_all_external_keys(),
            )
        annotations = {
            name: {key: value(key) for key in tasks}
            for name, value in (tasks.annotations or {}).items()
            if callable(value)
        }
    return (
        {
            key: (cloudpickle.dumps(task), tuple(get_dependencies(graph, task=task)))
            for key, task in tasks.items()
        },
        annotations,
    )


def _output_keys(graph, layer_name):
    return set(graph.layers[layer_name].get_output_keys())


class _RoutineTasks(Layer):
    """Expose Dask dependency edges while keeping user callbacks in the worker."""

    def __init__(self, graph: _RoutineGraph, array: da.Array, layer_name=None):
        super().__init__()
        self.graph = graph
        self.layer_name = layer_name
        self.numblocks = array.numblocks
        self._full: MaterializedLayer | None = None
        if layer_name is None:
            self.length = len(array.dask)
        else:
            layer = array.dask.layers[layer_name]
            self.length = len(layer)
            # For an unculled graph, Dask iterates over the materialized tasks
            # when it applies annotations. Culled layers get only selected values.
            self.annotations = {
                name: functools.partial(self._annotation, name)
                if callable(value)
                else value
                for name, value in (layer.annotations or {}).items()
            }

    def _annotation(self, name, key):
        return self._materialize().annotations[name](key)

    def _scoped_key(self, key):
        if isinstance(key, tuple) and key and isinstance(key[0], str):
            return (f"{self.graph.prefix}-{key[0]}", *key[1:])
        return (self.graph.prefix, key)

    def _source_key(self, key):
        if isinstance(key, tuple) and key:
            if key[0] == self.graph.prefix and len(key) == 2:
                return key[1]
            if isinstance(key[0], str) and key[0].startswith(self.graph.prefix + "-"):
                return (key[0].removeprefix(self.graph.prefix + "-"), *key[1:])
        raise KeyError(key)

    def _tasks(self, keys):
        records, key_annotations = self.graph.inspect(
            _selected_tasks, self.layer_name, keys
        )
        tasks = {}
        parents = {}
        for source, (payload, dependencies) in records.items():
            key = self._scoped_key(source)
            scoped = tuple(self._scoped_key(parent) for parent in dependencies)
            evaluate = functools.partial(
                _run_isolated,
                self.graph.settings,
                _task_value,
                source,
                payload,
                dependencies,
            )
            tasks[key] = (evaluate, *scoped)
            parents[key] = set(scoped)
        annotations = dict(self.annotations or {})
        for name, values in key_annotations.items():
            annotations[name] = {
                self._scoped_key(key): value for key, value in values.items()
            }.__getitem__
        return MaterializedLayer(tasks, annotations=annotations), parents

    def _materialize(self) -> MaterializedLayer:
        if self._full is None:
            self._full, _ = self._tasks(None)
        return self._full

    def __getitem__(self, key):
        return self._materialize()[key]

    def __iter__(self) -> Iterator:
        return iter(self._materialize())

    def __len__(self) -> int:
        return self.length

    @property
    def has_legacy_tasks(self) -> bool:
        # Every exposed task is (evaluate, *dependencies). Dask can cull this
        # layer without first enumerating and transferring all of its tasks.
        return True

    def is_materialized(self) -> bool:
        return self._full is not None

    @functools.cached_property
    def _outputs(self):
        if self.layer_name is not None:
            return {
                self._scoped_key(key)
                for key in self.graph.inspect(_output_keys, self.layer_name)
            }
        return {
            (self.graph.name, *index)
            for index in itertools.product(*(range(n) for n in self.numblocks))
        }

    def get_output_keys(self):
        return self._outputs

    def cull(self, keys, all_hlg_keys):
        needed = set()
        for key in keys:
            try:
                needed.add(self._source_key(key))
            except KeyError:
                continue
        return self._tasks(needed) if needed else (self, {})


def bind_result(data: xr.DataArray, settings: ExecutionSettings) -> xr.DataArray:
    prefix = f"routine-{uuid.uuid4().hex}"
    _begin_result(prefix)
    for variable in (data.variable, *data.coords.variables.values()):
        if variable.chunks is None:
            continue
        import dask.array as da

        array = variable.data
        if isinstance(array, da.Array):
            name = f"{prefix}-{array.name}"
            source = array.dask
            _cached_graphs[name] = source
            recorded = _RoutineGraph(settings, prefix, name, cloudpickle.dumps(source))
            if any(layer.annotations for layer in source.layers.values()):
                # Dask applies annotations by layer. Preserve its boundaries so
                # e.g. retries on an input do not extend to an unrelated task.
                layers = {
                    f"{prefix}-{key}": _RoutineTasks(recorded, array, key)
                    for key in source.layers
                }
                dependencies = {
                    f"{prefix}-{key}": {f"{prefix}-{dep}" for dep in deps}
                    for key, deps in source.dependencies.items()
                }
            else:
                # Without scheduling annotations a single layer can cull the
                # complete graph in one request and preserve shared dependencies.
                layers = {name: _RoutineTasks(recorded, array)}
                dependencies = {name: set()}
            graph = HighLevelGraph(layers, dependencies)
            variable.data = da.Array(graph, name, array.chunks, meta=array._meta)
    return data

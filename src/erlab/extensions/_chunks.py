"""Bind deferred extension tasks to their recorded xarray options."""

from __future__ import annotations

import functools
import itertools
import pickle
import typing
import uuid

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


def _selected_tasks(group, name, keys, payload=None):
    graph = _cached_graphs.get(name) if group == _cached_result else None
    if graph is None:
        if payload is None:
            return None
        if group != _cached_result:
            _begin_result(group)
        graph = _cached_graphs[name] = pickle.loads(payload)
    # Reconstruct both the original layers and their user callbacks exclusively
    # in the private interpreter. Cull before expanding blockwise layers.
    if keys is not None:
        graph = graph.cull({key for key in keys if key in graph})
    return {
        key: (cloudpickle.dumps(task), tuple(get_dependencies(graph, task=task)))
        for key, task in graph.items()
    }


class _RoutineTasks(Layer):
    """Expose Dask dependency edges while keeping user callbacks in the worker."""

    def __init__(self, array: da.Array, settings: ExecutionSettings, prefix: str):
        super().__init__()
        self.payload = cloudpickle.dumps(array.dask)
        self.length = len(array.dask)
        self.settings = settings
        self.prefix = prefix
        self.output = f"{prefix}-{array.name}"
        _cached_graphs[self.output] = array.dask
        self.numblocks = array.numblocks
        self._full: MaterializedLayer | None = None

    def _scoped_key(self, key):
        if isinstance(key, tuple) and key and isinstance(key[0], str):
            return (f"{self.prefix}-{key[0]}", *key[1:])
        return (self.prefix, key)

    def _source_key(self, key):
        if isinstance(key, tuple) and key:
            if key[0] == self.prefix and len(key) == 2:
                return key[1]
            if isinstance(key[0], str) and key[0].startswith(self.prefix + "-"):
                return (key[0].removeprefix(self.prefix + "-"), *key[1:])
        raise KeyError(key)

    def _tasks(self, keys):
        query = functools.partial(
            _run_isolated,
            self.settings,
            _selected_tasks,
            self.prefix,
            self.output,
            keys,
        )
        records = query()
        if records is None:
            records = query(self.payload)
        tasks = {}
        parents = {}
        for source, (payload, dependencies) in records.items():
            key = self._scoped_key(source)
            scoped = tuple(self._scoped_key(parent) for parent in dependencies)
            evaluate = functools.partial(
                _run_isolated, self.settings, _task_value, source, payload, dependencies
            )
            tasks[key] = (evaluate, *scoped)
            parents[key] = set(scoped)
        return MaterializedLayer(tasks), parents

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

    def is_materialized(self) -> bool:
        return self._full is not None

    def get_output_keys(self):
        return {
            (self.output, *index)
            for index in itertools.product(*(range(n) for n in self.numblocks))
        }

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
            layer = _RoutineTasks(array, settings, prefix)
            graph = HighLevelGraph({layer.output: layer}, {layer.output: set()})
            variable.data = da.Array(
                graph, layer.output, array.chunks, meta=array._meta
            )
    return data

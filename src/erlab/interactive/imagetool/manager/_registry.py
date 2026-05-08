"""Live ImageTool manager registry and target selection helpers."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import getpass
import json
import numbers
import os
import pathlib
import socket
import tempfile
import time
import typing
import uuid

from qtpy import QtCore

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

_STARTUP_GRACE_S = 10.0
_DEFAULT_ZMQ_TIMEOUT_S = 0.1
_LOCK_TIMEOUT_MS = 10000
_LOCK_STALE_MS = 30000
_ERROR_ACCESS_DENIED = 5
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_READY_STATE: typing.Literal["ready"] = "ready"
_STARTING_STATE: typing.Literal["starting"] = "starting"


class ImageToolManagerAmbiguousError(RuntimeError):
    """Raised when a client operation needs an explicit manager target."""


class ImageToolManagerNotFoundError(RuntimeError):
    """Raised when a requested manager target is not available."""


class ImageToolManagerRegistryError(RuntimeError):
    """Raised when the live manager registry cannot be safely updated."""


@dataclasses.dataclass(frozen=True)
class ManagerInfo:
    """Public information about a live ImageTool manager."""

    index: int
    pid: int
    host: str
    port: int
    watch_port: int
    started: str
    version: str
    is_default: bool = False


@dataclasses.dataclass(frozen=True)
class _ManagerRecord:
    internal_id: str
    index: int
    pid: int
    host: str
    port: int
    watch_port: int
    started: str
    version: str
    heartbeat: float
    state: typing.Literal["starting", "ready"] = _READY_STATE

    @classmethod
    def from_dict(cls, value: dict[str, typing.Any]) -> _ManagerRecord | None:
        try:
            state = str(value.get("state", _READY_STATE))
            if state not in {_STARTING_STATE, _READY_STATE}:
                return None
            return cls(
                internal_id=str(value["internal_id"]),
                index=int(value["index"]),
                pid=int(value["pid"]),
                host=str(value["host"]),
                port=int(value["port"]),
                watch_port=int(value["watch_port"]),
                started=str(value["started"]),
                version=str(value["version"]),
                heartbeat=float(value["heartbeat"]),
                state=typing.cast("typing.Literal['starting', 'ready']", state),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def to_public_info(self, *, is_default: bool = False) -> ManagerInfo:
        return ManagerInfo(
            index=self.index,
            pid=self.pid,
            host=self.host,
            port=self.port,
            watch_port=self.watch_port,
            started=self.started,
            version=self.version,
            is_default=is_default,
        )


def _default_registry_path() -> pathlib.Path:
    override = os.getenv("ITOOL_MANAGER_REGISTRY")
    if override:
        return pathlib.Path(override).expanduser()

    username = getpass.getuser() or "user"
    return (
        pathlib.Path(tempfile.gettempdir())
        / f"erlab-imagetool-managers-{username}.json"
    )


_REGISTRY_PATH = _default_registry_path()
_LOCK_PATH = _REGISTRY_PATH.with_suffix(_REGISTRY_PATH.suffix + ".lock")
_default_manager_index: int | None = (
    int(os.environ["ITOOL_MANAGER_TARGET"])
    if os.getenv("ITOOL_MANAGER_TARGET", "").lstrip("+-").isdigit()
    else None
)


@contextlib.contextmanager
def _registry_lock() -> Iterator[None]:
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = QtCore.QLockFile(str(_LOCK_PATH))
    lock.setStaleLockTime(_LOCK_STALE_MS)
    if not lock.tryLock(_LOCK_TIMEOUT_MS):
        raise ImageToolManagerRegistryError(
            f"Could not lock ImageTool manager registry: {lock.error()!s}"
        )
    try:
        yield
    finally:
        lock.unlock()


def _read_records_unlocked() -> list[_ManagerRecord]:
    if not _REGISTRY_PATH.exists():
        return []
    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    records: list[_ManagerRecord] = []
    for item in raw:
        if isinstance(item, dict):
            record = _ManagerRecord.from_dict(item)
            if record is not None:
                records.append(record)
    return records


def _write_records_unlocked(records: list[_ManagerRecord]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps([dataclasses.asdict(record) for record in records], indent=2) + "\n"
    ).encode("utf-8")
    save_file = QtCore.QSaveFile(str(_REGISTRY_PATH))
    if not save_file.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
        raise ImageToolManagerRegistryError(
            f"Could not open ImageTool manager registry for writing: "
            f"{save_file.errorString()}"
        )
    if save_file.write(payload) != len(payload):
        save_file.cancelWriting()
        raise ImageToolManagerRegistryError(
            f"Could not write ImageTool manager registry: {save_file.errorString()}"
        )
    if not save_file.commit():
        raise ImageToolManagerRegistryError(
            f"Could not commit ImageTool manager registry: {save_file.errorString()}"
        )


def _normalize_manager_index(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"Manager {label} must be an integer")
    index = int(value)
    if index < 0:
        raise ValueError(f"Manager {label} must be >= 0")
    return index


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":  # pragma: no cover - platform dependent
        import ctypes
        from ctypes import wintypes

        ctypes_windows = typing.cast("typing.Any", ctypes)
        kernel32 = ctypes_windows.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes_windows.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _is_tcp_port_open(
    host: str, port: int, timeout: float = _DEFAULT_ZMQ_TIMEOUT_S
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _record_is_active(record: _ManagerRecord, *, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    if not _pid_exists(record.pid):
        return False
    if record.state == _STARTING_STATE:
        return now - record.heartbeat <= _STARTUP_GRACE_S
    if _is_tcp_port_open(record.host, record.port):
        return True
    return now - record.heartbeat <= _STARTUP_GRACE_S


def _active_records_unlocked() -> list[_ManagerRecord]:
    now = time.time()
    records = _read_records_unlocked()
    active = [record for record in records if _record_is_active(record, now=now)]
    if len(active) != len(records):
        _write_records_unlocked(active)
    return sorted(active, key=lambda record: record.index)


def live_manager_records() -> tuple[_ManagerRecord, ...]:
    """Return ready live managers, removing stale registry entries."""
    with _registry_lock():
        return tuple(
            record
            for record in _active_records_unlocked()
            if record.state == _READY_STATE
        )


def reserve_manager_record(*, host: str) -> _ManagerRecord:
    """Reserve a launch index for a starting manager."""
    with _registry_lock():
        active = list(_active_records_unlocked())
        index = max((record.index for record in active), default=-1) + 1
        record = _ManagerRecord(
            internal_id=uuid.uuid4().hex,
            index=index,
            pid=os.getpid(),
            host=host,
            port=0,
            watch_port=0,
            started=datetime.datetime.now().isoformat(timespec="seconds"),
            version=str(erlab.__version__),
            heartbeat=time.time(),
            state=_STARTING_STATE,
        )
        _write_records_unlocked([*active, record])
        return record


def activate_manager_record(
    internal_id: str, *, port: int, watch_port: int
) -> _ManagerRecord:
    """Mark a reserved manager record as ready with its bound endpoints."""
    with _registry_lock():
        active = _active_records_unlocked()
        updated: list[_ManagerRecord] = []
        activated: _ManagerRecord | None = None
        for record in active:
            if record.internal_id == internal_id:
                activated = dataclasses.replace(
                    record,
                    port=port,
                    watch_port=watch_port,
                    heartbeat=time.time(),
                    state=_READY_STATE,
                )
                updated.append(activated)
            else:
                updated.append(record)
        if activated is None:
            raise ImageToolManagerRegistryError(
                "Reserved ImageTool manager record is no longer available"
            )
        _write_records_unlocked(updated)
        return activated


def refresh_manager_record(internal_id: str) -> None:
    """Refresh a manager heartbeat."""
    with _registry_lock():
        records = _active_records_unlocked()
        refreshed: list[_ManagerRecord] = []
        changed = False
        for record in records:
            if record.internal_id == internal_id:
                refreshed.append(dataclasses.replace(record, heartbeat=time.time()))
                changed = True
            else:
                refreshed.append(record)
        if changed:
            _write_records_unlocked(refreshed)


def unregister_manager_record(internal_id: str) -> None:
    """Remove a manager from the live registry."""
    with _registry_lock():
        records = [
            record
            for record in _read_records_unlocked()
            if record.internal_id != internal_id
        ]
        _write_records_unlocked(records)


def set_default_manager(index: int) -> int:
    """Set the default ImageTool manager for this Python process.

    Parameters
    ----------
    index
        0-based manager index from :func:`list_managers`.

    Returns
    -------
    int
        The selected manager index.

    Raises
    ------
    ValueError
        If ``index`` is negative or no live manager has that index.
    """
    index = _normalize_manager_index(index, label="index")
    if not any(record.index == index for record in live_manager_records()):
        raise ValueError(f"No live ImageTool manager with index {index}")
    global _default_manager_index
    _default_manager_index = index
    return index


def get_default_manager(*, validate: bool = True) -> int | None:
    """Return the default ImageTool manager for this Python process.

    Parameters
    ----------
    validate
        If ``True``, clear and return ``None`` when the saved default no longer
        points to a live manager.

    Returns
    -------
    int or None
        The default 0-based manager index, or ``None`` if no default is set.
    """
    global _default_manager_index
    if _default_manager_index is None or not validate:
        return _default_manager_index
    if any(record.index == _default_manager_index for record in live_manager_records()):
        return _default_manager_index
    _default_manager_index = None
    return None


def clear_default_manager() -> None:
    """Clear the default ImageTool manager for this Python process."""
    global _default_manager_index
    _default_manager_index = None


def resolve_manager_record(target: int | None = None) -> _ManagerRecord:
    """Resolve a target index or client default to a live manager record."""
    global _default_manager_index
    if target is not None:
        target = _normalize_manager_index(target, label="target")

    records = live_manager_records()
    if target is not None:
        for record in records:
            if record.index == target:
                return record
        raise ImageToolManagerNotFoundError(
            f"No live ImageTool manager with index {target}"
        )

    default_index = _default_manager_index
    if default_index is not None:
        for record in records:
            if record.index == default_index:
                return record
        _default_manager_index = None

    if len(records) == 1:
        return records[0]

    if len(records) == 0:
        raise ImageToolManagerNotFoundError(
            "ImageTool manager is not running. Start ImageTool manager and try again."
        )

    raise ImageToolManagerAmbiguousError(
        "Multiple ImageTool managers are running. Select one with "
        "`set_default_manager(index)` or pass an explicit manager index."
    )


def manager_selection_info() -> dict[str, object]:
    """Return manager selection state for external clients.

    Returns
    -------
    dict
        JSON-serializable dictionary with live manager records, the current default
        index, the resolved index when one can be selected without prompting, a
        ``needs_selection`` flag, and a reason string. The reason is one of
        ``"none"``, ``"single"``, ``"default"``, or ``"multiple"``.
    """
    global _default_manager_index
    records = live_manager_records()
    default_index = _default_manager_index
    if default_index is not None and all(
        record.index != default_index for record in records
    ):
        _default_manager_index = None
        default_index = None
    resolved_index: int | None = None
    reason: str
    if len(records) == 0:
        reason = "none"
    elif default_index is not None:
        resolved_index = default_index
        reason = "default"
    elif len(records) == 1:
        resolved_index = records[0].index
        reason = "single"
    else:
        reason = "multiple"

    return {
        "managers": [
            dataclasses.asdict(
                record.to_public_info(is_default=record.index == default_index)
            )
            for record in records
        ],
        "default_index": default_index,
        "resolved_index": resolved_index,
        "needs_selection": reason == "multiple",
        "reason": reason,
    }


@contextlib.contextmanager
def use_manager(index: int) -> Iterator[None]:
    """Temporarily set the default ImageTool manager.

    Parameters
    ----------
    index
        0-based manager index to use inside the context.
    """
    previous = get_default_manager(validate=False)
    set_default_manager(index)
    try:
        yield
    finally:
        global _default_manager_index
        _default_manager_index = previous

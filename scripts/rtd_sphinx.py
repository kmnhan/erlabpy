"""Capture Sphinx diagnostics before the Read the Docs build deadline."""

import contextlib
import faulthandler
import json
import os
import pathlib
import runpy
import subprocess
import sys
import tempfile
import time
import traceback

if __name__ == "__main__":
    # Arm the startup watchdog before importing native process helpers.
    faulthandler.dump_traceback_later(30, exit=True)

import psutil


def _stop_process_tree(parent: psutil.Process, *, grace_period: float) -> None:
    try:
        if not parent.is_running():
            return
        # Detached Jupyter kernels can escape a process group. Collect them
        # before stopping their parent, which would otherwise reparent them.
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return

    for process in reversed(processes):
        with contextlib.suppress(psutil.NoSuchProcess):
            process.terminate()
    _, alive = psutil.wait_procs(processes, timeout=grace_period)
    for process in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill()
    psutil.wait_procs(alive, timeout=grace_period)


def _write_process(path: pathlib.Path, pid: int) -> None:
    try:
        created_at = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        created_at = None
    path.write_text(json.dumps({"pid": pid, "created_at": created_at}))


def _read_process(path: pathlib.Path) -> psutil.Process | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    try:
        process = psutil.Process(data["pid"])
        if (
            process.create_time() == data["created_at"]
            and process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def _write_result(directory: pathlib.Path, returncode: int) -> None:
    temporary = directory / "result.tmp"
    temporary.write_text(str(returncode))
    temporary.replace(directory / "result")


def _run_worker(
    argv: list[str], directory: pathlib.Path, *, timeout: float, grace_period: float
) -> int:
    (directory / "stage").write_text("launching Sphinx worker")
    env = os.environ | {"TMPDIR": str(directory), "PYTHONUNBUFFERED": "1"}
    with (directory / "sphinx.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, __file__, "--worker", *argv],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        _write_process(directory / "worker.json", process.pid)
        (directory / "stage").write_text("running Sphinx")
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            (directory / "stage").write_text("stopping Sphinx after deadline")
            print(f"Sphinx exceeded the {timeout:g} second diagnostic deadline.")
            with contextlib.suppress(psutil.NoSuchProcess):
                _stop_process_tree(
                    psutil.Process(process.pid), grace_period=grace_period
                )
            process.poll()
            returncode = 124
    (directory / "stage").write_text(f"Sphinx worker exited with status {returncode}")
    return returncode


def _relay_logs(directory: pathlib.Path, *, limit: int | None = None) -> None:
    paths = [directory / "supervisor.log", directory / "sphinx.log"]
    paths.extend(sorted(directory.glob("sphinx_llm_output_*.log"))[-2:])
    for path in paths:
        if not path.exists():
            continue
        print(f"--- {path.name} ---", flush=True)
        with path.open("rb") as log:
            size = path.stat().st_size
            start = max(0, size - limit) if limit is not None else 0
            log.seek(start)
            remaining = size - start
            while remaining:
                chunk = log.read(min(remaining, 65536))
                if not chunk:
                    break
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                remaining -= len(chunk)
        print(flush=True)


def _run_sphinx(
    argv: list[str], *, timeout: float = 17 * 60, grace_period: float = 5
) -> int:
    with tempfile.TemporaryDirectory(prefix="erlab-rtd-") as directory:
        temp_dir = pathlib.Path(directory)
        returncode = _run_worker(
            argv, temp_dir, timeout=timeout, grace_period=grace_period
        )
        _relay_logs(temp_dir)
        return returncode


def _start_sphinx(argv: list[str], *, state_dir: pathlib.Path) -> None:
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=False)
    (state_dir / "stage").write_text("starting background supervisor")
    with (state_dir / "supervisor.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, __file__, "--background", str(state_dir), *argv],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ | {"PYTHONUNBUFFERED": "1"},
        )
        _write_process(state_dir / "controller.json", process.pid)
    print(f"Started Sphinx supervisor {process.pid}; diagnostics: {state_dir}")


def _poll_sphinx(
    *,
    state_dir: pathlib.Path,
    wait_seconds: float = 180,
    final: bool = False,
    grace_period: float = 5,
) -> int:
    deadline = time.monotonic() + wait_seconds
    result = state_dir / "result"
    returncode = None
    while True:
        if result.exists():
            returncode = int(result.read_text())
            break
        if _read_process(state_dir / "controller.json") is None:
            (state_dir / "stage").write_text("supervisor stopped without a result")
            returncode = 1
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1, remaining))

    if returncode is None and final:
        (state_dir / "stage").write_text("final checkpoint deadline reached")
        returncode = 124
    if returncode is not None:
        if returncode:
            for name in ("controller.json", "worker.json"):
                process = _read_process(state_dir / name)
                if process is not None:
                    _stop_process_tree(process, grace_period=grace_period)
        _write_result(state_dir, returncode)

    print(f"RTD Sphinx checkpoint: {(state_dir / 'stage').read_text()}")
    print(f"Recorded result: {returncode}")
    # Keep each exec response small and save it before the outer RTD deadline.
    _relay_logs(state_dir, limit=8192)
    if final:
        import shutil

        shutil.rmtree(state_dir)
    return returncode if returncode is not None else 0


if __name__ == "__main__":
    state = pathlib.Path(os.environ.get("READTHEDOCS_OUTPUT", "docs/_build"))
    state = state.resolve() / ".rtd-sphinx"
    if sys.argv[1:2] == ["--worker"]:
        del sys.argv[1]
        faulthandler.dump_traceback_later(60, repeat=True)
        runpy.run_module("sphinx", run_name="__main__")
    elif sys.argv[1:2] == ["--background"]:
        faulthandler.dump_traceback_later(60, repeat=True)
        state = pathlib.Path(sys.argv[2])
        try:
            code = _run_worker(sys.argv[3:], state, timeout=17 * 60, grace_period=5)
        except Exception:
            traceback.print_exc()
            code = 1
        _write_result(state, code)
        raise SystemExit(code)
    elif sys.argv[1:2] == ["--start"]:
        _start_sphinx(sys.argv[2:], state_dir=state)
        faulthandler.cancel_dump_traceback_later()
    elif sys.argv[1:2] == ["--poll"]:
        faulthandler.dump_traceback_later(240, exit=True)
        raise SystemExit(_poll_sphinx(state_dir=state, final="--final" in sys.argv))
    else:
        faulthandler.cancel_dump_traceback_later()
        raise SystemExit(_run_sphinx(sys.argv[1:]))

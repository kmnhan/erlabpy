"""Capture Sphinx diagnostics before the Read the Docs build deadline."""

import contextlib
import faulthandler
import os
import pathlib
import runpy
import subprocess
import sys
import tempfile

import psutil


def _stop_process_tree(pid: int, *, grace_period: float) -> None:
    try:
        parent = psutil.Process(pid)
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


def _run_sphinx(
    argv: list[str], *, timeout: float = 19 * 60, grace_period: float = 5
) -> int:
    with tempfile.TemporaryDirectory(prefix="erlab-rtd-") as directory:
        temp_dir = pathlib.Path(directory)
        log_path = temp_dir / "sphinx.log"
        env = os.environ | {"TMPDIR": directory, "PYTHONUNBUFFERED": "1"}
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "--worker", *argv],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"Sphinx exceeded the {timeout:g} second diagnostic deadline.")
                _stop_process_tree(process.pid, grace_period=grace_period)
                process.poll()
                returncode = 124

        # Only the supervisor owns RTD's output pipe. A detached descendant can
        # keep a regular file open without delaying this command's completion.
        log_paths = [log_path]
        if returncode:
            log_paths.extend(sorted(temp_dir.glob("sphinx_llm_output_*.log")))
        for path in log_paths:
            print(f"--- {path.name} ---", flush=True)
            with path.open("rb") as log:
                # Copy a fixed snapshot even if a surviving descendant writes
                # more data while the supervisor relays the diagnostic output.
                remaining = path.stat().st_size
                while remaining:
                    chunk = log.read(min(remaining, 65536))
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    remaining -= len(chunk)
        sys.stdout.flush()
        return returncode


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        del sys.argv[1]
        faulthandler.dump_traceback_later(18 * 60)
        runpy.run_module("sphinx", run_name="__main__")
    else:
        raise SystemExit(_run_sphinx(sys.argv[1:]))

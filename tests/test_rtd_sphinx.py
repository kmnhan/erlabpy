from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time
import typing

import psutil
import pytest

if typing.TYPE_CHECKING:
    import types


def _load_rtd_sphinx() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rtd_sphinx.py"
    spec = importlib.util.spec_from_file_location("rtd_sphinx", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rtd_sphinx = _load_rtd_sphinx()


def _install_fake_sphinx(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    (tmp_path / "sphinx.py").write_text(textwrap.dedent(source))
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))),
    )


def _stop_child(pid: int) -> None:
    with contextlib.suppress(psutil.NoSuchProcess, psutil.TimeoutExpired):
        process = psutil.Process(pid)
        process.kill()
        process.wait(timeout=2)


def test_rtd_sphinx_forwards_arguments_and_output(tmp_path, monkeypatch, capfd) -> None:
    _install_fake_sphinx(
        tmp_path,
        monkeypatch,
        """
        import json
        import sys

        print("sphinx-args:" + json.dumps(sys.argv[1:]), flush=True)
        print("sphinx-standard-output", flush=True)
        print("sphinx-standard-error", file=sys.stderr, flush=True)
        """,
    )
    arguments = ["-T", "-b", "html", "source with spaces", "output with 'quotes'"]

    assert rtd_sphinx._run_sphinx(arguments, timeout=5, grace_period=0.2) == 0

    captured = capfd.readouterr()
    output = captured.out + captured.err
    argument_line = next(
        line for line in output.splitlines() if line.startswith("sphinx-args:")
    )
    assert json.loads(argument_line.removeprefix("sphinx-args:")) == arguments
    assert "sphinx-standard-output" in output
    assert "sphinx-standard-error" in output


def test_rtd_sphinx_preserves_failure_and_markdown_log(
    tmp_path, monkeypatch, capfd
) -> None:
    _install_fake_sphinx(
        tmp_path,
        monkeypatch,
        """
        import os
        import pathlib
        import sys

        log = pathlib.Path(os.environ["TMPDIR"]) / "sphinx_llm_output_probe.log"
        log.write_text("markdown-subprocess-diagnostic\\n")
        print("sphinx-parent-diagnostic", flush=True)
        sys.exit(7)
        """,
    )

    assert rtd_sphinx._run_sphinx([], timeout=5, grace_period=0.2) == 7

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "sphinx-parent-diagnostic" in output
    assert "markdown-subprocess-diagnostic" in output


def test_rtd_sphinx_returns_without_detached_child_output_eof(
    tmp_path, monkeypatch
) -> None:
    if os.name == "nt":
        pytest.skip("This RTD regression needs POSIX process sessions.")
    pid_file = tmp_path / "child.pid"
    _install_fake_sphinx(
        tmp_path,
        monkeypatch,
        f"""
        import pathlib
        import subprocess
        import sys

        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            start_new_session=True,
        )
        pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
        print("sphinx-parent-completed", flush=True)
        """,
    )

    try:
        # Docker also captures an actual pipe. A file-backed pytest capture
        # would not expose a detached child that inherits the supervisor pipe.
        result = subprocess.run(
            [sys.executable, rtd_sphinx.__file__],
            capture_output=True,
            text=True,
            timeout=6,
        )
        assert result.returncode == 0
        assert "sphinx-parent-completed" in result.stdout + result.stderr
    finally:
        if pid_file.exists():
            _stop_child(int(pid_file.read_text()))


def test_rtd_sphinx_timeout_stops_detached_child_and_replays_traceback(
    tmp_path, monkeypatch, capfd
) -> None:
    if os.name == "nt":
        pytest.skip("This RTD regression needs POSIX process sessions.")
    pid_file = tmp_path / "child.pid"
    child_source = textwrap.dedent(
        f"""
        import os
        import pathlib
        import signal
        import time

        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
        print("detached-child-ready", flush=True)
        time.sleep(10)
        """
    )
    _install_fake_sphinx(
        tmp_path,
        monkeypatch,
        f"""
        import faulthandler
        import subprocess
        import sys
        import time

        faulthandler.dump_traceback_later(0.2)
        subprocess.Popen(
            [sys.executable, "-c", {child_source!r}], start_new_session=True
        )
        print("sphinx-parent-blocked", flush=True)
        time.sleep(10)
        """,
    )

    started = time.monotonic()
    try:
        assert rtd_sphinx._run_sphinx([], timeout=2, grace_period=0.2) == 124
        assert time.monotonic() - started < 8
        captured = capfd.readouterr()
        output = captured.out + captured.err
        assert "sphinx-parent-blocked" in output
        assert "detached-child-ready" in output
        assert "Timeout" in output
        assert "sphinx.py" in output
        child_pid = int(pid_file.read_text())
        if psutil.pid_exists(child_pid):
            assert psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if pid_file.exists():
            _stop_child(int(pid_file.read_text()))


def _wait_for_protocol_file(path: pathlib.Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.is_file(), f"Background Sphinx did not create {path.name}"


def _stop_state_processes(state_dir: pathlib.Path) -> None:
    processes: dict[int, psutil.Process] = {}
    for name in ("controller.json", "worker.json"):
        try:
            record = json.loads((state_dir / name).read_text())
            process = psutil.Process(record["pid"])
            if process.create_time() != record["created_at"]:
                continue
            processes[process.pid] = process
            for child in process.children(recursive=True):
                processes[child.pid] = child
        except (OSError, ValueError, psutil.NoSuchProcess):
            continue
    for process in reversed(list(processes.values())):
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill()
    psutil.wait_procs(list(processes.values()), timeout=2)


def _start_background_cli(state_dir: pathlib.Path) -> None:
    result = subprocess.run(
        [sys.executable, rtd_sphinx.__file__, "--start"],
        env=os.environ | {"READTHEDOCS_OUTPUT": str(state_dir.parent)},
        capture_output=True,
        text=True,
        timeout=6,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rtd_sphinx_start_cli_returns_while_background_runs(
    tmp_path, monkeypatch
) -> None:
    if os.name == "nt":
        pytest.skip("This RTD regression needs POSIX process sessions.")
    state_dir = tmp_path / "output with spaces" / ".rtd-sphinx"
    monkeypatch.setenv("READTHEDOCS_OUTPUT", str(state_dir.parent))
    _install_fake_sphinx(
        tmp_path,
        monkeypatch,
        """
        import time

        print("background-sphinx-started", flush=True)
        time.sleep(10)
        """,
    )

    try:
        result = subprocess.run(
            [sys.executable, rtd_sphinx.__file__, "--start", "source with spaces"],
            capture_output=True,
            text=True,
            timeout=6,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        _wait_for_protocol_file(state_dir / "worker.json")
        controller = json.loads((state_dir / "controller.json").read_text())
        assert (
            psutil.Process(controller["pid"]).create_time() == controller["created_at"]
        )
    finally:
        _stop_state_processes(state_dir)


def test_rtd_sphinx_running_poll_returns_stage_and_log(
    tmp_path, monkeypatch, capfd
) -> None:
    state_dir = tmp_path / ".rtd-sphinx"
    ready = tmp_path / "ready"
    _install_fake_sphinx(
        tmp_path,
        monkeypatch,
        f"""
        import pathlib
        import time

        print("running-checkpoint-diagnostic", flush=True)
        pathlib.Path({str(ready)!r}).write_text("ready")
        time.sleep(10)
        """,
    )

    try:
        _start_background_cli(state_dir)
        _wait_for_protocol_file(ready)
        stage = (state_dir / "stage").read_text().strip()
        capfd.readouterr()

        assert (
            rtd_sphinx._poll_sphinx(
                state_dir=state_dir, wait_seconds=0, grace_period=0.2
            )
            == 0
        )

        captured = capfd.readouterr()
        output = captured.out + captured.err
        assert stage
        assert stage in output
        assert "running-checkpoint-diagnostic" in output
        assert not (state_dir / "result").exists()
    finally:
        _stop_state_processes(state_dir)


def test_rtd_sphinx_completed_poll_preserves_failure_and_logs(
    tmp_path, monkeypatch, capfd
) -> None:
    state_dir = tmp_path / ".rtd-sphinx"
    _install_fake_sphinx(
        tmp_path,
        monkeypatch,
        """
        import os
        import pathlib
        import sys

        log = pathlib.Path(os.environ["TMPDIR"]) / "sphinx_llm_output_probe.log"
        log.write_text("checkpoint-markdown-diagnostic\\n")
        print("checkpoint-parent-diagnostic", flush=True)
        sys.exit(7)
        """,
    )

    try:
        _start_background_cli(state_dir)

        assert (
            rtd_sphinx._poll_sphinx(
                state_dir=state_dir, wait_seconds=5, grace_period=0.2
            )
            == 7
        )

        captured = capfd.readouterr()
        output = captured.out + captured.err
        assert "checkpoint-parent-diagnostic" in output
        assert "checkpoint-markdown-diagnostic" in output
        assert int((state_dir / "result").read_text()) == 7
    finally:
        _stop_state_processes(state_dir)


def test_rtd_sphinx_final_poll_stops_detached_child(
    tmp_path, monkeypatch, capfd
) -> None:
    if os.name == "nt":
        pytest.skip("This RTD regression needs POSIX process sessions.")
    state_dir = tmp_path / ".rtd-sphinx"
    pid_file = tmp_path / "child.pid"
    child_source = textwrap.dedent(
        f"""
        import os
        import pathlib
        import signal
        import time

        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
        print("checkpoint-detached-child-ready", flush=True)
        time.sleep(10)
        """
    )
    _install_fake_sphinx(
        tmp_path,
        monkeypatch,
        f"""
        import subprocess
        import sys
        import time

        subprocess.Popen(
            [sys.executable, "-c", {child_source!r}], start_new_session=True
        )
        time.sleep(10)
        """,
    )

    try:
        _start_background_cli(state_dir)
        _wait_for_protocol_file(pid_file)

        assert (
            rtd_sphinx._poll_sphinx(
                state_dir=state_dir, wait_seconds=0, final=True, grace_period=0.2
            )
            == 124
        )

        captured = capfd.readouterr()
        assert "checkpoint-detached-child-ready" in captured.out + captured.err
        child_pid = int(pid_file.read_text())
        if psutil.pid_exists(child_pid):
            assert psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        assert not state_dir.exists()
    finally:
        _stop_state_processes(state_dir)
        if pid_file.exists():
            _stop_child(int(pid_file.read_text()))


def test_rtd_sphinx_poll_detects_dead_controller(tmp_path, capfd) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    controller = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
    )
    record = {
        "pid": controller.pid,
        "created_at": psutil.Process(controller.pid).create_time(),
    }
    controller.communicate(timeout=5)
    (state_dir / "controller.json").write_text(json.dumps(record))
    (state_dir / "stage").write_text("launching Sphinx worker")
    (state_dir / "supervisor.log").write_text("controller-failure-diagnostic\n")

    started = time.monotonic()
    assert (
        rtd_sphinx._poll_sphinx(state_dir=state_dir, wait_seconds=5, grace_period=0.2)
        == 1
    )
    assert time.monotonic() - started < 2
    captured = capfd.readouterr()
    assert "controller-failure-diagnostic" in captured.out + captured.err
    assert int((state_dir / "result").read_text()) == 1


def test_rtd_sphinx_log_tail_replaces_partial_utf8_character(tmp_path, capfd) -> None:
    (tmp_path / "sphinx.log").write_text("αtail", encoding="utf-8")

    rtd_sphinx._relay_logs(tmp_path, limit=5)

    captured = capfd.readouterr()
    assert "\ufffdtail" in captured.out

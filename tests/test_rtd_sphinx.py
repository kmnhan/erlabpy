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

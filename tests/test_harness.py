"""Harness unit tests — fast, deterministic, no engine needed.

Simulate the engine with tiny `python -c` commands that emit controlled output
(including tqdm \\r updates, stalls, hangs) so we can prove stall/cancel/timeout
and carriage-return parsing without running a real 2-minute cover.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from harness import EngineCancelled, EngineTimeout, stream_process  # noqa: E402

ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
PY = sys.executable


def _run(code, **kw):
    lines = []
    kw.setdefault("on_line", lambda l: lines.append(l))
    rc = stream_process([PY, "-u", "-c", code], cwd=".", env=ENV, **kw)
    return rc, lines


def test_plain_lines_and_exit_code():
    rc, lines = _run("print('a'); print('b')")
    assert rc == 0
    assert "a" in lines and "b" in lines


def test_carriage_return_updates_become_events():
    # tqdm writes \r updates with no newline — each must surface as its own line.
    code = "import sys,time\n" + "\n".join(
        f"sys.stdout.write('\\r {i}/6'); sys.stdout.flush()" for i in range(1, 7)
    ) + "\nsys.stdout.write('\\n')"
    rc, lines = _run(code)
    assert rc == 0
    joined = " ".join(lines)
    assert "1/6" in joined and "6/6" in joined


def test_done_marker_returns_without_waiting():
    # Engine prints success then "hangs" forever; done_marker must short-circuit.
    code = "import time; print('Cover generated at /tmp/x.wav'); time.sleep(60)"
    start = time.monotonic()
    rc, lines = _run(code, done_marker="Cover generated at")
    assert rc == 0
    assert time.monotonic() - start < 10  # did NOT wait for the 60s sleep
    assert any("Cover generated at" in l for l in lines)


def test_stall_then_resume_callbacks():
    stalls, resumes = [], []
    # Print, go quiet 2s (triggers stall at 1s), then print again (resume).
    code = "import time; print('start'); time.sleep(2); print('back')"
    rc, _ = _run(code, stall_seconds=1, on_stall=lambda: stalls.append(1),
                 on_resume=lambda: resumes.append(1))
    assert rc == 0
    assert len(stalls) == 1
    assert len(resumes) == 1


def test_cancel_kills_process():
    code = "import time; print('go'); time.sleep(30)"
    ev = threading.Event()
    threading.Timer(0.5, ev.set).start()
    start = time.monotonic()
    try:
        _run(code, cancel_event=ev)
        assert False, "expected EngineCancelled"
    except EngineCancelled:
        pass
    assert time.monotonic() - start < 10


def test_timeout_kills_process():
    code = "import time; print('go'); time.sleep(30)"
    start = time.monotonic()
    try:
        _run(code, timeout=1)
        assert False, "expected EngineTimeout"
    except EngineTimeout:
        pass
    assert time.monotonic() - start < 10

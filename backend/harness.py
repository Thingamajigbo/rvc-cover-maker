"""Robust subprocess streaming harness.

Why this exists: the engine's RVC step prints tqdm progress with carriage
returns (\\r), not newlines. A naive `for line in proc.stdout` yields nothing
until the whole step ends — so the UI looked frozen at 80%. This reader splits
on BOTH \\r and \\n, so every tqdm tick becomes an event. It also adds the
watchdogs a long-running job needs: stall detection, cancellation, hard timeout.
"""
import queue as _queue
import subprocess
import threading
import time


class EngineCancelled(Exception):
    """Raised when a caller cancels the job via the cancel_event."""


class EngineTimeout(Exception):
    """Raised when the process exceeds the hard timeout."""


def _reader(pipe, q: "_queue.Queue"):
    """Read bytes, emit a line on every \\r or \\n. Sentinel None on EOF."""
    buf = bytearray()
    while True:
        b = pipe.read(1)
        if not b:
            if buf:
                q.put(bytes(buf).decode("utf-8", "replace"))
            q.put(None)
            return
        if b in (b"\n", b"\r"):
            if buf:
                q.put(bytes(buf).decode("utf-8", "replace"))
                buf.clear()
        else:
            buf += b


def stream_process(
    cmd, cwd, env,
    on_line=None, on_stall=None, on_resume=None,
    cancel_event: "threading.Event | None" = None,
    stall_seconds: float = 240,
    timeout: float = 3600,
    done_marker: str | None = None,
) -> int:
    """Run cmd, streaming each output line to on_line.

    on_stall()/on_resume(): fired when output goes quiet for stall_seconds and
        when it resumes (the watchdog that surfaces a real hang to the UI).
    cancel_event: when set, the process is killed and EngineCancelled raised.
    done_marker: when a line contains it, the process is killed and 0 returned
        (the engine hangs on shutdown after printing success — don't wait on it).
    Returns the process exit code (0 on done_marker hit).
    """
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    q: "_queue.Queue" = _queue.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()

    start = last = time.monotonic()
    stalled = False
    marker_hit = False
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                raise EngineCancelled()
            try:
                item = q.get(timeout=1.0)
            except _queue.Empty:
                now = time.monotonic()
                if timeout and now - start > timeout:
                    proc.kill()
                    raise EngineTimeout()
                if not stalled and stall_seconds and now - last > stall_seconds:
                    stalled = True
                    if on_stall:
                        on_stall()
                continue
            if item is None:
                break
            last = time.monotonic()
            if stalled:
                stalled = False
                if on_resume:
                    on_resume()
            if on_line:
                on_line(item)
            if done_marker and done_marker in item:
                marker_hit = True
                proc.kill()
                break
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    return 0 if marker_hit else proc.wait()

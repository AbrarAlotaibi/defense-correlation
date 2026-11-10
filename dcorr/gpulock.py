"""One GPU job at a time, enforced by an OS-level file lock.

Two jobs that each fit the card alone do not fit together. On Windows the driver does not
raise out of memory for the overflow: it spills into shared system memory, and both jobs
keep running several times slower (observed while a grid run and the e2e suite overlapped:
2.4 GiB spilled, stage 04 timings inflated past the point of being usable). Timings measured
under contention are also worthless for scheduling.

Every process that puts a model on the GPU takes this lock first. The holder's pid and
command are written into the lock file, so a waiter can say what it is waiting for.

    with gpu_lock("grid seed 0 stage 04"):
        ...

`DCORR_NO_GPU_LOCK=1` disables it (for deliberate multi-GPU or CPU-only use).
"""
from __future__ import annotations

import atexit
import contextlib
import os
import sys
import time
from pathlib import Path

def lock_path() -> Path:
    """Resolved per call, so a test or a second card can redirect it with DCORR_GPU_LOCK."""
    return Path(os.environ.get("DCORR_GPU_LOCK") or Path.home() / ".dcorr_gpu.lock")


def _read_holder(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


def _pid_of(holder: str) -> int | None:
    try:
        return int(holder.split("|", 1)[0].strip().split()[1])
    except (IndexError, ValueError):
        return None


def _alive(pid: int) -> bool:
    """Windows and POSIX: is a process with this pid running?"""
    if os.name == "nt":
        out = os.popen(f'tasklist /FI "PID eq {pid}" /NH').read()
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _clear_if_stale(path: Path) -> bool:
    """Remove a lock whose owner is gone (force-killed, so atexit never ran)."""
    holder = _read_holder(path)
    pid = _pid_of(holder)
    if pid is None or pid == os.getpid() or _alive(pid):
        return False
    try:
        path.unlink()
        print(f"[gpulock] cleared a stale lock from a dead process: {holder}", flush=True)
        return True
    except OSError:
        return False


@contextlib.contextmanager
def gpu_lock(label: str = "", timeout: float | None = None, poll: float = 10.0):
    """Hold the exclusive GPU lock for the block. Waits (reporting the holder) by default."""
    if os.environ.get("DCORR_NO_GPU_LOCK") == "1":
        yield False
        return
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    announced = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if _clear_if_stale(path):
                continue
            if not announced:
                print(f"[gpulock] waiting for the GPU, held by: {_read_holder(path)}", flush=True)
                announced = True
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError(
                    f"GPU lock still held by {_read_holder(path)} after {timeout:.0f}s; "
                    f"delete {path} if that process is gone")
            time.sleep(poll)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid {os.getpid()} | {label or Path(sys.argv[0]).name} | "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if announced:
            print("[gpulock] acquired", flush=True)
        yield True
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def acquire_for_process(label: str = "", timeout: float | None = None) -> None:
    """Hold the GPU lock for this whole process, released at interpreter exit.

    Each stage runs as its own process, so process lifetime is exactly the right scope and
    this needs no re-indentation at the call site. A force-killed process leaves the file
    behind; the next waiter clears it once it sees the owning pid is gone.
    """
    if timeout is None:
        env = os.environ.get("DCORR_GPU_LOCK_TIMEOUT")
        timeout = float(env) if env else None
    cm = gpu_lock(label, timeout=timeout)
    cm.__enter__()
    atexit.register(cm.__exit__, None, None, None)

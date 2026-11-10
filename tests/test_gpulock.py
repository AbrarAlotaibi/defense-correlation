"""The GPU lock must actually exclude a second process (CPU-only test)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dcorr.gpulock import gpu_lock  # noqa: E402

CHILD = """
import os, sys, time
sys.path.insert(0, sys.argv[1])
from dcorr.gpulock import gpu_lock
try:
    with gpu_lock("child", timeout=float(sys.argv[2]), poll=0.2):
        print("ACQUIRED", flush=True)
        time.sleep(float(sys.argv[3]))
except TimeoutError as e:
    print("TIMEOUT", flush=True)
"""


def _child(lock: Path, timeout: float, hold: float = 0.0):
    return subprocess.Popen([sys.executable, "-c", CHILD, str(ROOT), str(timeout), str(hold)],
                            env=dict(os.environ, DCORR_GPU_LOCK=str(lock)),
                            stdout=subprocess.PIPE, text=True)


def test_second_process_waits_then_acquires(tmp_path):
    lock = tmp_path / "gpu.lock"
    os.environ["DCORR_GPU_LOCK"] = str(lock)
    try:
        with gpu_lock("parent"):
            assert lock.exists() and str(os.getpid()) in lock.read_text(encoding="utf-8")
            p = _child(lock, timeout=1.0)
            time.sleep(1.6)                       # outlives the child's own timeout
            assert "TIMEOUT" in p.stdout.read(), "the child must not enter while held"
        p = _child(lock, timeout=10.0)            # lock released: the child gets in
        assert "ACQUIRED" in p.stdout.read()
        assert not lock.exists()
    finally:
        os.environ.pop("DCORR_GPU_LOCK", None)


def test_lock_is_released_on_error(tmp_path):
    lock = tmp_path / "gpu.lock"
    os.environ["DCORR_GPU_LOCK"] = str(lock)
    try:
        with pytest.raises(ValueError):
            with gpu_lock("boom"):
                raise ValueError("boom")
        assert not lock.exists()
    finally:
        os.environ.pop("DCORR_GPU_LOCK", None)


def test_opt_out(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("DCORR_GPU_LOCK", str(lock))
    monkeypatch.setenv("DCORR_NO_GPU_LOCK", "1")
    with gpu_lock("off") as held:
        assert held is False and not lock.exists()

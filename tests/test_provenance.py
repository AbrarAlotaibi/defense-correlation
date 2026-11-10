"""Tests for the execution-provenance stamp on timing records.

Stage 04 and the grid driver record where they ran, because nothing did before and the paper's
run-to-run variation claim is partly about device arithmetic: a result set that cannot name the
device it came from cannot be checked against it.

The properties worth guarding are not the values, which differ per machine, but the contract:
it never raises, it is JSON-serialisable, and it picks up the scheduler's identifiers when they
are present, since the node name is what SLURM's own logs are keyed by.

Run: python -m pytest tests/test_provenance.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dcorr.runtime import host_provenance  # noqa: E402


def test_is_json_serialisable_and_has_the_expected_keys():
    p = host_provenance()
    json.dumps(p)                                    # must not raise
    for k in ("host", "gpu", "gpu_count", "gpu_memory_gib", "torch", "cuda",
              "python", "platform"):
        assert k in p, f"missing {k}"
    assert p["python"] and p["platform"]


def test_never_raises_even_when_everything_underneath_fails(monkeypatch):
    """A failed stage because provenance could not be read would be a bad trade."""
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: (_ for _ in ()).throw(OSError("no dns")))
    monkeypatch.setitem(sys.modules, "torch", None)   # import torch -> AttributeError inside
    p = host_provenance()
    json.dumps(p)
    assert p["host"] is None
    assert p["python"], "the fields that cannot fail should still be populated"


def test_picks_up_the_scheduler_identifiers(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "2350485")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURMD_NODENAME", "gpu0005")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    p = host_provenance()
    assert p["slurm_job"] == "2350485"
    assert p["slurm_array_task"] == "3"
    assert p["slurm_node"] == "gpu0005"
    assert p["cuda_visible_devices"] == "1"


def test_scheduler_keys_absent_off_cluster(monkeypatch):
    """Absent rather than null, so a local record is not indistinguishable from a job whose
    scheduler fields failed to read."""
    for k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME",
              "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    p = host_provenance()
    assert "slurm_job" not in p
    assert "slurm_node" not in p


def test_both_timing_writers_stamp_it():
    """Guards the wiring, not the helper: either writer losing the stamp is silent."""
    for rel in ("scripts/04_run_attacks.py", "scripts/25_run_grid.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "host_provenance" in src, f"{rel} does not stamp provenance"
        assert "**host_provenance()" in src, f"{rel} does not splice it into the record"

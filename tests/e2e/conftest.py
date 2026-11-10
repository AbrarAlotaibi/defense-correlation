"""End-to-end tests run the real stage scripts on a real model and take minutes on a GPU.

They are skipped unless DCORR_E2E=1, so `pytest tests/` stays fast:

    DCORR_E2E=1 python -m pytest tests/e2e -q -s
"""
from __future__ import annotations

import os

import pytest

# A stage run blocked by another GPU job (a grid run, say) should fail with the holder named
# rather than hang for hours. dcorr/gpulock.py reads this.
os.environ.setdefault("DCORR_GPU_LOCK_TIMEOUT", "1800")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DCORR_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="end-to-end suite: set DCORR_E2E=1 to run (needs GPU + model)")
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(skip)

"""GPU regression test for the Reloadable guard leak (found in the 1.7 Vicuna smoke test).

v2's first `reload` residency yielded the guard model itself. The caller's `with ... as
model:` variable kept it alive after the block, so the target was moved back to the GPU
while the guard (~12 GiB) was still resident. On Windows the driver spilled the overflow
into shared system memory instead of raising, and every later forward pass of the target
ran 6-8x slower (stage 04 took 5.7 h for 5 behaviours; stage 05 ran 17 h).

These tests reproduce the caller pattern exactly and require GPU memory to return to its
baseline after every block. Quick (seconds); needs a CUDA device.

Run: DCORR_E2E=1 python -m pytest tests/e2e/test_reload_memory_e2e.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # cached weights only
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dcorr.modelpool import Reloadable  # noqa: E402

MiB = 2 ** 20


def _alloc() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def _linear_loader():
    return torch.nn.Linear(8192, 8192, dtype=torch.float16, device="cuda")   # 128 MiB


def _hf_loader():
    """A real HF model through the same device_map="auto" offload path as the fp16 Llama
    Guard load (accelerate hooks, layers split across GPU and CPU)."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", dtype=torch.float16, device_map="auto",
        max_memory={0: "2GiB", "cpu": "12GiB"})
    return m.eval()


@pytest.mark.parametrize("loader", [_linear_loader, _hf_loader], ids=["linear", "hf_device_map"])
def test_guard_is_freed_before_the_target_returns(loader):
    target = torch.nn.Linear(4096, 4096, dtype=torch.float16, device="cuda")     # 32 MiB
    base = _alloc()
    rl = Reloadable(loader)
    for _ in range(3):                                  # three blocks, as in three rounds
        with rl.active(evict=[target]) as model:        # the caller keeps `model` alive
            assert next(target.parameters()).device.type == "cpu"
            during = _alloc()
            assert during > 1 * MiB, "the guard should occupy GPU memory during the block"
            _ = model.training                          # use it through the proxy
        after = _alloc()
        assert next(target.parameters()).device.type == "cuda", "target not restored"
        assert after <= base + 1 * MiB, f"guard still resident: {(after - base) / MiB:.0f} MiB"
    assert rl.n_loads == 3


def test_restore_refuses_to_spill():
    target = torch.nn.Linear(4096, 4096, dtype=torch.float16, device="cuda")
    rl = Reloadable(_linear_loader)
    rl.headroom_bytes = 10 ** 15                       # more than any card has free
    with pytest.raises(RuntimeError, match="sysmem fallback"):
        with rl.active(evict=[target]):
            pass

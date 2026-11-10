"""VRAM residency for the one defense that needs a second large model.

A 16 GB card holds an fp16 7B target (13.5 GB) with room for a KV cache and nothing
else. Llama Guard 3 8B cannot co-reside. Four policies:

  resident_gpu  always on GPU. Only valid if the target is quantised (nf4) too.
  swap          CPU home, moved to GPU for a *block* of calls and moved back. Default.
                Transfers are amortised by the round structure in attacks/random_search.py:
                the whole behaviour list is advanced one round, then every candidate from
                every behaviour is scored in one guard block. That is a handful of
                transfers per run, not one per iteration.
  resident_cpu  stays on CPU, inference on CPU. No VRAM, ~seconds per call.
  remote        OpenAI-compatible endpoint (e.g. the existing exllamav3/tabbyAPI server).

`swap` also evicts the target to CPU first, so peak VRAM is max(target, guard) rather
than their sum.
"""
from __future__ import annotations

import contextlib
import gc
import itertools
import weakref
from typing import Any

import torch


def _empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Swappable:
    """Wraps an HF model whose device residency is managed explicitly."""

    # "offload" is the config-facing name for the default swap policy.
    _ALIASES = {"offload": "swap", "resident": "resident_gpu"}

    def __init__(self, model: Any, policy: str = "swap", gpu: str = "cuda"):
        self.model = model
        self.policy = self._ALIASES.get(policy, policy)
        if self.policy not in ("swap", "resident_gpu", "resident_cpu", "remote"):
            raise ValueError(
                f"unknown residency policy {policy!r}; expected one of "
                "swap/offload, resident_gpu/resident, resident_cpu, remote"
            )
        self.gpu = gpu
        if self.policy in ("swap", "resident_cpu"):
            self.model.to("cpu")
        _empty_cache()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to_gpu(self) -> None:
        if self.policy in ("resident_cpu", "remote"):
            return
        if self.device.type != "cuda":
            self.model.to(self.gpu)

    def to_cpu(self) -> None:
        if self.policy in ("resident_gpu", "remote"):
            return
        if self.device.type == "cuda":
            self.model.to("cpu")
            _empty_cache()

    @contextlib.contextmanager
    def active(self, evict: list["Swappable | Any"] | None = None):
        """Bring this model to GPU for a block of work, evicting others first.

        `evict` entries may be Swappable or anything with a `.model` attribute (e.g. the
        HFTarget), which is moved to CPU and restored on exit.
        """
        restored = []
        for other in evict or []:
            m = getattr(other, "model", other)
            try:
                if next(m.parameters()).device.type == "cuda":
                    m.to("cpu")
                    restored.append(m)
            except (StopIteration, AttributeError):
                continue
        _empty_cache()
        self.to_gpu()
        try:
            yield self.model
        finally:
            self.to_cpu()
            for m in restored:
                m.to(self.gpu)
            _empty_cache()

    def free(self) -> None:
        del self.model
        _empty_cache()


class Reloadable:
    """A model loaded from disk for each block of work and freed afterwards.

    For a guard that fits neither beside the target nor wholly on the card by itself: fp16
    Llama Guard 3 8B (~15 GiB) on a 16 GB card that also drives a display. The loader may
    place a few layers on CPU (device_map="auto" with max_memory), which `Swappable` cannot
    move. Peak VRAM is the guard alone, because the target is evicted to CPU for the block.
    Blocks are amortised by round structure (attacks/autodan.py), so a run loads the guard
    a handful of times; `n_loads` counts them so tests and logs can check that.

    The block yields a WEAK proxy, never the model itself. A caller's `with ... as model:`
    variable outlives the block, so yielding the model let v2's first version restore the
    target while the guard (~12 GiB) was still resident. On Windows the driver then spills
    the overflow into shared system memory instead of raising out-of-memory, and every later
    forward pass of the target ran 6-8x slower. The guard is now freed before the target
    returns, and the restore refuses to proceed if the target would not fit in free VRAM.
    """

    policy = "reload"
    # Free VRAM the restored target must leave, so activations do not spill either.
    headroom_bytes = 512 * 2 ** 20

    def __init__(self, loader, gpu: str = "cuda"):
        self.loader = loader
        self.gpu = gpu
        self.n_loads = 0

    @staticmethod
    def _nbytes(m) -> int:
        return sum(t.numel() * t.element_size()
                   for t in itertools.chain(m.parameters(), m.buffers()))

    @contextlib.contextmanager
    def active(self, evict: list[Any] | None = None):
        restored = []
        for other in evict or []:
            m = getattr(other, "model", other)
            try:
                if next(m.parameters()).device.type == "cuda":
                    m.to("cpu")
                    restored.append(m)
            except (StopIteration, AttributeError):
                continue
        _empty_cache()
        holder = [self.loader()]
        self.n_loads += 1
        try:
            yield weakref.proxy(holder[0])
        finally:
            holder.clear()                      # the only strong reference to the guard
            _empty_cache()
            if restored and torch.cuda.is_available():
                torch.cuda.synchronize()
                need = sum(self._nbytes(m) for m in restored) + self.headroom_bytes
                free, _ = torch.cuda.mem_get_info()
                if free < need:
                    raise RuntimeError(
                        f"cannot restore the target: {free / 2**30:.2f} GiB free VRAM, "
                        f"{need / 2**30:.2f} GiB needed. The guard may still be resident; "
                        "restoring now would spill into shared system memory (sysmem "
                        "fallback) and slow every later pass instead of failing.")
            for m in restored:
                m.to(self.gpu)
            _empty_cache()

    def free(self) -> None:
        _empty_cache()

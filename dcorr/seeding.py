"""Deterministic, process-independent seeds for the attack searches.

v1 seeded every defense's search with the same `random.Random(cfg.seed)` and reset it at
every behaviour chunk. Two defenses that neither change the objective nor reject a
candidate therefore replayed the *identical* search, and their breach vectors differed only
by grader noise (REVIEW F2). v2 derives one seed per (seed_index, defense, behaviour):

  * independent across defenses, so a defense's breach vector reflects its own search;
  * independent of chunking and of resume points, because no RNG state crosses behaviours;
  * reproducible across processes, because it uses SHA-256 rather than Python's `hash()`,
    which is salted per process unless PYTHONHASHSEED is set before the interpreter starts.

`common_random_numbers=True` drops the defense name from the seed, so every defense starts
from the same random stream. That is a legitimate variance-reduction design, but it must be
chosen and analysed deliberately rather than happen by accident.
"""
from __future__ import annotations

import hashlib

_SEP = "\x1f"   # unit separator: cannot occur in a behaviour id or defense name


def stable_seed(*parts: object) -> int:
    """63-bit seed from any sequence of parts, identical in every process and platform."""
    digest = hashlib.sha256(_SEP.join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def search_seed(base_seed: int, seed_index: int, defense_name: str, behaviour_id: str,
                *, common_random_numbers: bool = False) -> int:
    """Seed for one adaptive search: one defense, one behaviour, one replicate."""
    if common_random_numbers:
        return stable_seed("search", base_seed, seed_index, behaviour_id)
    return stable_seed("search", base_seed, seed_index, defense_name, behaviour_id)

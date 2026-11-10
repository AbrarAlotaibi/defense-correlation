"""JSONL append/read with resume keys, plus determinism helpers.

Every stage that costs GPU time writes one JSONL row per unit of work and skips units
whose key is already present. Aggregation is per *epoch* (one full sweep over the
behaviour list) rather than per task, so an interrupted run truncates a whole sweep
instead of silently undersampling the tail of the behaviour list.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line from a killed process: drop it, keep the rest.
                continue
    return rows


def append_jsonl(path: str | os.PathLike, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def _json_default(o: Any):
    """Coerce NumPy scalars to native types.

    The analysis path mixes NumPy and Python values (e.g. `x or np.bool_(False)` returns the
    NumPy value), and json.dump rejects np.bool_ / np.integer / np.floating with a confusing
    "Object of type bool is not JSON serializable". Handling it here keeps every writer safe
    rather than requiring a bool()/float() wrap at each call site.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        raise TypeError(f"not JSON serializable: {type(o).__name__}")
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, default=_json_default)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def result_key(row: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(row.get(f) for f in fields)


def done_keys(path: str | os.PathLike, fields: tuple[str, ...]) -> set[tuple]:
    """Keys already present in a results JSONL, for resume."""
    return {result_key(r, fields) for r in read_jsonl(path)}


def epochs(items: list, size: int) -> Iterator[tuple[int, list]]:
    """Yield (epoch_index, chunk). Used to aggregate per sweep, not per task."""
    if size <= 0:
        yield 0, items
        return
    for i in range(0, len(items), size):
        yield i // size, items[i : i + size]

"""Config loading: YAML with single-inheritance `extends`, deep-merged child over parent."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """dict with dotted access: cfg.get_path('target.model_id')."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # --- resolved directories -------------------------------------------------
    @property
    def data_dir(self) -> Path:
        p = REPO_ROOT / self.get_path("paths.data", "data")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_dir(self) -> Path:
        p = REPO_ROOT / self.get_path("paths.results", "results") / self.get_path("run_name", "run")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def enabled_defenses(self) -> list[str]:
        d = self.get_path("defenses", {}) or {}
        return [name for name, spec in d.items() if (spec or {}).get("enabled")]

    def defense_row(self, name: str) -> str:
        return self.get_path(f"defenses.{name}.row", "unassigned")


def load_config(path: str | os.PathLike, _seen: set[str] | None = None) -> Config:
    """Load a config, resolving one `extends:` chain relative to the config's own dir."""
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    _seen = _seen or set()
    if str(p) in _seen:
        raise ValueError(f"circular extends chain at {p}")
    _seen.add(str(p))

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    parent_name = raw.pop("extends", None)
    if parent_name:
        parent_path = (p.parent / parent_name).resolve()
        parent = load_config(parent_path, _seen)
        merged = _deep_merge(parent, raw)
    else:
        merged = raw

    cfg = Config(merged)
    cfg["_config_path"] = str(p)
    return cfg


def save_resolved(cfg: Config, dest: Path) -> None:
    """Write the fully-resolved config next to the results so a run is reconstructible."""
    plain = {k: v for k, v in cfg.items() if not k.startswith("_")}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(plain, f, sort_keys=False, allow_unicode=True)

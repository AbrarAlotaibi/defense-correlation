"""Shared runtime plumbing for the GPU stages: build target, load data, respond+judge glue."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .io_utils import read_jsonl
from .target import HFTarget


def build_target(cfg: Config) -> HFTarget:
    t = cfg.get_path("target", {})
    return HFTarget(
        model_id=t["model_id"],
        dtype=t.get("dtype", "float16"),
        device=t.get("device", "cuda"),
        probe_layer_frac=float(t.get("probe_layer_frac", 0.5)),
        system_prompt=t.get("system_prompt"),
        ppl_window_tokens=int(cfg.get_path("defenses.ppl_filter.window_tokens", 16)),
        batch_size=int(t.get("batch_size", 8)),
        gen_batch_size=int(t.get("gen_batch_size", 4)),
    )


def load_eval(cfg: Config):
    data = Path(cfg.data_dir)
    harmful = read_jsonl(data / "eval_harmful.jsonl")
    benign = read_jsonl(data / "eval_benign.jsonl")
    if not harmful:
        raise RuntimeError("data/eval_harmful.jsonl is empty - run scripts/00_prepare_data.py")
    # Optional runtime cap, used by replication targets that do not need the full n=100.
    # Applied here rather than by regenerating the split, so every run evaluates the SAME
    # first-k behaviours and the shared eval files are never rewritten.
    lim = cfg.get_path("data.eval_limit")
    if lim:
        harmful = harmful[: int(lim)]
        benign = benign[: int(lim)]
    return harmful, benign


def load_gcg_suffix(cfg: Config) -> str:
    p = Path(cfg.data_dir) / "gcg_suffix.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def behaviours_from(rows: list[dict]):
    from .attacks import Behaviour

    return [Behaviour(behaviour_id=r["behaviour_id"], prompt=r["prompt"],
                      target_str=r.get("target") or "Sure, here is",
                      category=r.get("category", "")) for r in rows]

"""Shared runtime plumbing for the GPU stages: build target, load data, respond+judge glue."""
from __future__ import annotations

import os
import platform
import socket
from pathlib import Path

from .config import Config
from .io_utils import read_jsonl
from .target import HFTarget


def host_provenance() -> dict:
    """Where this process is running, for stamping into timing and result records.

    Nothing in this project recorded the machine, so which GPU produced a given result was
    recoverable only from the sbatch files and a directory named after a faulty node. That
    matters more than it sounds: the paper reports that 6.3% of greedy responses differ
    between runs of the same prompt because of batch-dependent arithmetic, and arithmetic is
    a property of the device. A result set that cannot say which device it came from cannot
    be checked against that claim.

    The GPU name comes from torch rather than nvidia-smi, so under SLURM it reports the
    device this process was actually allocated rather than every card in the node. SLURM's
    own identifiers are included when present, because the node name is what the scheduler's
    logs are keyed by.

    Never raises: provenance is a nice-to-have on a record whose point is the measurement, so
    a missing field is always better than a failed stage.
    """
    out: dict = {"host": None, "gpu": None, "gpu_count": None, "gpu_memory_gib": None,
                 "torch": None, "cuda": None, "python": platform.python_version(),
                 "platform": platform.system()}
    try:
        out["host"] = socket.gethostname()
    except Exception:
        pass
    try:
        import torch

        out["torch"] = torch.__version__
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            out["gpu_memory_gib"] = round(props.total_memory / 1024**3, 1)
            out["cuda"] = torch.version.cuda
    except Exception:
        pass
    # SLURM, when this is a scheduled job. CUDA_VISIBLE_DEVICES is recorded because it is what
    # makes "device 0" mean a particular physical card.
    for key, env in (("slurm_job", "SLURM_JOB_ID"),
                     ("slurm_array_task", "SLURM_ARRAY_TASK_ID"),
                     ("slurm_node", "SLURMD_NODENAME"),
                     ("cuda_visible_devices", "CUDA_VISIBLE_DEVICES")):
        v = os.environ.get(env)
        if v:
            out[key] = v
    return out


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


# Benign evaluation sets for false-refusal rates. "jbb" is JailbreakBench's matched benign
# set (v1's only set, n=100); "orbench" is a held-out OR-Bench-Hard-1K sample (n=400, built
# by scripts/27_build_benign_orbench.py; REVIEW R5-5). data.eval_limit caps both.
BENIGN_SETS = {"jbb": None, "orbench": "eval_benign_orbench.jsonl"}


def load_benign(cfg: Config, name: str = "jbb") -> list[dict]:
    if name not in BENIGN_SETS:
        raise KeyError(f"unknown benign set {name!r}; known: {sorted(BENIGN_SETS)}")
    if BENIGN_SETS[name] is None:
        return load_eval(cfg)[1]
    rows = read_jsonl(Path(cfg.data_dir) / BENIGN_SETS[name])
    if not rows:
        raise RuntimeError(f"data/{BENIGN_SETS[name]} is empty - run scripts/27_build_benign_orbench.py")
    lim = cfg.get_path("data.eval_limit")
    return rows[: int(lim)] if lim else rows


def benign_suffix(name: str) -> str:
    """File suffix: stage04_<defense><suffix>.jsonl (v1 name kept for the JBB set)."""
    return "_benign" if name == "jbb" else f"_benign_{name}"


def load_gcg_suffix(cfg: Config) -> str:
    p = Path(cfg.data_dir) / "gcg_suffix.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def behaviours_from(rows: list[dict]):
    from .attacks import Behaviour

    return [Behaviour(behaviour_id=r["behaviour_id"], prompt=r["prompt"],
                      target_str=r.get("target") or "Sure, here is",
                      category=r.get("category", "")) for r in rows]

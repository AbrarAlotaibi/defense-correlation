"""Helpers shared by the end-to-end suites: configs, running stage scripts, reading results."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CALIBRATION = REPO / "results" / "primary_llama32_3b" / "calibration.json"
N_BEHAVIOURS = 4
SINGLE = ["ppl_filter", "token_anomaly", "probe", "refusal_prime", "smoothllm"]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config(results: Path, run_name: str, *, chunk: int = 2, method: str = "autodan",
           stack_order: list[str] | None = None, adaptive: dict | None = None,
           defenses: dict | None = None) -> str:
    """A run config on Llama-3.2-3B with a small budget. Llama Guard is off unless enabled
    in `defenses`; enabled defenses absent from `stack_order` are disabled (the stack
    refuses those)."""
    ad = {"method": method, "self_transfer": False, "target_prefix_tokens": 8}
    if method == "autodan":
        ad.update(use_template=False, generations=5, population=8, use_affirm_tail=True)
    else:
        ad.update(use_template=True, steps=4, batch_size=16, top_k=32, suffix_n_tokens=10,
                  eval_chunk=16, early_stop_loss=0.0)
    ad = _merge(ad, adaptive or {})
    defs = {"llamaguard": {"enabled": False}}
    defs = _merge(defs, defenses or {})
    if stack_order is not None:
        for d in SINGLE + ["llamaguard"]:
            if d not in stack_order:
                defs = _merge(defs, {d: {"enabled": False}})
    cfg = {
        "extends": str(REPO / "configs" / "primary_llama32_3b.yaml").replace("\\", "/"),
        "run_name": run_name,
        "paths": {"results": str(results).replace("\\", "/")},
        "data": {"eval_limit": N_BEHAVIOURS},
        "target": {"max_new_tokens": 24, "batch_size": 8, "gen_batch_size": 8},
        "attack": {"behaviour_chunk": chunk, "adaptive": ad,
                   "static": {"include_plain": True, "include_gcg": False}},
        "defenses": defs,
    }
    if stack_order is not None:
        cfg["stack"] = {"order": stack_order}
    return json.dumps(cfg)   # JSON is valid YAML


def run(code_root: Path, script: str, cfg_path: Path, args: list[str], log: Path) -> str:
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONIOENCODING="utf-8")
    env.pop("PYTHONHASHSEED", None)            # let every process salt hash() differently
    cmd = [sys.executable, str(code_root / "scripts" / script), "--config", str(cfg_path), *args]
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=code_root, env=env, stdout=f, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{script} {' '.join(args)} failed ({proc.returncode}):\n{text[-3000:]}")
    return text


def adaptive_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["behaviour_id"]: r for r in rows if r["attack"] == "adaptive"}


def extract_v1(dest: Path) -> Path:
    """The submitted code (tag v1-infus-submitted), unpacked into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "v1-infus-submitted"],
                             check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest)
    assert (dest / "scripts" / "_bootstrap.py").is_file(), "v1 extraction incomplete"
    return dest


def feasibility_mismatches(rows: dict[str, dict]) -> list[str]:
    """Rows where in-loop feasibility disagrees with what deployment's input stage did."""
    bad = []
    for i, x in rows.items():
        admitted = x["blocked_stage"] != "input"
        if bool(x["found_feasible"]) != admitted:
            bad.append(f"{i}: found_feasible={x['found_feasible']} blocked_stage={x['blocked_stage']}")
    return bad

"""Guard: the gold judge must never be reachable from the attack loop.

This is the single rule the previous line of work was burned on. The attack modules and
the target wrapper must not import the StrongREJECT judge, directly or transitively at
module load. Enforced by static import inspection so it cannot silently regress.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = "strongreject"
# Files that run inside, or are imported by, the attack loop.
IN_LOOP = [
    ROOT / "dcorr" / "attacks" / "random_search.py",
    ROOT / "dcorr" / "attacks" / "static.py",
    ROOT / "dcorr" / "target.py",
    ROOT / "dcorr" / "run_defense.py",
    ROOT / "dcorr" / "defenses" / "base.py",
    ROOT / "dcorr" / "defenses" / "ppl_filter.py",
    ROOT / "dcorr" / "defenses" / "token_anomaly.py",
    ROOT / "dcorr" / "defenses" / "probe.py",
    ROOT / "dcorr" / "defenses" / "llamaguard.py",
    ROOT / "dcorr" / "defenses" / "promptguard.py",
    ROOT / "dcorr" / "defenses" / "refusal_prime.py",
    ROOT / "dcorr" / "defenses" / "smoothllm.py",
    ROOT / "dcorr" / "defenses" / "stack.py",
]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    return mods


def test_no_strongreject_import_in_loop():
    offenders = []
    for f in IN_LOOP:
        if not f.exists():
            continue
        for m in _imports(f):
            if FORBIDDEN in (m or "").lower():
                offenders.append((f.name, m))
    assert not offenders, f"gold judge imported inside the attack loop: {offenders}"


def test_no_judge_module_import_in_loop():
    # Even the judge package (which re-exports StrongReject) must not be pulled in-loop.
    offenders = []
    for f in IN_LOOP:
        if not f.exists():
            continue
        for m in _imports(f):
            if m.endswith("judge") or ".judge" in m:
                offenders.append((f.name, m))
    assert not offenders, f"judge package imported inside the attack loop: {offenders}"


if __name__ == "__main__":
    test_no_strongreject_import_in_loop()
    test_no_judge_module_import_in_loop()
    print("ok: no judge in the attack loop")

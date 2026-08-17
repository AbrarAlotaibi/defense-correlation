"""Self-contained refusal pattern-bank matcher (no external dependencies).

Loads JSON banks from dcorr/judge/patterns/ and matches responses by model name. A bank
is a list of {type, match} rules with an optional per-model max_response_length cap. The
cap is essential for Llama-2: it opens with a refusal template ("As a responsible and
ethical AI...") then often complies for thousands of characters, so a real refusal is
short.

This replaces the earlier optional import of the gambit judge_prefilter - the banks now
travel with this repo, so the better-calibrated Llama-2/3 detection is available on any
machine (including Kaggle) with nothing optional about it.

Match types:
  exact_prefix : response.lstrip().startswith(match)
  exact        : response.strip() == match
  regex        : re.search(match, response)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PATTERNS_DIR = Path(__file__).parent / "patterns"


@dataclass(frozen=True)
class PrefilterResult:
    matched: bool
    pattern_id: Optional[str] = None
    pattern_type: Optional[str] = None


class _Rule:
    __slots__ = ("type", "raw", "regex", "id", "max_len")

    def __init__(self, type_: str, raw: str, idx: int, max_len: Optional[int]):
        self.type = type_
        self.raw = raw
        self.id = f"rule_{idx}"
        self.max_len = max_len
        self.regex = re.compile(raw) if type_ == "regex" else None

    def matches(self, response: str, default_max_len: Optional[int]) -> bool:
        cap = self.max_len if self.max_len is not None else default_max_len
        if cap is not None and len(response) > cap:
            return False
        if self.type == "exact_prefix":
            return response.lstrip().startswith(self.raw)
        if self.type == "exact":
            return response.strip() == self.raw
        if self.type == "regex":
            return self.regex.search(response) is not None
        return False


class RefusalPrefilter:
    def __init__(self, patterns_dir: Path = PATTERNS_DIR):
        self._by_model: dict[str, list[_Rule]] = {}
        self._max_len_by_model: dict[str, Optional[int]] = {}
        if not patterns_dir.exists():
            return
        for p in sorted(patterns_dir.glob("*.json")):
            try:
                self._load(p)
            except Exception:
                continue

    def _load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models") or []
        raw = data.get("patterns") or []
        bank_max = data.get("max_response_length")
        if not models or not raw:
            return
        rules: list[_Rule] = []
        for i, p in enumerate(raw):
            m = p.get("match", "")
            if not m:
                continue
            try:
                rules.append(_Rule(p.get("type", "exact_prefix"), m, i,
                                   p.get("max_response_length")))
            except re.error:
                continue
        for model in models:
            self._by_model[model] = rules
            self._max_len_by_model[model] = bank_max

    def classify(self, model_name: str, response: str) -> PrefilterResult:
        if not response or not isinstance(response, str):
            return PrefilterResult(False)
        rules = self._by_model.get(model_name)
        if not rules:
            return PrefilterResult(False)
        cap = self._max_len_by_model.get(model_name)
        for rule in rules:
            try:
                if rule.matches(response, cap):
                    return PrefilterResult(True, rule.id, rule.type)
            except Exception:
                continue
        return PrefilterResult(False)


_INSTANCE: Optional[RefusalPrefilter] = None


def get_prefilter() -> RefusalPrefilter:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RefusalPrefilter()
    return _INSTANCE

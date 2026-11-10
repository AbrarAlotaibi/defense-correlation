"""Shared CPU fakes for tests that drive the real attack and defense code without a model."""
from __future__ import annotations

import string

import torch

from dcorr.seeding import stable_seed

FLUENT = set(string.ascii_letters + string.digits + " .,'\":-")


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class Scored:
    def __init__(self, lp, nll):
        self.target_logprob = torch.tensor(lp)
        self.hidden = None
        self.window_nll = None if nll is None else torch.tensor(nll)


class FakeTarget:
    """Fitness: a fixed pseudo-random function of the text. window_nll: 10 x the fraction of
    non-fluent characters (so 10%-swap SmoothLLM copies score high, fluent prompts ~0).
    generate(): deterministic text; a prompt whose hash is in `refuse_mod` refuses."""
    model_id = "fake/target"
    tokenizer = CharTokenizer()

    def __init__(self, marker_mod: int = 0):
        self.calls: list[dict] = []
        self.marker_mod = marker_mod      # >0: every k-th (by hash) response carries MARKER

    MARKER = "UNSAFE-CONTENT"

    @staticmethod
    def nll(text: str) -> float:
        return 10.0 * sum(c not in FLUENT for c in text) / max(1, len(text))

    def score_chunked(self, users, target_str, need_hidden=False, need_window_nll=False,
                      system="__default__", **_kw):
        self.calls.append({"users": list(users), "system": system,
                           "need_window_nll": need_window_nll})
        lp = [-(stable_seed("fit", u) % 10_000) / 2_000.0 for u in users]
        return Scored(lp, [self.nll(u) for u in users] if need_window_nll else None)

    def generate(self, prompts, max_new_tokens=256, system="__default__"):
        out = []
        for p in prompts:
            marked = self.marker_mod and stable_seed("resp", p) % self.marker_mod == 0
            out.append(("Sure, here is " + (self.MARKER if marked else "an answer")) + ".")
        return out

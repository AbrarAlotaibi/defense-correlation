"""HF target-model wrapper.

One batched forward pass yields everything the attack loop and the cheap defenses need:

  * `target_logprob`  - the in-loop attack signal (mean logprob of the affirmative prefix)
  * `hidden`          - mid-layer hidden state at the last prompt token (probe feature)
  * `window_nll`      - max windowed NLL over the user-content token span (perplexity filter)

Causal attention means the hidden state at the last prompt token does not depend on the
forced target tokens that follow it, so the probe feature computed here is identical to
one computed on the prompt alone. That is what makes the grid affordable.

NO judge model is ever consulted here. The only in-loop signal is target logprob.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat import render
from .compat import dtype_kwargs


@dataclass
class ScoreOut:
    target_logprob: torch.Tensor            # (B,) mean logprob per target token
    target_logprob_sum: torch.Tensor        # (B,) sum logprob
    hidden: torch.Tensor | None             # (B, H) mid-layer state at last prompt token
    window_nll: torch.Tensor | None         # (B,) max windowed NLL over user span
    n_user_tokens: torch.Tensor | None      # (B,)


class HFTarget:
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "cuda",
                 probe_layer_frac: float = 0.5, system_prompt: str | None = None,
                 ppl_window_tokens: int = 16, batch_size: int = 8, gen_batch_size: int = 4):
        self.model_id = model_id
        self.device = device
        self.system_prompt = system_prompt
        self.ppl_window_tokens = ppl_window_tokens
        self.batch_size = batch_size
        self.gen_batch_size = gen_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"device_map": {"": device} if device != "cpu" else "cpu"}
        if str(dtype).lower() in ("nf4", "int4", "4bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.dtype_name = "nf4"
        else:
            kwargs.update(dtype_kwargs(dtype))
            self.dtype_name = str(dtype)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.model.requires_grad_(False)

        self.n_layers = int(getattr(self.model.config, "num_hidden_layers"))
        self.probe_layer = max(1, min(self.n_layers, int(self.n_layers * probe_layer_frac)))
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))

        self._allowed_token_ids: list[int] | None = None

    # ------------------------------------------------------------------ vocab
    def allowed_suffix_tokens(self) -> list[int]:
        """Token ids usable in an adversarial suffix: printable ASCII, no specials, no
        whitespace-only, no partial-byte tokens. Cached."""
        if self._allowed_token_ids is not None:
            return self._allowed_token_ids
        tok = self.tokenizer
        special = set(tok.all_special_ids or [])
        allowed = []
        vocab_size = len(tok)
        for tid in range(vocab_size):
            if tid in special:
                continue
            s = tok.decode([tid], skip_special_tokens=True)
            if not s or not s.strip():
                continue
            if not all(32 <= ord(c) < 127 for c in s):
                continue
            # Reject tokens that do not round-trip: they corrupt the suffix on re-encode.
            if tok.encode(s, add_special_tokens=False) != [tid]:
                continue
            allowed.append(tid)
        if not allowed:
            raise RuntimeError(f"no usable suffix tokens found for {self.model_id}")
        self._allowed_token_ids = allowed
        return allowed

    def decode_suffix(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score(self, users: list[str], target_str: str, system: str | None = "__default__",
              assistant_prefix: str = "", need_hidden: bool = False,
              need_window_nll: bool = False) -> ScoreOut:
        """Teacher-forced score of `target_str` after each prompt in `users`.

        `system=None` means explicitly no system prompt; the sentinel "__default__" means
        use the target's configured default.
        """
        if system == "__default__":
            system = self.system_prompt

        tok = self.tokenizer
        tgt_ids = tok.encode(target_str, add_special_tokens=False)
        if not tgt_ids:
            raise ValueError("empty target string")
        n_tgt = len(tgt_ids)

        seqs, user_tok_spans, prompt_lens = [], [], []
        for u in users:
            rp = render(tok, u, system=system, assistant_prefix=assistant_prefix)
            enc = tok(rp.text, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            # Token indices whose char span lies inside the user turn.
            lo, hi = None, None
            for i, (a, b) in enumerate(offsets):
                if b <= a:
                    continue
                if a >= rp.user_start and b <= rp.user_end:
                    lo = i if lo is None else lo
                    hi = i
            user_tok_spans.append((lo, hi))
            prompt_lens.append(len(ids))
            seqs.append(ids + tgt_ids)

        maxlen = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        input_ids = input_ids.to(self.model.device)
        attn = attn.to(self.model.device)

        # Capture ONLY the probe layer via a forward hook. output_hidden_states=True retains
        # every layer's activations (33 tensors for a 7B), which is pure waste when one layer
        # is wanted and was enough to OOM the constrained-probe runs on long prompts. The hook
        # is equivalent: HF's hidden_states[L] is the output of decoder layer L-1.
        _captured: dict = {}
        _handle = None
        _hook_ok = False
        if need_hidden:
            try:
                _layer = self.model.model.layers[self.probe_layer - 1]

                def _grab(_m, _i, o):
                    _captured["h"] = o[0] if isinstance(o, tuple) else o

                _handle = _layer.register_forward_hook(_grab)
                _hook_ok = True
            except (AttributeError, IndexError):
                _hook_ok = False   # unknown architecture: fall back to output_hidden_states

        out = self.model(input_ids=input_ids, attention_mask=attn,
                         output_hidden_states=(need_hidden and not _hook_ok), use_cache=False)
        if _handle is not None:
            _handle.remove()
        logits = out.logits
        device = logits.device
        B, T = logits.shape[0], logits.shape[1]

        # Per-token NLL of every input token, predicted from the previous position, computed
        # for the WHOLE batch in ONE pass:  nll_all[:, t] = -log P(token_t | context) =
        # logsumexp(logits_{t-1}) - logits_{t-1}[token_t]. This replaces the per-sequence
        # full-vocab log_softmax loops that left the GPU idle on Python and dominated
        # constrained-defense GCG. Everything below is then cheap indexing on nll_all.
        # Chunk over time: logits.float() on the whole [B, T, V] tensor is a float32 copy of
        # the entire logit block (4x the fp16 original) and OOMs on long prompts / large
        # batches. Chunking bounds the temporary to [B, CHUNK, V] while giving identical
        # results.
        lse = torch.empty(B, T, device=device, dtype=torch.float32)
        _CH = max(1, int(2_000_000 // max(1, logits.shape[-1])))   # ~2M float32 per chunk
        for s in range(0, T, _CH):
            lse[:, s : s + _CH] = torch.logsumexp(logits[:, s : s + _CH, :].float(), dim=-1)
        next_logit = logits[:, :-1, :].gather(
            -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).float()            # [B, T-1]
        nll_all = torch.zeros(B, T, device=device)
        nll_all[:, 1:] = lse[:, :-1] - next_logit                             # NLL at pos t

        # target logprob: the n_tgt target tokens sit at positions [p, p+n_tgt).
        lp_sum = torch.zeros(B, device=device)
        for i in range(B):
            p = prompt_lens[i]
            lp_sum[i] = -nll_all[i, p : p + n_tgt].sum()

        hidden = None
        if need_hidden:
            hs = _captured["h"] if _hook_ok else out.hidden_states[self.probe_layer]  # (B,T,H)
            idx = torch.tensor([p - 1 for p in prompt_lens], device=hs.device)
            hidden = hs[torch.arange(B, device=hs.device), idx, :].float()

        window_nll, n_user = None, None
        if need_window_nll:
            window_nll = torch.zeros(B, device=device)
            n_user = torch.zeros(B, dtype=torch.long)
            w = self.ppl_window_tokens
            for i in range(B):
                lo, hi = user_tok_spans[i]
                if lo is None or hi is None or hi <= lo:
                    continue
                nll = nll_all[i, lo : hi + 1]      # already computed; no vocab op here
                n_user[i] = nll.numel()
                if nll.numel() <= w:
                    window_nll[i] = nll.mean()
                else:
                    # max over sliding windows: the filter flags a prompt if ANY window is
                    # anomalous, which is what makes it defeatable by short suffixes.
                    csum = torch.cat([torch.zeros(1, device=device), nll.cumsum(0)])
                    means = (csum[w:] - csum[:-w]) / w
                    window_nll[i] = means.max()

        del out, logits, lse, next_logit
        return ScoreOut(
            target_logprob=(lp_sum / n_tgt).detach().cpu(),
            target_logprob_sum=lp_sum.detach().cpu(),
            hidden=hidden.detach().cpu() if hidden is not None else None,
            window_nll=window_nll.detach().cpu() if window_nll is not None else None,
            n_user_tokens=n_user,
        )

    def score_chunked(self, users: list[str], target_str: str, **kw) -> ScoreOut:
        """score() over arbitrarily many prompts, batched at self.batch_size."""
        outs = []
        for i in range(0, len(users), self.batch_size):
            outs.append(self.score(users[i : i + self.batch_size], target_str, **kw))
        cat = lambda attr: (  # noqa: E731
            torch.cat([getattr(o, attr) for o in outs], dim=0)
            if getattr(outs[0], attr) is not None else None
        )
        return ScoreOut(
            target_logprob=cat("target_logprob"),
            target_logprob_sum=cat("target_logprob_sum"),
            hidden=cat("hidden"),
            window_nll=cat("window_nll"),
            n_user_tokens=cat("n_user_tokens"),
        )

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, users: list[str], max_new_tokens: int = 256,
                 system: str | None = "__default__", assistant_prefix: str = "") -> list[str]:
        """Greedy decode. Deterministic: no sampling anywhere in this project."""
        if system == "__default__":
            system = self.system_prompt
        tok = self.tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"          # required for correct batched generation
        try:
            outputs: list[str] = []
            for i in range(0, len(users), self.gen_batch_size):
                chunk = users[i : i + self.gen_batch_size]
                texts = [render(tok, u, system=system, assistant_prefix=assistant_prefix).text
                         for u in chunk]
                enc = tok(texts, add_special_tokens=False, return_tensors="pt",
                          padding=True).to(self.model.device)
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tok.pad_token_id,
                )
                new = gen[:, enc["input_ids"].shape[1]:]
                outputs.extend(tok.batch_decode(new, skip_special_tokens=True))
            return outputs
        finally:
            tok.padding_side = prev_side

    def free(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

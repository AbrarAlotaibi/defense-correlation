"""GCG - Greedy Coordinate Gradient adversarial suffix attack (Zou et al., 2023).

Random search cannot break a refusal-robust target (Llama-2-7b-chat) within budget. GCG
uses the gradient of the target cross-entropy w.r.t. the one-hot suffix to propose the
most loss-reducing token swaps each step, then evaluates a batch of them and keeps the
best. This is the canonical strong suffix attack.

Design for a 16 GB card where the fp16 target already nearly fills VRAM:
  * gradient step is batch-1 (one sequence) with gradient checkpointing -> bounded memory;
  * candidate evaluation is no-grad, chunked;
  * the objective is the *truncated* affirmative target under the compliance template;
  * per behaviour, with self-transfer seeding so later behaviours converge in few steps.

Same defense-aware feasibility as random search: cheap constraints filter candidates each
step; expensive constraints (Llama Guard) are checked on the final suffix. The in-loop
signal is ONLY the target loss - no judge is ever consulted here.
"""
from __future__ import annotations

import gc

import torch

from ..defenses.base import Defense
from ..target import HFTarget
from .random_search import Behaviour, _compose, _truncate_target

_SENTINEL = "\x00SFX\x00"


def _split_around_suffix(target: HFTarget, behaviour_prompt: str, target_str: str,
                         template: str | None) -> tuple[list[int], list[int]]:
    """Token ids of the rendered prompt before / after the suffix insertion point."""
    from ..chat import render

    composed = _compose(target, behaviour_prompt, [], template, target_str)  # suffix -> ""
    # Rebuild with a sentinel in place of the (empty) suffix so we can locate it robustly.
    if template:
        marked = template.format(behaviour=behaviour_prompt, target=target_str, suffix=_SENTINEL)
    else:
        marked = f"{behaviour_prompt} {_SENTINEL}"
    text = render(target.tokenizer, marked, system=target.system_prompt).text
    if _SENTINEL not in text:
        # Fallback: no clean split; put everything in prefix, empty post.
        ids = target.tokenizer.encode(text.replace(_SENTINEL, ""), add_special_tokens=False)
        return ids, []
    left, right = text.split(_SENTINEL, 1)
    pre = target.tokenizer.encode(left, add_special_tokens=False)
    post = target.tokenizer.encode(right, add_special_tokens=False)
    return pre, post


class _GCGState:
    __slots__ = ("b", "pre", "post", "tgt", "suffix", "best_suffix", "best_loss",
                 "feas_suffix", "history", "n_evals")

    def __init__(self, b, pre, post, tgt, suffix):
        self.b = b
        self.pre = pre
        self.post = post
        self.tgt = tgt
        self.suffix = list(suffix)
        self.best_suffix = list(suffix)
        self.best_loss = float("inf")
        self.feas_suffix = None
        self.history = []
        self.n_evals = 0


def _init_suffix(target: HFTarget, n: int, allowed: list[int], seed_from) -> list[int]:
    if seed_from:
        return list(seed_from)[:n] if len(seed_from) >= n else list(seed_from)
    bang = target.tokenizer.encode("!", add_special_tokens=False)
    if len(bang) == 1 and bang[0] in allowed:
        return [bang[0]] * n
    return [allowed[i % len(allowed)] for i in range(n)]


@torch.no_grad()
def _candidate_losses(model, embed_w, seqs: torch.Tensor, tgt_len: int, chunk: int) -> torch.Tensor:
    """Mean CE loss on the last tgt_len tokens for each row of `seqs` [B, T]."""
    device = embed_w.device
    losses = torch.empty(seqs.shape[0], device=device)
    for i in range(0, seqs.shape[0], chunk):
        batch = seqs[i:i + chunk].to(device)
        # Slice to the target positions BEFORE upcasting: full float32 logits over a large
        # vocab (Llama-3.2 = 128k) would be many GB and OOM the card.
        logits = model(input_ids=batch).logits
        sl = logits[:, -tgt_len - 1:-1, :].float()          # [b, tgt_len, V]
        lp = torch.log_softmax(sl, dim=-1)
        tgt = batch[:, -tgt_len:]
        ce = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean(dim=1)
        losses[i:i + chunk] = ce
        del logits, sl, lp
    return losses


def run_gcg(
    target: HFTarget,
    defense: Defense,
    behaviours: list[Behaviour],
    *,
    steps: int = 150,
    suffix_n_tokens: int = 20,
    top_k: int = 96,
    batch_size: int = 32,
    target_prefix_tokens: int = 12,
    template: str | None = None,
    eval_chunk: int = 32,
    early_stop_loss: float = 0.5,
    init_suffixes: dict[str, list[int]] | None = None,
    seed: int = 0,
    progress=None,
) -> list[dict]:
    """Per-behaviour GCG. Returns one record per behaviour (same schema as run_adaptive)."""
    import random as _random

    rng = _random.Random(seed)
    model = target.model
    tok = target.tokenizer
    device = model.device
    embed_w = model.get_input_embeddings().weight  # [V, d]
    vocab = embed_w.shape[0]

    allowed = target.allowed_suffix_tokens()
    allowed_mask = torch.zeros(vocab, dtype=torch.bool, device=device)
    allowed_mask[torch.tensor([a for a in allowed if a < vocab], device=device)] = True

    ckpt_was = getattr(model, "is_gradient_checkpointing", False)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    if hasattr(model, "config"):
        model.config.use_cache = False

    records = []
    carry = None  # self-transfer within this run, behaviour-to-behaviour
    for bi, b in enumerate(behaviours):
        seed_from = (init_suffixes or {}).get(b.behaviour_id) or carry
        pre, post = _split_around_suffix(target, b.prompt, b.target_str, template)
        tgt_ids = tok.encode(_truncate_target(target, b.target_str, target_prefix_tokens),
                             add_special_tokens=False)
        if not tgt_ids:
            tgt_ids = tok.encode("Sure", add_special_tokens=False)
        st = _GCGState(b, pre, post, tgt_ids,
                       _init_suffix(target, suffix_n_tokens, allowed, seed_from))

        pre_t = torch.tensor(pre, device=device)
        post_t = torch.tensor(post, device=device)
        tgt_t = torch.tensor(tgt_ids, device=device)
        tgt_len = len(tgt_ids)

        for step in range(steps):
            suffix_t = torch.tensor(st.suffix, device=device)
            L = suffix_t.shape[0]

            # ---- gradient of target CE w.r.t. one-hot suffix ----
            one_hot = torch.zeros(L, vocab, device=device, dtype=embed_w.dtype)
            one_hot.scatter_(1, suffix_t.unsqueeze(1), 1.0)
            one_hot.requires_grad_(True)
            suf_embed = one_hot @ embed_w                       # [L, d]
            cat = torch.cat([
                embed_w[pre_t], suf_embed, embed_w[post_t], embed_w[tgt_t]
            ], dim=0).unsqueeze(0)                              # [1, T, d]
            logits = model(inputs_embeds=cat).logits
            lp = torch.log_softmax(logits[0, -tgt_len - 1:-1, :].float(), dim=-1)
            loss = -lp.gather(-1, tgt_t.unsqueeze(-1)).squeeze(-1).mean()
            loss.backward()
            grad = one_hot.grad.detach()                        # [L, V]
            grad[:, ~allowed_mask] = float("inf")               # only allowed tokens
            del one_hot, suf_embed, cat, logits, lp, loss
            top = (-grad).topk(min(top_k, int(allowed_mask.sum())), dim=1).indices  # [L, top_k]

            # ---- sample a batch of single-position swaps ----
            cands = []
            for _ in range(batch_size):
                pos = rng.randrange(L)
                new_tok = int(top[pos, rng.randrange(top.shape[1])])
                c = list(st.suffix)
                c[pos] = new_tok
                cands.append(c)
            # include current suffix so best never regresses
            cands.append(list(st.suffix))

            # ---- feasibility filter (cheap constraints) ----
            prompts = [_compose(target, b.prompt, c, template, b.target_str) for c in cands]
            if not defense.expensive_constraint:
                sc = target.score_chunked(
                    prompts, tok.decode(tgt_ids),
                    need_hidden=defense.needs_hidden(),
                    need_window_nll=defense.needs_window_nll(), **defense.score_kwargs())
                feas = defense.cheap_feasible(sc, prompts)
            else:
                feas = [True] * len(cands)

            # ---- evaluate true target loss for feasible candidates ----
            keep = [i for i, f in enumerate(feas) if f] or list(range(len(cands)))
            seqs = torch.stack([
                torch.cat([pre_t, torch.tensor(cands[i], device=device), post_t, tgt_t])
                for i in keep
            ])
            losses = _candidate_losses(model, embed_w, seqs, tgt_len, eval_chunk)
            st.n_evals += len(keep)
            bidx = int(losses.argmin())
            best_c = cands[keep[bidx]]
            best_l = float(losses[bidx])
            st.suffix = best_c
            if best_l < st.best_loss:
                st.best_loss = best_l
                st.best_suffix = list(best_c)
                if feas[keep[bidx]]:
                    st.feas_suffix = list(best_c)
            st.history.append(round(best_l, 4))

            del grad, top, seqs, losses
            if step % 20 == 0:
                gc.collect(); torch.cuda.empty_cache()

            if progress is not None:
                progress(bi, len(behaviours), step + 1, steps, st.best_loss)

            # Early stop: once the affirmative-target loss is low the greedy generation
            # complies, so more steps are wasted. Self-transfer makes most behaviours hit
            # this within a handful of steps.
            if st.best_loss <= early_stop_loss and (st.feas_suffix is not None
                                                    or not defense.has_input_filter):
                break

        sfx = st.feas_suffix if st.feas_suffix is not None else st.best_suffix
        carry = list(st.best_suffix)
        records.append({
            "behaviour_id": b.behaviour_id,
            "attack": "adaptive",
            "suffix_token_ids": sfx,
            "suffix": target.decode_suffix(sfx),
            "prompt": _compose(target, b.prompt, sfx, template, b.target_str),
            "in_loop_logprob": -float(st.best_loss),   # -CE = mean target logprob
            "found_feasible": st.feas_suffix is not None,
            "iterations": steps,
            "n_evals": st.n_evals,
            "loss_trace": st.history,
        })

    if not ckpt_was:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    gc.collect(); torch.cuda.empty_cache()
    return records

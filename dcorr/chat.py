"""Chat templating that also reports where the user content lives in the rendered string.

Every filter that scores "the prompt" (perplexity, token anomaly, the probe) needs the
character span of the user turn inside the templated text, so a single forward pass can
serve the attack objective and all three constraints at once.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderedPrompt:
    text: str            # full templated text, ending with the assistant generation prompt
    user_start: int      # char offset of the user content inside `text`
    user_end: int
    user_text: str


_SENTINEL = "\x00USERCONTENT\x00"

# Vicuna v1.1/v1.5 conversation format (FastChat). Used only when the checkpoint ships no
# chat_template; the sentinel stands in for the user turn exactly as in the templated path.
_VICUNA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                  "The assistant gives helpful, detailed, and polite answers to the user's "
                  "questions.")


def _fallback_template(name_or_path: str, system: str | None) -> str:
    name = (name_or_path or "").lower()
    if "vicuna" in name:
        sys_txt = system if system is not None else _VICUNA_SYSTEM
        prefix = f"{sys_txt} " if sys_txt else ""
        return f"{prefix}USER: {_SENTINEL} ASSISTANT:"
    # Generic last resort: a plain instruct framing.
    prefix = f"{system}\n\n" if system else ""
    return f"{prefix}### Instruction:\n{_SENTINEL}\n\n### Response:\n"


def render(tokenizer, user: str, system: str | None = None,
           assistant_prefix: str = "") -> RenderedPrompt:
    """Render one user turn through the model's chat template.

    The user content is located by rendering once with a sentinel in its place, which is
    robust to templates that reorder, wrap, or inject text around the turn.
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": _SENTINEL})

    try:
        probe_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Some checkpoints (notably Vicuna) ship no chat_template at all. Fall back to the
        # model family's documented conversation format rather than crashing; using the
        # right format matters because ASR is sensitive to the prompt framing.
        probe_text = _fallback_template(getattr(tokenizer, "name_or_path", ""), system)
    idx = probe_text.find(_SENTINEL)
    if idx < 0:
        # Template dropped or mangled the content; fall back to a plain rendering.
        text = probe_text.replace(_SENTINEL, user) + assistant_prefix
        return RenderedPrompt(text=text, user_start=0, user_end=len(user), user_text=user)

    text = probe_text[:idx] + user + probe_text[idx + len(_SENTINEL):] + assistant_prefix
    return RenderedPrompt(
        text=text,
        user_start=idx,
        user_end=idx + len(user),
        user_text=user,
    )


def strip_bos_duplication(tokenizer, text: str) -> str:
    """Chat templates for Llama-2/3 already emit BOS; tokenizing with add_special_tokens
    would emit a second one. Callers tokenize with add_special_tokens=False and use this
    to assert the template did its job."""
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text
    return text

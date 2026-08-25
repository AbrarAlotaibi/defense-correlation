"""Cross-version transformers shims.

The code targets transformers 5.x (local: 5.14), where from_pretrained takes `dtype=`.
Kaggle images often ship transformers 4.4x, where the kwarg is `torch_dtype=`. This picks
the right one at runtime so the same code loads models on either.
"""
from __future__ import annotations

import inspect
from functools import lru_cache

import torch

_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(name):
    if isinstance(name, torch.dtype):
        return name
    return _DTYPES.get(str(name).lower(), torch.float16)


@lru_cache(maxsize=1)
def _dtype_kwarg_name() -> str:
    from transformers import AutoModelForCausalLM

    try:
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in sig.parameters:
            return "dtype"
    except (ValueError, TypeError):
        pass
    # from_pretrained often uses **kwargs, so signature inspection can miss it. Fall back
    # on the transformers version: `dtype` was introduced in 5.x.
    try:
        import transformers

        major = int(str(transformers.__version__).split(".")[0])
        return "dtype" if major >= 5 else "torch_dtype"
    except Exception:
        return "torch_dtype"


def dtype_kwargs(name) -> dict:
    """{'dtype': x} on transformers>=5, {'torch_dtype': x} on 4.x."""
    return {_dtype_kwarg_name(): resolve_dtype(name)}

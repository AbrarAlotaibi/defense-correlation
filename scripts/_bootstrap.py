"""Shared script preamble: put the repo root on sys.path and load credentials.

Portable across Windows (local) and Linux (Kaggle). Credentials are resolved in this
order, first hit wins, and values are NEVER printed:

  1. environment variables already set
  2. a .env file: $DCORR_ENV, else <repo>/.env

No dependency on any external project. HF token aliases are normalised so transformers
and huggingface_hub both find it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Credential names this project consumes.
_SECRET_KEYS = ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN", "OPENAI_API_KEY")


def _from_env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    candidates = [
        Path(os.environ.get("DCORR_ENV", "")),
        REPO_ROOT / ".env",
        Path(r"C:\Users\Abrar\Desktop\PhD\.env"),   # local convenience; absent on Kaggle
    ]
    for cand in candidates:
        if cand and str(cand) not in ("", ".") and cand.is_file():
            for line in cand.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break
    return out


def load_env() -> None:
    """Populate os.environ with credentials from the first available source. Never prints."""
    resolved: dict[str, str] = {}
    resolved.update(_from_env_file())
    for k, v in resolved.items():
        if k and v and k not in os.environ:
            os.environ[k] = v

    tok = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if tok:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
        os.environ.setdefault("HF_TOKEN", tok)


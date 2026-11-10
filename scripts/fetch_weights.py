"""Fetch gated HF weights needed by the run into the local HF cache.

Uses the authorized HUGGING_FACE_HUB_TOKEN from Desktop/PhD/.env (never printed). Only
pulls the transformers-native files (safetensors + configs + tokenizer), skipping any
original/ *.pth consolidated checkpoints to save disk.

  gambit-abl python scripts/fetch_weights.py --set primary
  gambit-abl python scripts/fetch_weights.py --set replication
  gambit-abl python scripts/fetch_weights.py --models meta-llama/Llama-Guard-3-8B
"""
from __future__ import annotations

import argparse

from _bootstrap import load_env  # noqa: E402

SETS = {
    "primary": [
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Llama-Prompt-Guard-2-86M",
    ],
    "replication": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
}

# NOTE: *.bin is ALLOWED. Some checkpoints (e.g. lmsys/vicuna-7b-v1.5) ship only PyTorch
# .bin shards and no safetensors; ignoring *.bin silently fetches configs with no weights,
# which then fails at load time with a confusing "couldn't connect to huggingface.co" error.
ALLOW = ["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt", "*.md", "*.bin"]
IGNORE = ["original/*", "*.pth", "consolidated*", "*.gguf"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default=None)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    load_env()
    from huggingface_hub import snapshot_download

    models = list(args.models or [])
    if args.set:
        models += SETS[args.set]
    if not models:
        models = SETS["primary"]

    for repo in models:
        print(f"[fetch] {repo} ...", flush=True)
        path = snapshot_download(
            repo_id=repo,
            allow_patterns=ALLOW,
            ignore_patterns=IGNORE,
            local_files_only=False,
        )
        print(f"[fetch] {repo} -> {path}", flush=True)
    print("[fetch] done.")


if __name__ == "__main__":
    main()

"""Stage 02: train the linear probe on mid-layer hidden states.

Feature = layer-L hidden state at the last prompt token of the templated prompt (see
dcorr/target.py). Training data is data/probe_train.jsonl, which stage 00 already
deduplicated against every JBB evaluation behaviour; this script re-asserts that the
dedup report exists and that no training text collides with an eval behaviour, and
refuses to train otherwise. Contamination here silently breaks H2.

Trains a logistic-regression probe (standardised features) with an internal train/val
split, reports val AUROC/accuracy, and saves weights + the standardiser + the layer index
+ the model id, so the probe defense can refuse to load against the wrong model or layer.

Writes: data/probe_<tag>.pt , results/<run>/probe_train_report.json
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from _bootstrap import REPO_ROOT, load_env  # noqa: E402

from dcorr.config import load_config
from dcorr.io_utils import read_json, read_jsonl, set_all_seeds, write_json
from dcorr.runtime import build_target, load_eval

_TOKEN = re.compile(r"[a-z0-9]+")


def _toks(s: str) -> set[str]:
    return set(_TOKEN.findall((s or "").lower()))


def _extract(target, texts: list[str], batch: int) -> np.ndarray:
    feats = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        out = target.score(chunk, target_str="Sure", need_hidden=True)
        feats.append(out.hidden.numpy())
    return np.concatenate(feats, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_llama2.yaml")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--layer", type=int, default=None,
                    help="train at this layer instead of the config mid-layer "
                         "(used for the second internal-representations instance)")
    ap.add_argument("--tag", default=None, help="override the weights filename tag")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    set_all_seeds(seed)

    data = REPO_ROOT / cfg.get_path("paths.data", "data")
    report = read_json(data / "probe_dedup_report.json")
    if report is None:
        raise RuntimeError("data/probe_dedup_report.json missing - run scripts/00_prepare_data.py")

    rows = read_jsonl(data / "probe_train.jsonl")
    if len(rows) < 40:
        raise RuntimeError(f"probe_train.jsonl has only {len(rows)} rows - too few to train")

    # Re-assert disjointness against eval behaviours.
    harmful, benign = load_eval(cfg)
    eval_sets = [_toks(h["prompt"]) for h in harmful] + [_toks(b["prompt"]) for b in benign]
    thr = float(cfg.get_path("data.dedup_jaccard", 0.6))
    for r in rows:
        ts = _toks(r["text"])
        jmax = max((len(ts & e) / len(ts | e) if ts and e else 0.0 for e in eval_sets),
                   default=0.0)
        if jmax > thr:
            raise RuntimeError(
                f"probe training text collides with an eval behaviour (Jaccard {jmax:.2f} > "
                f"{thr}): {r['text'][:80]!r}. Rerun stage 00 - H2 would be contaminated."
            )

    target = build_target(cfg)
    layer_cfg = args.layer if args.layer is not None else cfg.get_path("defenses.probe.layer")
    if layer_cfg is not None:
        target.probe_layer = int(layer_cfg)
    layer = target.probe_layer
    print(f"[02] extracting features at layer {layer} for {len(rows)} prompts ...")

    texts = [r["text"] for r in rows]
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.float32)
    X = _extract(target, texts, batch=cfg.get_path("target.batch_size", 8)).astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    mu = X[tr_idx].mean(0)
    sd = X[tr_idx].std(0) + 1e-6
    Xn = (X - mu) / sd

    Xt = torch.tensor(Xn[tr_idx]); yt = torch.tensor(labels[tr_idx])
    Xv = torch.tensor(Xn[val_idx]); yv = torch.tensor(labels[val_idx])

    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        opt.zero_grad()
        logit = Xt @ w + b
        loss = lossf(logit, yt) + args.l2 * (w * w).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pv = torch.sigmoid(Xv @ w + b).numpy()
    from sklearn.metrics import roc_auc_score  # sklearn ships with the env

    try:
        auroc = float(roc_auc_score(labels[val_idx], pv)) if len(set(labels[val_idx])) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    acc = float(np.mean((pv >= 0.5) == labels[val_idx]))

    tag = {"meta-llama/Llama-2-7b-chat-hf": "llama2",
           "lmsys/vicuna-7b-v1.5": "vicuna",
           "meta-llama/Llama-3.2-3B-Instruct": "llama32_3b",
           "meta-llama/Llama-3.1-8B-Instruct": "llama31",
           "mistralai/Mistral-7B-Instruct-v0.3": "mistral"}.get(target.model_id, "target")
    if args.tag:
        tag = args.tag
    weights_path = data / f"probe_{tag}.pt"
    torch.save({
        "w": w.detach(), "b": float(b.detach().item()),
        "mu": torch.tensor(mu), "sd": torch.tensor(sd),
        "layer": int(layer), "model_id": target.model_id,
        "hidden_size": int(X.shape[1]),
    }, weights_path)

    rep = {
        "model_id": target.model_id, "layer": int(layer),
        "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)),
        "val_auroc": auroc, "val_acc": acc,
        "weights_path": str(weights_path),
        "dedup_reasserted": True,
    }
    write_json(cfg.results_dir / "probe_train_report.json", rep)
    print(f"[02] probe layer={layer} val AUROC={auroc:.3f} acc={acc:.3f} -> {weights_path}")
    target.free()


if __name__ == "__main__":
    main()

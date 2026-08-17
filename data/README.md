# data/

These files are **tracked in git as run artifacts** so the experiment is reproducible from
the repo alone. They are also fully **regenerable** from the local Hugging Face cache:

```
python scripts/00_prepare_data.py --config configs/primary_llama2.yaml
```

It builds, from datasets in the local Hugging Face cache:

- `eval_harmful.jsonl` — 100 JailbreakBench harmful behaviours (evaluation set)
- `eval_benign.jsonl` — 100 JBB benign prompts (for H3 false-refusal rates)
- `probe_train.jsonl` — probe-training pool, **deduplicated against every eval behaviour**
  by normalised token Jaccard (see `probe_dedup_report.json`)
- `probe_dedup_report.json` — the contamination audit; H2 depends on this disjointness
- `gcg_suffix.txt` — the fixed transfer GCG suffix for the static baseline (also embedded
  as a constant in `scripts/00_prepare_data.py`, so it is reproducible either way)

Downloaded base-model weights are **not** here and are never committed — those are run
*inputs* (from the HF cache), not artifacts. The small trained linear probe (`probe_*.pt`)
*is* committed, since it is a run output.

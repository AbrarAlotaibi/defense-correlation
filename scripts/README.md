# Scripts

Run every script from the repository root, for example
`python scripts/26_v2_analysis.py --config configs/v2_vicuna.yaml --seeds 0 1 2`.
Each file starts with a docstring that gives its inputs, outputs and usage.

## Running the experiment

| Script | Purpose |
| --- | --- |
| `fetch_weights.py` | Download the gated model weights into the local Hugging Face cache |
| `00_prepare_data.py` | Build the evaluation splits and the probe-training pool |
| `14_build_external_benign.py`, `15_calibrate_external.py` | Build the external benign corpus and calibrate thresholds on it |
| `27_build_benign_orbench.py` | Build the 400-prompt OR-Bench benign set |
| `01_calibrate.py`, `02_train_probe.py` | Calibrate the input filters and train the hidden-state probes |
| `03_positive_control.py` | Attack the undefended target, the gate for everything after it |
| `04_run_attacks.py` | Attack every safeguard, and run the benign sets through each |
| `05_run_stack.py` | Attack an assembled stack end to end |
| `12_undefended_benign.py` | Run the benign sets through the undefended target |
| `24_cross_eval.py` | Score every safeguard on every search's prompts (shared inputs) |
| `25_run_grid.py` | Drive the stages above for one config, seed by seed |
| `06_judge_gold.py`, `35_resolve_abstentions.py` | Grade the responses, then resolve grader abstentions with a second grader |
| `08_smoke_test.py`, `ppl_precheck.py`, `29_project_schedule.py` | Shape checks and run-time projection before a full run |
| `slurm_v2_*.sbatch` | The SLURM jobs that produced the reported runs |

## Analysis for the paper

| Script | Purpose |
| --- | --- |
| `26_v2_analysis.py` | Attack reproducibility, the behavior-level estimand and the multilevel model |
| `44_e2_passthrough.py`, `46_transfer_structure.py` | The prompt-level estimand and the structure of the transfer matrix |
| `47_e3_table.py` | The assembled stack against the estimate its layers imply |
| `43_frr_composition.py` | Over-refusal: composition estimate against the measured stack |
| `48_family_comparison.py` | The two attack families compared |
| `30_probe_validity.py` | What the hidden-state probes separate |
| `36_check_independence.py`, `38_grader_ceiling.py` | Controls: independence of the searches, and the grader's test-retest ceiling |
| `34_figures_v2.py`, `41_fig_replication.py`, `42_appendix_bootstrap.py` | Figures and the appendix table |

## Human adjudication of the grader

| Script | Purpose |
| --- | --- |
| `28_adjudication.py` | Draw the blinded sample; score two annotators |
| `28b_adjudication_single.py` | Build the retest sample; merge and score a single annotator |
| `28c_build_labeling_tool.py`, `templates/labeling_tool.html` | Build the offline labeling page |

## Manuscript checks

| Script | Purpose |
| --- | --- |
| `39_verify_claims.py` | Recompute the paper's quantitative claims and check them against the text |
| `40_verify_design.py` | Check the method description against the configs that ran |
| `31_verify_refs.py`, `33_checkcites.py` | Verify bibliography entries and citation keys |
| `32_texcheck.py`, `37_tablecheck.py`, `45_floatcheck.py` | Structural checks on the LaTeX source |

`39_verify_claims.py` and `40_verify_design.py` read the manuscript from a sibling folder.
Set `DCORR_MANUSCRIPT` to point them at another copy.

## Earlier version

These produced the material in `earlier_version/` and are not used by the current paper:
`07_analyze.py`, `09_build_fusion_inputs.py`, `09_confound_check.py`,
`10_build_cross_breach.py`, `11_manuscript_numbers.py`, `13_b3_d2_analysis.py`,
`16_judge_determinism.py`, `17_audit_inserts.py`, `18_task_a_variance.py`,
`19_rejudge_multi.py`, `20_task_b_grader_variance.py`,
`21_compare_published_vs_majority.py`, `22_task_d_external.py`,
`23_robustness_figures.py`, `fusion_analysis.py`, `input_level_phi.py`, `make_figures.py`,
`make_graphical_abstract.py`, `run_pipeline.sh`, `slurm_pipeline.sbatch`,
`slurm_b3_d2.sbatch`, `slurm_d3.sbatch` and `slurm_taskD.sbatch`.

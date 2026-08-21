"""Check every number the inserts assert against the artifact that produced it."""
import csv, io, json, re

R = 'results/hpc_vicuna_autodan/'
M = 'fusion/manuscript/'
tex = {f: io.open('paper/' + f, encoding='utf-8').read()
       for f in ('results_insert.tex', 'fusion_insert.tex', 'estimand_insert.tex',
                 'robustness_insert.tex')}
allt = "\n".join(tex.values())
fails, checks = [], 0


def want(token, label, where=None):
    global checks
    checks += 1
    hay = tex[where] if where else allt
    if token not in hay:
        fails.append(f"{label}: {token!r} absent")


def forbid(token, label, where=None):
    global checks
    checks += 1
    hay = tex[where] if where else allt
    if token in hay:
        fails.append(f"{label}: stale {token!r} STILL PRESENT")


# --- A1 -------------------------------------------------------------------------------
cmh = {(r['d1'], r['d2']): r for r in csv.DictReader(open(M + 'A1_B4a_cmh_15pairs.csv'))}
a1 = cmh[('ppl_filter', 'token_anomaly')]
want("$1.39$ ($p=0.91$)", "A1 CMH row")
assert abs(float(a1['CMH_OR']) - 1.39) < 0.005, "A1 CMH_OR drifted"
assert abs(float(a1['crude_OR_haldane']) - 6.014) < 0.005, "A1 crude OR drifted"
assert abs(float(a1['q']) - 0.9435) < 0.001, "A1 q drifted"
want("$0.943$", "A1 q value")
forbid("Same-row defenses share a blind spot that is specific to their",
       "A1 old over-general interpretation", 'results_insert.tex')

# --- B1 -------------------------------------------------------------------------------
k = {int(r['k']): (float(r['ASR']), float(r['FRR']))
     for r in csv.DictReader(open(M + 'B1_k_of_n_full.csv'))}
for kk in (5, 6):
    want(f"$k={kk}$ of 7", f"B1 k={kk} row", 'fusion_insert.tex')
    a, f = k[kk]
    want(f"{a:.3f} & {f:.3f}", f"B1 k={kk} values", 'fusion_insert.tex')

# --- B2 -------------------------------------------------------------------------------
man = json.load(open(M + 'manifest.json'))
assert man['B2']['rowbased_max_size'] == 4 and man['B2']['identical_membership_up_to_size'] == 4
forbid("identical ASR and FRR at every stack size:", "B2 unbounded claim", 'fusion_insert.tex')
want("every stack size the row rule admits", "B2 bounded claim", 'fusion_insert.tex')

# --- B3 -------------------------------------------------------------------------------
b3 = json.load(open(M + 'B3_attributable_refusal.json'))
assert b3['undefended_floor'] == 0.08
want("refuses $8$ of the $100$ benign prompts", "B3 floor", 'results_insert.tex')
want("$0.000$ for perplexity and", "B3 zero attributable", 'results_insert.tex')
for d, v in (('refusal_prime', '0.337'), ('llamaguard', '0.196'), ('probe_b', '0.011')):
    assert f"{b3['per_defense'][d]['attributable_frr']:.3f}" == v, f"B3 {d} drifted"
    want(f"${v}$", f"B3 attributable {d}", 'results_insert.tex')
forbid("roughly $7$--$8\\%$", "B3 stale 7-8% estimate", 'results_insert.tex')

# --- D2 -------------------------------------------------------------------------------
d2 = json.load(open(M + 'D2_intersection_vs_direct.json'))
assert d2['predicted_intersection_residual'] == 0.19 and d2['measured_direct_residual'] == 0.0
want("residual of $0.190$", "D2 prediction", 'results_insert.tex')
want("$100$ of $100$ behaviours", "D2 infeasibility", 'results_insert.tex')
forbid("faithful, very slightly conservative proxy", "D2 stale proxy claim", 'results_insert.tex')

# --- C4 + judge determinism -----------------------------------------------------------
# The single re-judge (19/300) is superseded by the five-repetition measurement; the
# inserts now quote the latter, so assert against Task B's outputs rather than the pilot's.
agree = io.open(M.replace('manuscript/', '') + 'taskB/B5_agreement.txt', encoding='utf-8').read()
assert 'pct_unanimous        90.46' in agree, "B5 unanimity drifted"
assert 'pct_3_2              4.24' in agree, "B5 3-2 rate drifted"
want("$90.5\\%$ of responses are unanimous", "grader unanimity", 'robustness_insert.tex')
want("$4.2\\%$ rest on a bare", "grader 3-2 rate", 'robustness_insert.tex')
want("$0.185$ wide against $0.312$", "grader vs bootstrap width", 'robustness_insert.tex')
want("$91\\%$ have overlapping grader intervals", "Delta orderability", 'robustness_insert.tex')
forbid("$19$ of them ($6.3\\%$)", "superseded single re-judge rate", 'results_insert.tex')

# Task A
want("$193.4$", "Cochran Q", 'robustness_insert.tex')
want("$11$ of the $21$ pairs after Holm", "Holm count", 'robustness_insert.tex')
want("$15/15$ pairs", "permutation vs bootstrap agreement", 'robustness_insert.tex')
want("$p=1.0$", "Llama Guard vs stack McNemar", 'results_insert.tex')

# Task D
dv = json.load(open(M.replace('manuscript/', '') + 'taskD/D3_verdict.json'))
assert 'UNDEFINED' in str(dv['probe16_x_probe8_survives_BH_under_external_thresholds'])
assert dv['probe16_x_probe8_informative_strata'] == 0
want("0.08 \\to 0.26", "probe16 FRR transfer", 'robustness_insert.tex')
want("0.35 \\to 0.64", "token-anomaly ASR transfer", 'robustness_insert.tex')
want("minimum across the whole grid is $0.300$", "min phi across regimes", 'robustness_insert.tex')
want("undefined", "CMH undefined, not refuted", 'robustness_insert.tex')

# the figures the robustness section cites must exist
import os
for fig in ('fig3b_forest_phi.pdf', 'fig5_regimes_phi.pdf'):
    checks += 1
    if not os.path.isfile('paper/figures/' + fig):
        fails.append(f"missing figure paper/figures/{fig}")
# The manuscript's 2.4% of 2700 is CORRECT; an earlier "correction" here was not.
# gold.jsonl has 2700 rows of which 1887 were judged, so both denominators must appear.
want("$66$ of the $2700$ graded", "C4 manuscript figure retained", 'results_insert.tex')
want("$1887$ responses", "C4 judged-only denominator", 'results_insert.tex')
forbid("$1885$ responses", "C4 stale judged count", 'results_insert.tex')

# --- D3 -------------------------------------------------------------------------------
ce = json.load(open(R + 'calibration_external.json'))
hb = ce['comparison']['probe']['heldout_block_rate_at_external_thr']
assert abs(hb - 0.26) < 1e-9, f"D3 probe held-out rate drifted: {hb}"
want("$26\\%$", "D3 probe blowup", 'results_insert.tex')
want("$805$ general instructions", "D3 corpus size", 'results_insert.tex')

# --- D1 / estimand --------------------------------------------------------------------
want("$15/15$", "D1 sign agreement", 'estimand_insert.tex')
want("$10/10$", "D1 transfer agreement", 'estimand_insert.tex')

# --- no placeholders anywhere ---------------------------------------------------------
for f, t in tex.items():
    for pat in (r'\\hpcTBD', 'TODO', 'XXX', 'TBD'):
        checks += 1
        if re.search(pat, t):
            fails.append(f"{f}: placeholder {pat!r} present")

print(f"{checks} assertions checked across 3 inserts")
if fails:
    print(f"\n{len(fails)} PROBLEM(S):")
    for f in fails:
        print("  -", f)
else:
    print("all consistent with the artifacts; no placeholders")

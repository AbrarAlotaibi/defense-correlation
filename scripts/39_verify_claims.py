"""Check the manuscript's quantitative claims against the stored results.

The paper has been revised from two replicates to three, had tables rebuilt, and had numbers
restated in several sections. Every figure quoted in prose should still match the artefact it
came from. This recomputes each one and reports mismatches rather than trusting the text.
"""
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results" / "v2_vicuna"
# The manuscript lives in a sibling repository. Set DCORR_MANUSCRIPT to check another copy.
TEX = Path(os.environ.get(
    "DCORR_MANUSCRIPT", REPO.parent / "defense-correlation-paper" / "paper_A_mlj" / "main.tex",
)).read_text(encoding="utf-8")
d = json.loads((R / "analysis_v2_resolved.json").read_text(encoding="utf-8"))

gold = []
for s in (0, 1, 2):
    for line in (R / f"seed{s}" / "gold_resolved.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            gold.append(json.loads(line))

checks = []


def claim(name, text_must_contain, computed, fmt="{:.3f}"):
    """text_must_contain: the literal string the paper should hold for `computed`.

    Accepts the manuscript's own conventions as equivalent to the bare string, so that a
    formatting difference is not reported as a disagreement about a number:

      * a four-digit count written with LaTeX's thousands separator, 2{,}646 for 2646,
        which is how the paper sets every count above a thousand;
    Deliberately NOT relaxed for precision. Accepting a shorter rendering looked reasonable
    until tested: 0.4242 then matches "0.42", and since the ICC is 0.693 the manuscript
    already contains "0.69" for an unrelated reason, so a p of 0.6875 would have passed
    against the wrong number. Two-decimal values are not distinctive in a paper full of them.
    Where the paper shows a value at lower precision than it is computed, the call site passes
    the displayed string, or skips the check if the exact value carries no claim.
    """
    want = fmt.format(computed) if not isinstance(text_must_contain, str) else text_must_contain
    present = want in TEX
    if not present and want.isdigit() and len(want) > 3:
        grouped = f"{int(want):,}".replace(",", "{,}")
        present = grouped in TEX
        if present:
            want = grouped
    checks.append((name, want, present, computed))


null = d["null"]["mean"]
nullp = sorted(d["null"]["pairs"], reverse=True)
claim("null mean", f"{null:.3f}", null)
for v in nullp:
    claim(f"null estimate {v:.3f}", f"{v:.3f}", v)
claim("null spread", f"{max(nullp)-min(nullp):.3f}", max(nullp) - min(nullp))

e1 = d["e1"]
mx = max(v["mean_phi"] for v in e1.values())
claim("largest excess over null", f"{mx-null:.3f}", mx - null)
claim("largest pair phi", f"{mx:.3f}", mx)
widths = [x["ci"][1] - x["ci"][0] for v in e1.values() for x in v["per_seed"].values()]
claim("median CI width", f"{statistics.median(widths):.3f}", statistics.median(widths))
n_above = sum(1 for v in e1.values() if v["mean_phi"] > null)
claim("pairs above null (as a word)", "five", n_above)

ng = [k for k in e1 if "llamaguard" not in k]
claim("non-guard mean phi", f"{statistics.mean(e1[k]['mean_phi'] for k in ng):.3f}",
      statistics.mean(e1[k]["mean_phi"] for k in ng))
parts = [d["partial"][k]["mean"] for k in ng]
claim("mean partial", f"{statistics.mean(parts):.3f}", statistics.mean(parts))
claim("min partial", f"{min(parts):.3f}", min(parts))
claim("max partial", f"{max(parts):.3f}", max(parts))
claim("difficulty removes pct", f"{100*(1-statistics.mean(parts)/statistics.mean(e1[k]['mean_phi'] for k in ng)):.0f}\\%",
      100 * (1 - statistics.mean(parts) / statistics.mean(e1[k]["mean_phi"] for k in ng)))

fit = d["difficulty"]["fit"]
claim("ICC", f"{fit['icc']:.3f}", fit["icc"])
claim("sd behaviour", f"{fit['sd_behaviour']:.2f}", fit["sd_behaviour"])
dp = [(v["observed"], v["predicted_difficulty_only"], v["excess"])
      for k, v in d["difficulty"]["pairs"].items() if "llamaguard" not in k]
claim("difficulty-only predicted", f"{statistics.mean(x[1] for x in dp):.3f}",
      statistics.mean(x[1] for x in dp))
claim("difficulty excess", f"{statistics.mean(x[2] for x in dp):.3f}",
      statistics.mean(x[2] for x in dp))
gd = [v["excess"] for k, v in d["difficulty"]["pairs"].items() if "llamaguard" in k]
claim("guard excess", f"{statistics.mean(gd):.3f}", statistics.mean(gd))

sat = sum(1 for v in e1.values() if v.get("disattenuation_saturated"))
claim("saturated count (word)", "eight", sat)
claim("max disattenuation", f"{max(v['disattenuated'] for v in e1.values() if v['disattenuated'] == v['disattenuated']):.2f}",
      max(v["disattenuated"] for v in e1.values() if v["disattenuated"] == v["disattenuated"]))

# The abstention figures in Section 5.2 describe the eight single-member arms, which is the
# population that section is about. Once the assembled stacks were graded, gold_resolved also
# carries their rows, and counting those in made these four claims disagree with a manuscript
# that was not wrong. Split by population instead: 221 of 5,627 for the arms, 37 of 890 for
# the stacks, which belongs in the E3 subsection rather than beside Table 4.
EIGHT_ARMS = {"undefended", "llamaguard", "ppl_filter", "probe", "probe_b",
              "refusal_prime", "smoothllm", "token_anomaly"}
arms = [r for r in gold if r["defense"] in EIGHT_ARMS]
stacks = [r for r in gold if r["defense"] not in EIGHT_ARMS]

ab = [r for r in arms if r.get("label_source") == "second_grader"]
claim("abstentions total", str(len(ab)), len(ab))
claim("judged rows", str(sum(1 for r in arms if r.get("judged"))),
      sum(1 for r in arms if r.get("judged")))
claim("abstentions fluent", str(sum(1 for r in ab if r["attack"] == "adaptive")),
      sum(1 for r in ab if r["attack"] == "adaptive"))
claim("abstentions breach", str(sum(1 for r in ab if r["breach"])),
      sum(1 for r in ab if r["breach"]))

if stacks:
    sab = [r for r in stacks if r.get("label_source") == "second_grader"]
    claim("stack judged rows", str(sum(1 for r in stacks if r.get("judged"))),
          sum(1 for r in stacks if r.get("judged")))
    claim("stack abstentions", str(len(sab)), len(sab))
    claim("stack abstentions breach", str(sum(1 for r in sab if r["breach"])),
          sum(1 for r in sab if r["breach"]))


def asr(defense, attack):
    sel = [r for r in gold if r["defense"] == defense and r["attack"] == attack]
    return sum(1 for r in sel if r["breach"]) / len(sel)


for dn in ("undefended", "llamaguard", "smoothllm"):
    claim(f"ASR {dn} adaptive", f"{asr(dn,'adaptive'):.2f}", asr(dn, "adaptive"))

# ---- the replication target -------------------------------------------------
# These went into the paper from a two-replicate analysis that the third seed overturned, so
# they are exactly the claims most worth a standing check. The subsection's argument is that
# the null here is too unstable to support the comparison, which means the spread and the
# largest excess both have to be quoted correctly or the argument reads as the opposite.
L = json.loads((REPO / "results" / "v2_llama2"
                / "analysis_v2.json").read_text(encoding="utf-8"))
lnull = L["null"]["mean"]
lpairs = sorted(L["null"]["pairs"], reverse=True)
claim("L2 null mean", f"{lnull:.3f}", lnull)
for v in lpairs:
    claim(f"L2 null estimate {v:.3f}", f"{v:.3f}", v)
claim("L2 null spread", f"{max(lpairs)-min(lpairs):.3f}", max(lpairs) - min(lpairs))

le1 = L["e1"]
lmx = max(v["mean_phi"] for v in le1.values())
claim("L2 largest excess", f"{lmx-lnull:.3f}", lmx - lnull)
claim("L2 pairs above null (word)", "Eight", sum(1 for v in le1.values()
                                                 if v["mean_phi"] > lnull))
lng = [k for k in le1 if "llamaguard" not in k]
lphi = statistics.mean(le1[k]["mean_phi"] for k in lng)
lpar = statistics.mean(L["partial"][k]["mean"] for k in lng)
claim("L2 non-guard mean phi", f"{lphi:.3f}", lphi)
claim("L2 mean partial", f"{lpar:.3f}", lpar)
claim("L2 difficulty removes pct", f"{100*(1-lpar/lphi):.0f}\\%", 100 * (1 - lpar / lphi))
claim("L2 ICC", f"{L['difficulty']['fit']['icc']:.3f}", L["difficulty"]["fit"]["icc"])
for v in L["reliability"]["llamaguard"]["pairs"]:
    claim(f"L2 guard reliability {v:.3f}", f"{abs(v):.3f}", v)
claim("L2 guard reliability mean", f"{L['reliability']['llamaguard']['mean']:.3f}",
      L["reliability"]["llamaguard"]["mean"])

# ---- over-refusal: composition estimate against the measured stack ----------
# The paper previously quoted only the union of the members' own refusals and drew a
# conclusion from it that the measured stack reverses, so both numbers and the gap between
# them now need to stay correct.
FR = R / "frr_composition.json"
if FR.exists():
    F = json.loads(FR.read_text(encoding="utf-8"))
    for bset, rows in F["sets"].items():
        r = rows[0]
        n = r["n"]
        tag = "matched" if bset == "jbb" else "orbench"
        claim(f"FRR {tag} union7", f"{r['union7']/n:.2f}", r["union7"] / n)
        claim(f"FRR {tag} if-independent7", f"{r['if_independent7']:.2f}",
              r["if_independent7"])
        k, nn = r["measured7"]
        if nn:
            claim(f"FRR {tag} measured7", f"{k/nn:.2f}", k / nn)
        if bset == "jbb":
            claim("FRR matched union6", f"{r['union6']/n:.2f}", r["union6"] / n)
            claim("FRR matched if-independent6", f"{r['if_independent6']:.2f}",
                  r["if_independent6"])
            k6, n6 = r["measured6"]
            if n6:
                claim("FRR matched measured6", f"{k6/n6:.2f}", k6 / n6)
            claim("FRR stack-only refusals", str(r["stack_only"]), r["stack_only"])
        else:
            claim("FRR orbench measured count", f"{k} of {nn}", k)

# ---- E2 on shared inputs ----------------------------------------------------
# The section's argument is that phi here measures how much each member alters the pipeline
# rather than co-failure between mechanisms, so the phi-with-undefended column and the
# tightness of the min bound both have to stay correct. These are on cross/gold.jsonl until
# the cross abstentions are resolved, at which point a shift shows up here rather than in
# the prose.
EP = R / "e2_passthrough.json"
if EP.exists():
    E = json.loads(EP.read_text(encoding="utf-8"))
    claim("E2 shared inputs", str(E["n"]), E["n"])
    for dn, m in E["members"].items():
        if m["phi_with_undefended"] is not None:
            claim(f"E2 {dn} phi w/ undefended", f"{m['phi_with_undefended']:.3f}",
                  m["phi_with_undefended"])
        # Only where the count is distinctive enough for a substring search to mean
        # something. "0", "3" and "14" occur all over a 13,000-word paper, so checking for
        # them would always pass and would be worse than not checking: a test that cannot
        # fail reads as coverage. Those four rows are covered by the table-column check.
        ident = m["identical_to_undefended"]
        if ident >= 100:
            claim(f"E2 {dn} returns target response", str(ident), ident)
    mr = E.get("min_rule", {})
    if mr.get("at_most_one_intervenes", {}).get("n"):
        a = mr["at_most_one_intervenes"]
        claim("E2 min-rule max deviation", f"{a['max_abs_deviation']:.3f}",
              a["max_abs_deviation"])
        claim("E2 min-rule mean deviation", f"{a['mean_abs_deviation']:.3f}",
              a["mean_abs_deviation"])

# ---- the transfer matrix's structure --------------------------------------------
# The claim is that the matrix is a function of its column, so the two variance shares and
# the diagonal advantage are the numbers carrying it.
TS = R / "transfer_structure.json"
if TS.exists():
    T2 = json.loads(TS.read_text(encoding="utf-8"))
    for k, label in (("variance_explained_by_column_all", "transfer column share"),
                     ("variance_explained_by_row_all", "transfer row share"),
                     ("variance_explained_by_column_no_guard", "transfer column, no guard"),
                     ("variance_explained_by_row_no_guard", "transfer row, no guard")):
        claim(label, f"{100 * T2[k]:.1f}\%", 100 * T2[k])
    claim("transfer diagonal, no guard", f"{T2['diagonal_mean_no_guard']:.3f}",
          T2["diagonal_mean_no_guard"])
    claim("transfer off-diagonal, no guard", f"{T2['offdiagonal_mean_no_guard']:.3f}",
          T2["offdiagonal_mean_no_guard"])
    claim("transfer diagonal advantage",
          f"{T2['diagonal_mean_no_guard'] - T2['offdiagonal_mean_no_guard']:.3f}",
          T2["diagonal_mean_no_guard"] - T2["offdiagonal_mean_no_guard"])

# ---- the second attack family ---------------------------------------------------
# Two claims, pulling opposite ways, so both need guarding. The method's verdict replicates
# (each family's own null and largest pair excess), and the practical one does not (the
# guard's reversal, which is the paper's single most consequential number).
FC = R / "family_comparison.json"
if FC.exists():
    F2 = json.loads(FC.read_text(encoding="utf-8"))
    for att, tag in (("adaptive", "fam genetic"), ("pair", "fam LLM-driven")):
        nul = F2["null"].get(att) or {}
        if nul.get("mean") is not None:
            claim(f"{tag} null", f"{nul['mean']:.3f}", nul["mean"])
        if nul.get("spread") is not None:
            claim(f"{tag} null spread", f"{nul['spread']:.3f}", nul["spread"])
        ng = {k: v for k, v in (F2["pairs"].get(att) or {}).items()
              if "llamaguard" not in k}
        if ng and nul.get("mean") is not None:
            mx = max(v["mean_phi"] for v in ng.values())
            claim(f"{tag} largest pair", f"{mx:.3f}", mx)
            claim(f"{tag} largest excess", f"{mx - nul['mean']:.3f}", mx - nul["mean"])
    # The guard reversal.
    g = F2["marginals"].get("llamaguard") or {}
    if g.get("adaptive") and g.get("pair"):
        claim("fam guard genetic ASR", f"{g['adaptive']['asr']:.2f}", g["adaptive"]["asr"])
        claim("fam guard LLM ASR", f"{g['pair']['asr']:.2f}", g["pair"]["asr"])
    cg = F2["cross_family"].get("llamaguard") or {}
    if cg:
        claim("fam guard LLM-only behaviours", str(cg["pair_only"]), cg["pair_only"])
        claim("fam guard genetic-only behaviours", str(cg["ga_only"]), cg["ga_only"])
    for att, tag in (("adaptive", "fam genetic budget"), ("pair", "fam LLM budget")):
        b = F2["budget"].get(att) or {}
        if b.get("median_evals"):
            claim(tag, f"{b['median_evals']:.0f}", b["median_evals"])

# ---- E2 on the replication target ----------------------------------------------
# The claim there is that blocking rate and effect on the outcome are close to orthogonal,
# carried by one pair of numbers: the mid-layer probe blocks 514 of 800 and still correlates
# 0.965 with no defense, while the token statistic blocks 295 and falls to 0.520. The min
# bound's two regimes carry the second claim.
L2EP = R.parent / "v2_llama2" / "e2_passthrough.json"
if L2EP.exists():
    LE = json.loads(L2EP.read_text(encoding="utf-8"))
    for dn in ("probe", "token_anomaly"):
        m = LE["members"].get(dn)
        if not m:
            continue
        claim(f"L2 E2 {dn} blocks", str(m["blocked"]), m["blocked"])
        if m["phi_with_undefended"] is not None:
            claim(f"L2 E2 {dn} phi w/ undefended", f"{m['phi_with_undefended']:.3f}",
                  m["phi_with_undefended"])
    mr = LE.get("min_rule") or {}
    one, both = mr.get("at_most_one_intervenes") or {}, mr.get("both_intervene") or {}
    if one.get("n"):
        claim("L2 E2 min-rule max dev, one intervenes",
              f"{one['max_abs_deviation']:.3f}", one["max_abs_deviation"])
    if both.get("n"):
        claim("L2 E2 min-rule mean dev, both intervene",
              f"{both['mean_abs_deviation']:.3f}", both["mean_abs_deviation"])
        claim("L2 E2 min-rule max dev, both intervene",
              f"{both['max_abs_deviation']:.3f}", both["max_abs_deviation"])

# ---- over-refusal on the replication target ------------------------------------
# The replication's cost-axis argument is that every quantity saturates, so the undefended
# baseline and the measured stack are what carry it. The undefended rate is checked too,
# because "the target refuses half the benign set before any safeguard" is the claim in that
# subsection a reader is most likely to disbelieve.
L2FR = R.parent / "v2_llama2" / "frr_composition.json"
if L2FR.exists():
    LF = json.loads(L2FR.read_text(encoding="utf-8"))
    for bset, tag in (("jbb", "L2 FRR matched"), ("orbench", "L2 FRR orbench")):
        rows = LF["sets"].get(bset)
        if not rows:
            continue
        row = rows[0]
        uk, un = row["undefended"]
        if un:
            claim(f"{tag} undefended", f"{uk / un:.3f}", uk / un)
        mk, mn = row["measured7"]
        if mn:
            claim(f"{tag} measured count", f"{mk} of {mn}", mk)

# ---- E3: the measured stack against the estimate its layers imply --------------
# This is the paper's practical claim, so the measured value, the intersection and the gap
# all need guarding -- and so does the McNemar p, which is the one significance statement in
# the section.
E3 = R / "e3_table.json"
if E3.exists():
    for row in json.loads(E3.read_text(encoding="utf-8")):
        tag = "E3 " + ("seven" if row["name"].startswith("Seven")
                       else "six" if row["name"].startswith("Six")
                       else "+".join(m[:4] for m in row["members"]))
        claim(f"{tag} measured", f"{row['measured']:.2f}", row["measured"])
        claim(f"{tag} intersection", f"{row['intersection']:.2f}", row["intersection"])
        claim(f"{tag} strongest", f"{row['strongest']:.2f}", row["strongest"])
        claim(f"{tag} gap", f"{row['gap']:.3f}", row["gap"])
        if len(row["seeds"]) == 1:
            pv = row["p_vs_intersection"][0]
            if pv < 0.05:
                claim(f"{tag} McNemar p", f"{pv:.3f}", pv)
        pooled = row.get("pooled_vs_intersection")
        if pooled:
            # The pooled test is what the section leads with, so its p and both discordant
            # counts have to stay correct in the text.
            # Only where the p carries a claim. For a floored comparison the paper's
            # statement is "no resolution", which the measured and intersection values
            # already pin; checking a p of 0.69 to four places would just invite a false
            # pass against the ICC's 0.693.
            if pooled["p"] < 0.05:
                claim(f"{tag} pooled p", f"{pooled['p']:.4f}", pooled["p"])
            claim(f"{tag} pooled discordant", str(pooled["n10"]), pooled["n10"])
            claim(f"{tag} pooled reverse", str(pooled["n01"]), pooled["n01"])

# ---- E3 on the replication target ----------------------------------------------
# The six-layer row is the one comparison that resolves on this target, and it is the
# newest claim in the paper, so every figure the prose quotes is pinned here. The
# seven-member row is skipped deliberately: its measured residual is 0.000 and the text
# says "no successful attack within the budget" rather than quoting a rate.
L2E3 = R.parent / "v2_llama2" / "e3_table.json"
if L2E3.exists():
    for row in json.loads(L2E3.read_text(encoding="utf-8")):
        if not row["name"].startswith("Six"):
            continue
        claim("L2 E3 six measured", f"{row['measured']:.3f}", row["measured"])
        claim("L2 E3 six intersection", f"{row['intersection']:.3f}", row["intersection"])
        claim("L2 E3 six gap", f"{row['gap']:.3f}", row["gap"])
        claim("L2 E3 six strongest", f"{row['strongest']:.3f}", row["strongest"])
        # Per-replicate gaps: the prose claims all three are positive, which is a stronger
        # statement than the mean and is the reason the row is reportable at all.
        for i, (m, x) in enumerate(zip(row["measured_per_seed"], row["intersection_per_seed"])):
            claim(f"L2 E3 six gap seed{i}", f"{m - x:+.2f}", m - x)
        pooled = row.get("pooled_vs_intersection")
        if pooled:
            claim("L2 E3 six pooled p", f"{pooled['p']:.4f}", pooled["p"])
            claim("L2 E3 six pooled discordant", str(pooled["n10"]), pooled["n10"])
            claim("L2 E3 six pooled reverse", str(pooled["n01"]), pooled["n01"])

# ---- grader agreement ----------------------------------------------------------
# These are the numbers that bound every phi in the paper, and they are the ones that went
# stale: the Vicuna triple survived a re-grade unrevised because only the abstention counts
# were guarded. Script 35 computes them over the rows the primary grader labelled without
# abstaining, which is the population 5.2 describes, so the stored value is the claim.
for run, tag in ((R, "agreement"), (R.parent / "v2_llama2", "L2 agreement")):
    raws, kappas = [], []
    for seed in (0, 1, 2):
        ap = run / f"seed{seed}" / "abstentions.json"
        if not ap.exists():
            continue
        A = json.loads(ap.read_text(encoding="utf-8"))
        r, k = A.get("raw_agreement"), A.get("cohen_kappa")
        if r is not None:
            claim(f"{tag} raw seed{seed}", f"{r:.3f}", r)
            raws.append(r)
        if k is not None:
            claim(f"{tag} kappa seed{seed}", f"{k:.3f}", k)
            kappas.append(k)
    # The paragraph argues from the SPREAD, not the mean, so the spread is a claim too.
    if len(kappas) > 1 and tag == "agreement":
        claim("kappa spread", f"{max(kappas) - min(kappas):.3f}", max(kappas) - min(kappas))

# The cross-evaluation is where the graders agree least, which is the paragraph's point.
XAB = R / "seed0" / "cross" / "abstentions.json"
if XAB.exists():
    A = json.loads(XAB.read_text(encoding="utf-8"))
    if A.get("raw_agreement") is not None:
        claim("cross raw agreement", f"{A['raw_agreement']:.3f}", A["raw_agreement"])
    if A.get("cohen_kappa") is not None:
        claim("cross kappa", f"{A['cohen_kappa']:.3f}", A["cohen_kappa"])

print(f"{'claim':<34}{'expected in text':>18}  present?")
bad = 0
for name, want, present, val in checks:
    flag = "yes" if present else "NO"
    if not present:
        bad += 1
    print(f"{name:<34}{want:>18}  {flag}")
print()
print(f"{len(checks)} claims checked, {bad} not found verbatim in the manuscript")
print("(a 'NO' may be a rounding or wording difference, not necessarily an error:"
      " inspect each)")
short = [n for n, w, p, _ in checks if p and len(w.strip("\\%")) <= 3]
if short:
    print(f"\n({len(short)} of the passes are substring matches on three characters or fewer,"
          f" so they are weak evidence: {', '.join(short[:4])}"
          f"{', ...' if len(short) > 4 else ''})")

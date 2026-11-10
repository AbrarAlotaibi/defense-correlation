import csv
import numpy as np

name = {"perplexity": "ppl_filter", "token-anomaly": "token_anomaly", "probe16": "probe",
        "probe8": "probe_b", "refusal-prime": "refusal_prime", "smoothllm": "smoothllm"}

orig = {}
for r in csv.DictReader(open('results/hpc_vicuna_autodan/table6.csv')):
    orig[frozenset((r['d1'], r['d2']))] = (float(r['phi']), float(r['p1']), float(r['p2']),
                                           float(r['excess']))

maj = list(csv.DictReader(open('fusion/taskB/B3_table10_majority.csv')))
gr = {r['pair']: r for r in csv.DictReader(open('fusion/taskB/B4_grader_intervals.csv'))}

print("phi: published single-judgment vs majority-of-5, and whether the published value")
print("falls inside the grader-only interval\n")
print(f"{'pair':30} {'pub':>6} {'maj':>6} {'shift':>7}  grader interval   pub inside?")
outside = 0
for r in maj:
    a, b = [name[x.strip()] for x in r['pair'].split(' x ')]
    p_pub = orig[frozenset((a, b))][0]
    p_maj = float(r['phi'])
    lo, hi = float(gr[r['pair']]['phi_grader_lo']), float(gr[r['pair']]['phi_grader_hi'])
    inside = lo <= p_pub <= hi
    outside += (not inside)
    print(f"{r['pair']:30} {p_pub:6.3f} {p_maj:6.3f} {p_maj-p_pub:+7.3f}  "
          f"[{lo:.3f},{hi:.3f}]  {'yes' if inside else 'NO'}")
print(f"\npublished phi outside the grader interval for {outside}/15 pairs")

# interval widths: grader vs behaviour bootstrap
gw = [float(gr[r['pair']]['phi_grader_hi']) - float(gr[r['pair']]['phi_grader_lo']) for r in maj]
bw = [float(r['phi_hi']) - float(r['phi_lo']) for r in maj]
print(f"\nmean interval width  grader {np.mean(gw):.3f}   behaviour bootstrap {np.mean(bw):.3f}"
      f"   ratio {np.mean(gw)/np.mean(bw):.2f}")

# can Delta values be ordered? count pairs of pairs whose grader intervals overlap
print("\nDelta orderability under grader noise:")
D = [(r['pair'], float(r['Delta']), float(gr[r['pair']]['Delta_grader_lo']),
      float(gr[r['pair']]['Delta_grader_hi'])) for r in maj]
D.sort(key=lambda t: -t[1])
tot = ov = 0
close_ov = close_tot = 0
for i in range(len(D)):
    for j in range(i + 1, len(D)):
        tot += 1
        overlap = not (D[i][2] > D[j][3] or D[j][2] > D[i][3])
        ov += overlap
        if abs(D[i][1] - D[j][1]) < 0.05:
            close_tot += 1
            close_ov += overlap
print(f"  all pairs-of-pairs: {ov}/{tot} have overlapping grader intervals "
      f"({100*ov/tot:.0f}%)")
print(f"  those whose Delta differs by < 0.05: {close_ov}/{close_tot} overlap "
      f"({100*close_ov/max(1,close_tot):.0f}%)")
print("\n  -> the 'do not order Delta values within 0.05' caveat is SUPPORTED by the grader")
print("     intervals, not removed by them." if close_ov / max(1, close_tot) > 0.8 else "")

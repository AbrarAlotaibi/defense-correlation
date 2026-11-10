"""Check that every figure and table is referenced in the prose, and vice versa.

An unreferenced float is a defect a reviewer notices and a copy-editor queries. A \\ref to a
label that is attached to nothing is caught by texcheck; this catches the opposite case, and
also reports where each float is first mentioned so the ordering can be sanity-checked.
"""
import re
import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text(encoding="utf-8")
s = re.sub(r"(?m)^\s*%.*$", "", s)

floats = []
for m in re.finditer(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", s, re.S):
    kind, body = m.group(1), m.group(2)
    lab = re.search(r"\\label\{([^}]*)\}", body)
    floats.append((kind, lab.group(1) if lab else None, m.start()))

refs = {}
for m in re.finditer(r"\\(?:ref|autoref|eqref)\{([^}]*)\}", s):
    refs.setdefault(m.group(1), []).append(m.start())

print(f"{'kind':<8}{'label':<22}{'refs':>5}  first mention")
bad = 0
for kind, lab, pos in floats:
    if lab is None:
        print(f"{kind:<8}{'(no label)':<22}{'-':>5}  cannot be referenced")
        bad += 1
        continue
    where = refs.get(lab, [])
    n = len(where)
    if n == 0:
        print(f"{kind:<8}{lab:<22}{n:>5}  *** NEVER REFERENCED ***")
        bad += 1
        continue
    first = min(where)
    rel = "before the float" if first < pos else "after the float"
    print(f"{kind:<8}{lab:<22}{n:>5}  {rel}")

labels = {lab for _, lab, _ in floats if lab}
# Every label in the file that is not attached to a float: sections, equations, appendices.
# Enumerating prefixes by hand meant app: labels were reported as dangling, and a check that
# cries wolf is a check people stop reading.
sec_labels = set(re.findall(r"\\label\{([^}]*)\}", s)) - labels
dangling = [r for r in refs if r not in labels and r not in sec_labels]
if dangling:
    print(f"\nreferences to labels that are defined nowhere in the file: {dangling}")
    bad += len(dangling)

print(f"\n{bad} problem(s)" if bad else "\nevery float is labelled and referenced")

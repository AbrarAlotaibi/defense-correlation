"""Check every tabular's declared column count against the fields in each row.

texcheck.py catches unbalanced environments and bad labels, but a row with the wrong number of
ampersands compiles to "Extra alignment tab has been changed" or silently misaligns, which is
exactly the failure a programmatically generated table introduces.

The column spec needs balanced-brace parsing, not a regex: tabular* takes a width argument
first, and the spec itself contains nested braces such as @{\\extracolsep{\\fill}}.
"""
import re
import sys
from pathlib import Path


def braced(s: str, i: int):
    """Read a {...} group starting at s[i] == '{'; return (contents, index after it)."""
    assert s[i] == "{", s[i:i + 20]
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced")


def ncols(spec: str) -> int:
    spec = re.sub(r"@\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)   # drop @{...}, one nesting level
    spec = re.sub(r"[pmb]\{[^{}]*\}", "P", spec)             # p{3cm} is one column
    spec = re.sub(r"\*\{(\d+)\}\{([^{}]*)\}",
                  lambda m: m.group(2) * int(m.group(1)), spec)
    return len(re.findall(r"[lcrXP]", spec))


s = Path(sys.argv[1]).read_text(encoding="utf-8")
s = re.sub(r"(?m)^\s*%.*$", "", s)
problems = 0

for m in re.finditer(r"\\begin\{(tabular\*|tabularx|tabular)\}", s):
    env = m.group(1)
    i = m.end()
    try:
        if env in ("tabular*", "tabularx"):
            _, i = braced(s, s.index("{", i))      # the width argument
        while s[i] in " \n\t[":                     # optional [t]/[b] position
            if s[i] == "[":
                i = s.index("]", i) + 1
            else:
                i += 1
        spec, body_start = braced(s, s.index("{", i))
    except Exception as e:                                       # noqa: BLE001
        print(f"  line {s[:m.start()].count(chr(10))+1}: cannot parse {env} spec: {e}")
        problems += 1
        continue

    n = ncols(spec)
    end = s.index(f"\\end{{{env}}}", body_start)
    body = s[body_start:end]
    line_no = s[:m.start()].count("\n") + 1

    bad = []
    for raw in body.split("\\\\"):
        r = raw
        for kill in ("\\toprule", "\\midrule", "\\bottomrule", "\\addlinespace", "\\hline"):
            r = r.replace(kill, "")
        r = re.sub(r"\\cmidrule(\([^)]*\))?\{[^{}]*\}", "", r)
        if not r.strip():
            continue
        fields = re.split(r"(?<!\\)&", r)
        got = len(fields)
        for f in fields:                     # a multicolumn spans several declared columns
            mc = re.search(r"\\multicolumn\{(\d+)\}", f)
            if mc:
                got += int(mc.group(1)) - 1
        if got != n:
            bad.append((got, " ".join(r.split())[:66]))
    if bad:
        problems += len(bad)
        print(f"  MISMATCH {env} at line {line_no}, spec '{spec}' -> {n} columns:")
        for got, txt in bad[:8]:
            print(f"      {got} fields: {txt}")
    else:
        print(f"  OK  {env} line {line_no}: {n} columns, all rows match")

print()
if problems:
    print(f"{problems} row(s) whose field count does not match the column spec")
    sys.exit(1)
print("every tabular row matches its column spec")

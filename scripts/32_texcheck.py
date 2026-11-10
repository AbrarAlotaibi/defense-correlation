"""Structural checks on a LaTeX file, for when no local TeX toolchain exists.

Catches the mistakes that cost an Overleaf round trip: unbalanced environments, unbalanced
braces, undefined labels, duplicate labels, and references to labels that do not exist.
"""
import re
import sys
from collections import Counter
from pathlib import Path

s = Path(sys.argv[1]).read_text(encoding="utf-8")
body = re.sub(r"(?m)^\s*%.*$", "", s)
problems = []

stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    kind, env = m.group(1), m.group(2)
    if kind == "begin":
        stack.append((env, m.start()))
    else:
        if not stack:
            problems.append(f"\\end{{{env}}} with no matching \\begin")
        elif stack[-1][0] != env:
            problems.append(f"\\end{{{env}}} closes \\begin{{{stack[-1][0]}}}")
            stack.pop()
        else:
            stack.pop()
for env, pos in stack:
    line = body[:pos].count("\n") + 1
    problems.append(f"\\begin{{{env}}} never closed (line ~{line})")

depth = 0
for i, ch in enumerate(body):
    if ch == "{" and (i == 0 or body[i - 1] != "\\"):
        depth += 1
    elif ch == "}" and (i == 0 or body[i - 1] != "\\"):
        depth -= 1
        if depth < 0:
            problems.append(f"unbalanced closing brace near line {body[:i].count(chr(10)) + 1}")
            depth = 0
if depth:
    problems.append(f"{depth} unclosed brace(s)")

labels = re.findall(r"\\label\{([^}]+)\}", body)
for lab, n in Counter(labels).items():
    if n > 1:
        problems.append(f"duplicate label: {lab} ({n} times)")
refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", body))
for r in sorted(refs - set(labels)):
    problems.append(f"reference to an undefined label: {r}")

for req in (r"\begin{document}", r"\end{document}", r"\bibliography{"):
    if req not in body:
        problems.append(f"missing {req}")

print(f"environments balanced: {not stack}")
print(f"labels: {len(labels)}   references: {len(refs)}")
if problems:
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nno structural problems found")

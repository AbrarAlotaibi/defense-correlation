"""Check that every key cited in a .tex exists in the .bib.

Run before any Overleaf build: an undefined key is a silent [?] in the PDF, and with an
author-year style it also breaks the sentence it sits in.

Usage: python scripts/33_checkcites.py main.tex refs.bib
"""
import re
import sys
from pathlib import Path

tex = Path(sys.argv[1]).read_text(encoding="utf-8")
bib = Path(sys.argv[2]).read_text(encoding="utf-8", errors="ignore")

keys = set()
for m in re.finditer(r"\\cite[pt]?\s*(?:\[[^\]]*\])?\{([^}]*)\}", tex, re.S):
    keys |= {k.strip() for k in m.group(1).split(",") if k.strip()}
have = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))

missing = sorted(keys - have)
print(f"cited: {len(keys)}   bib entries: {len(have)}   missing: {len(missing)}")
for k in sorted(keys):
    print(("  ok      " if k in have else "  MISSING ") + k)
if missing:
    sys.exit(1)

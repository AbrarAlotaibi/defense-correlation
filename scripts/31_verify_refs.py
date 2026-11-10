"""Verify every bibliography entry against Crossref, DataCite and arXiv.

Journals require that "every reference included must correspond to a real, existing
source", with DOIs where available, and that preprints are marked as preprints and replaced by
the published version where one exists. Reference lists carried across drafts, especially ones
assembled with model assistance, can contain entries that do not exist or whose metadata has
drifted, so this checks each one against an authority rather than trusting the file.

For each entry:
  * if it has a DOI, resolve it (Crossref, then DataCite) and compare the title and year;
  * else if it is an arXiv preprint, resolve the arXiv id and report whether a journal version
    now exists (arXiv records the DOI when there is one);
  * else search Crossref by title and report the best match and its similarity.

Writes a CSV and prints a summary. Network-bound, so it is slow and polite: one request at a
time with a small delay.

Known limitation: the title-search fallback (used when an entry has neither a DOI nor an
arXiv id) relies on Crossref, whose coverage of machine-learning conference proceedings is
poor. NeurIPS, ICML and USENIX papers routinely come back NOT_FOUND even though they are real
and well known, so that status means "could not confirm here", not "does not exist". The
decisive signal is TITLE_MISMATCH on an entry that HAS an identifier: the identifier is in the
file and it resolves to something else.

Usage:
  python scripts/31_verify_refs.py refs.bib --out refs_report.csv
  python scripts/31_verify_refs.py refs.bib --only-cited main.tex
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "defense-correlation-ref-check/1.0 (mailto:amotaibi@iau.edu.sa)"
ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", re.S)
FIELD = re.compile(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*\n", re.S)


def get(url: str, timeout: float = 25.0):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def norm(s: str) -> str:
    s = re.sub(r"\{|\}|\\[a-zA-Z]+|\$", "", s or "")
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def parse_bib(text: str) -> list[dict]:
    out = []
    for kind, key, body in ENTRY.findall(text):
        f = {k.lower(): re.sub(r"\s+", " ", v).strip()
             for k, v in FIELD.findall(body + "\n")}
        f.update(_type=kind.lower(), _key=key)
        out.append(f)
    return out


def check(e: dict) -> dict:
    title, year = e.get("title", ""), e.get("year", "")
    doi = (e.get("doi") or "").strip().replace("https://doi.org/", "")
    res = {"key": e["_key"], "type": e["_type"], "title": title[:70], "year": year,
           "doi": doi, "status": "", "found_title": "", "similarity": "", "note": ""}

    if doi:
        raw = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
        src = "crossref"
        if raw is None:
            raw = get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
            src = "datacite"
        if raw is None:
            res["status"] = "DOI_NOT_FOUND"
            return res
        d = json.loads(raw)
        if src == "crossref":
            m = d["message"]
            found = (m.get("title") or [""])[0]
            fy = str((m.get("issued", {}).get("date-parts") or [[""]])[0][0])
        else:
            a = d["data"]["attributes"]
            t = a.get("titles") or [{}]
            found = t[0].get("title", "")
            fy = str(a.get("publicationYear", ""))
        sim = similar(title, found)
        res.update(found_title=found[:70], similarity=f"{sim:.2f}",
                   status="OK" if sim >= 0.80 else "TITLE_MISMATCH")
        if year and fy and year != fy:
            res["note"] = f"year {year} vs {fy}"
        return res

    arx = None
    for field in ("eprint", "journal", "note", "howpublished", "url"):
        m = re.search(r"(\d{4}\.\d{4,5})", e.get(field, "") or "")
        if m:
            arx = m.group(1)
            break
    if arx:
        # arXiv's own API is unreachable from some networks (it returns an empty body rather
        # than an error), so resolve the arXiv DOI through DataCite, which mints them.
        res["doi"] = f"10.48550/arXiv.{arx}"
        raw = get(f"https://api.datacite.org/dois/10.48550%2FarXiv.{arx}")
        if raw is None:
            res["status"] = "ARXIV_NOT_FOUND"
            return res
        a = json.loads(raw)["data"]["attributes"]
        found = ((a.get("titles") or [{}])[0]).get("title", "")
        sim = similar(title, found)
        # a journal version, if one exists, appears as a related identifier
        pub = ""
        for rel in a.get("relatedIdentifiers") or []:
            if rel.get("relationType") in ("IsIdenticalTo", "IsPreviousVersionOf",
                                           "IsSupplementTo", "IsVersionOf"):
                pub = rel.get("relatedIdentifier", "")
                break
        res.update(found_title=found[:70], similarity=f"{sim:.2f}",
                   status="OK_PREPRINT" if sim >= 0.80 else "TITLE_MISMATCH",
                   note=f"also {pub}" if pub else "preprint; check for a published version")
        return res

    if not title:
        res["status"] = "NO_TITLE"
        return res
    raw = get("https://api.crossref.org/works?rows=3&query.bibliographic="
              + urllib.parse.quote(title))
    if raw is None:
        res["status"] = "SEARCH_FAILED"
        return res
    items = json.loads(raw)["message"].get("items", [])
    best, bs = "", 0.0
    for it in items:
        t = (it.get("title") or [""])[0]
        s = similar(title, t)
        if s > bs:
            best, bs = t, s
            res["doi"] = it.get("DOI", "")
    res.update(found_title=best[:70], similarity=f"{bs:.2f}",
               status="MATCH_NO_DOI_IN_BIB" if bs >= 0.85 else "NOT_FOUND",
               note="add the DOI" if bs >= 0.85 else "no close match on title")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bib", type=Path)
    ap.add_argument("--out", type=Path, default=Path("refs_report.csv"))
    ap.add_argument("--only-cited", type=Path, default=None,
                    help="a .tex: check only the entries it cites")
    ap.add_argument("--delay", type=float, default=0.4)
    a = ap.parse_args()

    entries = parse_bib(a.bib.read_text(encoding="utf-8", errors="ignore"))
    if a.only_cited:
        tex = a.only_cited.read_text(encoding="utf-8")
        cited = set()
        for m in re.finditer(r"\\cite[pt]?\s*(?:\[[^\]]*\])?\{([^}]*)\}", tex, re.S):
            cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
        entries = [e for e in entries if e["_key"] in cited]
    print(f"checking {len(entries)} entries against Crossref, DataCite and arXiv", flush=True)

    rows = []
    for i, e in enumerate(entries, 1):
        r = check(e)
        rows.append(r)
        flag = "" if r["status"].startswith("OK") else "  <-- "
        print(f"[{i:>3}/{len(entries)}] {r['status']:<21} {r['key']}{flag}", flush=True)
        time.sleep(a.delay)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nsummary")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v}")
    bad = [r for r in rows if not r["status"].startswith("OK")]
    print(f"\n{len(bad)} entries need attention -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

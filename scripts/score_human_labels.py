"""Compare filled human label sheets with the AI judges, and compare arms under human labels.

  python scripts/score_human_labels.py rater1.csv [rater2.csv ...]

Reads the blind key (results/human_label_key.csv) and, if present, the Gemma re-judging (runs/exp5b-gemma-judge).
Reports: agreement of each rater with the Qwen judge (and Gemma), agreement between raters, and the mean human
safety / reinforcing rate per arm (never vs always C vs U-CTSC v4).
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
key = {r["id"]: r for r in csv.DictReader(open(ROOT / "results" / "human_label_key.csv"))}
gem = {}
for p in (ROOT / "runs" / "exp5b-gemma-judge").glob("*.jsonl"):
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "turn" and r.get("label"):
            gem[(p.stem, r["persona"], r["day"], r["turn"])] = r


def read(path):
    out = {}
    for r in csv.reader(open(path)):
        if r and r[0].startswith("H") and len(r) >= 7 and r[5].strip():
            out[r[0]] = (r[5].strip().lower(), int(r[6]) if r[6].strip().isdigit() else None)
    return out


raters = {Path(p).stem: read(p) for p in sys.argv[1:]}
for name, lab in raters.items():
    ids = [i for i in lab if i in key]
    q = [key[i]["qwen_label"] for i in ids]
    h = [lab[i][0] for i in ids]
    print(f"{name}: {len(ids)} labelled; label kappa vs Qwen judge {cohen_kappa_score(h, q):.2f}")
    g = [(lab[i][0], gem[k]["label"]) for i in ids
         if (k := (key[i]["arm"], key[i]["persona"], int(key[i]["day"]), int(key[i]["turn"]))) in gem]
    if g:
        print(f"   label kappa vs Gemma judge {cohen_kappa_score(*zip(*g)):.2f} (n={len(g)})")
    by = defaultdict(list)
    for i in ids:
        by[key[i]["arm"]].append(lab[i])
    for arm, xs in sorted(by.items()):
        s = [x[1] for x in xs if x[1] is not None]
        print(f"   {arm:10} n={len(xs):3}  human reinforcing {st.mean(x[0] == 'reinforcing' for x in xs):.0%}"
              f"  human safety {st.mean(s):.2f}")
names = list(raters)
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        ids = [i for i in raters[names[a]] if i in raters[names[b]]]
        print(f"{names[a]} vs {names[b]}: label kappa {cohen_kappa_score([raters[names[a]][i][0] for i in ids], [raters[names[b]][i][0] for i in ids]):.2f} (n={len(ids)})")

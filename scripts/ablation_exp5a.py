"""Experiment 5a: paired ablation of the learned controller vs the reply check (same 54 children, seed 21)."""
import json, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
rows = {}
for f, ren in [("runs/online-exp4/outcomes.jsonl", {"uctsc_v4": "uctsc_v4_exp4"}), ("runs/online-exp5a/outcomes.jsonl", {})]:
    for l in open(ROOT / f):
        r = json.loads(l)
        if r["status"] != "done": continue
        pol = ren.get(r["policy"], r["policy"])
        rows.setdefault(pol, {})[r["persona"]] = r
M = ["utility", "turn_safety", "reinforcing_rate", "flagged_days", "end_ai_attachment" , "d_loneliness", "intervention_rate", "rewrite_rate"]
def val(r, m):
    if m == "d_loneliness": return r["end_loneliness"] - r["start_loneliness"]
    return r.get(m)
ARMS = [("never", "no controller, no check"), ("uctsc_v3", "learned controller only"), ("guard_only", "reply check only"),
        ("uctsc_v4", "controller + check (v4)"), ("always_C", "always C"), ("always_C_guard", "always C + check")]
out = {"arms": {}, "pairs": {}}
common = set.intersection(*(set(rows[a]) for a, _ in ARMS))
print("children in all arms:", len(common))
for a, lab in ARMS:
    out["arms"][a] = {"label": lab, **{m: st.mean(val(rows[a][p], m) or 0 for p in common) for m in M}}
    print(f"{lab:28}" + " ".join(f"{m[:10]}={out['arms'][a][m]:.3f}" for m in M))
def pair(a, b):
    d = {}
    for m in M:
        x = [(val(rows[a][p], m) or 0) - (val(rows[b][p], m) or 0) for p in common]
        d[m] = {"mean": st.mean(x), "se": st.stdev(x) / len(x) ** .5}
    return d
for a, b in [("guard_only", "never"), ("uctsc_v3", "never"), ("uctsc_v4", "never"), ("uctsc_v4", "guard_only"), ("uctsc_v4", "uctsc_v3"),
             ("always_C_guard", "always_C"), ("uctsc_v4", "always_C_guard"), ("uctsc_v4", "uctsc_v4_exp4")]:
    out["pairs"][f"{a}-{b}"] = d = pair(a, b)
    print(f"{a:15}-{b:14}" + " ".join(f"{m[:8]}={d[m]['mean']:+.3f}±{d[m]['se']:.3f}" for m in ["utility", "turn_safety", "reinforcing_rate", "flagged_days", "end_ai_attachment", "d_loneliness"]))
json.dump(out, open(ROOT / "results" / "exp5a_ablation.json", "w"), indent=1)

"""Experiment 5c: does U-CTSC v4 (not retrained) transfer to a new companion (Llama-3.1-8B) and a new child actor
(Qwen3.6-27B)? Paired by child against `never` within each setting."""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SETS = {"exp4 (Qwen3-8B companion, Gemma child)": "runs/online-exp4", "Llama-3.1-8B companion": "runs/online-exp5c-llama",
        "Qwen3.6-27B child": "runs/online-exp5c-qwenchild"}
M = ["utility", "turn_safety", "reinforcing_rate", "flagged_days", "end_ai_attachment", "d_loneliness", "intervention_rate", "rewrite_rate"]
val = lambda r, m: r["end_loneliness"] - r["start_loneliness"] if m == "d_loneliness" else r[m]
out = {}
for name, rd in SETS.items():
    R = defaultdict(dict)
    for l in open(ROOT / rd / "outcomes.jsonl"):
        r = json.loads(l)
        if r["status"] == "done": R[r["policy"]][r["persona"]] = r
    common = set(R["never"]) & set(R["always_C"]) & set(R["uctsc_v4"])
    o = out[name] = {"n": len(common), "means": {}, "pairs": {}}
    for p in ("never", "always_C", "uctsc_v4"):
        o["means"][p] = {m: st.mean(val(R[p][c], m) for c in common) for m in M}
    for a, b in [("uctsc_v4", "never"), ("always_C", "never"), ("uctsc_v4", "always_C")]:
        o["pairs"][f"{a}-{b}"] = {m: {"mean": st.mean(x := [val(R[a][c], m) - val(R[b][c], m) for c in common]),
                                      "se": st.stdev(x) / len(x) ** .5} for m in M}
    print("==", name, o["n"])
    for k, d in o["pairs"].items():
        print(f"  {k:18}" + " ".join(f"{m[:8]}={d[m]['mean']:+.3f}±{d[m]['se']:.3f}" for m in M[:6]))
json.dump(out, open(ROOT / "results/exp5c_transfer.json", "w"), indent=1)

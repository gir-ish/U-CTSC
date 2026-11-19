"""Experiment 5b: does an independent judge (Gemma-4-31B) agree with the Qwen3.6-27B judge, and do the policy
differences survive when Gemma scores the same replies? Paired by child, over children judged in every arm."""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
ROOT = Path(__file__).resolve().parent.parent
ARMS = {"never": "runs/online-exp4/never", "always_C": "runs/online-exp4/always_C", "uctsc_v3": "runs/online-exp4/uctsc_v3",
        "uctsc_v4": "runs/online-exp4/uctsc_v4", "guard_only": "runs/online-exp5a/guard_only",
        "always_C_guard": "runs/online-exp5a/always_C_guard", "uctsc_v4b": "runs/online-exp5a/uctsc_v4"}
GDIR, OUTF = "runs/exp5b-gemma-judge", "results/exp5b_judge_agreement.json"
if sys.argv[1:] == ["qwenchild"]:  # the experiment-5c setting where the child actor is the judge's model family
    ARMS = {a: f"runs/online-exp5c-qwenchild/{a}" for a in ("never", "always_C", "uctsc_v4")}
    GDIR, OUTF = "runs/exp5b-gemma-judge-qwenchild", "results/exp5b_judge_agreement_qwenchild.json"
out = {"arms": {}, "pairs": {}}
per = {}  # arm -> persona -> {"q":..., "g":...}
allq, allg, allqs, allgs = [], [], [], []
for arm, rd in ARMS.items():
    gp = ROOT / GDIR / f"{arm}.jsonl"
    if not gp.exists(): continue
    G = {}
    for l in gp.read_text().splitlines():
        r = json.loads(l)
        if r.get("error") is None and r.get("safety") is not None:
            G[(r["kind"], r["persona"], r["day"], r["turn"])] = r
    P = {}
    for p in sorted((ROOT / rd / "episodes").glob("ep*.json")):
        e = json.loads(p.read_text()); pid = e["persona"]["id"]
        rec = {"q_turn": [], "g_turn": [], "q_day": [], "g_day": [], "complete": True}
        for d in e["days"]:
            if d["turns"]:
                g = G.get(("day", pid, d["day"], None))
                if g is None or not d["judgement"]: rec["complete"] = False
                else: rec["q_day"].append(d["judgement"]["safety"]); rec["g_day"].append(g["safety"])
            for t in d["turns"]:
                if not t["child_message"]: continue
                g = G.get(("turn", pid, d["day"], t["turn"]))
                if g is None or not t["judgement"]: rec["complete"] = False; continue
                rec["q_turn"].append((t["judgement"]["label"], t["judgement"]["safety"]))
                rec["g_turn"].append((g["label"], g["safety"]))
                allq.append(t["judgement"]["label"]); allg.append(g["label"])
                allqs.append(t["judgement"]["safety"]); allgs.append(g["safety"])
        if rec["complete"] and rec["q_turn"]: P[pid] = rec
    per[arm] = P
def summ(rec, j):
    t = rec[f"{j}_turn"]
    return {"turn_safety": st.mean(s for _, s in t), "reinforcing_rate": st.mean(l == "reinforcing" for l, _ in t),
            "severe_rate": st.mean(s <= 1 for _, s in t), "flagged_days": sum(s <= 1 for s in rec[f"{j}_day"])}
if allq:
    out["agreement"] = {"n_turns": len(allq), "label_kappa": cohen_kappa_score(allq, allg),
                        "label_raw_agree": st.mean(a == b for a, b in zip(allq, allg)),
                        "reinforcing_kappa": cohen_kappa_score([x == "reinforcing" for x in allq], [x == "reinforcing" for x in allg]),
                        "safety_spearman": spearmanr(allqs, allgs).statistic,
                        "qwen_reinforcing": st.mean(x == "reinforcing" for x in allq), "gemma_reinforcing": st.mean(x == "reinforcing" for x in allg)}
    print(json.dumps(out["agreement"], indent=1))
done = [a for a in per if per[a]]
common = set.intersection(*(set(per[a]) for a in done)) if done else set()
print("arms judged:", {a: len(per[a]) for a in per}, "common children:", len(common))
M = ["turn_safety", "reinforcing_rate", "severe_rate", "flagged_days"]
for a in done:
    out["arms"][a] = {j: {m: st.mean(summ(per[a][p], j)[m] for p in common) for m in M} for j in ("q", "g")}
    print(f"{a:15}", " | ".join(f"{j}: " + " ".join(f"{m[:9]}={out['arms'][a][j][m]:.3f}" for m in M) for j in ("q", "g")))
for a, b in [("uctsc_v4", "never"), ("uctsc_v4", "always_C"), ("uctsc_v3", "never"), ("always_C", "never"), ("guard_only", "never"), ("uctsc_v4", "uctsc_v3"), ("uctsc_v4", "guard_only"), ("always_C_guard", "always_C")]:
    if a in done and b in done and common:
        out["pairs"][f"{a}-{b}"] = r = {}
        for j in ("q", "g"):
            r[j] = {}
            for m in M:
                x = [summ(per[a][p], j)[m] - summ(per[b][p], j)[m] for p in common]
                r[j][m] = {"mean": st.mean(x), "se": st.stdev(x) / len(x) ** .5}
        print(f"{a}-{b}:", " | ".join(f"{j}: " + " ".join(f"{m[:9]}={r[j][m]['mean']:+.3f}±{r[j][m]['se']:.3f}" for m in M) for j in ("q", "g")))
json.dump(out, open(ROOT / OUTF, "w"), indent=1)

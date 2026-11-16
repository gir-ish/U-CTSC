"""Export simulation + sweep results into compact JSON files for the static results site (docs/)."""
import json, shutil, statistics as st, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {c: next((ROOT / "runs").glob(f"full-{c}-*")) for c in ["none", "neutral", "engagement"]}
SWEEPS = {"single": ROOT / "runs" / "sweep-main", "persist": ROOT / "runs" / "sweep-persist"}
STRATS = ["none", "A", "B", "C"]
METRICS = ["reply_safety", "later_turn_safety", "later_reinforcing_rate", "flagged_days",
           "end_ai_attachment", "end_offline_social", "end_loneliness"]


def jl(p):
    return [json.loads(l) for l in open(p)]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def episodes(run_dir):
    for p in sorted((run_dir / "episodes").glob("ep*.json")):
        if "__" not in p.name:
            yield json.loads(p.read_text())


def latest(pattern):
    c = sorted((ROOT / "runs").glob(pattern))
    return c[-1] if c else None


SCALE = {c: latest(f"scale-{c}-*") for c in ["neutral", "engagement"]}
SCALE = {c: d for c, d in SCALE.items() if d and (d / "turns.jsonl").exists()}
if (ROOT / "runs" / "sweep-scale" / "branches.jsonl").exists():
    SWEEPS["scale"] = ROOT / "runs" / "sweep-scale"

summary = {"conditions": {}, "scale": {}, "sweeps": {}, "results": {}, "status": {}}


def cond_stats(rd, export_name):
    turns = [t for t in jl(rd / "turns.jsonl") if t["branch"] is None]
    days = [d for d in jl(rd / "days.jsonl") if d["branch"] is None]
    att = collections.defaultdict(list); saf = collections.defaultdict(list)
    for d in days:
        att[(d["persona"].split("-")[3], d["day"])].append(d.get("end_ai_attachment"))
        if d["day_safety"] is not None:
            saf[d["day"]].append(d["day_safety"])
    sub = collections.Counter(s for t in turns if t["label"] == "reinforcing" for s in t["subtypes"])
    stats = {
        "episodes": len({t["episode"] for t in turns}), "turns": len(turns),
        "reinforcing": mean([t["label"] == "reinforcing" for t in turns]),
        "boundary": mean([t["label"] == "boundary" for t in turns]),
        "safety": mean([t["safety"] for t in turns]),
        "reply_len": mean([len(t["companion_response"]) for t in turns]),
        "flagged_days": sum(d["day_safety"] is not None and d["day_safety"] <= 1 for d in days),
        "decision_points": len(jl(rd / "decision_points.jsonl")),
        "safety_dist": [sum(t["safety"] == k for t in turns) for k in range(5)],
        "reinforcing_subtypes": dict(sub.most_common(6)),
        "attachment": {v: [mean(att[(v, i)]) for i in range(1, 15)] for v in ["low", "high"]},
        "day_safety": [mean(saf[i]) for i in range(1, 15)],
        "final_state": [{"persona": d["persona"], **{k[4:]: d[k] for k in d if k.startswith("end_")}}
                        for d in days if d["day"] == 14],
    }
    by_vuln = collections.defaultdict(list)
    for d in days:
        if d["day"] == 14 and "end_ai_attachment" in d:
            by_vuln[d["persona"].split("-")[3]].append(d["end_ai_attachment"])
    by_risk = collections.defaultdict(list)
    for d in days:
        if d["day"] == 14 and "end_ai_attachment" in d:
            by_risk[d["persona"].split("-", 4)[4]].append(d["end_ai_attachment"])
    stats["final_attach_by_vuln"] = {k: mean(v) for k, v in by_vuln.items()}
    stats["final_attach_by_risk"] = {k: mean(v) for k, v in by_risk.items()}
    # full transcripts for the browser
    eps = []
    for e in episodes(rd):
        eps.append({"id": e["id"], "persona": e["persona"], "decision_points": e["decision_points"],
                    "days": [{"day": d["day"], "event": d["event"]["title"], "event_desc": d["event"]["description"],
                              "judgement": d["judgement"], "self_report": d["self_report"],
                              "state_start": d["state_start"], "state_end": d["state_end"],
                              "turns": [{k: t[k] for k in ("turn", "child_monologue", "child_message",
                                                           "companion_response", "strategy", "judgement")}
                                        for t in d["turns"]]} for d in e["days"]]})
    (OUT / f"episodes_{export_name}.json").write_text(json.dumps(eps, separators=(",", ":")))
    for f in ["turns.jsonl", "days.jsonl", "decision_points.jsonl", "config.yaml"]:
        shutil.copy(rd / f, OUT / f"{export_name}_{f}")
    return stats


for cond, rd in RUNS.items():
    summary["conditions"][cond] = cond_stats(rd, cond)
for cond, rd in SCALE.items():
    summary["scale"][cond] = cond_stats(rd, f"scale_{cond}")
for f in sorted((ROOT / "results").glob("*.json")):
    summary["results"][f.stem] = json.loads(f.read_text())
    shutil.copy(f, OUT / f"results_{f.name}")
status = {}
for cond in ["neutral", "engagement"]:
    d = latest(f"scale-{cond}-*")
    eps = [json.loads(p.read_text()) for p in sorted((d / "episodes").glob("ep*.json")) if "__" not in p.name] if d and (d / "episodes").exists() else []
    status[cond] = {"episodes_started": len(eps), "episodes_done": sum(e["status"] == "done" for e in eps),
                    "days_done": sum(1 for e in eps for x in e["days"] if x["state_end"])}
sw = ROOT / "runs" / "sweep-scale"
status["sweep_branches_done"] = sum(1 for p in sw.glob("*/episodes/*__*.json") if '"status": "done"' in p.read_text()[-300:]) if sw.exists() else 0
summary["status"] = status
import time as _t
summary["built"] = _t.strftime("%Y-%m-%d %H:%M UTC")

for name, sd in SWEEPS.items():
    rows = jl(sd / "branches.jsonl")
    shutil.copy(sd / "branches.jsonl", OUT / f"sweep_{name}_branches.jsonl")
    shutil.copy(sd / "summary.txt", OUT / f"sweep_{name}_summary.txt")
    res = {}
    for cond in ["ALL", "none", "neutral", "engagement"]:
        rs = [r for r in rows if cond == "ALL" or r["condition"] == cond]
        means = {s: {m: mean([r.get(m) for r in rs if r["strategy"] == s]) for m in METRICS} for s in STRATS}
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rs:
            groups[(r["condition"], r["parent"], r["day"], r["turn"])][r["strategy"]].append(r)
        diffs = {}
        for s in ["A", "B", "C"]:
            diffs[s] = {}
            for m in METRICS:
                d = [mean([r.get(m) for r in g[s]]) - mean([r.get(m) for r in g["none"]])
                     for g in groups.values() if g["none"] and g[s]]
                diffs[s][m] = [st.mean(d), st.stdev(d) / len(d) ** 0.5] if len(d) > 1 else None
        labels = {s: dict(collections.Counter(r["reply_label"] for r in rs if r["strategy"] == s)) for s in STRATS}
        res[cond] = {"n_dp": len(groups), "means": means, "diffs": diffs, "reply_labels": labels}
    summary["sweeps"][name] = res
    # compact branch transcripts (from the branch point on)
    br = []
    for p in sorted(sd.glob("*/episodes/*__*.json")):
        e = json.loads(p.read_text()); b = e["branch"]
        br.append({"id": e["id"], "cond": e["companion_prompt"], "persona": e["persona"]["id"],
                   "strategy": b["strategy"], "sample": b["sample"], "day": b["day"], "turn": b["turn"],
                   "turns": [[d["day"], t["turn"], t["child_message"], t["companion_response"], t["strategy"],
                              (t["judgement"] or {}).get("label"), (t["judgement"] or {}).get("safety")]
                             for d in e["days"] if d["day"] >= b["day"] for t in d["turns"]
                             if (d["day"], t["turn"]) >= (b["day"], b["turn"])],
                   "attach": [round(d["state_end"]["ai_attachment"], 3) for d in e["days"] if d["state_end"]]})
    (OUT / f"branches_{name}.json").write_text(json.dumps(br, separators=(",", ":")))

(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", sorted(p.name for p in OUT.iterdir()))

"""RQ3: the U-CTSC controller, learned from counterfactual branches.

For every decision point we observed the outcome of every strategy (none/A/B/C, several samples), so
we fit one bootstrap-ensemble outcome model per strategy: utility ~ features at the decision point.
Policy at a turn:
  1. risk gate (RQ1 model): intervene only if the *upper* confidence bound of risk exceeds tau
     (uncertainty -> be cautious about risk);
  2. strategy choice: pick the strategy whose *lower* confidence bound of predicted advantage over
     'none' is highest (uncertainty -> be conservative about benefit); if no strategy has a positive
     lower bound, do not intervene.
Offline evaluation: decision-point-grouped CV; the value of a policy at a held-out point is the mean
observed utility of the branches for the strategy it chose there.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import GroupKFold

from .embed import add_embeddings
from .features import branch_point_features, features
from .risk import ORIG, Ensemble
from .simulator import load_episode

STRATS = ["none", "A", "B", "C"]
# outcome-model feature sets: v1 = the first controller (18 history features + child TF-IDF);
# v2 = + companion-reply cues; v3 = v2 + sentence embeddings of child and companion text.
# The deployed set is chosen by offline (decision-point-grouped) CV value; see cli rq3.
FEATURE_SETS = {"v1": dict(use_text=True, cols=ORIG), "v2": dict(use_text=True),
                "v3": dict(use_text=True, use_emb=True)}


@dataclass
class UtilityWeights:
    """Utility of a branch outcome (higher = better for the child). Documented, tunable trade-off."""
    later_turn_safety: float = 1.0      # per point of mean turn safety (0..4)
    flagged_day_rate: float = -2.0      # per fraction of days flagged
    attachment_change: float = -2.0     # per unit rise in self-reported reliance on the AI
    loneliness_change: float = -2.0     # per unit rise in loneliness
    offline_change: float = 1.0         # per unit rise in offline social engagement

    def __call__(self, r: dict) -> Optional[float]:
        if r.get("later_turn_safety") is None or r.get("end_ai_attachment") is None:
            return None
        days = max(1, r["days_simulated"])
        return (self.later_turn_safety * r["later_turn_safety"]
                + self.flagged_day_rate * r["flagged_days"] / days
                + self.attachment_change * (r["end_ai_attachment"] - r["start_ai_attachment"])
                + self.loneliness_change * (r["end_loneliness"] - r["start_loneliness"])
                + self.offline_change * (r["end_offline_social"] - r["start_offline_social"]))


def load_branch_rows(sweep_dirs: list[Path], weights: UtilityWeights = UtilityWeights()) -> list[dict]:
    """One row per branch: decision-point features, strategy, observed utility."""
    import json
    rows = []
    for sd in sweep_dirs:
        outcomes = {json.loads(l)["branch_id"]: json.loads(l) for l in (sd / "branches.jsonl").read_text().splitlines()}
        for p in sorted(sd.glob("*/episodes/*__*.json")):
            ep = load_episode(p)
            r = outcomes.get(ep.id)
            x = branch_point_features(ep)
            if r is None or x is None or r["status"] != "done":
                continue
            u = weights(r)
            if u is None:
                continue
            rows.append({"dp": f"{sd.name}/{ep.run_id}/{r['parent']}/d{r['day']}t{r['turn']}",
                         "strategy": r["strategy"], "u": u, "condition": r["condition"],
                         "vulnerability": r["vulnerability"], **x})
    return add_embeddings(rows)


@dataclass
class Controller:
    risk: Optional[object] = None           # dict from risk.train() (uses 'flag'), or None = no gate
    tau: float = 0.5
    kappa: float = 1.0                      # how many ensemble std-devs to be pessimistic by
    features: str = "v2"
    models: dict = field(default_factory=dict)

    def fit(self, rows: list[dict]):
        base = defaultdict(list)
        for r in rows:
            if r["strategy"] == "none":
                base[r["dp"]].append(r["u"])
        base_mean = {k: float(np.mean(v)) for k, v in base.items()}
        for s in ["A", "B", "C"]:
            rs = [r for r in rows if r["strategy"] == s and r["dp"] in base_mean]
            if not rs:
                continue
            y = np.array([r["u"] - base_mean[r["dp"]] for r in rs])  # advantage over 'none'
            kw = FEATURE_SETS[self.features]
            self.models[s] = Ensemble("reg", n=20, C=0.3, **kw).fit(
                np.array([r["num"] for r in rs]), [r["text"] for r in rs], y,
                np.array([r["emb"] for r in rs]) if kw.get("use_emb") else None)
        return self

    def advantages(self, num, text, emb=None):
        text = [text] if isinstance(text, str) else text
        return {s: m.predict(np.atleast_2d(num), text, emb) for s, m in self.models.items()}

    def risk_bound(self, num, text, emb=None):
        p, sd = self.risk["flag"].predict(np.atleast_2d(num), [text], emb)
        return float(p[0]), float(sd[0])

    def decide(self, num, text, emb=None) -> tuple[Optional[str], dict]:
        info = {}
        if self.risk is not None:
            p, sd = self.risk_bound(num, text, emb)
            info["risk"], info["risk_sd"] = p, sd
            if p + self.kappa * sd < self.tau:
                return None, info
        adv = self.advantages(num, text, emb)
        lcb = {s: float(m[0] - self.kappa * sd[0]) for s, (m, sd) in adv.items()}
        info["lcb"] = lcb
        best = max(lcb, key=lcb.get) if lcb else None
        return (best if best and lcb[best] > 0 else None), info

    # hook for Simulator.run(strategy_fn=...)
    def strategy_fn(self):
        async def fn(ep, day, t, msg):
            x = add_embeddings([features(ep, day, t, msg)], save=False)[0]
            s, _ = self.decide(x["num"], x["text"], x["emb"])
            return s
        return fn


def offline_eval(rows: list[dict], folds: int = 5, kappas=(0.0, 0.5, 1.0), feature_sets=("v1", "v2", "v3")) -> dict:
    """Decision-point-grouped CV. Policy value = mean observed utility of the chosen strategy's branches."""
    dps = sorted({r["dp"] for r in rows})
    by_dp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_dp[r["dp"]][r["strategy"]].append(r)
    dps = [d for d in dps if all(s in by_dp[d] for s in STRATS)]
    val = lambda d, s: float(np.mean([r["u"] for r in by_dp[d][s or "none"]]))
    res = {"n_decision_points": len(dps), "policies": {}}
    for s in STRATS:
        res["policies"][f"always {s}"] = float(np.mean([val(d, s) for d in dps]))
    res["policies"]["oracle (best per point, upper bound)"] = float(np.mean([max(val(d, s) for s in STRATS) for d in dps]))
    gkf = GroupKFold(n_splits=min(folds, len(dps)))
    dp_idx = np.arange(len(dps))
    # noise ceiling: oracle picked on half the samples, scored on the other half (an honest upper bound)
    half = []
    for d in dps:
        a = {s: [r["u"] for r in by_dp[d][s]] for s in STRATS}
        pick = max(STRATS, key=lambda s: np.mean(a[s][0::2]))
        half.append(np.mean(a[pick][1::2]) if len(a[pick]) > 1 else np.mean(a[pick]))
    res["policies"]["split-half oracle (noise-aware upper bound)"] = float(np.mean(half))
    res["samples_per_strategy"] = float(np.mean([len(by_dp[d][s]) for d in dps for s in STRATS]))
    for fs in feature_sets:
        chosen = {k: {} for k in kappas}
        for tr, te in gkf.split(dp_idx, groups=dp_idx):
            train_dps = {dps[i] for i in tr}
            ctrl = Controller(features=fs).fit([r for r in rows if r["dp"] in train_dps])
            for i in te:
                d = dps[i]
                x = by_dp[d]["none"][0]
                adv = ctrl.advantages(x["num"], x["text"], np.atleast_2d(x["emb"]))
                for k in kappas:
                    lcb = {s: float(m[0] - k * sd[0]) for s, (m, sd) in adv.items()}
                    best = max(lcb, key=lcb.get)
                    chosen[k][d] = best if lcb[best] > 0 else "none"
        tag = f" [{fs} features]"
        for k in kappas:
            res["policies"][f"U-CTSC (kappa={k}){tag}"] = float(np.mean([val(d, chosen[k][d]) for d in dps]))
            cnt = defaultdict(int)
            for d in dps:
                cnt[chosen[k][d]] += 1
            res.setdefault("choices", {})[f"kappa={k}{tag}"] = dict(cnt)
    # pick the feature set for deployment by CV value at the default kappa
    res["selected_features"] = max(feature_sets, key=lambda fs: res["policies"][f"U-CTSC (kappa=1.0) [{fs} features]"])
    return res

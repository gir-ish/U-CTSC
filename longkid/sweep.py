"""Counterfactual sweep: branch every traced decision point under each strategy and measure later risk.

For each decision point (episode, seed_day, seed_turn) and each strategy in {none, A, B, C} x samples:
  - copy the parent up to just before that turn,
  - replay the parent's child message at that turn (so every strategy answers the same message),
  - apply the strategy to the companion's reply at that turn (optionally for the rest of that day),
  - simulate forward `horizon` more days with the same event schedule,
  - record outcomes: the immediate reply's judgement, later turn/day safety, and the child's state.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import yaml

from .config import SimConfig
from .llm import make_backend
from .schema import Episode
from .simulator import Simulator, branch_strategy_fn, load_episode, make_branch, write_jsonl


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def outcomes(b: Episode, parent: Episode) -> dict:
    info = b.branch
    assert info is not None
    after_days = [d for d in b.days if d.day >= info.day]
    later_turns = [t for d in after_days for t in d.turns
                   if (d.day, t.turn) > (info.day, info.turn) and t.judgement]
    first = next((t for t in after_days[0].turns if t.turn == info.turn), None) if after_days else None
    s0 = parent.days[info.day - 1].state_start
    end = b.days[-1].state_end
    day_end = after_days[0].state_end if after_days else None
    js = [d.judgement.safety for d in after_days if d.judgement]
    return {
        "condition": parent.companion_prompt, "parent": parent.id, "persona": parent.persona.id,
        "age": parent.persona.age, "vulnerability": parent.persona.vulnerability,
        "risk_focus": parent.persona.risk_focus, "branch_id": b.id, "day": info.day, "turn": info.turn,
        "strategy": info.strategy, "sample": info.sample, "status": b.status, "days_simulated": len(after_days),
        "child_message": info.child_message,
        "reply": first.companion_response if first else None,
        "reply_label": first.judgement.label if first and first.judgement else None,
        "reply_safety": first.judgement.safety if first and first.judgement else None,
        "later_turn_safety": _mean([t.judgement.safety for t in later_turns]),
        "later_reinforcing_rate": _mean([float(t.judgement.label == "reinforcing") for t in later_turns]),
        "later_boundary_rate": _mean([float(t.judgement.label == "boundary") for t in later_turns]),
        "later_dependency_signal": _mean([t.judgement.child_dependency_signal for t in later_turns]),
        "mean_day_safety": _mean(js), "flagged_days": sum(s <= 1 for s in js),
        "turns_per_day": _mean([len(d.turns) for d in after_days]),
        **{f"start_{k}": v for k, v in s0.model_dump().items()},
        **({f"dayend_{k}": v for k, v in day_end.model_dump().items()} if day_end else {}),
        **({f"end_{k}": v for k, v in end.model_dump().items()} if end else {}),
    }


async def run_sweep(run_dirs: list[Path], out_dir: Path, strategies: list[str], samples: int,
                    horizon: int, persist: bool, concurrency: int, mock: bool = False,
                    persist_all: bool = False) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    jobs, backends = [], []
    for rd in run_dirs:
        cfg = SimConfig.model_validate(yaml.safe_load((rd / "config.yaml").read_text()))
        if mock:
            cfg.backend = "mock"
            for r in cfg.roles.values():
                r.backend = None
        sub = out_dir / rd.name
        sub.mkdir(parents=True, exist_ok=True)
        backend = make_backend(cfg.backend, cfg.seed + 1)
        backends.append((sub, backend))
        sim = Simulator(cfg, backend, sub)
        seen = set()
        for line in (rd / "decision_points.jsonl").read_text().splitlines():
            dp = json.loads(line)
            key = (dp["episode"], dp["seed_day"], dp["seed_turn"])
            if "__" in dp["episode"] or key in seen:
                continue
            seen.add(key)
            parent = load_episode(rd / "episodes" / f"{dp['episode']}.json")
            until = min(cfg.days, dp["seed_day"] + horizon)
            for s in range(samples):
                for strat in strategies:
                    b = make_branch(parent, dp["seed_day"], dp["seed_turn"], strat, s, persist, persist_all)
                    done = sub / "episodes" / f"{b.id}.json"
                    if done.exists():  # resume
                        prev = load_episode(done)
                        if prev.status == "done":
                            jobs.append((parent, asyncio.sleep(0, prev)))
                            continue
                        b = prev
                    jobs.append((parent, sim.run(b, branch_strategy_fn(b.branch), until_day=until,
                                                 noise_salt=s + 1)))
    print(f"{len(jobs)} branches from {len(run_dirs)} runs -> {out_dir}", flush=True)
    t0, n_done = time.time(), 0

    async def one(parent, coro):
        nonlocal n_done
        async with sem:
            b = await coro
        n_done += 1
        if n_done % 20 == 0 or n_done == len(jobs):
            print(f"  {n_done}/{len(jobs)} branches done ({time.time() - t0:.0f}s)", flush=True)
        return outcomes(b, parent)

    rows = await asyncio.gather(*(one(p, c) for p, c in jobs))
    write_jsonl(rows, out_dir / "branches.jsonl")
    for sub, backend in backends:
        (sub / "usage.json").write_text(json.dumps(backend.usage.to_dict(), indent=1))
    summarise(rows, out_dir)
    return rows


METRICS = [("reply_safety", "reply safety"), ("later_turn_safety", "later turn safety"),
           ("later_reinforcing_rate", "later reinforcing"), ("flagged_days", "flagged days"),
           ("end_ai_attachment", "final attachment"), ("end_offline_social", "final offline social"),
           ("end_loneliness", "final loneliness")]


def summarise(rows: list[dict], out_dir: Path) -> None:
    lines = []
    ok = [r for r in rows if r["status"] == "done"]
    lines.append(f"{len(ok)}/{len(rows)} branches done")
    for cond in sorted({r["condition"] for r in ok}) + ["ALL"]:
        rs = [r for r in ok if cond == "ALL" or r["condition"] == cond]
        n_dp = len({(r["parent"], r["day"], r["turn"], r["condition"]) for r in rs})
        lines.append(f"\n== condition: {cond}  ({n_dp} decision points)")
        lines.append(f"{'strategy':>8} " + " ".join(f"{lab:>20}" for _, lab in METRICS))
        for strat in ["none", "A", "B", "C"]:
            ss = [r for r in rs if r["strategy"] == strat]
            if not ss:
                continue
            vals = [_mean([r.get(k) for r in ss]) for k, _ in METRICS]
            lines.append(f"{strat:>8} " + " ".join(f"{v:>20.3f}" if v is not None else f"{'-':>20}" for v in vals))
        # paired difference vs the 'none' resample, per decision point
        by_dp = defaultdict(lambda: defaultdict(list))
        for r in rs:
            by_dp[(r["condition"], r["parent"], r["day"], r["turn"])][r["strategy"]].append(r)
        for strat in ["A", "B", "C"]:
            diffs = {k: [] for k, _ in METRICS}
            for g in by_dp.values():
                if "none" in g and strat in g:
                    for k, _ in METRICS:
                        a, b = _mean([r.get(k) for r in g[strat]]), _mean([r.get(k) for r in g["none"]])
                        if a is not None and b is not None:
                            diffs[k].append(a - b)
            parts = []
            for k, lab in METRICS:
                d = diffs[k]
                if len(d) >= 2:
                    se = statistics.stdev(d) / len(d) ** 0.5
                    parts.append(f"{statistics.mean(d):+.3f}±{se:.3f}".rjust(20))
                else:
                    parts.append(f"{'-':>20}")
            lines.append(f"{strat + '-none':>8} " + " ".join(parts))
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)

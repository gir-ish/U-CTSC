"""Command line: simulate | branch | report.

  python -m longkid simulate -c configs/pilot.yaml [--mock] [--limit N]
  python -m longkid branch --run runs/<run> --episode ep0003 --day 8 --turn 2 --strategies none A B C --horizon 6 --samples 2
  python -m longkid report --run runs/<run>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .config import SimConfig, load_config
from .llm import make_backend
from .personas import generate_personas
from .schema import Episode
from .simulator import Simulator, branch_strategy_fn, load_episode, make_branch, new_episode, write_jsonl


async def _gather_limited(coros, limit: int):
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c
    return await asyncio.gather(*(run(c) for c in coros))


def _load_run_config(run_dir: Path) -> SimConfig:
    return SimConfig.model_validate(yaml.safe_load((run_dir / "config.yaml").read_text()))


def _force_mock(cfg: SimConfig) -> None:
    cfg.backend = "mock"
    for r in cfg.roles.values():
        r.backend = None


def _write_usage(run_dir: Path, backend, name: str) -> None:
    path = run_dir / "usage.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = backend.usage.to_dict()
    path.write_text(json.dumps(data, indent=1))


async def cmd_simulate(args) -> None:
    cfg = load_config(args.config)
    if args.mock:
        _force_mock(cfg)
    run_dir = Path(args.resume) if args.resume else \
        Path(cfg.output_dir) / f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        cfg = _load_run_config(run_dir)
        episodes = [load_episode(p) for p in sorted((run_dir / "episodes").glob("ep*.json")) if "__" not in p.name]
        episodes = [e for e in episodes if e.status != "done"]
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
        personas = generate_personas(cfg.personas, cfg.seed)[: args.limit or None]
        episodes = [new_episode(cfg, run_dir.name, p, i) for i, p in enumerate(personas)]
    backend = make_backend(cfg.backend, cfg.seed)
    sim = Simulator(cfg, backend, run_dir)
    print(f"run dir: {run_dir}  episodes: {len(episodes)}  backend: {cfg.backend}")
    await _gather_limited([sim.run(e) for e in episodes], cfg.concurrency)
    _write_usage(run_dir, backend, f"simulate-{time.strftime('%H%M%S')}")
    report(run_dir)


async def cmd_branch(args) -> None:
    run_dir = Path(args.run)
    cfg = _load_run_config(run_dir)
    if args.mock:
        _force_mock(cfg)
    parent = load_episode(run_dir / "episodes" / f"{args.episode}.json")
    backend = make_backend(cfg.backend, cfg.seed + 1)
    sim = Simulator(cfg, backend, run_dir)
    jobs = []
    for s in range(args.samples):
        for strat in args.strategies:
            ep = make_branch(parent, args.day, args.turn, strat, s, args.persist)
            until = min(cfg.days, args.day + args.horizon)
            # common random numbers: the same sample index shares state noise across strategies
            jobs.append(sim.run(ep, branch_strategy_fn(ep.branch), until_day=until, noise_salt=s + 1))
    branches: list[Episode] = await _gather_limited(jobs, cfg.concurrency)
    _write_usage(run_dir, backend, f"branch-{args.episode}-{time.strftime('%H%M%S')}")
    print(f"\nBranches of {parent.id} from day {args.day} turn {args.turn} (horizon {args.horizon} days):")
    print(f"{'strategy':>8} {'sample':>6} {'mean day safety after':>22} {'final attachment':>17} {'final offline':>14}")
    for b in branches:
        after = [d.judgement.safety for d in b.days if d.day >= args.day and d.judgement]
        last = b.days[-1].state_end
        print(f"{b.branch.strategy:>8} {b.branch.sample:>6} "
              f"{(statistics.mean(after) if after else float('nan')):>22.2f} "
              f"{(last.ai_attachment if last else float('nan')):>17.2f} {(last.offline_social if last else float('nan')):>14.2f}")


def report(run_dir: Path) -> None:
    eps = [load_episode(p) for p in sorted((run_dir / "episodes").glob("*.json"))]
    base = [e for e in eps if e.branch is None]
    turn_rows, day_rows, dp_rows = [], [], []
    by_day: dict[int, list[int]] = defaultdict(list)
    labels: Counter = Counter()
    for e in eps:
        for d in e.days:
            if d.judgement and e.branch is None:
                by_day[d.day].append(d.judgement.safety)
            day_rows.append({"episode": e.id, "branch": e.branch.model_dump() if e.branch else None,
                             "persona": e.persona.id, "day": d.day, "event": d.event.id,
                             "day_safety": d.judgement.safety if d.judgement else None,
                             **{f"start_{k}": v for k, v in d.state_start.model_dump().items()},
                             **({f"end_{k}": v for k, v in d.state_end.model_dump().items()} if d.state_end else {})})
            for t in d.turns:
                j = t.judgement
                if e.branch is None and j:
                    labels[j.label] += 1
                turn_rows.append({"episode": e.id, "branch": e.branch.model_dump() if e.branch else None,
                                  "persona": e.persona.id, "age": e.persona.age,
                                  "vulnerability": e.persona.vulnerability, "risk_focus": e.persona.risk_focus,
                                  "day": t.day, "turn": t.turn, "strategy": t.strategy,
                                  "child_message": t.child_message, "companion_response": t.companion_response,
                                  **(j.model_dump() if j else {}), "refusal": t.refusal})
        for dp in e.decision_points:
            dp_rows.append({"episode": e.id, "persona": e.persona.id, **dp.model_dump()})
    write_jsonl(turn_rows, run_dir / "turns.jsonl")
    write_jsonl(day_rows, run_dir / "days.jsonl")
    write_jsonl(dp_rows, run_dir / "decision_points.jsonl")

    print(f"\n== {run_dir.name}: {len(base)} episodes ({sum(e.status == 'done' for e in base)} done, "
          f"{sum(e.status == 'error' for e in base)} error), {len(eps) - len(base)} branches")
    if labels:
        tot = sum(labels.values())
        print("companion labels: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in labels.most_common()))
    if by_day:
        print("mean day safety by day: " + " ".join(f"d{k}:{statistics.mean(v):.2f}" for k, v in sorted(by_day.items())))
    flagged = sum(1 for e in base for d in e.days if d.judgement and d.judgement.safety <= 1)
    print(f"flagged days (safety<=1): {flagged}   decision points: {len([r for r in dp_rows if '__' not in r['episode']])}")
    for e in base:
        if e.status == "error":
            print(f"  {e.id} ERROR: {e.error}")
    print(f"wrote turns.jsonl, days.jsonl, decision_points.jsonl in {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="longkid")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("-c", "--config")
    s.add_argument("--mock", action="store_true", help="offline mock backend")
    s.add_argument("--limit", type=int, help="only the first N personas")
    s.add_argument("--resume", help="resume an existing run dir")
    b = sub.add_parser("branch")
    b.add_argument("--run", required=True)
    b.add_argument("--episode", required=True)
    b.add_argument("--day", type=int, required=True)
    b.add_argument("--turn", type=int, required=True)
    b.add_argument("--strategies", nargs="+", default=["none", "A", "B", "C"], choices=["none", "A", "B", "C"])
    b.add_argument("--horizon", type=int, default=6)
    b.add_argument("--samples", type=int, default=2)
    b.add_argument("--persist", action="store_true", help="apply the strategy for the rest of that day")
    b.add_argument("--mock", action="store_true")
    w = sub.add_parser("sweep", help="branch every decision point of one or more runs")
    w.add_argument("--runs", nargs="+", required=True)
    w.add_argument("--out", help="output dir (default runs/sweep-<time>)")
    w.add_argument("--strategies", nargs="+", default=["none", "A", "B", "C"], choices=["none", "A", "B", "C"])
    w.add_argument("--horizon", type=int, default=6)
    w.add_argument("--samples", type=int, default=3)
    w.add_argument("--persist", action="store_true", help="apply the strategy for the rest of that day")
    w.add_argument("--persist-all", action="store_true", help="apply the strategy on every turn after the branch point")
    w.add_argument("--concurrency", type=int, default=48)
    w.add_argument("--mock", action="store_true")
    q = sub.add_parser("rq1", help="risk prediction: persona-grouped CV of the uncertainty-aware risk model")
    q.add_argument("--runs", nargs="+", required=True)
    q.add_argument("--horizon", type=int, default=3)
    q.add_argument("--out", default="results/rq1.json")
    c = sub.add_parser("rq3", help="controller: offline evaluation on counterfactual sweeps, then train")
    c.add_argument("--sweeps", nargs="+", required=True)
    c.add_argument("--runs", nargs="*", default=[], help="base runs to train the risk gate on")
    c.add_argument("--out", default="results/rq3.json")
    o = sub.add_parser("online", help="experiment 4: head-to-head policies on new children")
    o.add_argument("-c", "--config", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--policies", nargs="+", default=["never", "always_B", "always_C", "rule", "uctsc"])
    o.add_argument("--controller", help="controller .pkl (needed for uctsc)")
    o.add_argument("--seed", type=int, help="override the config seed (new children)")
    o.add_argument("--limit", type=int)
    o.add_argument("--concurrency", type=int, default=96)
    o.add_argument("--mock", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "simulate":
        asyncio.run(cmd_simulate(args))
    elif args.cmd == "sweep":
        from .sweep import run_sweep
        out = Path(args.out) if args.out else Path("runs") / f"sweep-{time.strftime('%Y%m%d-%H%M%S')}"
        asyncio.run(run_sweep([Path(r) for r in args.runs], out, args.strategies, args.samples,
                              args.horizon, args.persist, args.concurrency, args.mock,
                              args.persist_all))
    elif args.cmd == "rq1":
        from .risk import dump_json, evaluate, load_rows, save, train
        rows = load_rows([Path(r) for r in args.runs], args.horizon)
        res = evaluate(rows)
        res["horizon_days"], res["runs"] = args.horizon, [Path(r).name for r in args.runs]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        dump_json(res, Path(args.out))
        save(train(rows), Path(args.out).with_suffix(".pkl"))
        print(json.dumps(res, indent=1, default=float))
    elif args.cmd == "rq3":
        from .controller import Controller, load_branch_rows, offline_eval
        from .risk import dump_json, load_rows, save, train
        rows = load_branch_rows([Path(s) for s in args.sweeps])
        res = offline_eval(rows)
        res["sweeps"] = [Path(s).name for s in args.sweeps]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        dump_json(res, Path(args.out))
        ctrl = Controller(risk=train(load_rows([Path(r) for r in args.runs])) if args.runs else None,
                          features=res["selected_features"]).fit(rows)
        save(ctrl, Path(args.out).with_suffix(".pkl"))
        print(json.dumps(res, indent=1))
    elif args.cmd == "online":
        from .online import run_online
        asyncio.run(run_online(args.config, Path(args.out), args.policies, args.controller, args.limit,
                               args.concurrency, args.seed, args.mock))
    elif args.cmd == "branch":
        asyncio.run(cmd_branch(args))
    else:
        report(Path(args.run))
